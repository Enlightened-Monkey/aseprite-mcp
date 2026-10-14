"""AI-assisted element segmentation for pixel art (optional "ai" extra).

Pipeline:
 1. name    - a local vision-language model (Ollama, e.g. gemma3) lists the
              elements a designer would put on separate layers
 2. find    - an open-vocabulary detector (Grounding DINO) turns each name
              into boxes
 3. cut     - SAM 2.1 turns each box into a mask
 4. snap    - masks are brought back to the pixel grid (majority vote per
              pixel) and to the art's own colour clusters
 5. fill in - classical pixel-art analysis adds what models miss: repeated
              single-pixel particles (snow, stars) and the backdrop

Models run on the upscaled image (nearest neighbour): they were trained on
photos and need objects bigger than a few pixels.
Requires: pip install aseprite-mcp[ai]  (torch, transformers) and, for
naming, a running Ollama with a vision model.
"""
import base64
import io
import json
import urllib.request
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..pixelart import lint as L
from ..pixelart import segment as SG

SAM_MODEL = "facebook/sam2.1-hiera-large"
DETECTOR_MODEL = "IDEA-Research/grounding-dino-base"
OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class Element:
    name: str
    mask: np.ndarray
    score: float = 0.0
    source: str = "ai"
    boxes: list = field(default_factory=list)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def upscale(img: np.ndarray, target: int = 1024, smooth: bool = True) -> tuple[Image.Image, int]:
    """Upscale by an integer factor so the longer side is about `target`.

    smooth (default) uses Lanczos: detectors and SAM were trained on photos
    and see blocky nearest-neighbour pixels as noise (on a pixel-art scene a
    detector's confidence for "woolly mammoth" rose from 0.24 to 0.44).
    Masks are snapped back to the pixel grid afterwards anyway.
    """
    s = max(1, target // max(img.shape[:2]))
    rgb = img.copy()
    rgb[rgb[..., 3] == 0] = (255, 255, 255, 255)  # models expect no alpha
    im = Image.fromarray(rgb[..., :3])
    return im.resize((im.width * s, im.height * s), Image.LANCZOS if smooth else Image.NEAREST), s


# ------------------------------------------------------------- naming

def name_elements(img: np.ndarray, model: str = "gemma3:12b", max_elements: int = 12) -> list[str]:
    """Ask a local vision model which elements the picture contains."""
    im, _ = upscale(img, 768, smooth=False)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    prompt = ("This is a pixel-art image. List the distinct visual elements a designer would put on separate "
              "layers (characters, creatures, objects, effects, background parts). Answer ONLY with JSON: "
              '{"elements":["short concrete noun phrase", ...]}. Use 4-' + str(max_elements) + " elements, "
              "each a concrete noun phrase an object detector understands (e.g. 'woolly mammoth', 'rider', "
              "'spear', 'rocks', 'pine trees'). Do not list the same thing twice.")
    body = {"model": model, "prompt": prompt, "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False, "format": "json", "keep_alive": 0, "options": {"temperature": 0.2}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    data = json.loads(json.loads(urllib.request.urlopen(req, timeout=600).read())["response"])
    names = []
    for e in data.get("elements", []):
        n = (e["name"] if isinstance(e, dict) else str(e)).strip().lower()
        if n and n not in names:
            names.append(n)
    return names[:max_elements]


# ------------------------------------------------------------ detection

def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect_boxes(image: Image.Image, names: list[str], min_score: float = 0.2,
                 relative: float = 0.6, max_per_name: int = 4) -> dict[str, list]:
    """Grounding DINO: {name: [(score, [x0, y0, x1, y1]), ...]} in image px.

    Pixel art scores lower than photos, so thresholds are relative: per
    name keep boxes scoring at least `relative` of the best one (and at least
    min_score), without near-duplicates (IoU > 0.7).
    """
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    dev = _device()
    proc = AutoProcessor.from_pretrained(DETECTOR_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL).to(dev)
    out: dict[str, list] = {}
    for n in names:  # one query per name keeps labels unambiguous
        inputs = proc(images=image, text=n.rstrip(".") + ".", return_tensors="pt").to(dev)
        with torch.no_grad():
            res = model(**inputs)
        det = proc.post_process_grounded_object_detection(
            res, inputs.input_ids, threshold=min_score, text_threshold=min_score,
            target_sizes=[image.size[::-1]])[0]
        found = sorted(zip(det["scores"].tolist(), det["boxes"].tolist()), key=lambda d: -d[0])
        kept = []
        for score, box in found:
            if found and score < relative * found[0][0]:
                break
            if all(_iou(box, k[1]) < 0.7 for k in kept):
                kept.append((score, box))
            if len(kept) >= max_per_name:
                break
        out[n] = kept
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


# -------------------------------------------------------------- SAM 2

class Sam:
    def __init__(self, model_id: str = SAM_MODEL):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.dev = _device()
        self.proc = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.dev)

    def boxes(self, image: Image.Image, boxes: list) -> list[tuple[np.ndarray, float]]:
        """One mask per box (the best of SAM's candidates)."""
        if not boxes:
            return []
        inputs = self.proc(images=image, input_boxes=[[list(map(float, b)) for b in boxes]],
                           return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = out.iou_scores.cpu()[0]
        res = []
        for i in range(masks.shape[0]):
            j = int(scores[i].argmax())
            res.append((masks[i, j].numpy().astype(bool), float(scores[i, j])))
        return res

    def prompt(self, image: Image.Image, positive: list, negative: list = (),
               box: list | None = None, max_area: float = 0.4) -> tuple[np.ndarray, float]:
        """One mask from positive points (inside the element), negative
        points (inside other elements) and an optional box.

        SAM proposes three masks (part, object, whole); the most confident
        one smaller than max_area of the image wins, so a click on a rock
        does not return the whole ground. SAM was trained with a few points
        per prompt - pass only nearby negatives (see nearest()).
        """
        pts = [list(map(float, p)) for p in positive] + [list(map(float, p)) for p in negative]
        lbl = [1] * len(positive) + [0] * len(negative)
        kwargs = {"input_points": [[pts]], "input_labels": [[lbl]]}
        if box is not None:
            kwargs["input_boxes"] = [[list(map(float, box))]]
        inputs = self.proc(images=image, return_tensors="pt", **kwargs).to(self.dev)
        with self.torch.no_grad():
            out = self.model(**inputs, multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0]
        scores = out.iou_scores.cpu()[0, 0]
        area = image.size[0] * image.size[1]
        best = (np.zeros(masks.shape[1:], dtype=bool), 0.0)
        for j in range(masks.shape[0]):
            m = masks[j].numpy().astype(bool)
            if m.sum() <= max_area * area and float(scores[j]) > best[1]:
                best = (m, float(scores[j]))
        return best

    def points_grid(self, image: Image.Image, n: int = 24, batch: int = 64,
                    min_score: float = 0.85) -> list[tuple[np.ndarray, float]]:
        """Automatic masks: one positive point per grid cell."""
        w, h = image.size
        pts = [((x + 0.5) * w / n, (y + 0.5) * h / n) for y in range(n) for x in range(n)]
        res = []
        for i in range(0, len(pts), batch):
            chunk = pts[i:i + batch]
            inputs = self.proc(images=image, input_points=[[[list(p)] for p in chunk]],
                               input_labels=[[[1] for _ in chunk]], return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                out = self.model(**inputs, multimask_output=True)
            masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
            scores = out.iou_scores.cpu()[0]
            for k in range(masks.shape[0]):
                j = int(scores[k].argmax())
                if float(scores[k, j]) >= min_score:
                    res.append((masks[k, j].numpy().astype(bool), float(scores[k, j])))
        return res


def dedupe(masks: list[tuple[np.ndarray, float]], iou: float = 0.8) -> list[tuple[np.ndarray, float]]:
    """Drop masks that mostly repeat a better one."""
    kept = []
    for m, s in sorted(masks, key=lambda ms: -ms[1]):
        if m.sum() < 4:
            continue
        if all((m & k).sum() / max(1, (m | k).sum()) < iou for k, _ in kept):
            kept.append((m, s))
    return kept


# ------------------------------------------------------------- pipeline

def segment_elements(img: np.ndarray, names: list[str] | None = None, vlm: str = "gemma3:12b",
                     min_score: float = 0.2) -> tuple[np.ndarray, list[Element]]:
    """Label every pixel with an element; returns (labels, elements).

    labels: index into elements, -1 = transparent. The last elements are
    "falling particles" (when found) and "backdrop" (everything left).
    """
    image, s = upscale(img)
    names = names or name_elements(img, vlm)
    boxes = detect_boxes(image, names, min_score)
    sam = Sam()
    elements: list[Element] = []
    for n in names:
        found = boxes.get(n, [])
        if not found:
            continue
        masks = sam.boxes(image, [b for _, b in found])
        union = np.zeros(img.shape[:2], dtype=bool)
        best = 0.0
        for (m, sc), (ds, _) in zip(masks, found):
            union |= SG.downsample_mask(m, s)
            best = max(best, sc * ds)
        if union.sum() >= 4:
            elements.append(Element(n, union, best, "ai", [b for _, b in found]))
    del sam

    return _finish(img, elements)


def segment_with_hints(img: np.ndarray, hints: list[dict], negatives: bool = True,
                       min_score: float = 0.5) -> tuple[np.ndarray, list[Element]]:
    """Segment named elements from an agent's hints.

    hints: [{"name": str, "points": [[x, y], ...], "box": [x0, y0, x1, y1]?}]
    in sprite pixel coordinates - an agent (or a person) that understands
    the picture says what is where; SAM cuts the exact shapes and they are
    snapped to the pixel grid. With negatives, the nearest points of other
    elements are given as "not this", which keeps touching elements apart
    (only a few: SAM degrades with many negatives). Repeated
    single-pixel particles and the remaining backdrop are added classically.
    """
    image, s = upscale(img)
    sam = Sam()
    elements: list[Element] = []
    centre = lambda xy: (xy[0] * s + s / 2, xy[1] * s + s / 2)
    others = [[centre(p) for j, o in enumerate(hints) if j != i for p in o["points"]] for i in range(len(hints))]
    for i, h in enumerate(hints):
        pos = [centre(p) for p in h["points"]]
        neg = nearest(pos, others[i], 4) if negatives else []
        box = [v * s for v in h["box"]] if h.get("box") else None
        mask = np.zeros(img.shape[:2], dtype=bool)
        score = 0.0
        m, sc = sam.prompt(image, pos, neg, box)
        if sc >= min_score:
            mask |= SG.downsample_mask(m, s)
            score = sc
        # Elements made of separate parts (ribbons, rocks, two tusks): each
        # point also cuts its own part, and the parts are joined.
        if len(pos) > 1:
            for p in pos:
                pm, psc = sam.prompt(image, [p], nearest([p], others[i], 2) if negatives else [])
                if psc >= min_score:
                    mask |= SG.downsample_mask(pm, s)
                    score = max(score, psc)
        elements.append(Element(h["name"], mask, score, "agent+sam"))
    del sam
    return _finish(img, elements, vote=True)


def nearest(points: list, candidates: list, k: int) -> list:
    """The k candidates closest to any of the points."""
    if not candidates:
        return []
    d = [min((cx - px) ** 2 + (cy - py) ** 2 for px, py in points) for cx, cy in candidates]
    return [candidates[i] for i in np.argsort(d)[:k]]


def _finish(img: np.ndarray, elements: list[Element], vote: bool = False) -> tuple[np.ndarray, list[Element]]:
    """Resolve overlaps, add particles and backdrop, snap to colour clusters.

    vote: let fine colour regions (graph segmentation) join the element
    that covers most of them - fills holes and aligns borders to the art.
    """
    opaque = img[..., 3] > 0
    if vote:
        regions = SG.felzenszwalb(img, k=40, min_size=6)
        labels = SG.resolve_by_regions([e.mask for e in elements], regions)
        labels = SG.region_vote(labels, regions)
    else:
        labels = SG.masks_to_labels([e.mask for e in elements], img.shape[:2])
    particles = SG.particle_mask(img)
    if particles.sum() >= L.PARTICLE_REPEATS:
        elements.append(Element("falling particles", particles, 1.0, "classical"))
        labels[particles & ((labels < 0) | _is_backdrop_like(img, labels, elements))] = len(elements) - 1
    rest = opaque & (labels < 0)
    if rest.any():
        elements.append(Element("backdrop", rest, 1.0, "classical"))
        labels[rest] = len(elements) - 1
    labels = SG.snap_labels(img, labels)
    labels[~opaque] = -1
    for i, e in enumerate(elements):
        e.mask = labels == i
    return labels, elements


def _is_backdrop_like(img, labels, elements) -> np.ndarray:
    """Pixels whose current label is a large, frame-touching element."""
    out = np.zeros(labels.shape, dtype=bool)
    h, w = labels.shape
    for i, e in enumerate(elements):
        m = labels == i
        frame = m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()
        if m.sum() > 0.15 * h * w and frame > 0.1 * 2 * (h + w):
            out |= m
    return out


def automatic_masks(img: np.ndarray, grid: int = 24) -> np.ndarray:
    """Label image from SAM's automatic (prompt-free) masks, snapped to pixels."""
    image, s = upscale(img)
    sam = Sam()
    masks = dedupe(sam.points_grid(image, grid))
    small = [SG.downsample_mask(m, s) for m, _ in masks]
    labels = SG.masks_to_labels(small, img.shape[:2])
    labels = SG.snap_labels(img, labels)
    labels[img[..., 3] == 0] = -1
    return labels
