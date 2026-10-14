"""Injected-defect regression tests: each check finds its defect and the
clean bases stay quiet."""
import numpy as np
import pytest

from aseprite_mcp.pixelart import cleanup as CL
from aseprite_mcp.pixelart import lint as L
from benchmarks import defects as D

PIXEL = ["l_corners", "jaggies", "gaps", "thick_line", "full_step_aa", "outer_aa", "colour_noise"]
TRAIT = ["mixed_blacks", "pillow", "dither_small"]


def _bad(base, defect):
    return D.DEFECTS[defect][0](D.BASES[base](), np.random.default_rng(7))


@pytest.mark.parametrize("base", list(D.BASES))
def test_clean_bases_have_no_warnings(base):
    assert [i["check"] for i in L.lint(D.BASES[base]()) if i["severity"] == "warning"] == []


@pytest.mark.parametrize("base", list(D.BASES))
@pytest.mark.parametrize("defect", PIXEL + TRAIT)
def test_defect_is_reported(base, defect):
    bad, _ = _bad(base, defect)
    found = {i["check"] for i in L.lint(bad)}
    assert found & set(D.DEFECTS[defect][1])


@pytest.mark.parametrize("base", list(D.BASES))
@pytest.mark.parametrize("defect", ["gaps", "outer_aa", "colour_noise", "mixed_blacks"])
def test_cleanup_restores(base, defect):
    clean = D.BASES[base]()
    bad, mask = _bad(base, defect)
    fixed, _ = CL.cleanup(bad)
    assert (np.abs(fixed.astype(int) - clean)[mask].max(axis=-1) < 12).mean() >= 0.9


@pytest.mark.parametrize("base", list(D.BASES))
@pytest.mark.parametrize("defect", ["thick_line", "full_step_aa"])
def test_style_fixes_only_in_thorough_mode(base, defect):
    bad, mask = _bad(base, defect)
    gentle, _ = CL.cleanup(bad)
    thorough, _ = CL.cleanup(bad, keep_details=False)
    changed = lambda a: (np.abs(a.astype(int) - bad).max(axis=-1) > 0)[mask].sum()
    assert changed(thorough) > changed(gentle)
