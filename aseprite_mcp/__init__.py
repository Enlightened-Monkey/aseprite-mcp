"""Aseprite MCP - Model Context Protocol implementation for Aseprite."""

import os

from mcp.server.fastmcp import FastMCP

_CORE_RULES = os.path.join(os.path.dirname(__file__), "knowledge", "core.md")
with open(_CORE_RULES, encoding="utf-8") as _f:
    INSTRUCTIONS = (
        "Tools for drawing and animating pixel art in Aseprite. "
        "Follow these pixel-art rules; check your work with lint_pixel_art and readability_preview.\n\n"
        + _f.read()
    )

mcp = FastMCP("aseprite", instructions=INSTRUCTIONS)

__version__ = "0.1.0"
__author__ = "Divyansh Singh"