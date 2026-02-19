"""Filesystem pattern helpers.

Rules:
- write raw artifacts under workspace/
- print only small summaries/previews (never dump huge payloads to stdout)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

WORKSPACE_DIR = Path("workspace")


def ensure_workspace() -> Path:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def workspace_root() -> Path:
    return ensure_workspace()


def _resolve_path(filename: str) -> Path:
    ensure_workspace()
    path = WORKSPACE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(filename: str, text: str) -> Path:
    path = _resolve_path(filename)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(filename: str, obj: Any) -> Path:
    path = _resolve_path(filename)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def write_lines(filename: str, lines: Iterable[str]) -> Path:
    text = "\n".join(lines) + ("\n" if lines else "")
    return write_text(filename, text)


def safe_preview_text(text: str, max_bytes: int = 512) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    cut = data[:max_bytes]
    return cut.decode("utf-8", errors="ignore") + "..."


def safe_preview_json(obj: Any, max_bytes: int = 512) -> str:
    rendered = json.dumps(obj, ensure_ascii=True, indent=2)
    return safe_preview_text(rendered, max_bytes=max_bytes)
