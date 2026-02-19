from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


IR_VERSION = 8


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_path(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def file_id(path: str) -> str:
    return f"file:{path}"


def symbol_id(path: str, line: int | None, name: str) -> str:
    line_part = line if line is not None and line > 0 else 0
    return f"sym:{path}#L{line_part}:{name}"


def new_ir(repo: Path, languages: List[str], file_hashes: Dict[str, str]) -> Dict[str, Any]:
    return {
        "meta": {
            "version": IR_VERSION,
            "generated_at": now_iso(),
            "repo": repo.as_posix(),
            "languages": sorted(set(languages)),
            "file_hashes": file_hashes,
        },
        "entities": {},
        "files": {},
        "symbols": {},
        "edges": {
            "file_dep": [],
            "external_dep": [],
            "symbol_ref": [],
            "dataflow": [],
            "type_ref": [],
            "entity_use": [],
        },
    }


def load_ir(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_ir(path: Path, ir: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ir, ensure_ascii=True, indent=2), encoding="utf-8")
