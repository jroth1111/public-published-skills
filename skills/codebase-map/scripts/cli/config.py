from __future__ import annotations

from pathlib import Path
from typing import Optional, Set


def parse_languages(value: str) -> Set[str]:
    if value == "auto":
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def auto_precise_tokens(enabled: bool) -> bool:
    if enabled:
        return True
    try:
        import tiktoken  # type: ignore

        return True
    except Exception:
        return False


def resolve_out_dir(repo: Path, out_arg: Optional[str], *, workspace_root: Path) -> Path:
    if out_arg:
        out_path = Path(out_arg)
        if out_path.is_absolute():
            return out_path
        out_str = out_path.as_posix()
        if out_str.startswith("workspace/"):
            out_path = Path(out_str[len("workspace/") :])
        return (workspace_root / out_path).resolve()
    return (workspace_root / "codebase-map" / repo.name).resolve()
