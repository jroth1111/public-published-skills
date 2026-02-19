from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from .constants import CONFIG_FILES
from .imports import match_globs


def filter_docs(
    files: Iterable[str],
    *,
    extra_globs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> List[str]:
    docs: List[str] = []
    extra = list(extra_globs or [])
    exclude = list(exclude_globs or [])
    for path in files:
        if match_globs(path, exclude):
            continue
        if match_globs(path, extra):
            docs.append(path)
            continue
        lower = path.lower()
        name = Path(path).name.lower()
        if name in {"skill.md", "agents.md"}:
            docs.append(path)
        elif "/references/" in lower or lower.startswith("references/"):
            docs.append(path)
        elif name.startswith("readme") or name.startswith("changelog"):
            docs.append(path)
        elif "/docs/" in lower or lower.startswith("docs/") or "/doc/" in lower or lower.startswith("doc/"):
            docs.append(path)
        elif "/adr/" in lower or name.startswith("adr"):
            docs.append(path)
    return sorted(set(docs))


def filter_tests(
    files: Iterable[str],
    *,
    extra_globs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> List[str]:
    tests: List[str] = []
    extra = list(extra_globs or [])
    exclude = list(exclude_globs or [])
    for path in files:
        if match_globs(path, exclude):
            continue
        if match_globs(path, extra):
            tests.append(path)
            continue
        lower = path.lower()
        name = Path(path).name.lower()
        suffix = Path(path).suffix.lower()
        if "/test/" in lower or lower.startswith("test/") or "/tests/" in lower or lower.startswith("tests/"):
            tests.append(path)
        elif ".spec." in lower or ".test." in lower:
            if suffix in {".json", ".yml", ".yaml"}:
                continue
            tests.append(path)
        elif name.startswith("test_"):
            tests.append(path)
    return sorted(set(tests))


def filter_configs(
    files: Iterable[str],
    *,
    extra_globs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> List[str]:
    configs: List[str] = []
    extra = list(extra_globs or [])
    exclude = list(exclude_globs or [])
    for path in files:
        if match_globs(path, exclude):
            continue
        if match_globs(path, extra):
            configs.append(path)
            continue
        name = Path(path).name
        if name.lower() in CONFIG_FILES:
            configs.append(path)
    return sorted(set(configs))


def role_for_file(path: str, tests: Set[str], configs: Set[str], entrypoints: Set[str]) -> str:
    if path in tests:
        return "test"
    if path in configs:
        return "config"
    if path in entrypoints:
        return "entrypoint"
    return "library"
