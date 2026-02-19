from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils import ToolState, progress, run_cmd

from .constants import (
    EXCLUDE_DIRS,
    FD_EXCLUDES,
    RG_EXCLUDES,
    UNSUPPORTED_LANG_EXTS,
)


NOISE_FILE_NAMES = {
    ".coverage",
    ".dmypy.json",
    ".pdm-python",
    ".pnp.cjs",
    ".pnp.js",
    "coverage.xml",
    "manifest",
    "next-env.d.ts",
}
NOISE_FILE_SUFFIXES = (
    ".egg",
    ".profraw",
    ".pyc",
    ".pyd",
    ".pyo",
    ".so",
    ".tsbuildinfo",
)


def is_generated_noise_file(path: str) -> bool:
    lower = path.lower()
    name = Path(path).name.lower()
    if name in NOISE_FILE_NAMES:
        return True
    if name.endswith(".egg-info"):
        return True
    if any(lower.endswith(suffix) for suffix in NOISE_FILE_SUFFIXES):
        return True
    if lower.endswith(".yarn/install-state.gz") or "/.yarn/install-state.gz" in lower:
        return True
    if ".egg-info/" in lower:
        return True
    return False


def list_repo_files(repo: Path, warnings: List[str], tools: ToolState) -> List[str]:
    progress("Discovering files...")
    fd_cmd = ["fd", "--type", "f", "--hidden"]
    for exclude in FD_EXCLUDES:
        fd_cmd.extend(["--exclude", exclude])
    result = run_cmd(fd_cmd, cwd=repo, warnings=warnings, tools=tools, capture=True)
    if result and result.returncode == 0:
        files = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not is_generated_noise_file(line.strip())
        ]
        progress(f"Found {len(files)} files", done=True)
        return sorted(set(files))
    if result is not None and result.returncode != 0:
        warnings.append("fd failed; falling back to rg")

    cmd = ["rg", "--files"]
    for glob in RG_EXCLUDES:
        cmd.extend(["-g", glob])
    result = run_cmd(cmd, cwd=repo, warnings=warnings, tools=tools, capture=True)
    if result and result.returncode == 0:
        files = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not is_generated_noise_file(line.strip())
        ]
        progress(f"Found {len(files)} files", done=True)
        return sorted(set(files))
    if result is not None and result.returncode != 0:
        warnings.append("rg failed; falling back to Python file walk")
    return list_repo_files_fallback(repo)


def list_repo_files_fallback(repo: Path) -> List[str]:
    files: List[str] = []
    skipped_symlinks = 0
    max_filename_length = 255  # Common limit, truncate if exceeded

    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for filename in filenames:
            full = Path(root) / filename

            if full.is_symlink():
                skipped_symlinks += 1
                continue

            try:
                rel = full.relative_to(repo).as_posix()
                if is_generated_noise_file(rel):
                    continue
                if len(rel) > max_filename_length:
                    rel = (
                        rel[: max_filename_length - 8]
                        + "..."
                        + hashlib.md5(rel.encode()).hexdigest()[:4]
                    )
                files.append(rel)
            except (ValueError, OSError):
                continue

    if skipped_symlinks > 0:
        progress(
            f"Found {len(files)} files (fallback, skipped {skipped_symlinks} symlinks)",
            done=True,
        )
    else:
        progress(f"Found {len(files)} files (fallback)", done=True)
    return sorted(set(files))


def detect_languages(files: Iterable[str], arg_value: Set[str]) -> Set[str]:
    if arg_value:
        return arg_value
    languages: Set[str] = set()
    if any(path.endswith(".ts") for path in files):
        languages.add("ts")
    if any(path.endswith(".tsx") for path in files):
        languages.add("tsx")
    if any(path.endswith((".js", ".mjs", ".cjs")) for path in files):
        languages.add("js")
    if any(path.endswith(".jsx") for path in files):
        languages.add("jsx")
    if any(path.endswith(".py") for path in files):
        languages.add("py")
    if any(path.endswith(".go") for path in files):
        languages.add("go")
    if any(path.endswith(".rs") for path in files):
        languages.add("rs")
    return languages


def code_files_for_languages(files: Iterable[str], languages: Set[str]) -> List[str]:
    exts: Set[str] = set()
    if "ts" in languages:
        exts.add(".ts")
    if "tsx" in languages:
        exts.add(".tsx")
    if "js" in languages:
        exts.update({".js", ".mjs", ".cjs"})
    if "jsx" in languages:
        exts.add(".jsx")
    if "py" in languages:
        exts.add(".py")
    if "go" in languages:
        exts.add(".go")
    if "rs" in languages:
        exts.add(".rs")
    results: List[str] = []
    for path in files:
        suffix = Path(path).suffix
        if suffix == ".d.ts":
            continue
        if suffix in exts:
            results.append(path)
    return results


def code_files_size_mb(repo: Path, files: Iterable[str]) -> float:
    total = 0
    for rel in files:
        try:
            total += (repo / rel).stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)


def unsupported_language_counts(files: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix in UNSUPPORTED_LANG_EXTS:
            lang = UNSUPPORTED_LANG_EXTS[suffix]
            counts[lang] = counts.get(lang, 0) + 1
    return counts
