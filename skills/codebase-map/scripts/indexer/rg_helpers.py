from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils import ToolState, run_cmd


def rg_json_matches(
    repo: Path,
    pattern: str,
    *,
    globs: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    tools: Optional[ToolState] = None,
) -> List[Dict[str, Any]]:
    cmd = ["rg", "--json", pattern]
    for glob in globs or []:
        cmd.extend(["-g", glob])
    result = run_cmd(cmd, cwd=repo, warnings=warnings or [], tools=tools or ToolState(), capture=True)
    if not result:
        return []
    if result.returncode not in (0, 1):
        if warnings is not None:
            warnings.append(f"rg returned {result.returncode} for pattern: {pattern}")
    matches: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "match":
            matches.append(payload)
    return matches


def rg_collect_imports(
    repo: Path,
    *,
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for entry in rg_json_matches(
        repo,
        r"(?:import|export)\s+(?:type\s+)?[^;]*?from\s+['\"]([^'\"]+)['\"]",
        globs=globs,
        warnings=warnings,
        tools=tools,
    ):
        path = entry.get("data", {}).get("path", {}).get("text")
        if not isinstance(path, str):
            continue
        submatches = entry.get("data", {}).get("submatches", [])
        if not submatches:
            continue
        module = submatches[0].get("match", {}).get("text")
        if isinstance(module, str):
            results.append((path, module))
    return results


def rg_collect_py_imports(
    repo: Path,
    *,
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for entry in rg_json_matches(
        repo,
        r"(?:import|from)\s+([A-Za-z0-9_\\.]+)",
        globs=globs,
        warnings=warnings,
        tools=tools,
    ):
        path = entry.get("data", {}).get("path", {}).get("text")
        if not isinstance(path, str):
            continue
        submatches = entry.get("data", {}).get("submatches", [])
        if not submatches:
            continue
        module = submatches[0].get("match", {}).get("text")
        if isinstance(module, str):
            results.append((path, module))
    return results


def rg_collect_reexports(
    repo: Path,
    *,
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for entry in rg_json_matches(
        repo,
        r"export\s+(?:type\s+)?\\{[^}]*\\}\\s+from\\s+['\"]([^'\"]+)['\"]",
        globs=globs,
        warnings=warnings,
        tools=tools,
    ):
        path = entry.get("data", {}).get("path", {}).get("text")
        if not isinstance(path, str):
            continue
        submatches = entry.get("data", {}).get("submatches", [])
        if not submatches:
            continue
        module = submatches[0].get("match", {}).get("text")
        if isinstance(module, str):
            results.append((path, module))
    return results


def rg_collect_symbols(
    repo: Path,
    *,
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    patterns = [
        (r"^\\s*export\\s+class\\s+(\\w+)", "class"),
        (r"^\\s*export\\s+(?:async\\s+)?function\\s+(\\w+)", "function"),
        (r"^\\s*export\\s+const\\s+(\\w+)", "const"),
    ]
    for pattern, kind in patterns:
        for entry in rg_json_matches(
            repo,
            pattern,
            globs=globs,
            warnings=warnings,
            tools=tools,
        ):
            path = entry.get("data", {}).get("path", {}).get("text")
            if not isinstance(path, str):
                continue
            submatches = entry.get("data", {}).get("submatches", [])
            if not submatches:
                continue
            name = submatches[0].get("match", {}).get("text")
            if not isinstance(name, str):
                continue
            line_num = entry.get("data", {}).get("line_number")
            if not isinstance(line_num, int):
                line_num = 0
            results.append(
                {
                    "file": path,
                    "name": name,
                    "kind": kind,
                    "exported": True,
                    "line": line_num,
                    "signature": f"{kind} {name}",
                }
            )
    return results


def rg_count_matches(
    repo: Path,
    pattern: str,
    *,
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> int:
    matches = rg_json_matches(
        repo,
        pattern,
        globs=globs,
        warnings=warnings,
        tools=tools,
    )
    return len(matches)


def estimate_import_count_rg(
    repo: Path,
    languages: Optional[Set[str]] = None,
    *,
    globs: Optional[List[str]] = None,
    warnings: List[str],
    tools: ToolState,
) -> int:
    if globs is None and languages:
        globs = []
        if "ts" in languages:
            globs.append("**/*.ts")
        if "tsx" in languages:
            globs.append("**/*.tsx")
        if "js" in languages:
            globs.extend(["**/*.js", "**/*.mjs", "**/*.cjs"])
        if "jsx" in languages:
            globs.append("**/*.jsx")
        if "py" in languages:
            globs.append("**/*.py")
    import_count = rg_count_matches(
        repo,
        r"\\bimport\\b|\\brequire\\(",
        globs=globs,
        warnings=warnings,
        tools=tools,
    )
    return import_count
