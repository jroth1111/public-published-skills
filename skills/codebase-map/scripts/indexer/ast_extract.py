from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils import ToolState, run_cmd


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def run_ast_grep(
    repo: Path,
    pattern: str,
    lang: str,
    *,
    files: Optional[List[str]],
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Dict[str, Any]]:
    cmd = ["ast-grep", "--lang", lang, "-p", pattern, "--json=stream"]
    if files:
        cmd.extend(files)
    else:
        for glob in globs or []:
            cmd.extend(["--globs", glob])
    result = run_cmd(cmd, cwd=repo, warnings=warnings, tools=tools, capture=True)
    if not result:
        return []
    if result.returncode not in (0, 1):
        warnings.append(f"ast-grep returned {result.returncode} for pattern: {pattern}")
    matches: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            matches.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.append("Failed to parse ast-grep JSON output line")
            break
    return matches


def run_ast_grep_scan(
    repo: Path,
    rules_yaml: str,
    *,
    files: Optional[List[str]],
    globs: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    if files:
        batches = list(chunked(files, 200 if workers and workers > 1 else 500))
        if not batches:
            return matches

        def scan_batch(batch: List[str]) -> List[Dict[str, Any]]:
            batch_matches: List[Dict[str, Any]] = []
            batch_warnings: List[str] = []
            cmd = ["ast-grep", "scan", "--inline-rules", rules_yaml, "--json=stream"]
            cmd.extend(batch)
            result = run_cmd(cmd, cwd=repo, warnings=batch_warnings, tools=tools, capture=True)
            if not result:
                warnings.extend(batch_warnings)
                return batch_matches
            if result.returncode not in (0, 1):
                batch_warnings.append(f"ast-grep scan returned {result.returncode}")
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    batch_matches.append(json.loads(line))
                except json.JSONDecodeError:
                    batch_warnings.append("Failed to parse ast-grep JSON output line")
                    break
            warnings.extend(batch_warnings)
            return batch_matches

        first_batch = batches.pop(0)
        matches.extend(scan_batch(first_batch))
        if "ast-grep" in tools.missing:
            return matches

        if batches and workers and workers > 1:
            max_workers = max(1, int(workers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(scan_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    matches.extend(future.result())
        else:
            for batch in batches:
                matches.extend(scan_batch(batch))
    else:
        cmd = ["ast-grep", "scan", "--inline-rules", rules_yaml, "--json=stream"]
        for glob in globs or []:
            cmd.extend(["--globs", glob])
        result = run_cmd(cmd, cwd=repo, warnings=warnings, tools=tools, capture=True)
        if not result:
            return matches
        if result.returncode not in (0, 1):
            warnings.append(f"ast-grep scan returned {result.returncode}")
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                matches.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append("Failed to parse ast-grep JSON output line")
                break

    if matches:
        def match_key(item: Dict[str, Any]) -> tuple[str, int, str]:
            file_path = item.get("file") or ""
            range_data = item.get("range") or {}
            start = range_data.get("start") if isinstance(range_data, dict) else {}
            line = start.get("line") if isinstance(start, dict) else 0
            rule_id = item.get("ruleId") or ""
            return (str(file_path), int(line or 0), str(rule_id))

        matches.sort(key=match_key)
    return matches
