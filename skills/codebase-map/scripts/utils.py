from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set


@dataclass
class ToolState:
    missing: Set[str] = field(default_factory=set)
    used: Set[str] = field(default_factory=set)


def progress(message: str, done: bool = False) -> None:
    """Print a progress message to stderr (doesn't interfere with stdout output)."""
    if done:
        print(f"  [done] {message}", file=sys.stderr)
    else:
        print(f"  [....] {message}", file=sys.stderr)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path],
    warnings: List[str],
    tools: ToolState,
    capture: bool = True,
) -> Optional[subprocess.CompletedProcess]:
    tool = cmd[0]
    if tool in tools.missing:
        return None
    tools.used.add(tool)
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError:
        tools.missing.add(tool)
        warnings.append(f"Missing tool: {tool}")
        return None


def run_cmd_logged(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path],
    log_path: Path,
    warnings: List[str],
    tools: ToolState,
) -> Optional[int]:
    tool = cmd[0]
    if tool in tools.missing:
        return None
    tools.used.add(tool)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=False,
                text=True,
                stdout=log_file,
                stderr=log_file,
            )
        return proc.returncode
    except FileNotFoundError:
        tools.missing.add(tool)
        warnings.append(f"Missing tool: {tool}")
        return None


def tool_version(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path],
    warnings: List[str],
    tools: ToolState,
) -> Optional[str]:
    result = run_cmd(cmd, cwd=cwd, warnings=warnings, tools=tools, capture=True)
    if not result or not result.stdout:
        return None
    return result.stdout.splitlines()[0].strip()
