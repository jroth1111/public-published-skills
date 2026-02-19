from __future__ import annotations

import ast
import bisect
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple


_SKIP_PATH_SEGMENTS = ("/.tmp-codebase-map-", "/.code-state/", "/.beads/")

_TS_DEF_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"
    ),
)
_TS_CALL_PATTERN = re.compile(r"(?:\.|\b)([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_TS_ASSIGN_PATTERN = re.compile(r"^\s*(?:const|let|var\s+)?[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*[^=].*")
_TS_CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "new",
    "typeof",
    "await",
    "function",
    "class",
    "super",
    "import",
}

_RS_DEF_PATTERN = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_RS_CALL_PATTERN = re.compile(r"(?:\.|::|\b)([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_RS_ASSIGN_PATTERN = re.compile(r"^\s*let\s+(?:mut\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+;")
_RS_CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "loop",
    "match",
    "fn",
    "struct",
    "enum",
    "trait",
    "impl",
    "return",
    "let",
    "Some",
    "Ok",
    "Err",
}


def _eligible_path(path: str) -> bool:
    lowered = f"/{path.lower().lstrip('/')}"
    if lowered.startswith("/tests/") or "/test/" in lowered or "/tests/" in lowered:
        return False
    return not any(token in lowered for token in _SKIP_PATH_SEGMENTS)


def _read_source(repo: Path, rel_path: str) -> str:
    try:
        return (repo / rel_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _nearest_scope(defs: Sequence[Tuple[int, str]], line_no: int) -> Tuple[int, str] | None:
    if not defs:
        return None
    line_index = [item[0] for item in defs]
    pos = bisect.bisect_right(line_index, line_no) - 1
    if pos < 0:
        return None
    return defs[pos]


def _build_python_edges(
    repo: Path,
    files: Sequence[str],
    *,
    max_call_edges: int,
    max_flow_edges: int,
    provenance: str,
) -> Dict[str, Any]:
    py_files = [path for path in files if path.endswith(".py") and _eligible_path(path)]
    function_defs: Dict[str, List[Tuple[str, int, str]]] = {}
    defs_by_path: Dict[str, List[Tuple[int, str]]] = {}
    call_sites: List[Tuple[str, int, str, str]] = []
    flow_pairs: Set[Tuple[str, int, str, int]] = set()

    class Analyzer(ast.NodeVisitor):
        def __init__(self, rel_path: str) -> None:
            self.rel_path = rel_path
            self.scope_stack: List[Tuple[str, int]] = []

        def _qual_name_for(self, name: str) -> str:
            if not self.scope_stack:
                return name
            return ".".join([part for part, _ in self.scope_stack] + [name])

        def _current_scope(self, fallback_line: int) -> Tuple[str, int] | None:
            if not self.scope_stack:
                return None
            scope_name = ".".join(part for part, _ in self.scope_stack)
            scope_line = self.scope_stack[-1][1]
            return (scope_name, max(1, scope_line or fallback_line))

        def _record_function(self, node: ast.AST) -> None:
            if not hasattr(node, "name"):
                return
            name = str(getattr(node, "name", "") or "")
            if not name:
                return
            line = int(getattr(node, "lineno", 0) or 0)
            qual_name = self._qual_name_for(name)
            function_defs.setdefault(name, []).append((self.rel_path, line, qual_name))
            defs_by_path.setdefault(self.rel_path, []).append((line, qual_name))
            self.scope_stack.append((name, line))
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record_function(node)

        def visit_Call(self, node: ast.Call) -> None:
            callee_name = ""
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_name = node.func.attr
            if callee_name:
                call_line = int(getattr(node, "lineno", 0) or 0)
                scope = self._current_scope(call_line)
                if scope:
                    caller_name, caller_line = scope
                    call_sites.append((self.rel_path, caller_line, caller_name, callee_name))
            self.generic_visit(node)

        def _add_flow_pair(self, value_node: ast.AST, target_node: ast.AST, stmt_line: int) -> None:
            from_line = int(getattr(value_node, "lineno", 0) or stmt_line or 0)
            to_line = int(getattr(target_node, "lineno", 0) or stmt_line or 0)
            if from_line <= 0 or to_line <= 0:
                return
            flow_pairs.add((self.rel_path, from_line, self.rel_path, to_line))

        def visit_Assign(self, node: ast.Assign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            for target in node.targets:
                self._add_flow_pair(node.value, target, stmt_line)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            if node.value is not None:
                self._add_flow_pair(node.value, node.target, stmt_line)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            stmt_line = int(getattr(node, "lineno", 0) or 0)
            self._add_flow_pair(node.value, node.target, stmt_line)
            self.generic_visit(node)

    for rel_path in py_files:
        source = _read_source(repo, rel_path)
        if not source:
            continue
        try:
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, ValueError):
            continue
        Analyzer(rel_path).visit(tree)

    call_edges: List[Dict[str, Any]] = []
    seen_call_edges: Set[Tuple[str, int, str, str, int, str]] = set()
    for caller_path, caller_line, caller_name, callee_name in call_sites:
        candidates = function_defs.get(callee_name, [])
        if not candidates:
            continue
        same_file = [entry for entry in candidates if entry[0] == caller_path]
        selected = same_file if same_file else candidates
        for callee_path, callee_line, callee_qual_name in selected[:3]:
            edge_key = (
                caller_path,
                caller_line,
                caller_name,
                callee_path,
                callee_line,
                callee_qual_name,
            )
            if edge_key in seen_call_edges:
                continue
            seen_call_edges.add(edge_key)
            call_edges.append(
                {
                    "caller_path": caller_path,
                    "caller_line": caller_line,
                    "caller_name": caller_name,
                    "callee_path": callee_path,
                    "callee_line": callee_line,
                    "callee_name": callee_qual_name,
                    "provenance": provenance,
                }
            )
            if len(call_edges) >= max_call_edges:
                break
        if len(call_edges) >= max_call_edges:
            break

    dataflow_edges: List[Dict[str, Any]] = []
    for from_path, from_line, to_path, to_line in sorted(flow_pairs):
        dataflow_edges.append(
            {
                "from_path": from_path,
                "from_line": from_line,
                "to_path": to_path,
                "to_line": to_line,
                "provenance": provenance,
            }
        )
        if len(dataflow_edges) >= max_flow_edges:
            break

    return {
        "call_edges": call_edges,
        "dataflow_edges": dataflow_edges,
        "files_scanned": len(py_files),
    }


def _build_text_language_edges(
    repo: Path,
    files: Sequence[str],
    *,
    exts: Set[str],
    def_patterns: Sequence[re.Pattern[str]],
    call_pattern: re.Pattern[str],
    call_keywords: Set[str],
    assign_pattern: re.Pattern[str],
    max_call_edges: int,
    max_flow_edges: int,
    provenance: str,
) -> Dict[str, Any]:
    lang_files = [path for path in files if Path(path).suffix in exts and _eligible_path(path)]
    defs_by_name: Dict[str, List[Tuple[str, int, str]]] = {}
    defs_by_path: Dict[str, List[Tuple[int, str]]] = {}
    file_lines: Dict[str, List[str]] = {}

    for rel_path in lang_files:
        source = _read_source(repo, rel_path)
        if not source:
            continue
        lines = source.splitlines()
        file_lines[rel_path] = lines
        defs: List[Tuple[int, str]] = []
        for index, line in enumerate(lines, start=1):
            for pattern in def_patterns:
                match = pattern.match(line)
                if not match:
                    continue
                name = match.group(1)
                defs.append((index, name))
                defs_by_name.setdefault(name, []).append((rel_path, index, name))
                break
        defs.sort(key=lambda item: item[0])
        if defs:
            defs_by_path[rel_path] = defs

    call_edges: List[Dict[str, Any]] = []
    seen_call_edges: Set[Tuple[str, int, str, str, int, str]] = set()
    flow_pairs: Set[Tuple[str, int, str, int]] = set()

    for rel_path, lines in file_lines.items():
        scope_defs = defs_by_path.get(rel_path, [])
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if assign_pattern.match(stripped):
                flow_pairs.add((rel_path, index, rel_path, index))

            scope = _nearest_scope(scope_defs, index)
            if not scope:
                continue
            caller_line, caller_name = scope
            for match in call_pattern.finditer(line):
                callee_name = (match.group(1) or "").strip()
                if not callee_name or callee_name in call_keywords:
                    continue
                candidates = defs_by_name.get(callee_name, [])
                if not candidates:
                    continue
                same_file = [entry for entry in candidates if entry[0] == rel_path]
                selected = same_file if same_file else candidates
                for callee_path, callee_line, callee_qual_name in selected[:3]:
                    edge_key = (
                        rel_path,
                        caller_line,
                        caller_name,
                        callee_path,
                        callee_line,
                        callee_qual_name,
                    )
                    if edge_key in seen_call_edges:
                        continue
                    seen_call_edges.add(edge_key)
                    call_edges.append(
                        {
                            "caller_path": rel_path,
                            "caller_line": caller_line,
                            "caller_name": caller_name,
                            "callee_path": callee_path,
                            "callee_line": callee_line,
                            "callee_name": callee_qual_name,
                            "provenance": provenance,
                        }
                    )
                    if len(call_edges) >= max_call_edges:
                        break
                if len(call_edges) >= max_call_edges:
                    break
            if len(call_edges) >= max_call_edges:
                break
        if len(call_edges) >= max_call_edges:
            break

    dataflow_edges: List[Dict[str, Any]] = []
    for from_path, from_line, to_path, to_line in sorted(flow_pairs):
        dataflow_edges.append(
            {
                "from_path": from_path,
                "from_line": from_line,
                "to_path": to_path,
                "to_line": to_line,
                "provenance": provenance,
            }
        )
        if len(dataflow_edges) >= max_flow_edges:
            break

    return {
        "call_edges": call_edges,
        "dataflow_edges": dataflow_edges,
        "files_scanned": len(lang_files),
    }


def _classify_language_confidence(file_count: int, call_edges: int, dataflow_edges: int) -> str:
    if file_count <= 0:
        return "n/a"
    call_density = call_edges / max(file_count, 1)
    flow_density = dataflow_edges / max(file_count, 1)
    score = 0.0
    if call_density >= 1.0:
        score += 2.0
    elif call_density >= 0.2:
        score += 1.0
    if flow_density >= 0.5:
        score += 1.0
    elif flow_density >= 0.1:
        score += 0.5
    if score >= 2.5:
        return "high"
    if score >= 1.0:
        return "medium"
    return "low"


def build_ast_primary_edges(
    repo: Path,
    files: Sequence[str],
    languages: Set[str],
    *,
    provenance: str = "ast",
    max_py_call_edges: int = 20000,
    max_py_flow_edges: int = 30000,
    max_ts_call_edges: int = 12000,
    max_ts_flow_edges: int = 12000,
    max_rs_call_edges: int = 8000,
    max_rs_flow_edges: int = 8000,
) -> Dict[str, Any]:
    call_edges: List[Dict[str, Any]] = []
    dataflow_edges: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    by_language: Dict[str, Dict[str, Any]] = {}

    if "py" in languages:
        py_info = _build_python_edges(
            repo,
            files,
            max_call_edges=max_py_call_edges,
            max_flow_edges=max_py_flow_edges,
            provenance=provenance,
        )
        call_edges.extend(py_info.get("call_edges", []))
        dataflow_edges.extend(py_info.get("dataflow_edges", []))
        counts["py_call_edges_ast"] = len(py_info.get("call_edges", []))
        counts["py_dataflow_edges_ast"] = len(py_info.get("dataflow_edges", []))
        by_language["py"] = {
            "files": int(py_info.get("files_scanned", 0)),
            "call_edges": counts["py_call_edges_ast"],
            "dataflow_edges": counts["py_dataflow_edges_ast"],
        }

    if languages & {"ts", "tsx", "js", "jsx"}:
        ts_info = _build_text_language_edges(
            repo,
            files,
            exts={".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"},
            def_patterns=_TS_DEF_PATTERNS,
            call_pattern=_TS_CALL_PATTERN,
            call_keywords=_TS_CALL_KEYWORDS,
            assign_pattern=_TS_ASSIGN_PATTERN,
            max_call_edges=max_ts_call_edges,
            max_flow_edges=max_ts_flow_edges,
            provenance=provenance,
        )
        call_edges.extend(ts_info.get("call_edges", []))
        dataflow_edges.extend(ts_info.get("dataflow_edges", []))
        counts["ts_call_edges_ast"] = len(ts_info.get("call_edges", []))
        counts["ts_dataflow_edges_ast"] = len(ts_info.get("dataflow_edges", []))
        by_language["ts"] = {
            "files": int(ts_info.get("files_scanned", 0)),
            "call_edges": counts["ts_call_edges_ast"],
            "dataflow_edges": counts["ts_dataflow_edges_ast"],
        }

    if "rs" in languages:
        rs_info = _build_text_language_edges(
            repo,
            files,
            exts={".rs"},
            def_patterns=(_RS_DEF_PATTERN,),
            call_pattern=_RS_CALL_PATTERN,
            call_keywords=_RS_CALL_KEYWORDS,
            assign_pattern=_RS_ASSIGN_PATTERN,
            max_call_edges=max_rs_call_edges,
            max_flow_edges=max_rs_flow_edges,
            provenance=provenance,
        )
        call_edges.extend(rs_info.get("call_edges", []))
        dataflow_edges.extend(rs_info.get("dataflow_edges", []))
        counts["rs_call_edges_ast"] = len(rs_info.get("call_edges", []))
        counts["rs_dataflow_edges_ast"] = len(rs_info.get("dataflow_edges", []))
        by_language["rs"] = {
            "files": int(rs_info.get("files_scanned", 0)),
            "call_edges": counts["rs_call_edges_ast"],
            "dataflow_edges": counts["rs_dataflow_edges_ast"],
        }

    low_languages: List[str] = []
    confidence_by_lang: Dict[str, str] = {}
    for lang_key, info in by_language.items():
        conf = _classify_language_confidence(
            int(info.get("files", 0)),
            int(info.get("call_edges", 0)),
            int(info.get("dataflow_edges", 0)),
        )
        confidence_by_lang[lang_key] = conf
        if conf == "low" and int(info.get("files", 0)) >= 2:
            low_languages.append(lang_key)

    if not confidence_by_lang:
        overall_conf = "n/a"
    elif low_languages:
        overall_conf = "low"
    elif any(conf == "medium" for conf in confidence_by_lang.values()):
        overall_conf = "medium"
    else:
        overall_conf = "high"

    return {
        "call_edges": call_edges,
        "dataflow_edges": dataflow_edges,
        "counts": counts,
        "by_language": by_language,
        "confidence": overall_conf,
        "confidence_by_language": confidence_by_lang,
        "low_confidence": bool(low_languages),
        "low_confidence_languages": low_languages,
    }

