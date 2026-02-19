from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import ToolState

from .ast_extract import run_ast_grep
from .constants import ast_globs_for


def line_number_for_offset(content: str, offset: int) -> int:
    return content[:offset].count("\n") + 1


def clean_type_name(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    raw = raw.split("<", 1)[0]
    raw = re.sub(r"[^A-Za-z0-9_]", "", raw)
    return raw


def extract_type_edges(content: str, path: str) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    for match in re.finditer(
        r"\bclass\s+(\w+)\s+extends\s+([A-Za-z0-9_<>,\s]+)(?:\s+implements\s+([^{]+))?",
        content,
    ):
        name = match.group(1)
        base = clean_type_name(match.group(2))
        if base:
            edges.append(
                {
                    "from_name": name,
                    "to_name": base,
                    "kind": "extends",
                    "from_path": path,
                    "line": line_number_for_offset(content, match.start()),
                }
            )
        impls_raw = match.group(3) or ""
        for impl in impls_raw.split(","):
            impl_name = clean_type_name(impl)
            if impl_name:
                edges.append(
                    {
                        "from_name": name,
                        "to_name": impl_name,
                        "kind": "implements",
                        "from_path": path,
                        "line": line_number_for_offset(content, match.start()),
                    }
                )
    for match in re.finditer(r"\binterface\s+(\w+)\s+extends\s+([^{]+)\{", content):
        name = match.group(1)
        bases = match.group(2)
        for base in bases.split(","):
            base_name = clean_type_name(base)
            if base_name:
                edges.append(
                    {
                        "from_name": name,
                        "to_name": base_name,
                        "kind": "extends",
                        "from_path": path,
                        "line": line_number_for_offset(content, match.start()),
                    }
                )
    for match in re.finditer(r"\btype\s+(\w+)\s*=\s*([A-Za-z0-9_<>&| ]+)", content):
        name = match.group(1)
        rhs = match.group(2)
        for token in re.split(r"[|&]", rhs):
            token_name = clean_type_name(token)
            if token_name and token_name != name:
                edges.append(
                    {
                        "from_name": name,
                        "to_name": token_name,
                        "kind": "alias",
                        "from_path": path,
                        "line": line_number_for_offset(content, match.start()),
                    }
                )
    for match in re.finditer(r"\bclass\s+(\w+)\s*\(([^)]*)\)\s*:", content):
        name = match.group(1)
        bases = match.group(2)
        for base in bases.split(","):
            base_name = clean_type_name(base)
            if base_name and base_name != "object":
                edges.append(
                    {
                        "from_name": name,
                        "to_name": base_name,
                        "kind": "extends",
                        "from_path": path,
                        "line": line_number_for_offset(content, match.start()),
                    }
                )
    return edges


def match_line(match: Dict[str, Any]) -> Optional[int]:
    range_data = match.get("range") or {}
    if not isinstance(range_data, dict):
        return None
    start = range_data.get("start") or {}
    if not isinstance(start, dict):
        return None
    line = start.get("line")
    if not isinstance(line, int):
        return None
    return line + 1


def meta_single(match: Dict[str, Any], name: str) -> Optional[str]:
    meta = match.get("metaVariables") or {}
    if not isinstance(meta, dict):
        return None
    single = meta.get("single") or {}
    if not isinstance(single, dict):
        return None
    entry = single.get(name)
    if not isinstance(entry, dict):
        return None
    value = entry.get("text")
    if isinstance(value, str):
        return value
    return None


def meta_multi(match: Dict[str, Any], name: str) -> Optional[List[str]]:
    meta = match.get("metaVariables") or {}
    if not isinstance(meta, dict):
        return None
    multiple = meta.get("multi") or {}
    if not isinstance(multiple, dict):
        return None
    entry = multiple.get(name)
    if not isinstance(entry, dict):
        return None
    text = entry.get("text")
    if isinstance(text, str):
        return [text]
    return None


def normalize_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    joined = " ".join(token.strip() for token in tokens if token and token.strip())
    joined = re.sub(r"\s*,\s*", ", ", joined)
    joined = re.sub(r"\s*=\s*", " = ", joined)
    joined = re.sub(r"\s+", " ", joined)
    return joined.strip()


def signature_from_match(lang: str, kind: str, name: str, match: Dict[str, Any]) -> str:
    params_tokens = meta_multi(match, "PARAMS")
    params = normalize_tokens(params_tokens or [])
    ret = meta_single(match, "RET")
    base = meta_single(match, "BASE")
    bases_tokens = meta_multi(match, "BASES")
    bases = normalize_tokens(bases_tokens or [])

    if kind in {"function", "const"}:
        if params_tokens is not None:
            signature = f"{name}({params})"
        elif kind == "const":
            signature = f"const {name}"
        else:
            signature = name
        if ret:
            signature = f"{signature} -> {ret}"
        return signature

    if kind == "class":
        if bases_tokens is not None and bases:
            return f"class {name}({bases})"
        if base:
            return f"class {name} extends {base}"
        return f"class {name}"

    if kind == "interface":
        if base:
            return f"interface {name} extends {base}"
        return f"interface {name}"

    if kind == "type":
        return f"type {name}"

    if kind == "struct":
        return f"struct {name}"

    if kind == "trait":
        return f"trait {name}"

    if kind == "enum":
        return f"enum {name}"

    return name


def symbol_visibility(name: str, exported: bool) -> str:
    if exported:
        return "public"
    if name.startswith("_"):
        return "private"
    return "internal"


def is_pascal(name: str) -> bool:
    return bool(re.match(r"[A-Z][A-Za-z0-9]*$", name)) and not name.isupper()


def maybe_component(lang: str, kind: str, name: str) -> str:
    if lang in {"tsx", "jsx"} and kind in {"function", "class", "const"} and is_pascal(name):
        return "component"
    return kind


def ensure_file_data(file_data: Dict[str, Dict[str, Any]], path: str) -> Dict[str, Any]:
    if path not in file_data:
        file_data[path] = {"imports": [], "re_exports": [], "exports": [], "defines": []}
    return file_data[path]


def collect_imports(
    repo: Path,
    lang: str,
    patterns: Iterable[str],
    files: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for pattern in patterns:
        for match in run_ast_grep(
            repo,
            pattern,
            lang,
            files=files,
            globs=ast_globs_for(lang),
            warnings=warnings,
            tools=tools,
        ):
            file_path = match.get("file")
            module = meta_single(match, "MOD")
            if isinstance(file_path, str) and module:
                results.append((file_path, module))
    return results


def collect_symbols(
    repo: Path,
    lang: str,
    patterns: Iterable[Tuple[str, str, bool]],
    files: Optional[List[str]],
    warnings: List[str],
    tools: ToolState,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for pattern, kind, exported in patterns:
        for match in run_ast_grep(
            repo,
            pattern,
            lang,
            files=files,
            globs=ast_globs_for(lang),
            warnings=warnings,
            tools=tools,
        ):
            file_path = match.get("file")
            name = meta_single(match, "NAME")
            if not isinstance(file_path, str) or not name:
                continue
            line = match_line(match)
            signature = signature_from_match(lang, kind, name, match)
            results.append(
                {
                    "file": file_path,
                    "name": name,
                    "kind": kind,
                    "exported": exported,
                    "line": line,
                    "signature": signature,
                }
            )
    return results


def regex_js_imports(content: str) -> List[str]:
    modules: List[str] = []
    for match in re.finditer(
        r"^\\s*(?:import|export)\\s+(?:.+?\\s+from\\s+)?[\\\"']([^\\\"']+)[\\\"']",
        content,
        re.MULTILINE,
    ):
        modules.append(match.group(1))
    for match in re.finditer(r"require\\(\\s*[\\\"']([^\\\"']+)[\\\"']\\s*\\)", content):
        modules.append(match.group(1))
    for match in re.finditer(r"import\\(\\s*[\\\"']([^\\\"']+)[\\\"']\\s*\\)", content):
        modules.append(match.group(1))
    return modules


def regex_js_symbols(content: str) -> List[Dict[str, Any]]:
    symbols: List[Dict[str, Any]] = []
    patterns = [
        (r"^\\s*export\\s+(?:async\\s+)?function\\s+(\\w+)", "function"),
        (r"^\\s*export\\s+class\\s+(\\w+)", "class"),
        (r"^\\s*export\\s+const\\s+(\\w+)", "const"),
        (r"^\\s*export\\s+default\\s+(?:async\\s+)?function\\s+(\\w+)?", "function"),
        (r"^\\s*export\\s+default\\s+class\\s+(\\w+)?", "class"),
    ]
    for pattern, kind in patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            name = match.group(1) or "default"
            line = content[: match.start()].count("\\n") + 1
            symbols.append(
                {
                    "name": name,
                    "kind": kind,
                    "exported": True,
                    "line": line,
                    "signature": f"{kind} {name}",
                }
            )
    return symbols
