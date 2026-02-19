from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def truncate_doc(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3].rstrip() + "..."


def parse_jsdoc_param(rest: str) -> Dict[str, str]:
    param_type = ""
    optional = False
    name = ""
    desc = ""
    rest = rest.strip()
    if rest.startswith("{"):
        type_end = rest.find("}")
        if type_end != -1:
            param_type = rest[1:type_end].strip()
            rest = rest[type_end + 1 :].strip()
    if rest:
        parts = rest.split(maxsplit=1)
        name_token = parts[0]
        desc = parts[1] if len(parts) > 1 else ""
        if name_token.startswith("[") and name_token.endswith("]"):
            optional = True
            name_token = name_token[1:-1]
            if "=" in name_token:
                name_token = name_token.split("=", 1)[0]
        name = name_token.strip()
    return {
        "name": name,
        "description": desc,
        "type": param_type,
        "optional": optional,
    }


def parse_jsdoc_returns(rest: str) -> Tuple[str, str]:
    rest = rest.strip()
    ret_type = ""
    desc = ""
    if rest.startswith("{"):
        type_end = rest.find("}")
        if type_end != -1:
            ret_type = rest[1:type_end].strip()
            rest = rest[type_end + 1 :].strip()
    desc = rest
    return ret_type, desc


def parse_jsdoc_raises(rest: str) -> Dict[str, str]:
    rest = rest.strip()
    exc_type = ""
    desc = ""
    if rest.startswith("{"):
        type_end = rest.find("}")
        if type_end != -1:
            exc_type = rest[1:type_end].strip()
            rest = rest[type_end + 1 :].strip()
    if rest:
        parts = rest.split(maxsplit=1)
        if not exc_type:
            exc_type = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
        else:
            desc = rest
    return {"type": exc_type, "description": desc}


def parse_jsdoc_block(text: str) -> Dict[str, object]:
    lines = []
    for raw in text.splitlines():
        cleaned = raw.strip()
        if cleaned.startswith("/**"):
            cleaned = cleaned[3:]
        if cleaned.endswith("*/"):
            cleaned = cleaned[:-2]
        cleaned = cleaned.lstrip("*").strip()
        lines.append(cleaned)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    summary = ""
    description_lines: List[str] = []
    params: List[Dict[str, str]] = []
    returns = ""
    returns_type = ""
    raises: List[Dict[str, str]] = []
    examples: List[str] = []
    in_example = False
    for line in lines:
        if not line:
            if in_example:
                examples.append("")
            elif summary:
                description_lines.append("")
            continue
        if line.startswith("@"):
            in_example = False
            tag, *_rest = line.split(maxsplit=1)
            tag = tag.lower()
            rest = line[len(tag) :].strip()
            if tag in {"@param", "@arg", "@argument"}:
                param = parse_jsdoc_param(rest)
                if param.get("name"):
                    params.append(param)
            elif tag in {"@return", "@returns"}:
                returns_type, returns = parse_jsdoc_returns(rest)
            elif tag in {"@throws", "@exception", "@raise"}:
                exc = parse_jsdoc_raises(rest)
                if exc.get("type") or exc.get("description"):
                    raises.append(exc)
            elif tag == "@example":
                in_example = True
                if rest:
                    examples.append(rest)
            continue
        if in_example:
            examples.append(line)
            continue
        if not summary:
            summary = line
        else:
            description_lines.append(line)

    description = "\n".join(description_lines).strip()
    if summary:
        summary = truncate_doc(summary, 160)
    if description:
        description = truncate_doc(description, 400)
    if len(params) > 6:
        params = params[:6]
    if len(examples) > 2:
        examples = examples[:2]
    examples = [truncate_doc(example, 200) for example in examples if example]
    return {
        "summary": summary,
        "description": description,
        "params": params,
        "returns": returns,
        "returns_type": returns_type,
        "raises": raises[:4],
        "examples": examples,
    }


def jsdoc_blocks(lines: List[str]) -> List[Dict[str, object]]:
    blocks: List[Dict[str, object]] = []
    in_block = False
    start_line = 0
    buf: List[str] = []
    for idx, line in enumerate(lines, start=1):
        if not in_block and "/**" in line:
            in_block = True
            start_line = idx
            buf = [line]
            if "*/" in line and line.find("/**") < line.find("*/"):
                blocks.append(
                    {
                        "start": start_line,
                        "end": idx,
                        "raw": "\n".join(buf),
                    }
                )
                in_block = False
                buf = []
            continue
        if in_block:
            buf.append(line)
            if "*/" in line:
                blocks.append(
                    {
                        "start": start_line,
                        "end": idx,
                        "raw": "\n".join(buf),
                    }
                )
                in_block = False
                buf = []
    return blocks


def jsdoc_for_symbol_line(
    lines: List[str],
    blocks: List[Dict[str, object]],
    line_no: int,
) -> Optional[Dict[str, object]]:
    if line_no <= 1:
        return None
    candidate: Optional[Dict[str, object]] = None
    for block in blocks:
        if int(block["end"]) < line_no:
            candidate = block
        else:
            break
    if not candidate:
        return None
    start = int(candidate["end"]) + 1
    for idx in range(start, line_no):
        stripped = lines[idx - 1].strip()
        if not stripped:
            continue
        if stripped.startswith("@"):
            continue
        return None
    parsed = parse_jsdoc_block(str(candidate.get("raw") or ""))
    if not parsed.get("summary") and not parsed.get("description"):
        return None
    return parsed


def extract_py_docstring(lines: List[str], start_line: int) -> Optional[Dict[str, object]]:
    if start_line <= 0 or start_line > len(lines):
        return None
    header = lines[start_line - 1]
    indent = len(header) - len(header.lstrip())
    idx = start_line
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if len(line) - len(line.lstrip()) <= indent:
            return None
        stripped = line.lstrip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            content = stripped[3:]
            doc_lines: List[str] = []
            if content.endswith(quote):
                doc_lines.append(content[:-3])
                doc_text = "\n".join(doc_lines)
                return parse_py_docstring(doc_text)
            doc_lines.append(content)
            idx += 1
            while idx < len(lines):
                next_line = lines[idx]
                if quote in next_line:
                    before, _sep, _after = next_line.partition(quote)
                    doc_lines.append(before)
                    doc_text = "\n".join(doc_lines)
                    return parse_py_docstring(doc_text)
                doc_lines.append(next_line)
                idx += 1
            return parse_py_docstring("\n".join(doc_lines))
        return None
    return None


def parse_py_docstring(text: str) -> Dict[str, object]:
    raw_lines = [line.rstrip("\n") for line in text.splitlines()]
    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    summary = ""
    description_lines: List[str] = []
    params: List[Dict[str, str]] = []
    returns = ""
    returns_type = ""
    raises: List[Dict[str, str]] = []
    examples: List[str] = []

    def heading_kind(line: str, next_line: str) -> Optional[str]:
        stripped = line.strip()
        lower = stripped.lower().rstrip(":")
        if stripped.endswith(":") and lower in {
            "args",
            "arguments",
            "parameters",
            "params",
            "returns",
            "return",
            "raises",
            "raise",
            "exceptions",
            "example",
            "examples",
        }:
            return lower.rstrip("s")
        if lower in {
            "parameters",
            "returns",
            "raises",
            "examples",
        } and re.match(r"^[-=]{3,}$", next_line.strip() or ""):
            return lower.rstrip("s")
        return None

    for line in raw_lines:
        match = re.match(r"^\s*:param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if match:
            params.append(
                {
                    "name": match.group(1),
                    "description": match.group(2).strip(),
                }
            )
            continue
        match = re.match(r"^\s*:returns?\s*:\s*(.*)$", line)
        if match:
            returns = match.group(1).strip()
            continue
        match = re.match(r"^\s*:rtype\s*:\s*(.*)$", line)
        if match:
            returns_type = match.group(1).strip()
            continue
        match = re.match(r"^\s*:raises?\s+([A-Za-z_][A-Za-z0-9_\.]*)\s*:\s*(.*)$", line)
        if match:
            raises.append({"type": match.group(1), "description": match.group(2).strip()})

    idx = 0
    while idx < len(raw_lines) and not raw_lines[idx].strip():
        idx += 1
    if idx < len(raw_lines):
        summary = raw_lines[idx].strip()
        idx += 1

    while idx < len(raw_lines):
        line = raw_lines[idx]
        next_line = raw_lines[idx + 1] if idx + 1 < len(raw_lines) else ""
        if heading_kind(line, next_line):
            break
        if line.strip().startswith(":param") or line.strip().startswith(":return") or line.strip().startswith(":rtype") or line.strip().startswith(":raises"):
            idx += 1
            continue
        description_lines.append(line.strip())
        idx += 1

    while idx < len(raw_lines):
        line = raw_lines[idx]
        next_line = raw_lines[idx + 1] if idx + 1 < len(raw_lines) else ""
        kind = heading_kind(line, next_line)
        if not kind:
            idx += 1
            continue
        idx += 1
        if next_line and re.match(r"^[-=]{3,}$", next_line.strip() or ""):
            idx += 1

        block: List[str] = []
        while idx < len(raw_lines):
            line = raw_lines[idx]
            next_line = raw_lines[idx + 1] if idx + 1 < len(raw_lines) else ""
            if heading_kind(line, next_line):
                break
            block.append(line)
            idx += 1

        if kind in {"arg", "argument", "parameter", "param"}:
            current: Optional[Dict[str, str]] = None
            current_indent = 0
            for raw in block:
                if not raw.strip():
                    continue
                indent = len(raw) - len(raw.lstrip())
                text = raw.strip()
                match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$", text)
                if match:
                    if current:
                        params.append(current)
                    current = {
                        "name": match.group(1),
                        "type": (match.group(2) or "").strip(),
                        "description": match.group(3).strip(),
                    }
                    current_indent = indent
                    continue
                match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^#]+)$", text)
                if match:
                    if current:
                        params.append(current)
                    current = {
                        "name": match.group(1),
                        "type": match.group(2).strip(),
                        "description": "",
                    }
                    current_indent = indent
                    continue
                if current and indent > current_indent:
                    current["description"] = (current.get("description", "") + " " + text).strip()
            if current:
                params.append(current)
        elif kind == "return":
            for raw in block:
                if not raw.strip():
                    continue
                text = raw.strip()
                match = re.match(r"^([A-Za-z_][A-Za-z0-9_\[\], ]*)\s*:\s*(.*)$", text)
                if match and not returns_type:
                    returns_type = match.group(1).strip()
                    returns = match.group(2).strip()
                elif not returns:
                    returns = text
                else:
                    returns = (returns + " " + text).strip()
        elif kind == "raise":
            current: Optional[Dict[str, str]] = None
            current_indent = 0
            for raw in block:
                if not raw.strip():
                    continue
                indent = len(raw) - len(raw.lstrip())
                text = raw.strip()
                match = re.match(r"^([A-Za-z_][A-Za-z0-9_\.]*)\s*:\s*(.*)$", text)
                if match:
                    if current:
                        raises.append(current)
                    current = {"type": match.group(1), "description": match.group(2).strip()}
                    current_indent = indent
                    continue
                if current and indent > current_indent:
                    current["description"] = (current.get("description", "") + " " + text).strip()
            if current:
                raises.append(current)
        elif kind == "example":
            example_text = "\\n".join(line.rstrip() for line in block).strip()
            if example_text:
                examples.append(example_text)

    summary = truncate_doc(summary, 160) if summary else ""
    description = truncate_doc("\\n".join([line for line in description_lines if line]).strip(), 400)
    if len(params) > 6:
        params = params[:6]
    if len(examples) > 2:
        examples = examples[:2]
    examples = [truncate_doc(example, 200) for example in examples if example]
    return {
        "summary": summary,
        "description": description,
        "params": params,
        "returns": returns,
        "returns_type": returns_type,
        "raises": raises[:4],
        "examples": examples,
    }


def attach_symbol_docs(
    repo: Path,
    symbol_nodes: Dict[str, Dict[str, Any]],
    *,
    limit_paths: Optional[Set[str]] = None,
) -> None:
    symbols_by_path: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for sid, sym in symbol_nodes.items():
        if sym.get("doc_1l"):
            continue
        defined_in = sym.get("defined_in") or {}
        path = defined_in.get("path")
        line = defined_in.get("line", 0)
        if not isinstance(path, str) or not path or not isinstance(line, int):
            continue
        if limit_paths is not None and path not in limit_paths:
            continue
        if line <= 0:
            continue
        symbols_by_path[path].append((sid, line))

    for path, items in symbols_by_path.items():
        file_path = repo / path
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(content) > 400_000:
            continue
        lines = content.splitlines()
        ext = Path(path).suffix.lower()
        if ext in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
            blocks = jsdoc_blocks(lines)
            blocks.sort(key=lambda block: int(block["end"]))
            for sid, line in items:
                doc = jsdoc_for_symbol_line(lines, blocks, line)
                if not doc:
                    continue
                symbol_nodes[sid]["doc_1l"] = doc.get("summary", "")
                symbol_nodes[sid]["documentation"] = doc
        elif ext == ".py":
            for sid, line in items:
                doc = extract_py_docstring(lines, line)
                if not doc:
                    continue
                symbol_nodes[sid]["doc_1l"] = doc.get("summary", "")
                symbol_nodes[sid]["documentation"] = doc
