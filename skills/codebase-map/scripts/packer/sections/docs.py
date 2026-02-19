from __future__ import annotations

import re
from typing import Any, Dict, List

from ..digest_core import compact_doc


def split_params(params_text: str) -> List[str]:
    parts: List[str] = []
    current = ""
    depth = 0
    for ch in params_text:
        if ch in {"(", "[", "{"}:
            depth += 1
        elif ch in {")", "]", "}"}:
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def signature_param_details(signature: str) -> List[Dict[str, object]]:
    if "(" not in signature or ")" not in signature:
        return []
    params_text = signature[signature.find("(") + 1 : signature.rfind(")")]
    if not params_text.strip():
        return []
    details: List[Dict[str, object]] = []
    for raw in split_params(params_text):
        part = raw.strip()
        if not part or part in {"*", "/"}:
            continue
        variadic = part.startswith("...") or part.startswith("*")
        part = part.lstrip(".")
        part = part.lstrip("*")
        if part.startswith("{") or part.startswith("["):
            continue
        has_default = "=" in part
        name_part = part
        type_part = ""
        if ":" in part:
            name_part, type_part = part.split(":", 1)
        if "=" in name_part:
            name_part = name_part.split("=", 1)[0]
        name = name_part.strip().rstrip("?")
        if not name or name in {"self", "cls"}:
            continue
        optional = ("?" in name_part) or has_default
        type_part = type_part.split("=", 1)[0].strip()
        details.append(
            {
                "name": name,
                "optional": optional,
                "variadic": variadic,
                "has_default": has_default,
                "type": type_part,
            }
        )
    return details


def normalize_type_name(value: str) -> str:
    if not value:
        return ""
    raw = value.strip().lower()
    if not raw:
        return ""
    token_match = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", raw)
    if not token_match:
        return ""
    token = token_match[0]
    if "." in token:
        token = token.split(".")[-1]
    synonyms = {
        "str": "string",
        "string": "string",
        "bool": "boolean",
        "boolean": "boolean",
        "int": "int",
        "integer": "int",
        "float": "float",
        "number": "number",
        "list": "list",
        "dict": "dict",
        "map": "dict",
        "object": "object",
        "any": "any",
        "unknown": "unknown",
        "none": "none",
        "void": "void",
        "null": "null",
        "undefined": "undefined",
    }
    return synonyms.get(token, token)


def doc_quality_lines(
    symbols: Dict[str, Dict[str, Any]],
    files: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    strict: bool = False,
    max_items: int = 6,
) -> List[str]:
    if not symbols:
        return []
    total = 0
    missing = 0
    empty_summary = 0
    param_mismatch = 0
    param_missing_all = 0
    return_missing = 0
    type_mismatch = 0
    type_missing = 0
    return_type_missing = 0
    offenders: List[tuple[float, str, Dict[str, Any]]] = []
    for sym in symbols.values():
        if sym.get("visibility") == "private":
            continue
        kind = sym.get("kind", "")
        if kind not in {"function", "class", "component", "const"}:
            continue
        total += 1
        doc = sym.get("documentation") if isinstance(sym.get("documentation"), dict) else {}
        summary = str(doc.get("summary") or sym.get("doc_1l") or "").strip()
        if not summary:
            missing += 1
            offenders.append((float(sym.get("score", 0.0)), "missing_doc", sym))
            continue
        if not summary.strip():
            empty_summary += 1
        signature = sym.get("signature", "")
        sig_params = signature_param_details(signature)
        sig_param_names = [p["name"] for p in sig_params if p.get("name")]
        required_params = [
            p["name"]
            for p in sig_params
            if not p.get("optional") and not p.get("variadic") and p.get("name")
        ]
        doc_params = [
            p.get("name")
            for p in doc.get("params", [])
            if isinstance(p, dict) and p.get("name")
        ]
        if required_params and not doc_params:
            param_missing_all += 1
            offenders.append((float(sym.get("score", 0.0)), "param_missing_all", sym))
        elif doc_params:
            missing_params = set(required_params) - set(doc_params)
            extra_params = set(doc_params) - set(sig_param_names)
            if missing_params or extra_params:
                param_mismatch += 1
                offenders.append((float(sym.get("score", 0.0)), "param_mismatch", sym))
        # Type checks
        for param in sig_params:
            name = param.get("name")
            if not name:
                continue
            sig_type = normalize_type_name(str(param.get("type") or ""))
            doc_type = ""
            for doc_param in doc.get("params", []):
                if isinstance(doc_param, dict) and doc_param.get("name") == name:
                    doc_type = normalize_type_name(str(doc_param.get("type") or ""))
                    break
            if sig_type and doc_type:
                if (
                    sig_type != doc_type
                    and sig_type not in {"any", "unknown", "object"}
                    and doc_type not in {"any", "unknown", "object"}
                ):
                    type_mismatch += 1
                    offenders.append((float(sym.get("score", 0.0)), "type_mismatch", sym))
            elif strict and sig_type and not doc_type:
                type_missing += 1
                offenders.append((float(sym.get("score", 0.0)), "type_missing", sym))

        ret_type = ""
        if "->" in signature:
            ret_type = signature.split("->", 1)[1].strip()
        ret_type_norm = normalize_type_name(ret_type)
        if ret_type_norm in {"none", "void", "undefined", "null"}:
            ret_type_norm = ""
        doc_returns = str(doc.get("returns") or "").strip()
        if ret_type_norm and not doc_returns:
            return_missing += 1
            offenders.append((float(sym.get("score", 0.0)), "return_missing", sym))
        if strict and ret_type_norm and doc_returns:
            doc_ret_norm = normalize_type_name(doc_returns)
            if doc_ret_norm and doc_ret_norm != ret_type_norm:
                return_type_missing += 1
                offenders.append((float(sym.get("score", 0.0)), "return_type_mismatch", sym))

    if total == 0:
        return []
    lines = ["[DOCS_QUALITY]"]
    lines.append(
        "summary="
        + ",".join(
            [
                f"total={total}",
                f"missing={missing}",
                f"empty_summary={empty_summary}",
                f"param_missing_all={param_missing_all}",
                f"param_mismatch={param_mismatch}",
                f"return_missing={return_missing}",
                f"type_missing={type_missing}",
                f"type_mismatch={type_mismatch}",
                f"return_type_mismatch={return_type_missing}",
            ]
        )
    )
    offenders.sort(key=lambda item: item[0], reverse=True)
    for score, issue, sym in offenders[:max_items]:
        name = sym.get("name") or ""
        path = sym.get("defined_in", {}).get("path") if isinstance(sym.get("defined_in"), dict) else ""
        if not name:
            continue
        if path:
            lines.append(f"{issue} {name} {path}")
        else:
            lines.append(f"{issue} {name}")
    return lines if len(lines) > 1 else []


def doc_section_lines(
    symbols: Dict[str, Dict],
    symbol_alias: Dict[str, str],
    *,
    max_items: int = 8,
) -> List[str]:
    if not symbols or not symbol_alias:
        return []
    lines = ["[DOCS]"]
    for sid, alias in symbol_alias.items():
        sym = symbols.get(sid)
        if not sym:
            continue
        doc = sym.get("documentation") if isinstance(sym.get("documentation"), dict) else {}
        summary = str(doc.get("summary") or sym.get("doc_1l") or "").strip()
        if not summary:
            continue
        description = str(doc.get("description") or "").strip()
        line = f"{alias} {compact_doc(summary, 90)}"
        if description:
            line += " :: " + compact_doc(description, 120)
        lines.append(line)
        if len(lines) - 1 >= max_items:
            break
    return lines if len(lines) > 1 else []
