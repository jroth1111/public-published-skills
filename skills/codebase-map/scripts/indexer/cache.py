from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def language_for_path(path: str) -> str:
    if path.endswith(".tsx"):
        return "tsx"
    if path.endswith(".ts"):
        return "ts"
    if path.endswith(".jsx"):
        return "jsx"
    if path.endswith((".js", ".mjs", ".cjs")):
        return "js"
    if path.endswith(".py"):
        return "py"
    if path.endswith(".go"):
        return "go"
    if path.endswith(".rs"):
        return "rs"
    return "unknown"


def hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_file_hashes(repo: Path, files: Iterable[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for rel in files:
        full = repo / rel
        if full.is_file():
            hashes[rel] = hash_file(full)
    return hashes


def strip_js_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def strip_py_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "#" in line:
            before = line.split("#", 1)[0]
            lines.append(before)
        else:
            lines.append(line)
    return "\n".join(lines)


def semantic_hash(content: str, lang: str) -> str:
    if lang in {"ts", "tsx", "js", "jsx", "javascript"}:
        stripped = strip_js_comments(content)
    elif lang == "py" or lang == "python":
        stripped = strip_py_comments(content)
    else:
        stripped = content
    stripped = re.sub(r"\s+", "", stripped)
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()


def load_semantic_hash_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != 1:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    return payload


def save_semantic_hash_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except OSError:
        pass


def build_semantic_hashes(
    repo: Path,
    files: Iterable[str],
    *,
    file_hashes: Optional[Dict[str, str]] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    hashes: Dict[str, str] = {}
    cache_files: Dict[str, Any] = {}
    if isinstance(cache, dict):
        cached_files = cache.get("files")
        if isinstance(cached_files, dict):
            cache_files = cached_files
    for rel in files:
        full = repo / rel
        if not full.is_file():
            continue
        file_hash = file_hashes.get(rel) if isinstance(file_hashes, dict) else None
        cached = cache_files.get(rel) if isinstance(cache_files, dict) else None
        if (
            file_hash
            and isinstance(cached, dict)
            and cached.get("hash") == file_hash
            and cached.get("semantic")
        ):
            hashes[rel] = str(cached.get("semantic"))
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lang = language_for_path(rel)
        if not lang:
            continue
        sem = semantic_hash(content, lang)
        hashes[rel] = sem
        cache_files[rel] = {"hash": file_hash or "", "semantic": sem}
    return hashes, {"version": 1, "files": cache_files}


def load_ast_cache(path: Path, include_internal: bool) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != 1:
        return {}
    if payload.get("include_internal") != include_internal:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    return payload


def build_ast_cache(
    file_data: Dict[str, Dict[str, Any]],
    symbol_nodes: Dict[str, Dict[str, Any]],
    file_hashes: Dict[str, str],
    *,
    include_internal: bool,
) -> Dict[str, Any]:
    cache_files: Dict[str, Any] = {}
    for path, data in file_data.items():
        if path not in file_hashes:
            continue
        symbols = [symbol_nodes[sid] for sid in data.get("defines", []) if sid in symbol_nodes]
        cache_files[path] = {
            "hash": file_hashes.get(path, ""),
            "imports": list(data.get("imports", [])),
            "re_exports": list(data.get("re_exports", [])),
            "exports": list(data.get("exports", [])),
            "defines": list(data.get("defines", [])),
            "symbols": symbols,
        }
    return {
        "version": 1,
        "include_internal": include_internal,
        "files": cache_files,
    }
