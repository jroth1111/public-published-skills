from __future__ import annotations

import fnmatch
import json
import posixpath
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from evidence_model import score_candidate

from ..digest_core import classify_path_role, short_path
from .traces import is_test_path


def normalize_dynamic_segment(segment: str) -> str:
    if segment.startswith("[[...") and segment.endswith("]]"):
        return "*" + segment[5:-2]
    if segment.startswith("[...") and segment.endswith("]"):
        return "*" + segment[4:-1]
    if segment.startswith("[") and segment.endswith("]"):
        return ":" + segment[1:-1]
    return segment


def nextjs_route_path(path: str, prefix: str) -> Optional[str]:
    candidates = [path]
    if prefix and path.startswith(prefix):
        candidates.append(path[len(prefix) :])
    for candidate in candidates:
        for root in ("src/app/", "app/"):
            if candidate.startswith(root):
                base = candidate[len(root) :]
                return nextjs_route_path_from_base(base)
            marker = f"/{root}"
            idx = candidate.find(marker)
            if idx != -1:
                base = candidate[idx + len(marker) :]
                return nextjs_route_path_from_base(base)
        for root in ("src/pages/", "pages/"):
            if candidate.startswith(root):
                base = candidate[len(root) :]
                return nextjs_pages_path_from_base(base)
            marker = f"/{root}"
            idx = candidate.find(marker)
            if idx != -1:
                base = candidate[idx + len(marker) :]
                return nextjs_pages_path_from_base(base)
    return None


def nextjs_route_path_from_base(base: str) -> str:
    base = base.rsplit(".", 1)[0]
    if base.endswith("/page"):
        base = base[:-5]
    elif base.endswith("/route"):
        base = base[:-6]
    if base == "page" or base == "route":
        base = ""
    parts = [part for part in base.split("/") if part]
    parts = [normalize_dynamic_segment(part) for part in parts if part != "index"]
    return "/" + "/".join(parts) if parts else "/"


def nextjs_pages_path_from_base(base: str) -> str:
    base = base.rsplit(".", 1)[0]
    if base.endswith("/index"):
        base = base[:-6]
    elif base == "index":
        base = ""
    parts = [part for part in base.split("/") if part]
    parts = [normalize_dynamic_segment(part) for part in parts]
    return "/" + "/".join(parts) if parts else "/"


def nextjs_route_kind(path: str, prefix: str) -> str:
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    lower = path.lower()
    if lower.startswith(("pages/api/", "src/pages/api/")):
        return "api"
    if "/pages/api/" in lower:
        return "api"
    if lower.endswith(("/route.ts", "/route.tsx", "/route.js", "/route.jsx")):
        return "api"
    return "ui"


def infer_route_entry(path: str, prefix: str, routes_mode: str) -> Dict[str, str]:
    label: Optional[str] = None
    kind = "ui"
    if routes_mode == "nextjs":
        label = nextjs_route_path(path, prefix)
        kind = nextjs_route_kind(path, prefix)
    if not label:
        lower = path.lower()
        if "/api/" in lower or lower.endswith(("/api.py", "/api.ts", "/api.tsx", "/api.js", "/api.jsx")):
            kind = "api"
        canonical = path.lstrip("./")
        label = f"file:{canonical}"
    return {"label": label, "kind": kind}


def route_path_confidence(path: str) -> int:
    lower = path.lower()
    name = Path(lower).name
    if any(token in lower for token in ("/routes/", "/router/", "/routers/")):
        return 2
    if name in {
        "routes.py",
        "router.py",
        "urls.py",
        "routes.ts",
        "router.ts",
        "routes.js",
        "router.js",
        "routes.go",
        "router.go",
        "routes.rs",
        "router.rs",
    }:
        return 2
    if any(token in lower for token in ("/api/", "/pages/", "/controllers/", "/handlers/", "/endpoints/")):
        return 1
    return 0


def route_content_has_signal(path: str, content: str) -> bool:
    if not content:
        return False
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return bool(
            re.search(
                r"@(?:app|router|api_router|bp|blueprint)\.(?:get|post|put|delete|patch|route)\(",
                content,
            )
            or re.search(r"\b(?:path|re_path|url)\(\s*[\"']", content)
        )
    if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        if NEXT_ROUTE_METHODS_RE.search(content):
            return True
        return bool(
            re.search(r"\b(?:app|router|fastify|server)\.(?:get|post|put|delete|patch|use|all)\(", content)
            or re.search(r"create(?:Lazy)?FileRoute\(", content)
        )
    if ext == ".rs":
        return bool(
            re.search(r"#\[(?:get|post|put|patch|delete)\s*\([\"']", content)
            or re.search(r"\.route\(\s*[\"']", content)
            or re.search(r"\b(?:web::|routing::)(?:get|post|put|delete|patch|resource)\b", content)
        )
    if ext == ".go":
        return bool(
            re.search(r"\b(?:HandleFunc|Handle|GET|POST|PUT|DELETE|PATCH|Path|PathPrefix)\s*\(\s*[\"']", content)
            or re.search(r"\b(?:router|r|mux|engine|e)\.(?:GET|POST|PUT|DELETE|PATCH)\s*\(\s*[\"']", content)
        )
    return False


ROUTE_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
MAX_CONTENT_CACHE_BYTES = 200000
ROUTE_CONTENT_CACHE: Dict[str, str] = {}
AST_GREP_CACHE: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
FUNCTION_SPAN_CACHE: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
AST_GREP_AVAILABLE = True


def compact_route_label(label: str, max_len: int = 64) -> str:
    if len(label) <= max_len:
        return label
    if max_len <= 3:
        return label[:max_len]
    return label[: max_len - 3] + "..."


def read_route_content(repo_root: Optional[Path], path: str, limit: int = 20000) -> str:
    if not repo_root:
        return ""
    try:
        if path in ROUTE_CONTENT_CACHE:
            content = ROUTE_CONTENT_CACHE[path]
        else:
            content = (repo_root / path).read_text(encoding="utf-8", errors="ignore")
            if len(content) > MAX_CONTENT_CACHE_BYTES:
                content = content[:MAX_CONTENT_CACHE_BYTES]
            ROUTE_CONTENT_CACHE[path] = content
    except OSError:
        return ""
    if len(content) > limit:
        return content[:limit]
    return content


def lang_for_ast_grep(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".ts":
        return "ts"
    if suffix == ".tsx":
        return "tsx"
    if suffix == ".jsx":
        return "jsx"
    if suffix in {".js", ".mjs", ".cjs"}:
        return "js"
    if suffix == ".py":
        return "python"
    return ""


def run_ast_grep(repo_root: Optional[Path], path: str, lang: str, pattern: str) -> List[Dict[str, Any]]:
    global AST_GREP_AVAILABLE
    if not repo_root or not lang or not AST_GREP_AVAILABLE:
        return []
    key = (path, lang, pattern)
    if key in AST_GREP_CACHE:
        return AST_GREP_CACHE[key]
    cmd = ["ast-grep", "--lang", lang, "-p", pattern, "--json=stream", path]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        AST_GREP_AVAILABLE = False
        AST_GREP_CACHE[key] = []
        return []
    if result.returncode not in (0, 1):
        AST_GREP_CACHE[key] = []
        return []
    matches: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            matches.append(json.loads(line))
        except json.JSONDecodeError:
            break
    AST_GREP_CACHE[key] = matches
    return matches


def match_line(match: Dict[str, Any]) -> int:
    range_data = match.get("range")
    if not isinstance(range_data, dict):
        return 0
    start = range_data.get("start")
    if not isinstance(start, dict):
        return 0
    line = start.get("line")
    return int(line) + 1 if isinstance(line, int) else 0


def match_span(match: Dict[str, Any]) -> Tuple[int, int]:
    range_data = match.get("range")
    if not isinstance(range_data, dict):
        return 0, 0
    start = range_data.get("start")
    end = range_data.get("end")
    start_line = int(start.get("line")) + 1 if isinstance(start, dict) and isinstance(start.get("line"), int) else 0
    end_line = int(end.get("line")) + 1 if isinstance(end, dict) and isinstance(end.get("line"), int) else 0
    if end_line and start_line and end_line < start_line:
        end_line = start_line
    return start_line, end_line


def meta_single(match: Dict[str, Any], name: str) -> Optional[str]:
    meta = match.get("metaVariables") or {}
    single = meta.get("single") if isinstance(meta, dict) else {}
    if isinstance(single, dict):
        value = single.get(name) or {}
        if isinstance(value, dict):
            text = value.get("text")
            return str(text) if text is not None else None
    return None


def strip_string_literal(value: str) -> str:
    if not value:
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] in {"'", '"', "`"} and text[-1] == text[0]:
        return text[1:-1]
    return text


def ast_route_calls_for_file(
    path: str,
    *,
    repo_root: Optional[Path],
    frameworks: Optional[Set[str]],
) -> List[Tuple[str, str, int, bool]]:
    lang = lang_for_ast_grep(path)
    if not lang or lang == "python":
        return []
    allow = frameworks or {"hono", "express", "fastify"}
    if not allow.intersection({"hono", "express", "fastify"}):
        return []
    obj_names = {"app", "router", "fastify"}
    entries: List[Tuple[str, str, int, bool]] = []
    for match in run_ast_grep(repo_root, path, lang, "$OBJ.$METHOD($PATH, $$$)"):
        obj = (meta_single(match, "OBJ") or "").strip()
        method = (meta_single(match, "METHOD") or "").lower()
        raw_path = meta_single(match, "PATH") or ""
        if obj and obj not in obj_names:
            continue
        if method not in ROUTE_METHODS and method != "use":
            continue
        route_path = strip_string_literal(raw_path)
        if not route_path or not route_path.startswith("/"):
            continue
        line = match_line(match)
        entries.append((method, route_path, line, method == "use"))
    for match in run_ast_grep(repo_root, path, lang, "$OBJ.on($METHODS, $PATH, $$$)"):
        obj = (meta_single(match, "OBJ") or "").strip()
        if obj and obj not in obj_names:
            continue
        raw_methods = meta_single(match, "METHODS") or ""
        raw_path = meta_single(match, "PATH") or ""
        route_path = strip_string_literal(raw_path)
        if not route_path or not route_path.startswith("/"):
            continue
        methods = re.findall(r"[\"'`]([A-Za-z]+)[\"'`]", raw_methods)
        method_label = ",".join(sorted({m.lower() for m in methods if m})) or "on"
        line = match_line(match)
        entries.append((method_label, route_path, line, False))
    return entries


def function_spans_for_file(path: str, *, repo_root: Optional[Path]) -> List[Tuple[int, int]]:
    lang = lang_for_ast_grep(path)
    if not lang or not repo_root:
        return []
    key = (path, lang)
    if key in FUNCTION_SPAN_CACHE:
        return FUNCTION_SPAN_CACHE[key]
    patterns: List[str] = []
    if lang in {"ts", "tsx", "js", "jsx"}:
        patterns = [
            "function $NAME($$$) { $$$ }",
            "($$$) => { $$$ }",
            "async ($$$) => { $$$ }",
            "($$$) => $BODY",
            "async ($$$) => $BODY",
        ]
    elif lang == "python":
        patterns = [
            "def $NAME($$$):\n    $$$BODY",
            "async def $NAME($$$):\n    $$$BODY",
        ]
    spans: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for pattern in patterns:
        for match in run_ast_grep(repo_root, path, lang, pattern):
            start_line, end_line = match_span(match)
            if start_line and end_line:
                span = (start_line, end_line)
                if span not in seen:
                    seen.add(span)
                    spans.append(span)
    spans.sort(key=lambda item: (item[0], item[1]))
    FUNCTION_SPAN_CACHE[key] = spans
    return spans


def function_body_for_line(content: str, path: str, line_no: int, *, repo_root: Optional[Path]) -> str:
    if not content or line_no <= 0:
        return ""
    spans = function_spans_for_file(path, repo_root=repo_root)
    if not spans:
        return ""
    chosen: Optional[Tuple[int, int]] = None
    best_span = 1_000_000
    for start_line, end_line in spans:
        if start_line <= line_no <= end_line:
            span_size = end_line - start_line
            if span_size < best_span:
                best_span = span_size
                chosen = (start_line, end_line)
    if not chosen:
        return ""
    start_line, end_line = chosen
    lines = content.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def route_labels_from_content(
    path: str,
    content: str,
    routes_mode: str,
    *,
    repo_root: Optional[Path],
    max_labels: int = 24,
) -> List[Dict[str, str]]:
    if not content:
        return []
    ext = Path(path).suffix.lower()
    labels: List[Dict[str, str]] = []
    seen: Set[str] = set()
    label_limit = max_labels if max_labels > 0 else None

    def add(label: str, kind: str = "api") -> None:
        label = compact_route_label(label)
        if not label or label in seen:
            return
        seen.add(label)
        labels.append({"label": label, "kind": kind})

    def add_method(method: str, route_path: str) -> None:
        method = method.lower()
        if method in ROUTE_METHODS or method in {"route", "api_route", "use"}:
            add(f"{method}:{route_path}")

    def reached_limit() -> bool:
        return label_limit is not None and len(labels) >= label_limit

    if ext == ".py" and routes_mode in {"auto", "heuristic", "fastapi", "django"}:
        for match in re.finditer(
            r"@(?:app|router|api_router|bp|blueprint)\.([a-zA-Z_]+)\(\s*[\"']([^\"']+)[\"']",
            content,
        ):
            add_method(match.group(1), match.group(2))
            if reached_limit():
                return labels
        for match in re.finditer(r"\bpath\(\s*[\"']([^\"']+)[\"']", content):
            add(f"path:{match.group(1)}")
            if reached_limit():
                return labels
        for match in re.finditer(r"\b(?:re_path|url)\(\s*r?[\"']([^\"']+)[\"']", content):
            add(f"re_path:{match.group(1)}")
            if reached_limit():
                return labels
        for match in re.finditer(r"@(?:app|blueprint|bp)\.route\(\s*[\"']([^\"']+)[\"']", content):
            add(f"route:{match.group(1)}")
            if reached_limit():
                return labels

    if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"} and routes_mode in {
        "auto",
        "heuristic",
        "express",
    }:
        ast_entries = ast_route_calls_for_file(
            path,
            repo_root=repo_root,
            frameworks={"express", "fastify", "hono"},
        )
        for method, route_path, _line, is_middleware in ast_entries:
            if is_middleware:
                continue
            add_method(method, route_path)
            if reached_limit():
                return labels
        for match in re.finditer(
            r"\b(?:app|router|fastify)\.([a-zA-Z_]+)\(\s*[\"']([^\"']+)[\"']",
            content,
        ):
            add_method(match.group(1), match.group(2))
            if reached_limit():
                return labels

    if ext == ".rs" and routes_mode in {"auto", "heuristic", "rust"}:
        for match in re.finditer(r"#\[(get|post|put|patch|delete)\s*\(\s*[\"']([^\"']+)[\"']", content):
            add_method(match.group(1), match.group(2))
            if reached_limit():
                return labels
        for match in re.finditer(
            r"\.route\(\s*[\"']([^\"']+)[\"']\s*,\s*(get|post|put|patch|delete)\s*\(",
            content,
        ):
            add_method(match.group(2), match.group(1))
            if reached_limit():
                return labels
        for match in re.finditer(r"\b(?:web::resource|Router::route)\(\s*[\"']([^\"']+)[\"']", content):
            add(f"route:{match.group(1)}")
            if reached_limit():
                return labels

    if ext == ".go" and routes_mode in {"auto", "heuristic", "go"}:
        for match in re.finditer(
            r"\b(?:router|r|mux|engine|e)\.(GET|POST|PUT|DELETE|PATCH)\(\s*[\"']([^\"']+)[\"']",
            content,
        ):
            add_method(match.group(1).lower(), match.group(2))
            if reached_limit():
                return labels
        for match in re.finditer(
            r"\b(?:http|router|r|mux)\.(?:HandleFunc|Handle|Path|PathPrefix)\(\s*[\"']([^\"']+)[\"']",
            content,
        ):
            add(f"handle:{match.group(1)}")
            if reached_limit():
                return labels

    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        for match in re.finditer(
            r"create(?:Lazy)?FileRoute\(\s*[\"']([^\"']+)[\"']\s*\)",
            content,
        ):
            add(match.group(1), kind="ui")
            if reached_limit():
                return labels

    return labels


def route_entries_for_file(
    *,
    path: str,
    prefix: str,
    routes_mode: str,
    repo_root: Optional[Path],
    entrypoint_role: Optional[str] = None,
    max_labels: int = 24,
) -> List[Dict[str, str]]:
    if routes_mode == "nextjs":
        return [infer_route_entry(path, prefix, routes_mode)]
    content = read_route_content(repo_root, path)
    labels = route_labels_from_content(
        path,
        content,
        routes_mode,
        repo_root=repo_root,
        max_labels=max_labels,
    )
    if labels:
        inferred = infer_route_entry(path, prefix, routes_mode)
        inferred_label = str(inferred.get("label") or "")
        if inferred_label.startswith("file:") and not any(
            str(item.get("label") or "") == inferred_label for item in labels
        ):
            labels = [inferred] + labels
        return labels
    role = str(entrypoint_role or "").lower()
    if role in HTTP_ROUTE_ENTRYPOINT_ROLES:
        inferred = infer_route_entry(path, prefix, routes_mode)
        # Entry-point-derived routes are API-facing unless proven UI-specific.
        if inferred.get("kind") == "ui":
            inferred["kind"] = "api"
        return [inferred]
    confidence = route_path_confidence(path)
    if confidence >= 2:
        return [infer_route_entry(path, prefix, routes_mode)]
    if confidence >= 1 and route_content_has_signal(path, content):
        return [infer_route_entry(path, prefix, routes_mode)]
    return []


def line_number_for_offset(content: str, offset: int) -> int:
    return content[:offset].count("\n") + 1


def format_loc(path: str, line: int, prefix: str) -> str:
    if not path:
        return ""
    short = short_path(path, prefix, max_segments=4)
    return f"{short}:{line}" if line else short


def extract_app_router_keys(content: str) -> List[Tuple[str, int]]:
    lines = content.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if "appRouter" in line and "{" in line:
            start_idx = idx
            break
    if start_idx is None:
        return []
    keys: List[Tuple[str, int]] = []
    depth = 0
    for idx in range(start_idx, len(lines)):
        line = lines[idx]
        depth += line.count("{") - line.count("}")
        if idx == start_idx:
            continue
        if depth <= 0:
            break
        if depth == 1:
            match = re.match(r"\s*([A-Za-z0-9_]+)\s*:", line)
            if match:
                keys.append((match.group(1), idx + 1))
    return keys


def extract_router_object_keys(content: str) -> List[Tuple[str, int, str]]:
    lines = content.splitlines()
    entries: List[Tuple[str, int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"\s*export\s+const\s+([A-Za-z0-9_]+Router)\s*=\s*{", line)
        if not match:
            continue
        router_name = match.group(1)
        depth = line.count("{") - line.count("}")
        for jdx in range(idx + 1, len(lines)):
            current = lines[jdx]
            if depth == 1:
                key_match = re.match(r"\s*([A-Za-z0-9_]+)\s*:", current)
                if key_match:
                    entries.append((router_name, jdx + 1, key_match.group(1)))
            depth += current.count("{") - current.count("}")
            if depth <= 0:
                break
    return entries


def extract_router_object_keys_from_span(
    lines: List[str],
    *,
    start_line: int,
    router_name: str,
) -> List[Tuple[str, int, str]]:
    entries: List[Tuple[str, int, str]] = []
    depth = 0
    for idx, line in enumerate(lines):
        depth += line.count("{") - line.count("}")
        if depth == 1:
            key_match = re.match(r"\s*([A-Za-z0-9_]+)\s*:", line)
            if key_match:
                entries.append((router_name, start_line + idx, key_match.group(1)))
        if depth <= 0 and idx > 0:
            break
    return entries


def ast_router_object_entries_for_file(
    path: str,
    *,
    repo_root: Optional[Path],
) -> List[Tuple[str, int, str]]:
    lang = lang_for_ast_grep(path)
    if not lang or lang == "python":
        return []
    content = read_route_content(repo_root, path, limit=40000)
    if not content:
        return []
    entries: List[Tuple[str, int, str]] = []
    for pattern in ("export const $NAME = { $$$ }", "const $NAME = { $$$ }"):
        for match in run_ast_grep(repo_root, path, lang, pattern):
            name = (meta_single(match, "NAME") or "").strip()
            if not name or "router" not in name.lower():
                continue
            start_line, end_line = match_span(match)
            if not start_line or not end_line:
                continue
            lines = content.splitlines()[start_line - 1 : end_line]
            entries.extend(
                extract_router_object_keys_from_span(
                    lines,
                    start_line=start_line,
                    router_name=name,
                )
            )
    return entries


def normalize_router_name(name: str) -> str:
    base = name
    if base.endswith("Router"):
        base = base[: -len("Router")]
        if base:
            base = base[0].lower() + base[1:]
    return base or name


def extract_exported_functions(content: str) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    for match in re.finditer(r"^\s*export\s+(?:async\s+)?function\s+([A-Za-z0-9_]+)", content, re.MULTILINE):
        name = match.group(1)
        if name.endswith(("Input", "Schema", "Status", "Enum", "Config", "Error", "Errors")):
            continue
        entries.append((name, line_number_for_offset(content, match.start())))
    for match in re.finditer(
        r"^\s*export\s+const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(",
        content,
        re.MULTILINE,
    ):
        name = match.group(1)
        if name.endswith(("Input", "Schema", "Status", "Enum", "Config", "Error", "Errors")):
            continue
        entries.append((name, line_number_for_offset(content, match.start())))
    return entries


def extract_tanstack_routes(content: str) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    for match in re.finditer(
        r"create(?:Lazy)?FileRoute\(\s*['\"]([^'\"]+)['\"]\s*\)",
        content,
    ):
        entries.append((match.group(1), line_number_for_offset(content, match.start())))
    return entries


def extract_hono_entries(content: str) -> List[Tuple[str, str, int]]:
    entries: List[Tuple[str, str, int]] = []
    for match in re.finditer(
        r"\bapp\.on\(\s*\[([^\]]+)\]\s*,\s*([\"'`])([^\"'`]+)\2",
        content,
        re.IGNORECASE,
    ):
        methods = re.findall(r"[\"']([A-Z]+)[\"']", match.group(1))
        label = ",".join(methods) if methods else "on"
        entries.append((label.lower(), match.group(3), line_number_for_offset(content, match.start())))
    for match in re.finditer(
        r"\bapp\.(get|post|put|delete|patch|options|head)\(\s*([\"'`])([^\"'`]+)\2",
        content,
        re.IGNORECASE,
    ):
        entries.append((match.group(1).lower(), match.group(3), line_number_for_offset(content, match.start())))
    for match in re.finditer(
        r"\bapp\.use\(\s*([\"'`])([^\"'`]+)\1",
        content,
        re.IGNORECASE,
    ):
        entries.append(("use", match.group(2), line_number_for_offset(content, match.start())))
    return entries


def entrypoint_domain(kind: str, label: str, path: str) -> str:
    lower_path = path.lower()
    if kind in {"http", "ui", "job"}:
        return kind
    if "/routers/" in lower_path:
        if "/routers/index." in lower_path:
            return "rpc"
        name = Path(lower_path).stem
        name = name.replace("-router", "").replace("_router", "")
        if name.endswith("router"):
            name = name[: -len("router")]
        return name or "rpc"
    if "/services/" in lower_path:
        return Path(lower_path).stem
    if "/jobs/" in lower_path:
        return "jobs"
    if "." in label:
        base = label.split(".", 1)[0]
        base = base.replace("Router", "")
        return base or "rpc"
    return kind or "entry"


def entrypoint_group_lines(
    entries: List[Dict[str, object]],
    *,
    prefix: str,
    max_groups: int = 18,
    max_items: int = 8,
) -> List[str]:
    if not entries:
        return []
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in entries:
        domain = str(entry.get("domain") or "entry")
        groups[domain].append(entry)
    order = ["http", "rpc", "jobs", "ui"]
    ordered_domains = order + sorted(
        [name for name in groups.keys() if name not in order]
    )
    lines = ["[ENTRYPOINT_GROUPS]"]
    listed = 0
    for domain in ordered_domains:
        items = groups.get(domain, [])
        if not items:
            continue
        if listed >= max_groups:
            break
        items_sorted = sorted(
            items,
            key=lambda item: (
                0 if "/routers/" in str(item.get("path", "")) else 1,
                0 if str(item.get("kind")) == "http" else 1,
                str(item.get("label", "")),
            ),
        )
        rendered: List[str] = []
        seen: Set[str] = set()
        for item in items_sorted:
            if len(rendered) >= max_items:
                break
            path = str(item.get("path") or "")
            line = int(item.get("line") or 0)
            loc = format_loc(path, line, prefix)
            label = str(item.get("label") or "")
            kind = str(item.get("kind") or "")
            if domain == "ui":
                token = f"{label}@{loc}" if label else loc
            elif domain in {"jobs", "job"}:
                token = f"{label}@{loc}" if label else loc
            elif domain == "http":
                token = loc
            elif domain == "rpc":
                token = loc
            else:
                token = loc
            if token in seen or not token:
                continue
            seen.add(token)
            rendered.append(token)
        if not rendered:
            continue
        lines.append(f"G{listed + 1} {domain} " + ", ".join(rendered))
        listed += 1
    return lines if listed > 0 else []


def middleware_lines(
    entries: List[Dict[str, object]],
    *,
    prefix: str,
    max_items: int = 12,
) -> List[str]:
    if not entries:
        return []
    lines = ["[MIDDLEWARE]", f"count={len(entries)}"]
    listed = 0
    seen: Set[str] = set()
    for entry in entries:
        if listed >= max_items:
            break
        label = str(entry.get("label") or "")
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 0)
        loc = format_loc(path, line, prefix)
        token = f"{label} {loc}".strip()
        if not token or token in seen:
            continue
        seen.add(token)
        lines.append(f"M{listed + 1} {token}")
        listed += 1
    return lines if listed else []


def entrypoint_inventory_lines(
    files: Dict[str, Dict],
    *,
    prefix: str,
    repo_root: Optional[Path],
    max_items: int = 120,
    max_per_group: int = 24,
    frameworks: Optional[Set[str]] = None,
    include_middleware: bool = False,
) -> Tuple[List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    if not repo_root:
        return [], [], []
    paths = [node.get("path", "") for node in files.values() if node.get("path")]
    lines: List[str] = ["[ENTRYPOINTS]"]
    entries: List[Dict[str, object]] = []
    middleware_entries: List[Dict[str, object]] = []

    def add_entry(kind: str, label: str, path: str, line: int) -> None:
        domain = entrypoint_domain(kind, label, path)
        entries.append(
            {"kind": kind, "label": label, "path": path, "line": line, "domain": domain}
        )

    def add_middleware(label: str, path: str, line: int) -> None:
        middleware_entries.append({"label": label, "path": path, "line": line})

    active_frameworks = {item.lower() for item in frameworks} if frameworks else set()
    allow_http = not active_frameworks or active_frameworks.intersection({"hono", "express", "fastify"})
    allow_rpc = not active_frameworks or active_frameworks.intersection({"rpc", "router"})

    server_paths = [
        path
        for path in paths
        if re.search(r"/server/src/index\.(ts|tsx|js|jsx)$", path)
    ]
    for path in server_paths:
        content = read_route_content(repo_root, path, limit=40000)
        if not content:
            continue
        if allow_http and re.search(r"\bnew\s+Hono\b", content):
            match = re.search(r"\bnew\s+Hono\b", content)
            line = line_number_for_offset(content, match.start()) if match else 0
            add_entry("http", "hono-app", path, line)
        if allow_http:
            for match in re.finditer(r"\bnew\s+(RPCHandler|OpenAPIHandler)\b", content):
                add_entry(
                    "http",
                    match.group(1).lower(),
                    path,
                    line_number_for_offset(content, match.start()),
                )
            http_entries = ast_route_calls_for_file(
                path,
                repo_root=repo_root,
                frameworks=active_frameworks or None,
            )
            if not http_entries:
                http_entries = [
                    (method, route_path, line, method == "use")
                    for method, route_path, line in extract_hono_entries(content)
                ]
            for method, route_path, line, is_middleware in http_entries:
                label = f"{method}:{route_path}"
                if is_middleware or method == "use":
                    add_middleware(label, path, line)
                    if include_middleware:
                        add_entry("http", label, path, line)
                else:
                    add_entry("http", label, path, line)
        for match in re.finditer(r"\bexport\s+const\s+scheduled\b", content):
            add_entry(
                "job",
                "scheduled",
                path,
                line_number_for_offset(content, match.start()),
            )

    router_index_paths = [
        path
        for path in paths
        if re.search(r"/routers/index\.(ts|tsx|js|jsx)$", path)
    ]
    for path in router_index_paths:
        content = read_route_content(repo_root, path, limit=20000)
        if not content:
            continue
        if not allow_rpc:
            continue
        match = re.search(r"export\s+const\s+appRouter\b", content)
        if match:
            add_entry("rpc", "appRouter", path, line_number_for_offset(content, match.start()))
        ast_keys = [
            (router_name, line, key)
            for router_name, line, key in ast_router_object_entries_for_file(
                path, repo_root=repo_root
            )
            if router_name == "appRouter"
        ]
        if ast_keys:
            for _router_name, line, key in ast_keys[:max_per_group]:
                add_entry("rpc", key, path, line)
        else:
            keys = extract_app_router_keys(content)
            for key, line in keys[:max_per_group]:
                add_entry("rpc", key, path, line)

    router_paths = [
        path
        for path in paths
        if "/routers/" in path and path.endswith((".ts", ".tsx", ".js", ".jsx"))
    ]
    for path in sorted(router_paths):
        if path.endswith(("/routers/index.ts", "/routers/index.tsx", "/routers/index.js", "/routers/index.jsx")):
            continue
        if not allow_rpc:
            continue
        content = read_route_content(repo_root, path, limit=20000)
        if not content:
            continue
        ast_entries = ast_router_object_entries_for_file(path, repo_root=repo_root)
        if not ast_entries:
            ast_entries = extract_router_object_keys(content)
        for router_name, line, key in ast_entries[:max_per_group]:
            base = normalize_router_name(router_name)
            add_entry("rpc", f"{base}.{key}", path, line)

    service_paths = [
        path
        for path in paths
        if "/services/" in path and path.endswith((".ts", ".tsx", ".js", ".jsx"))
    ]
    job_paths = [
        path
        for path in paths
        if "/jobs/" in path and path.endswith((".ts", ".tsx", ".js", ".jsx"))
    ]
    job_entries: List[Tuple[str, str, str, int]] = []
    for path in sorted(job_paths):
        content = read_route_content(repo_root, path, limit=12000)
        if not content:
            continue
        for name, line in extract_exported_functions(content)[:3]:
            job_entries.append(("job", name, path, line))
    for kind, label, path, line in job_entries[:max_per_group]:
        add_entry(kind, label, path, line)

    service_entries: List[Tuple[str, str, str, int]] = []
    for path in sorted(service_paths):
        content = read_route_content(repo_root, path, limit=40000)
        if not content:
            continue
        if allow_rpc:
            router_keys = ast_router_object_entries_for_file(path, repo_root=repo_root)
            if not router_keys:
                router_keys = extract_router_object_keys(content)
            for router_name, line, key in router_keys[: max_per_group // 2]:
                base = normalize_router_name(router_name)
                service_entries.append(("rpc", f"{base}.{key}", path, line))
        funcs = extract_exported_functions(content)
        for name, line in funcs[: max_per_group // 2]:
            service_entries.append(("service", name, path, line))
    for kind, label, path, line in service_entries[: max_per_group * 2]:
        add_entry(kind, label, path, line)

    ui_route_paths = [
        path
        for path in paths
        if "/routes/" in path and path.endswith((".tsx", ".jsx", ".ts", ".js"))
    ]
    ui_entries: List[Tuple[str, str, str, int]] = []
    for path in sorted(ui_route_paths):
        content = read_route_content(repo_root, path, limit=12000)
        if not content:
            continue
        for route_path, line in extract_tanstack_routes(content)[:1]:
            ui_entries.append(("ui", route_path, path, line))
    for kind, label, path, line in ui_entries[:max_per_group]:
        add_entry(kind, label, path, line)

    if not entries:
        fallback_seen: Set[str] = set()
        for node in files.values():
            path = str(node.get("path") or "")
            if not path or is_test_path(path):
                continue
            role = str(node.get("entrypoint_role") or "")
            if node.get("role") != "entrypoint" and not role:
                continue
            kind = role or "entry"
            key = f"{kind}:{path}"
            if key in fallback_seen:
                continue
            fallback_seen.add(key)
            add_entry(kind, Path(path).name, path, 0)

    if not entries:
        return [], [], middleware_entries

    lines.append(f"count={len(entries)}")
    for idx, entry in enumerate(entries[:max_items], start=1):
        kind = str(entry.get("kind") or "")
        label = str(entry.get("label") or "")
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 0)
        loc = format_loc(path, line, prefix)
        lines.append(f"E{idx} {kind} {label} {loc}")
    return lines, entries, middleware_entries


def extract_function_body(content: str, start_offset: int) -> str:
    if start_offset < 0 or start_offset >= len(content):
        return ""
    brace_idx = content.find("{", start_offset)
    if brace_idx == -1:
        return ""
    depth = 0
    for idx in range(brace_idx, len(content)):
        ch = content[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[brace_idx + 1 : idx]
    return ""


def summarize_steps_from_body(
    body: str,
    max_steps: int = 4,
    *,
    skip_names: Optional[Set[str]] = None,
) -> List[str]:
    steps: List[str] = []
    seen: Set[str] = set()
    skip_tokens = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "catch",
        "then",
        "await",
        "const",
        "let",
        "function",
        "async",
        "new",
        "get",
        "post",
        "use",
        "on",
        "input",
        "handler",
    }
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "req.text" in stripped or "req.json" in stripped or "request.json" in stripped:
            token = "read_body"
        elif "JSON.parse" in stripped:
            token = "parse_json"
        elif "signature" in stripped and ("verify" in stripped or "hash" in stripped):
            token = "verify_signature"
        elif "return" in stripped and ".json" in stripped:
            token = "respond_json"
        elif "return" in stripped and ".text" in stripped:
            token = "respond_text"
        else:
            match = re.search(r"\bawait\s+([A-Za-z0-9_]+)\b", stripped)
            if match:
                token = f"call:{match.group(1)}"
            else:
                match = re.search(r"\b([A-Za-z0-9_]+)\s*\(", stripped)
                token = f"call:{match.group(1)}" if match else ""
        if token and token.startswith("call:"):
            name = token.split(":", 1)[1]
            if name in skip_tokens or (skip_names and name in skip_names):
                token = ""
        if not token or token in seen:
            continue
        seen.add(token)
        steps.append(token)
        if len(steps) >= max_steps:
            break
    return steps


def extract_preconditions(body: str, max_items: int = 3) -> List[str]:
    prefixes = ("require", "assert", "ensure", "validate", "check", "guard", "verify")
    seen: Set[str] = set()
    items: List[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = re.search(r"\bawait\s+([A-Za-z0-9_]+)\b", stripped)
        name = match.group(1) if match else ""
        if not name:
            match = re.search(r"\b([A-Za-z0-9_]+)\s*\(", stripped)
            name = match.group(1) if match else ""
        if not name:
            continue
        lower = name.lower()
        if not lower.startswith(prefixes):
            continue
        if name in seen:
            continue
        seen.add(name)
        items.append(name)
        if len(items) >= max_items:
            break
    return items


def extract_input_schema(content: str) -> str:
    match = re.search(r"\.input\(\s*([A-Za-z0-9_]+)\s*\)", content)
    if match:
        return match.group(1)
    match = re.search(r"input:\s*z\.infer<\s*typeof\s+([A-Za-z0-9_]+)\s*>", content)
    if match:
        return match.group(1)
    return ""


def window_from_line(content: str, line_no: int, span: int = 20) -> str:
    if line_no <= 0:
        return ""
    lines = content.splitlines()
    start = max(0, line_no - 1)
    end = min(len(lines), start + span)
    return "\n".join(lines[start:end])


def offset_for_line(content: str, line_no: int) -> int:
    if line_no <= 0:
        return 0
    lines = content.splitlines(keepends=True)
    return sum(len(line) for line in lines[: max(0, line_no - 1)])


def extract_call_window(content: str, line_no: int, max_lines: int = 40) -> str:
    if line_no <= 0:
        return ""
    lines = content.splitlines()
    start = max(0, line_no - 1)
    paren = 0
    started = False
    collected: List[str] = []
    for idx in range(start, min(len(lines), start + max_lines)):
        line = lines[idx]
        if not started and "(" in line:
            started = True
        if started:
            paren += line.count("(") - line.count(")")
            collected.append(line)
            if paren <= 0 and idx > start:
                break
        else:
            collected.append(line)
    return "\n".join(collected)


def entrypoint_detail_lines(
    entries: List[Dict[str, object]],
    *,
    repo_root: Optional[Path],
    prefix: str,
    max_entries: int = 12,
    max_per_kind: int = 4,
) -> List[str]:
    if not repo_root or not entries:
        return []
    lines = ["[ENTRY_DETAILS]"]
    listed = 0
    ordered: List[Dict[str, object]] = []

    def pick_diverse(candidates: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
        picked: List[Dict[str, object]] = []
        seen_domains: Set[str] = set()
        for entry in candidates:
            domain = str(entry.get("domain") or "")
            if domain and domain in seen_domains:
                continue
            picked.append(entry)
            if domain:
                seen_domains.add(domain)
            if len(picked) >= limit:
                return picked
        for entry in candidates:
            if entry in picked:
                continue
            picked.append(entry)
            if len(picked) >= limit:
                break
        return picked
    for kind in ("http", "rpc", "service"):
        candidates = [entry for entry in entries if str(entry.get("kind") or "") == kind]
        if kind == "http":
            route_entries = [
                entry
                for entry in candidates
                if ":" in str(entry.get("label") or "")
                and not str(entry.get("label") or "").startswith("use:")
            ]
            candidates = route_entries
        elif kind == "rpc":
            method_entries: List[Dict[str, object]] = []
            top_entries: List[Dict[str, object]] = []
            for entry in candidates:
                label = str(entry.get("label") or "")
                path = str(entry.get("path") or "")
                if "." in label or ("/routers/" in path and "/routers/index." not in path):
                    method_entries.append(entry)
                else:
                    top_entries.append(entry)
            candidates = method_entries + top_entries
        elif kind == "service":
            verb_first: List[Dict[str, object]] = []
            rest: List[Dict[str, object]] = []
            for entry in candidates:
                label = str(entry.get("label") or "")
                if re.search(r"(list|create|update|delete|start|finalize|authenticate|upload|import|refund|cancel|approve|verify|process)", label, re.IGNORECASE):
                    verb_first.append(entry)
                else:
                    rest.append(entry)
            candidates = verb_first + rest
        ordered.extend(pick_diverse(candidates, max_per_kind))
    for entry in ordered:
        if listed >= max_entries:
            break
        kind = str(entry.get("kind") or "")
        label = str(entry.get("label") or "")
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 0)
        if kind not in {"http", "rpc", "service"}:
            continue
        content = read_route_content(repo_root, path, limit=40000)
        if not content:
            continue
        detail = ""
        body = ""
        if kind == "http":
            if ":" not in label or label.startswith("use:"):
                continue
            method, route_path = label.split(":", 1)
            body = function_body_for_line(content, path, line, repo_root=repo_root)
            if not body:
                body = extract_call_window(content, line, max_lines=30)
            detail = f"{method}:{route_path}"
        elif kind in {"rpc", "service"}:
            body = function_body_for_line(content, path, line, repo_root=repo_root)
            detail = label
        if not body and line:
            body = window_from_line(content, line, span=20)
        skip_names = None
        if kind in {"rpc", "service"} and label:
            skip_names = {label.split(".", 1)[-1]}
        steps = summarize_steps_from_body(body, skip_names=skip_names) if body else []
        preconditions = extract_preconditions(body) if body else []
        input_schema = extract_input_schema(body) if body else ""
        loc = format_loc(path, line, prefix)
        line_parts = [f"D{listed + 1} {kind} {detail} {loc}"]
        if input_schema:
            line_parts.append(f"input={input_schema}")
        if preconditions:
            line_parts.append("pre=" + ",".join(preconditions))
        if steps:
            line_parts.append("steps=" + ",".join(steps))
        lines.append(" ".join(line_parts))
        listed += 1
    return lines if listed else []


def invariants_section_lines(
    entry_detail_lines: List[str],
    *,
    max_items: int = 6,
) -> List[str]:
    if not entry_detail_lines:
        return []
    preconditions: List[str] = []
    for line in entry_detail_lines:
        if " pre=" not in line:
            continue
        parts = line.split(" pre=", 1)
        if len(parts) < 2:
            continue
        rest = parts[1].split()[0]
        for name in rest.split(","):
            name = name.strip()
            if name and name not in preconditions:
                preconditions.append(name)
        if len(preconditions) >= max_items:
            break
    if not preconditions:
        return []
    lines = ["[INVARIANTS]"]
    for idx, name in enumerate(preconditions[:max_items], start=1):
        lower = name.lower()
        meaning = "precondition check"
        if "auth" in lower or "login" in lower:
            meaning = "user must be authenticated"
        elif "admin" in lower:
            meaning = "admin access required"
        elif "permission" in lower or "access" in lower or "role" in lower:
            meaning = "authorization required"
        elif "tenant" in lower or "org" in lower or "workspace" in lower:
            meaning = "tenant/org context required"
        elif "billing" in lower or "payment" in lower or "subscription" in lower:
            meaning = "billing/subscription required"
        elif "owner" in lower:
            meaning = "resource ownership required"
        elif "csrf" in lower:
            meaning = "CSRF check"
        elif "rate" in lower:
            meaning = "rate limit check"
        elif "feature" in lower or "flag" in lower:
            meaning = "feature flag enabled"
        lines.append(f"I{idx} {name} -> {meaning}")
    return lines


def group_route_entries(
    entries: List[Dict[str, str]],
    *,
    kind: str,
    max_groups: int = 6,
    samples_per_group: int = 2,
) -> List[str]:
    groups: Dict[str, Dict[str, object]] = {}
    for entry in entries:
        if entry.get("kind") != kind:
            continue
        route = entry.get("label") or ""
        if not route.startswith("/"):
            continue
        if route == "/":
            key = "/"
        else:
            parts = [part for part in route.split("/") if part]
            if kind == "api" and len(parts) >= 2:
                key = f"/{parts[0]}/{parts[1]}"
            elif parts:
                key = f"/{parts[0]}"
            else:
                key = "/"
        info = groups.setdefault(key, {"count": 0, "samples": []})
        info["count"] = int(info["count"]) + 1
        samples = info["samples"]
        if isinstance(samples, list) and len(samples) < samples_per_group and route not in samples:
            samples.append(route)
    ordered = sorted(groups.items(), key=lambda item: (-int(item[1]["count"]), item[0]))
    results: List[str] = []
    for key, info in ordered[:max_groups]:
        samples = ",".join(info["samples"]) if info.get("samples") else ""
        suffix = f":{samples}" if samples else ""
        results.append(f"{key}({info['count']}){suffix}")
    return results


def route_archetypes(
    entries: List[Dict[str, object]],
    files: Dict[str, Dict],
    edges: List[Dict],
    *,
    max_items: int = 4,
) -> List[str]:
    dep_map: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in files and dst in files:
            dep_map[src].append(dst)

    archetypes: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for entry in entries:
        fid = entry.get("fid")
        label = entry.get("label", "")
        if not isinstance(fid, str):
            continue
        if not isinstance(label, str) or not label.startswith("/"):
            continue
        deps = dep_map.get(fid, [])
        service = ""
        data = ""
        for dep in deps:
            dep_path = files.get(dep, {}).get("path", "")
            if classify_path_role(dep_path) == "services":
                service = dep
                break
        for dep in deps:
            dep_path = files.get(dep, {}).get("path", "")
            if classify_path_role(dep_path) == "data":
                data = dep
                break
        if service:
            for dep in dep_map.get(service, []):
                dep_path = files.get(dep, {}).get("path", "")
                if classify_path_role(dep_path) == "data":
                    data = dep
                    break
        if not service and not data:
            continue
        if service and data:
            signature = "routes->services->data"
        elif service:
            signature = "routes->services"
        else:
            signature = "routes->data"
        key = (signature, service, data)
        info = archetypes.setdefault(
            key,
            {"signature": signature, "count": 0, "samples": []},
        )
        info["count"] = int(info["count"]) + 1
        samples = info["samples"]
        if isinstance(samples, list) and len(samples) < 2 and label not in samples:
            samples.append(label)

    ordered = sorted(
        archetypes.values(),
        key=lambda item: (-int(item["count"]), str(item["signature"])),
    )
    results: List[str] = []
    for item in ordered[:max_items]:
        samples = ",".join(item.get("samples") or [])
        suffix = f":{samples}" if samples else ""
        results.append(f"{item['signature']}({item['count']}){suffix}")
    return results


def match_globs(path: str, globs: Iterable[str]) -> bool:
    lower_path = path.lower()
    lower_name = Path(path).name.lower()
    for pattern in globs:
        lowered = pattern.lower()
        if any(token in lowered for token in ("*", "?", "[")):
            if fnmatch.fnmatch(lower_path, lowered) or fnmatch.fnmatch(lower_name, lowered):
                return True
            continue
        if lowered in lower_path or lowered == lower_name:
            return True
    return False


ROUTE_MODE_PATTERNS = {
    "heuristic": {
        "globs": ["/routes/", "/route/", "/api/", "/handlers/", "/controllers/", "/router/", "/endpoints/", "/urls/"],
        "names": ["routes.py", "router.py", "handlers.py", "api.py", "urls.py", "routes.rs", "router.rs", "routes.go", "router.go"],
    },
    "nextjs": {
        "globs": [
            "app/**/page.*",
            "app/**/route.*",
            "pages/**",
            "pages/api/**",
            "src/app/**/page.*",
            "src/app/**/route.*",
            "src/pages/**",
            "src/pages/api/**",
        ],
        "names": ["page.tsx", "page.ts", "route.ts", "route.tsx"],
    },
    "fastapi": {
        "globs": ["**/routers.py", "**/router.py", "**/routes.py", "**/api.py", "**/endpoints.py"],
        "names": ["main.py", "app.py"],
    },
    "django": {
        "globs": ["**/urls.py", "**/views.py"],
        "names": ["urls.py", "views.py"],
    },
    "express": {
        "globs": ["**/routes/**", "**/router/**", "**/controllers/**", "**/api/**"],
        "names": [
            "routes.ts",
            "routes.tsx",
            "router.ts",
            "router.tsx",
            "routes.js",
            "router.js",
            "api.ts",
            "api.js",
        ],
    },
    "rust": {
        "globs": ["**/routes/**", "**/router/**", "**/api/**", "**/handlers/**", "**/controllers/**", "**/endpoints/**"],
        "names": ["routes.rs", "router.rs", "mod.rs", "main.rs", "lib.rs"],
    },
    "go": {
        "globs": ["**/routes/**", "**/router/**", "**/api/**", "**/handlers/**", "**/controllers/**"],
        "names": ["routes.go", "router.go", "handler.go", "handlers.go", "server.go", "main.go"],
    },
}

HTTP_ROUTE_ENTRYPOINT_ROLES = {"http", "api", "route", "web"}
NEXT_ROUTE_METHODS_RE = re.compile(
    r"\bexport\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)\b"
)


def route_patterns_for_mode(mode: str) -> Tuple[List[str], Set[str]]:
    base = ROUTE_MODE_PATTERNS["heuristic"]
    if not mode or mode == "heuristic":
        return list(base["globs"]), {name.lower() for name in base["names"]}
    extra = ROUTE_MODE_PATTERNS.get(mode)
    if not extra:
        return list(base["globs"]), {name.lower() for name in base["names"]}
    globs = list(dict.fromkeys(base["globs"] + extra["globs"]))
    names = {name.lower() for name in base["names"]} | {name.lower() for name in extra["names"]}
    return globs, names


def route_candidates(
    files: Dict[str, Dict],
    *,
    route_globs: Optional[Iterable[str]] = None,
    route_names: Optional[Iterable[str]] = None,
    exclude_globs: Optional[Iterable[str]] = None,
    routes_mode: str = "heuristic",
    include_entrypoint_routes: bool = True,
    repo_root: Optional[Path] = None,
) -> List[str]:
    hits: List[Tuple[float, str, str]] = []
    base_globs, base_names = route_patterns_for_mode(routes_mode)
    globs = base_globs + list(route_globs or [])
    names = base_names | {name.lower() for name in (route_names or [])}
    excludes = list(exclude_globs or [])
    for fid, node in files.items():
        path = node.get("path", "")
        if is_test_path(path):
            continue
        if match_globs(path, excludes):
            continue
        entry_role = str(node.get("entrypoint_role") or "").lower()
        from_entrypoint = include_entrypoint_routes and entry_role in HTTP_ROUTE_ENTRYPOINT_ROLES
        path_match = match_globs(path, globs)
        name_match = Path(path).name.lower() in names
        if path_match or name_match or from_entrypoint:
            bonus = float(route_path_confidence(path))
            if from_entrypoint:
                # Trust explicit HTTP-ish entrypoint roles when route naming is weak.
                bonus += 2.0
            ast = 0
            if repo_root:
                content = read_route_content(repo_root, path, limit=12000)
                if route_content_has_signal(path, content):
                    ast = 1
            lexical = 1 if (path_match or name_match) else 0
            graph = 1 if (from_entrypoint or float(node.get("score", 0.0)) >= 0.2) else 0
            config = 1 if routes_mode not in {"auto", "heuristic", ""} else 0
            evidence = score_candidate(
                "route",
                ast=ast,
                config=config,
                graph=graph,
                lexical=lexical,
            )
            root = path.split("/", 1)[0] if "/" in path else "."
            hits.append((float(node.get("score", 0.0)) + bonus + evidence.score, fid, root))
    hits.sort(key=lambda item: item[0], reverse=True)
    if not hits:
        return []

    # Interleave by top-level root to keep coverage across monorepo modules.
    by_root: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for weighted, fid, root in hits:
        by_root[root].append((weighted, fid))
    root_order = sorted(by_root.keys(), key=lambda key: by_root[key][0][0], reverse=True)

    ordered: List[str] = []
    while True:
        progressed = False
        for root in root_order:
            bucket = by_root[root]
            if not bucket:
                continue
            ordered.append(bucket.pop(0)[1])
            progressed = True
        if not progressed:
            break
    return ordered
