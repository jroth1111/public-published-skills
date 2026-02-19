from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..digest_core import file_id, short_path
from ..trace_graph import trace_node_label
from .flows import flow_edges_from_symbol_refs


def call_chain_lines(
    ir: Dict,
    *,
    file_alias: Dict[str, str],
    symbol_alias: Dict[str, str],
    entrypoints: List[str],
    prefix: str,
    max_items: int = 6,
    max_depth: int = 3,
) -> List[str]:
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {}).get("symbol_ref", [])
    if not symbols or not edges or not file_alias:
        return []

    has_codeql = any(edge.get("provenance") == "codeql" for edge in edges)
    selected_files = set(file_alias.keys())

    def collect_edge_weights(
        restrict_to_selected: bool,
        *,
        provenance_filter: Optional[str] = None,
    ) -> Dict[Tuple[str, str], int]:
        weights: Dict[Tuple[str, str], int] = defaultdict(int)
        for edge in edges:
            if provenance_filter and edge.get("provenance") != provenance_filter:
                continue
            src = edge.get("from")
            dst = edge.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            src_node = symbols.get(src)
            dst_node = symbols.get(dst)
            if not src_node or not dst_node:
                continue
            src_path = src_node.get("defined_in", {}).get("path")
            dst_path = dst_node.get("defined_in", {}).get("path")
            if not src_path or not dst_path:
                continue
            src_fid = file_id(str(src_path))
            dst_fid = file_id(str(dst_path))
            if restrict_to_selected and (
                src_fid not in selected_files or dst_fid not in selected_files
            ):
                continue
            weights[(src, dst)] += 1
        return weights

    def collect_with_scope(
        *,
        provenance_filter: Optional[str] = None,
    ) -> Tuple[Dict[Tuple[str, str], int], bool]:
        scoped = collect_edge_weights(
            restrict_to_selected=True,
            provenance_filter=provenance_filter,
        )
        expanded = False
        if not scoped:
            scoped = collect_edge_weights(
                restrict_to_selected=False,
                provenance_filter=provenance_filter,
            )
            expanded = bool(scoped)
        return scoped, expanded

    # Aggregate symbol-to-symbol edge weights for stable chain ranking.
    all_edge_weights, all_expanded_scope = collect_with_scope()
    codeql_edge_weights, codeql_expanded_scope = collect_with_scope(provenance_filter="codeql")

    # Prefer strict CodeQL chains only when semantic coverage is non-trivial.
    use_codeql_only = bool(codeql_edge_weights) and len(codeql_edge_weights) >= 12
    edge_weights = codeql_edge_weights if use_codeql_only else all_edge_weights
    expanded_scope = codeql_expanded_scope if use_codeql_only else all_expanded_scope

    if not edge_weights:
        return []

    outgoing: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (src, dst), weight in edge_weights.items():
        outgoing[src].append((dst, weight))
    for src in outgoing:
        outgoing[src].sort(key=lambda item: (-item[1], item[0]))

    # Prefer entrypoint-defined symbols as chain roots.
    root_symbols: List[str] = []
    for fid in entrypoints:
        if fid not in selected_files:
            continue
        file_node = ir.get("files", {}).get(fid, {})
        for sid in file_node.get("exports", []) + file_node.get("defines", []):
            if sid in outgoing and sid not in root_symbols:
                root_symbols.append(sid)

    # Fallback to highest fan-out symbols when entrypoints are sparse.
    if not root_symbols:
        ranked = sorted(
            outgoing.items(),
            key=lambda item: (-sum(weight for _dst, weight in item[1]), item[0]),
        )
        root_symbols = [sid for sid, _ in ranked]

    def symbol_label(symbol_id: str) -> str:
        node = symbols.get(symbol_id, {})
        name = str(node.get("name") or symbol_id)
        path = str(node.get("defined_in", {}).get("path") or "")
        fid = file_id(path) if path else ""
        file_ref = file_alias.get(fid) or ""
        if not file_ref and path:
            file_ref = short_path(path, prefix, max_segments=3)
        sym_ref = symbol_alias.get(symbol_id) or name
        if file_ref:
            return f"{file_ref}:{sym_ref}"
        return sym_ref

    lines: List[str] = ["[CALL_CHAINS]"]
    lines.append(f"confidence={'high' if use_codeql_only else 'medium'}")
    if has_codeql and not use_codeql_only:
        lines.append("provenance=mixed")
    if expanded_scope:
        lines.append("scope=expanded")
    seen_chains: Set[Tuple[str, ...]] = set()
    chain_idx = 1
    depth_cap = max(2, max_depth + 1)

    for root in root_symbols:
        if chain_idx > max_items:
            break
        chain = [root]
        current = root
        while len(chain) < depth_cap:
            choices = outgoing.get(current, [])
            if not choices:
                break
            next_symbol = None
            for candidate, _weight in choices:
                if candidate not in chain:
                    next_symbol = candidate
                    break
            if not next_symbol:
                break
            chain.append(next_symbol)
            current = next_symbol

        if len(chain) < 2:
            continue
        chain_key = tuple(chain)
        if chain_key in seen_chains:
            continue
        seen_chains.add(chain_key)
        chain_text = " -> ".join(symbol_label(sid) for sid in chain)
        lines.append(f"CC{chain_idx} depth={len(chain) - 1} {chain_text}")
        chain_idx += 1

    return lines if len(lines) > 2 else []


def build_trace_edges(
    ir: Dict,
    selected_files: Set[str],
) -> Tuple[List[Tuple[str, str, str]], str]:
    trace_edges: List[Tuple[str, str, str]] = []
    confidence = "low"
    conf_rank = {"low": 1, "medium": 2, "high": 3}
    edge_conf: Dict[Tuple[str, str], str] = {}

    def add_edge(src: str, dst: str, conf: str) -> None:
        if selected_files and (src not in selected_files or dst not in selected_files):
            return
        key = (src, dst)
        prev = edge_conf.get(key)
        if prev and conf_rank.get(conf, 1) <= conf_rank.get(prev, 1):
            return
        edge_conf[key] = conf

    dataflow_edges = ir.get("edges", {}).get("dataflow", [])
    for edge in dataflow_edges:
        from_path = edge.get("from_path")
        to_path = edge.get("to_path")
        if not from_path or not to_path:
            continue
        src = file_id(str(from_path))
        dst = file_id(str(to_path))
        add_edge(src, dst, "high")

    symbol_edges = flow_edges_from_symbol_refs(ir, selected_files)
    for edge in symbol_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        add_edge(src, dst, "medium")

    file_edges = ir.get("edges", {}).get("file_dep", [])
    for edge in file_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        add_edge(src, dst, "low")

    if edge_conf:
        if any(conf == "high" for conf in edge_conf.values()):
            confidence = "high"
        elif any(conf == "medium" for conf in edge_conf.values()):
            confidence = "medium"
    trace_edges = [
        (src, dst, conf) for (src, dst), conf in edge_conf.items() if src and dst
    ]
    trace_edges.sort(
        key=lambda item: (-conf_rank.get(item[2], 1), item[0], item[1])
    )
    return trace_edges, confidence


def normalize_trace_path(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    if "://" in text:
        try:
            _, rest = text.split("://", 1)
            if "/" in rest:
                text = "/" + rest.split("/", 1)[1]
            else:
                text = "/"
        except ValueError:
            pass
    text = text.split("?", 1)[0].split("#", 1)[0]
    if text and not text.startswith("/"):
        text = "/" + text
    return text


def is_test_path(path: str) -> bool:
    lowered = path.lower()
    return (
        "/__tests__/" in lowered
        or "/tests/" in lowered
        or lowered.endswith(".test.tsx")
        or lowered.endswith(".test.ts")
        or lowered.endswith(".test.jsx")
        or lowered.endswith(".test.js")
        or ".spec." in lowered
    )


def route_pattern_regex(route: str) -> re.Pattern:
    pattern = route
    pattern = re.sub(r":[^/]+", "[^/]+", pattern)
    pattern = re.sub(r"\{[^/]+\}", "[^/]+", pattern)
    pattern = re.sub(r"\[[^/]+\]", "[^/]+", pattern)
    pattern = re.escape(pattern)
    pattern = pattern.replace(r"\[\^/\]\+", "[^/]+")
    return re.compile("^" + pattern + "$")


def extract_client_calls(content: str) -> List[Tuple[str, str]]:
    calls: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def add_call(method: str, path: str) -> None:
        if not path:
            return
        key = (method, path)
        if key in seen:
            return
        seen.add(key)
        calls.append(key)

    for match in re.finditer(r"\bfetch\(\s*([\"'`])([^\"'`]+)\1", content):
        path = match.group(2)
        if "${" in path:
            continue
        add_call("get", normalize_trace_path(path))
    for match in re.finditer(
        r"\bfetch\(\s*new\s+Request\(\s*([\"'`])([^\"'`]+)\1",
        content,
    ):
        path = match.group(2)
        if "${" in path:
            continue
        add_call("get", normalize_trace_path(path))
    for match in re.finditer(
        r"\baxios\.(get|post|put|delete|patch|head|options)\(\s*([\"'`])([^\"'`]+)\2",
        content,
        re.IGNORECASE,
    ):
        method = match.group(1).lower()
        path = match.group(3)
        if "${" in path:
            continue
        add_call(method, normalize_trace_path(path))
    for match in re.finditer(r"\baxios\(\s*\{([^}]+)\}", content, re.DOTALL):
        block = match.group(1)
        method_match = re.search(r"method\s*:\s*['\"]([a-zA-Z]+)['\"]", block)
        url_match = re.search(r"url\s*:\s*['\"]([^'\"]+)['\"]", block)
        if not url_match:
            continue
        path = url_match.group(1)
        if "${" in path:
            continue
        method = method_match.group(1).lower() if method_match else "get"
        add_call(method, normalize_trace_path(path))
    for match in re.finditer(r"\b\w+\.fetch\(\s*([\"'`])([^\"'`]+)\1", content):
        path = match.group(2)
        if "${" in path:
            continue
        add_call("get", normalize_trace_path(path))
    for match in re.finditer(
        r"\b(api|apiClient|client|http)\.(get|post|put|patch|delete)\(\s*([\"'`])([^\"'`]+)\3",
        content,
    ):
        method = match.group(2).lower()
        path = match.group(4)
        if "${" in path:
            continue
        add_call(method, normalize_trace_path(path))
    return calls


def extract_import_paths(content: str) -> List[str]:
    imports: List[str] = []
    for match in re.finditer(
        r"\bimport\s+(?:type\s+)?[^;]*?\s+from\s*([\"'])([^\"']+)\1",
        content,
    ):
        imports.append(match.group(2))
    for match in re.finditer(
        r"\bexport\s+(?:type\s+)?[^;]*?\s+from\s*([\"'])([^\"']+)\1",
        content,
    ):
        imports.append(match.group(2))
    for match in re.finditer(r"\bimport\(\s*([\"'])([^\"']+)\1\s*\)", content):
        imports.append(match.group(2))
    return imports


def is_server_action_content(content: str) -> bool:
    lowered = content.lower()
    if '"use server"' in lowered or "'use server'" in lowered:
        return True
    if re.search(r"\bquery\s*\(", content) and "use server" in lowered:
        return True
    if re.search(r"\baction\s*\(", content) and "use server" in lowered:
        return True
    if re.search(r"\bloader\s*\(", content) and "use server" in lowered:
        return True
    if re.search(r"\bmutation\s*\(", content) and "use server" in lowered:
        return True
    if re.search(r"export\s+async\s+function\s+(action|loader)\b", content):
        return True
    return False


def build_client_call_edges(
    *,
    repo_root: Optional[Path],
    files: Dict[str, Dict],
    selected_files: Set[str],
    api_contracts: List[Dict[str, object]],
    route_entries: List[Dict[str, object]],
) -> List[Tuple[str, str, str]]:
    if not repo_root or not selected_files:
        if not repo_root:
            return []
        selected_files = set(files.keys())
    routes: List[Tuple[str, str, str]] = []
    if api_contracts:
        for entry in api_contracts:
            route = str(entry.get("route") or "")
            method = str(entry.get("method") or "").lower()
            path = str(entry.get("path") or "")
            if not route or not path:
                continue
            routes.append((method, normalize_trace_path(route), file_id(path)))
    else:
        for entry in route_entries:
            label = str(entry.get("label") or "")
            fid = entry.get("fid")
            if not label or not isinstance(fid, str):
                continue
            if ":" in label:
                method, route = label.split(":", 1)
                routes.append((method.lower(), normalize_trace_path(route), fid))
            else:
                routes.append(("", normalize_trace_path(label), fid))
    compiled: List[Tuple[str, str, re.Pattern, str]] = []
    for method, route, fid in routes:
        compiled.append((method, route, route_pattern_regex(route), fid))

    def is_local_module(module: str) -> bool:
        if module.startswith((".", "/", "~", "@/")):
            return True
        if module.startswith("@~"):
            return True
        return "/routes/" in module or "/actions" in module or "/api/" in module

    def looks_like_rpc_module(module: str) -> bool:
        lowered = module.lower()
        return (
            "/routes/api/" in lowered
            or lowered.startswith("routes/api/")
            or "/routes/" in lowered
            and "/api/" in lowered
            or "/actions" in lowered
            or "/server/actions" in lowered
            or "/rpc/" in lowered
            or "trpc" in lowered
            or "orpc" in lowered
        )

    def normalize_module_path(module: str) -> str:
        cleaned = module.split("?", 1)[0].split("#", 1)[0]
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return ""
        if cleaned.startswith("~/") or cleaned.startswith("@/"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("~"):
            cleaned = cleaned[1:]
        return cleaned.lstrip("/")

    path_index: Dict[str, str] = {}
    suffix_index: Dict[str, str] = {}
    for fid, node in files.items():
        path = node.get("path") or ""
        if not path:
            continue
        norm = path.lstrip("./")
        if norm not in path_index:
            path_index[norm] = fid
        stem = re.sub(r"\.(tsx?|jsx?|mjs|cjs)$", "", norm)
        if stem and stem not in path_index:
            path_index[stem] = fid
        if norm.endswith("/index.ts") or norm.endswith("/index.tsx") or norm.endswith("/index.js") or norm.endswith("/index.jsx"):
            base = re.sub(r"/index\.(tsx?|jsx?|mjs|cjs)$", "", norm)
            if base and base not in path_index:
                path_index[base] = fid
        if "/src/" in norm:
            suffix = norm.split("/src/", 1)[1]
            suffix_stem = re.sub(r"\.(tsx?|jsx?|mjs|cjs)$", "", suffix)
            if suffix and suffix not in suffix_index:
                suffix_index[suffix] = fid
            if suffix_stem and suffix_stem not in suffix_index:
                suffix_index[suffix_stem] = fid

    def package_root(path: str) -> str:
        parts = [part for part in path.split("/") if part]
        if not parts:
            return ""
        if parts[0] in {"apps", "packages", "workers"} and len(parts) > 1:
            return "/".join(parts[:2])
        return parts[0]

    def is_rpc_server_path(path: str) -> bool:
        lowered = path.lower()
        if "trpc" not in lowered and "orpc" not in lowered:
            return False
        return any(
            token in lowered
            for token in (
                "/server/",
                "/api/",
                "/routers/",
                "/router",
                "router.",
                "procedure",
            )
        )

    def is_rpc_server_content(content: str) -> bool:
        lowered = content.lower()
        if "trpc" not in lowered and "orpc" not in lowered:
            return False
        return (
            "inittrpc" in lowered
            or "createtrpc" in lowered
            or "@trpc/server" in lowered
            or "@orpc/server" in lowered
            or "orpc/server" in lowered
            or "trpc.router" in lowered
            or "procedure" in lowered and "trpc" in lowered
            or "orpc" in lowered and ("router" in lowered or "procedure" in lowered)
        )

    def is_rpc_client_content(content: str) -> bool:
        lowered = content.lower()
        if "trpc" not in lowered and "orpc" not in lowered:
            return False
        return (
            "trpc." in lowered
            or "orpc." in lowered
            or "@trpc/client" in lowered
            or "@orpc/client" in lowered
            or "trpc/client" in lowered
            or "orpc/client" in lowered
            or "createtrpcproxyclient" in lowered
            or "createtrpcclient" in lowered
            or "createtrpcreact" in lowered
            or "createorpcclient" in lowered
            or "trpcclient" in lowered
        )

    rpc_server_fids: Set[str] = set(
        fid
        for fid, node in files.items()
        if is_rpc_server_path(str(node.get("path") or "")) and not is_test_path(str(node.get("path") or ""))
    )
    rpc_client_fids: Set[str] = set()
    server_action_fids: Set[str] = set()

    def resolve_import_to_fid(module: str, importer_path: str) -> Optional[str]:
        cleaned = normalize_module_path(module)
        if not cleaned:
            return None
        if cleaned.startswith("."):
            base_dir = posixpath.dirname(importer_path)
            cleaned = posixpath.normpath(posixpath.join(base_dir, cleaned))
            cleaned = cleaned.lstrip("./")
        candidates = [cleaned]
        if cleaned.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
            candidates.append(re.sub(r"\.(tsx?|jsx?|mjs|cjs)$", "", cleaned))
        else:
            candidates.append(cleaned + ".ts")
            candidates.append(cleaned + ".tsx")
            candidates.append(cleaned + ".js")
            candidates.append(cleaned + ".jsx")
            candidates.append(posixpath.join(cleaned, "index.ts"))
            candidates.append(posixpath.join(cleaned, "index.tsx"))
            candidates.append(posixpath.join(cleaned, "index.js"))
            candidates.append(posixpath.join(cleaned, "index.jsx"))
        for cand in candidates:
            if cand in path_index:
                return path_index[cand]
            if cand in suffix_index:
                return suffix_index[cand]
        return None

    def scan_file_ids(file_ids: Iterable[str]) -> List[Tuple[str, str, str]]:
        results: List[Tuple[str, str, str]] = []
        for fid in file_ids:
            node = files.get(fid, {})
            path = node.get("path") or ""
            if not path or not path.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")):
                continue
            if is_test_path(path):
                continue
            file_path = repo_root / path
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(content) > 200_000:
                continue
            if is_rpc_server_content(content):
                rpc_server_fids.add(fid)
            if is_server_action_content(content):
                server_action_fids.add(fid)
            imports = extract_import_paths(content)
            if imports:
                importer_path = str(path)
                for module in imports:
                    if not is_local_module(module):
                        continue
                    if not looks_like_rpc_module(module):
                        continue
                    target = resolve_import_to_fid(module, importer_path)
                    if target and target != fid:
                        conf = (
                            "high"
                            if target in server_action_fids or target in rpc_server_fids
                            else "medium"
                        )
                        results.append((fid, target, conf))
            if is_rpc_client_content(content) or any(
                "trpc" in module.lower() or "orpc" in module.lower() for module in imports
            ):
                rpc_client_fids.add(fid)
            if compiled and (
                "fetch" in content
                or "axios" in content
                or "api." in content
                or "apiClient" in content
                or "client." in content
                or "http." in content
            ):
                for method, call_path in extract_client_calls(content):
                    for route_method, route_path, route_re, route_fid in compiled:
                        if route_method in {"all", "route"}:
                            if call_path == route_path or call_path.startswith(route_path.rstrip("/") + "/"):
                                results.append((fid, route_fid, "medium"))
                            continue
                        if route_method and route_method != method:
                            continue
                        if route_re.match(call_path):
                            results.append((fid, route_fid, "medium"))
        return results

    edges = scan_file_ids(selected_files)
    if len(selected_files) < len(files):
        extra = scan_file_ids(files.keys())
        if extra:
            edges.extend(extra)
    if rpc_client_fids and rpc_server_fids:
        server_by_root: Dict[str, List[str]] = defaultdict(list)
        for fid in rpc_server_fids:
            path = str(files.get(fid, {}).get("path") or "")
            server_by_root[package_root(path)].append(fid)
        for fid in rpc_client_fids:
            path = str(files.get(fid, {}).get("path") or "")
            root = package_root(path)
            targets = server_by_root.get(root) or list(rpc_server_fids)
            for target in targets:
                if fid != target:
                    conf = "high" if target in rpc_server_fids else "medium"
                    edges.append((fid, target, conf))
    seen: Set[Tuple[str, str, str]] = set()
    deduped: List[Tuple[str, str, str]] = []
    for edge in edges:
        if edge in seen:
            continue
        seen.add(edge)
        deduped.append(edge)
    return deduped


def type_ref_file_edges(
    ir: Dict,
    selected_files: Set[str],
) -> List[Tuple[str, str]]:
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {}).get("type_ref", [])
    results: List[Tuple[str, str]] = []
    for edge in edges:
        from_path = edge.get("from_path")
        to_id = edge.get("to")
        if not from_path or not to_id:
            continue
        if isinstance(to_id, str) and to_id in symbols:
            to_path = symbols[to_id].get("defined_in", {}).get("path")
            if not to_path:
                continue
            src = file_id(str(from_path))
            dst = file_id(str(to_path))
            if selected_files and (src not in selected_files or dst not in selected_files):
                continue
            results.append((src, dst))
    return results


def static_trace_paths(
    *,
    start_nodes: List[str],
    adj: Dict[str, List[str]],
    edge_conf: Dict[Tuple[str, str], str],
    max_depth: int,
    max_paths: int,
) -> List[Tuple[List[str], str]]:
    paths: List[Tuple[List[str], str]] = []
    conf_rank = {"low": 1, "medium": 2, "high": 3}

    def path_confidence(path: List[str]) -> str:
        if len(path) < 2:
            return "low"
        min_rank = 3
        for idx in range(len(path) - 1):
            conf = edge_conf.get((path[idx], path[idx + 1]), "low")
            min_rank = min(min_rank, conf_rank.get(conf, 1))
        for label, rank in conf_rank.items():
            if rank == min_rank:
                return label
        return "low"

    def dfs(path: List[str], depth: int) -> None:
        if len(paths) >= max_paths:
            return
        current = path[-1]
        if depth >= max_depth or current not in adj:
            paths.append((path, path_confidence(path)))
            return
        next_nodes = adj.get(current, [])
        if not next_nodes:
            paths.append((path, path_confidence(path)))
            return
        for nxt in next_nodes:
            if nxt in path:
                continue
            dfs(path + [nxt], depth + 1)
            if len(paths) >= max_paths:
                return

    for start in start_nodes:
        if len(paths) >= max_paths:
            break
        dfs([start], 1)
    return paths


def static_trace_lines(
    *,
    ir: Dict,
    files: Dict[str, Dict],
    entities: Dict[str, Dict[str, object]],
    entity_edges: List[Dict[str, object]],
    entrypoints: List[str],
    route_entries: List[Dict[str, object]],
    repo_root: Optional[Path],
    api_contracts: List[Dict[str, object]],
    file_alias: Dict[str, str],
    entity_alias: Dict[str, str],
    prefix: str,
    max_depth: int,
    max_items: int,
    trace_direction: str = "both",
    trace_start: Optional[str] = None,
    trace_end: Optional[str] = None,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    if not files:
        return [], []
    trace_direction = (trace_direction or "both").lower()
    if trace_direction not in {"forward", "reverse", "both"}:
        trace_direction = "both"
    alias_to_file = {alias: fid for fid, alias in file_alias.items()}
    alias_to_entity = {alias: ent_id for ent_id, alias in entity_alias.items()}

    def resolve_trace_nodes(value: Optional[str]) -> List[str]:
        if not value:
            return []
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        results: List[str] = []
        for token in tokens:
            if token in files or token in entities:
                results.append(token)
                continue
            if token in alias_to_file:
                results.append(alias_to_file[token])
                continue
            if token in alias_to_entity:
                results.append(alias_to_entity[token])
                continue
            lowered = token.lower()
            for fid, node in files.items():
                path = str(node.get("path") or "")
                if not path:
                    continue
                if lowered in path.lower() or lowered == Path(path).name.lower():
                    results.append(fid)
            for ent_id, node in entities.items():
                name = str(node.get("name") or "")
                if name and name.lower() == lowered:
                    results.append(ent_id)
        seen: Set[str] = set()
        unique: List[str] = []
        for node_id in results:
            if node_id in seen:
                continue
            seen.add(node_id)
            unique.append(node_id)
        return unique

    start_override = resolve_trace_nodes(trace_start)
    end_override = resolve_trace_nodes(trace_end)

    selected_files = set(file_alias.keys())
    if start_override or end_override:
        selected_files = set(files.keys())
    else:
        for node_id in start_override + end_override:
            if node_id in files:
                selected_files.add(node_id)

    trace_edges, confidence = build_trace_edges(ir, selected_files)
    graph_edges = trace_edges

    entity_conf: Dict[Tuple[str, str], str] = {}
    if entity_edges:
        for edge in entity_edges:
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            if src.startswith("file:") and dst.startswith("entity:"):
                graph_edges.append((src, dst, "medium"))
                entity_conf[(src, dst)] = "medium"
            elif src.startswith("entity:") and dst.startswith("file:"):
                graph_edges.append((src, dst, "medium"))
                entity_conf[(src, dst)] = "medium"

    client_edges = build_client_call_edges(
        repo_root=repo_root,
        files=files,
        selected_files=selected_files,
        api_contracts=api_contracts,
        route_entries=route_entries,
    )
    for src, dst, conf in client_edges:
        graph_edges.append((src, dst, conf))

    edge_conf: Dict[Tuple[str, str], str] = {}
    for src, dst, conf in graph_edges:
        edge_conf[(src, dst)] = conf
    for (src, dst), conf in entity_conf.items():
        edge_conf[(src, dst)] = conf

    adj: Dict[str, List[str]] = defaultdict(list)
    rev: Dict[str, List[str]] = defaultdict(list)
    for src, dst, _conf in graph_edges:
        if src == dst:
            continue
        if selected_files and (src not in selected_files and dst not in selected_files):
            continue
        adj[src].append(dst)
        rev[dst].append(src)

    route_fids = [entry.get("fid") for entry in route_entries if isinstance(entry.get("fid"), str)]
    start_nodes: List[str] = []
    for fid in entrypoints:
        if fid in files and fid not in start_nodes:
            start_nodes.append(fid)
    for fid in route_fids:
        if fid in files and fid not in start_nodes:
            start_nodes.append(fid)

    start_forward = start_nodes
    start_reverse = start_nodes
    end_forward = []
    end_reverse: List[str] = []
    if start_override:
        start_forward = start_override
        start_reverse = start_override
    if end_override:
        end_forward = end_override
        end_reverse = end_override

    lines = ["[STATIC_TRACES]"]
    lines.append(f"confidence={confidence}")

    def truncate_at_end(paths: List[Tuple[List[str], str]], end_nodes: List[str]) -> List[Tuple[List[str], str]]:
        if not end_nodes:
            return paths
        result: List[Tuple[List[str], str]] = []
        for path, conf in paths:
            trimmed = []
            for node in path:
                trimmed.append(node)
                if node in end_nodes:
                    break
            if len(trimmed) >= 2:
                result.append((trimmed, conf))
        return result

    def maybe_prefix_user(labels: List[str], node_id: str) -> List[str]:
        if not node_id or not labels:
            return labels
        label = labels[0]
        if "frontend" in label and "user" not in label:
            labels[0] = "user" + " -> " + label
        return labels

    def maybe_suffix_user(labels: List[str], node_id: str) -> List[str]:
        if not node_id or not labels:
            return labels
        label = labels[-1]
        if "frontend" in label and "user" not in label:
            labels[-1] = label + " -> user"
        return labels

    if trace_direction in {"forward", "both"}:
        forward_paths = static_trace_paths(
            start_nodes=start_forward,
            adj=adj,
            edge_conf=edge_conf,
            max_depth=max_depth,
            max_paths=max_items,
        )
        forward_paths = truncate_at_end(forward_paths, end_forward)
        idx = 1
        for path, conf in forward_paths:
            if len(path) < 2:
                continue
            labels = [
                trace_node_label(
                    node,
                    files=files,
                    entities=entities,
                    file_alias=file_alias,
                    entity_alias=entity_alias,
                    prefix=prefix,
                )
                for node in path
            ]
            labels = maybe_prefix_user(labels, path[0])
            labels = maybe_suffix_user(labels, path[-1])
            lines.append(f"ST{idx} forward conf={conf} " + " -> ".join(labels))
            idx += 1
            if idx > max_items:
                break

    if trace_direction in {"reverse", "both"}:
        reverse_paths = static_trace_paths(
            start_nodes=start_reverse,
            adj=rev,
            edge_conf=edge_conf,
            max_depth=max_depth,
            max_paths=max_items,
        )
        reverse_paths = truncate_at_end(reverse_paths, end_reverse)
        ridx = 1
        for path, conf in reverse_paths:
            if len(path) < 2:
                continue
            labels = [
                trace_node_label(
                    node,
                    files=files,
                    entities=entities,
                    file_alias=file_alias,
                    entity_alias=entity_alias,
                    prefix=prefix,
                )
                for node in path
            ]
            labels = maybe_prefix_user(labels, path[0])
            labels = maybe_suffix_user(labels, path[-1])
            lines.append(f"SR{ridx} reverse conf={conf} " + " -> ".join(labels))
            ridx += 1
            if ridx > max_items:
                break
    return lines if len(lines) > 1 else [], graph_edges
