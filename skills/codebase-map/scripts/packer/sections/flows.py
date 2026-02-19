from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ..digest_core import classify_path_role, file_id, short_path


def flow_lines(
    *,
    files: Dict[str, Dict],
    edges: List[Dict],
    file_alias: Dict[str, str],
    entrypoints: List[str],
    route_entries: List[Dict[str, object]],
    prefix: str,
    flow_symbols: bool = False,
    symbol_pairs: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    adj_cache: Optional[Dict[str, List[str]]] = None,
    out_degree_cache: Optional[Dict[str, int]] = None,
    max_flows: int = 6,
    max_depth: int = 3,
) -> List[str]:
    if not edges or not file_alias:
        return []
    adj: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    out_degree: Counter[str] = Counter()
    if adj_cache:
        for src, targets in adj_cache.items():
            for dst in targets:
                if src in file_alias and dst in file_alias:
                    adj[src].append((dst, 1))
                    out_degree[src] += 1
        if out_degree_cache:
            out_degree.update(out_degree_cache)
    else:
        for edge in edges:
            src = edge.get("from")
            dst = edge.get("to")
            if src not in file_alias or dst not in file_alias:
                continue
            weight = edge.get("weight", 1)
            try:
                weight_value = int(weight) if weight is not None else 1
            except (TypeError, ValueError):
                weight_value = 1
            adj[src].append((dst, weight_value))
            out_degree[src] += weight_value

    route_fids = [entry.get("fid") for entry in route_entries if isinstance(entry.get("fid"), str)]
    start_fids: List[str] = []
    for fid in entrypoints:
        if fid in file_alias and fid not in start_fids:
            start_fids.append(fid)
    for fid in route_fids:
        if fid in file_alias and fid not in start_fids:
            start_fids.append(fid)

    if not start_fids:
        return []

    lines: List[str] = ["[FLOWS]"]
    seen_paths: Set[Tuple[str, ...]] = set()
    flow_idx = 1

    for start in start_fids:
        if flow_idx > max_flows:
            break
        path = [start]
        current = start
        for _depth in range(max_depth - 1):
            choices = adj.get(current, [])
            if not choices:
                break
            choices.sort(
                key=lambda item: (-item[1], -out_degree.get(item[0], 0), file_alias.get(item[0], ""))
            )
            next_fid = None
            for candidate, _weight in choices:
                if candidate not in path:
                    next_fid = candidate
                    break
            if not next_fid:
                break
            path.append(next_fid)
            current = next_fid
        if len(path) < 2:
            continue
        path_key = tuple(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        entry_role = files.get(start, {}).get("entrypoint_role") or "entry"
        if start in route_fids:
            entry_role = "route"
        symbol_labels: List[Optional[str]] = [None for _ in path]
        if flow_symbols and symbol_pairs:
            for idx in range(len(path) - 1):
                pair = symbol_pairs.get((path[idx], path[idx + 1]))
                if not pair:
                    continue
                src_name, dst_name = pair
                if idx == 0 and src_name:
                    symbol_labels[idx] = src_name
                if dst_name:
                    symbol_labels[idx + 1] = dst_name
        path_refs = ">".join(
            (
                f"{file_alias.get(fid, short_path(files.get(fid, {}).get('path', ''), prefix))}:{symbol_labels[idx]}"
                if symbol_labels[idx]
                else file_alias.get(fid, short_path(files.get(fid, {}).get("path", ""), prefix))
            )
            for idx, fid in enumerate(path)
        )
        roles = [classify_path_role(files.get(fid, {}).get("path", "")) for fid in path]
        role_seq = ">".join(roles)
        line = f"FL{flow_idx} {entry_role} {path_refs}"
        if any(role != "other" for role in roles):
            line += f" roles={role_seq}"
        lines.append(line)
        flow_idx += 1
        if flow_idx > max_flows:
            break
    return lines


def flow_edges_from_symbol_refs(
    ir: Dict,
    selected_files: Set[str],
) -> List[Dict[str, object]]:
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {}).get("symbol_ref", [])
    if not symbols or not edges:
        return []
    has_codeql = any(edge.get("provenance") == "codeql" for edge in edges)
    counts: Counter[Tuple[str, str]] = Counter()
    for edge in edges:
        if has_codeql and edge.get("provenance") != "codeql":
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if src not in symbols or dst not in symbols:
            continue
        src_path = symbols[src].get("defined_in", {}).get("path")
        dst_path = symbols[dst].get("defined_in", {}).get("path")
        if not src_path or not dst_path:
            continue
        src_fid = file_id(src_path)
        dst_fid = file_id(dst_path)
        if src_fid == dst_fid:
            continue
        if selected_files and (src_fid not in selected_files or dst_fid not in selected_files):
            continue
        counts[(src_fid, dst_fid)] += 1
    return [
        {"from": src, "to": dst, "weight": weight}
        for (src, dst), weight in counts.items()
    ]


def flow_symbol_pairs(
    ir: Dict,
    selected_files: Set[str],
) -> Dict[Tuple[str, str], Tuple[str, str]]:
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {}).get("symbol_ref", [])
    if not symbols or not edges:
        return {}
    has_codeql = any(edge.get("provenance") == "codeql" for edge in edges)
    counts: Counter[Tuple[str, str, str, str]] = Counter()
    for edge in edges:
        if has_codeql and edge.get("provenance") != "codeql":
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if src not in symbols or dst not in symbols:
            continue
        src_path = symbols[src].get("defined_in", {}).get("path")
        dst_path = symbols[dst].get("defined_in", {}).get("path")
        if not src_path or not dst_path:
            continue
        src_fid = file_id(src_path)
        dst_fid = file_id(dst_path)
        if src_fid == dst_fid:
            continue
        if selected_files and (src_fid not in selected_files or dst_fid not in selected_files):
            continue
        src_name = symbols[src].get("name") or ""
        dst_name = symbols[dst].get("name") or ""
        if not src_name or not dst_name:
            continue
        counts[(src_fid, dst_fid, src_name, dst_name)] += 1
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for (src_fid, dst_fid, src_name, dst_name), _count in counts.most_common():
        key = (src_fid, dst_fid)
        if key not in pairs:
            pairs[key] = (src_name, dst_name)
    return pairs
