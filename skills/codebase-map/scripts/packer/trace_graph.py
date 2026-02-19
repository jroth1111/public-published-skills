from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .digest_core import layer_for_path, short_path


def trace_node_label(
    node_id: str,
    *,
    files: Dict[str, Dict],
    entities: Dict[str, Dict[str, object]],
    file_alias: Dict[str, str],
    entity_alias: Dict[str, str],
    prefix: str,
) -> str:
    if node_id.startswith("entity:"):
        label = entity_alias.get(node_id) or node_id
        name = entities.get(node_id, {}).get("name")
        if name:
            label = f"{label}:{name}"
        return f"{label}(entity)"
    node = files.get(node_id, {})
    path = str(node.get("path") or "")
    base = file_alias.get(node_id) or short_path(path, prefix, max_segments=3)
    layer = layer_for_path(path)
    if layer:
        return f"{base}({layer})"
    return base


def trace_graph_lines(
    *,
    files: Dict[str, Dict],
    entities: Dict[str, Dict[str, object]],
    trace_edges: List[Tuple[str, str, str]],
    entrypoints: List[str],
    route_entries: List[Dict[str, object]],
    file_alias: Dict[str, str],
    entity_alias: Dict[str, str],
    prefix: str,
    max_depth: int,
    max_items: int,
) -> List[str]:
    if not trace_edges:
        return []
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_kind: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for src, dst, kind in trace_edges:
        adj[src].append(dst)
        if kind:
            edge_kind[(src, dst)].add(kind)
    route_fids = [entry.get("fid") for entry in route_entries if isinstance(entry.get("fid"), str)]
    start_nodes = []
    for fid in entrypoints + route_fids:
        if isinstance(fid, str) and fid in files and fid not in start_nodes:
            start_nodes.append(fid)
    lines = ["[TRACE_GRAPH]"]
    edge_count = 0
    for start in start_nodes:
        frontier = [start]
        visited = {start}
        depth = 0
        while frontier and depth < max_depth and edge_count < max_items:
            next_frontier: List[str] = []
            for node in frontier:
                for nxt in adj.get(node, []):
                    if edge_count >= max_items:
                        break
                    edge_count += 1
                    src_label = trace_node_label(
                        node,
                        files=files,
                        entities=entities,
                        file_alias=file_alias,
                        entity_alias=entity_alias,
                        prefix=prefix,
                    )
                    dst_label = trace_node_label(
                        nxt,
                        files=files,
                        entities=entities,
                        file_alias=file_alias,
                        entity_alias=entity_alias,
                        prefix=prefix,
                    )
                    kinds = edge_kind.get((node, nxt), set())
                    suffix = ""
                    if kinds:
                        suffix = f" ({','.join(sorted(kinds))})"
                    lines.append(f"{src_label} -> {dst_label}{suffix}")
                    if nxt not in visited:
                        visited.add(nxt)
                        next_frontier.append(nxt)
                if edge_count >= max_items:
                    break
            frontier = next_frontier
            depth += 1
        if edge_count >= max_items:
            break
    return lines if len(lines) > 1 else []
