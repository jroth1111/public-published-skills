from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from ..digest_core import component_key


def diagram_data(
    files: Dict[str, Dict],
    edges: List[Dict],
    *,
    prefix: str,
    depth: int,
    max_nodes: int,
    max_edges: int,
) -> Tuple[List[str], Dict[str, int], List[Tuple[str, str, int]]]:
    comp_for_file: Dict[str, str] = {}
    comp_counts: Counter[str] = Counter()
    for fid, node in files.items():
        path = node.get("path", "")
        if not path:
            continue
        comp = component_key(path, prefix, depth)
        comp_for_file[fid] = comp
        comp_counts[comp] += 1

    if not comp_counts:
        return [], {}, []

    comp_degree: Counter[str] = Counter()
    comp_edges: Counter[Tuple[str, str]] = Counter()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in comp_for_file or dst not in comp_for_file:
            continue
        src_comp = comp_for_file[src]
        dst_comp = comp_for_file[dst]
        if src_comp == dst_comp:
            continue
        weight = edge.get("weight", 1)
        try:
            weight_value = int(weight) if weight is not None else 1
        except (TypeError, ValueError):
            weight_value = 1
        comp_edges[(src_comp, dst_comp)] += weight_value
        comp_degree[src_comp] += weight_value
        comp_degree[dst_comp] += weight_value

    def comp_sort(name: str) -> Tuple[int, int, str]:
        return (-comp_degree[name], -comp_counts[name], name)

    top_components = sorted(comp_counts.keys(), key=comp_sort)[:max_nodes]
    top_set = set(top_components)

    edge_list = [
        (src, dst, weight)
        for (src, dst), weight in comp_edges.items()
        if src in top_set and dst in top_set
    ]
    edge_list.sort(key=lambda item: (-item[2], item[0], item[1]))
    edge_list = edge_list[:max_edges]
    return top_components, dict(comp_counts), edge_list


def diagram_lines(
    files: Dict[str, Dict],
    edges: List[Dict],
    *,
    prefix: str,
    depth: int,
    max_nodes: int,
    max_edges: int,
    mode: str,
) -> List[str]:
    top_components, comp_counts, edge_list = diagram_data(
        files,
        edges,
        prefix=prefix,
        depth=depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    if not top_components:
        return []
    aliases = {name: f"C{idx}" for idx, name in enumerate(top_components, start=1)}
    lines: List[str] = ["[DIAGRAM]"]
    lines.append(f"mode={mode} depth={depth} nodes={len(top_components)} edges={len(edge_list)}")
    if mode == "mermaid":
        lines.append("```mermaid")
        lines.append("flowchart LR")
        for name in top_components:
            label = f"{name} ({comp_counts.get(name, 0)})"
            lines.append(f'{aliases[name]}["{label}"]')
        for src, dst, weight in edge_list:
            lines.append(f"{aliases[src]} -->|{weight}| {aliases[dst]}")
        lines.append("```")
        return lines
    for name in top_components:
        lines.append(f"{aliases[name]} {name} f={comp_counts.get(name, 0)}")
    for src, dst, weight in edge_list:
        lines.append(f"{aliases[src]}>{aliases[dst]}:{weight}")
    return lines


def sequence_lines_from_flows(
    flow_section: List[str],
    *,
    max_lines: int = 4,
) -> List[str]:
    if not flow_section or len(flow_section) < 2:
        return []
    lines = ["[DIAGRAM_SEQ]"]
    for line in flow_section[1:]:
        if len(lines) - 1 >= max_lines:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        lines.append(f"{parts[0]} {parts[2]}")
    return lines if len(lines) > 1 else []
