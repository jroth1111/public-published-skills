from __future__ import annotations

from typing import Dict, Iterable, List, Set

from ir import file_id


def normalize_edge_types(value: str) -> List[str]:
    if not value:
        return []
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        return []
    if "all" in raw:
        return ["file_dep", "symbol_ref", "dataflow", "external_dep", "entity_use"]
    seen: Set[str] = set()
    edge_types: List[str] = []
    for item in raw:
        if item not in seen:
            seen.add(item)
            edge_types.append(item)
    return edge_types


def build_graph(ir: Dict, *, edge_types: Iterable[str], include_all_nodes: bool) -> Dict[str, object]:
    files = ir.get("files", {})
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {})
    entities = ir.get("entities", {})

    nodes: Dict[str, Dict[str, str]] = {}
    edge_list: List[Dict[str, object]] = []
    edge_types_set = set(edge_types)

    def add_node(node_id: str, node_type: str, label: str) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": node_type, "label": label}

    def add_file_node(fid: str) -> None:
        file_node = files.get(fid, {})
        label = file_node.get("path") or fid
        add_node(fid, "file", label)

    def add_symbol_node(sid: str) -> None:
        sym = symbols.get(sid, {})
        label = sym.get("signature") or sym.get("name") or sid
        add_node(sid, "symbol", label)

    def add_external_node(module: str) -> str:
        ext_id = f"ext:{module}"
        add_node(ext_id, "external", module)
        return ext_id

    def add_entity_node(eid: str) -> None:
        ent = entities.get(eid, {})
        label = ent.get("name") or eid
        add_node(eid, "entity", label)

    if include_all_nodes:
        if edge_types_set & {"file_dep", "dataflow", "external_dep"}:
            for fid in files.keys():
                add_file_node(fid)
        if "symbol_ref" in edge_types_set:
            for sid in symbols.keys():
                add_symbol_node(sid)
        if "entity_use" in edge_types_set:
            for eid in entities.keys():
                add_entity_node(eid)

    if "file_dep" in edge_types_set:
        for edge in edges.get("file_dep", []):
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            add_file_node(src)
            add_file_node(dst)
            edge_list.append(
                {
                    "source": src,
                    "target": dst,
                    "type": "file_dep",
                    "weight": edge.get("weight"),
                    "provenance": edge.get("provenance"),
                }
            )

    if "symbol_ref" in edge_types_set:
        for edge in edges.get("symbol_ref", []):
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            add_symbol_node(src)
            add_symbol_node(dst)
            edge_list.append(
                {
                    "source": src,
                    "target": dst,
                    "type": "symbol_ref",
                    "kind": edge.get("kind"),
                    "provenance": edge.get("provenance"),
                }
            )

    if "dataflow" in edge_types_set:
        for edge in edges.get("dataflow", []):
            from_path = edge.get("from_path")
            to_path = edge.get("to_path")
            if not from_path or not to_path:
                continue
            src = file_id(from_path)
            dst = file_id(to_path)
            add_file_node(src)
            add_file_node(dst)
            edge_list.append(
                {
                    "source": src,
                    "target": dst,
                    "type": "dataflow",
                    "from_line": edge.get("from_line"),
                    "to_line": edge.get("to_line"),
                    "provenance": edge.get("provenance"),
                }
            )

    if "external_dep" in edge_types_set:
        for edge in edges.get("external_dep", []):
            src = edge.get("from")
            module = edge.get("module")
            if not src or not module:
                continue
            add_file_node(src)
            ext_id = add_external_node(str(module))
            edge_list.append(
                {
                    "source": src,
                    "target": ext_id,
                    "type": "external_dep",
                    "module": module,
                }
            )

    if "entity_use" in edge_types_set:
        for edge in edges.get("entity_use", []):
            src = edge.get("from")
            dst = edge.get("to")
            if not src or not dst:
                continue
            if src in files:
                add_file_node(src)
            elif src in symbols:
                add_symbol_node(src)
            if dst in entities:
                add_entity_node(dst)
            if src in nodes and dst in nodes:
                edge_list.append(
                    {
                        "source": src,
                        "target": dst,
                        "type": edge.get("kind", "entity_use"),
                    }
                )

    return {
        "directed": True,
        "nodes": list(nodes.values()),
        "edges": edge_list,
    }


def export_graph_json(graph: Dict[str, object]) -> str:
    import json

    return json.dumps(graph, ensure_ascii=True, indent=2)


def export_graphml(graph: Dict[str, object]) -> str:
    def esc(value: object) -> str:
        text = "" if value is None else str(value)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    keys = [
        ("n_label", "node", "label", "string"),
        ("n_type", "node", "type", "string"),
        ("e_type", "edge", "type", "string"),
        ("e_kind", "edge", "kind", "string"),
        ("e_weight", "edge", "weight", "double"),
        ("e_from_line", "edge", "from_line", "int"),
        ("e_to_line", "edge", "to_line", "int"),
        ("e_provenance", "edge", "provenance", "string"),
        ("e_module", "edge", "module", "string"),
    ]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for key_id, scope, name, key_type in keys:
        lines.append(
            f'<key id="{key_id}" for="{scope}" attr.name="{name}" attr.type="{key_type}"/>'
        )
    lines.append('<graph id="G" edgedefault="directed">')

    for node in graph.get("nodes", []):
        node_id = esc(node.get("id"))
        lines.append(f'<node id="{node_id}">')
        lines.append(f'  <data key="n_label">{esc(node.get("label"))}</data>')
        lines.append(f'  <data key="n_type">{esc(node.get("type"))}</data>')
        lines.append("</node>")

    for edge in graph.get("edges", []):
        src = esc(edge.get("source"))
        dst = esc(edge.get("target"))
        lines.append(f'<edge source="{src}" target="{dst}">')
        lines.append(f'  <data key="e_type">{esc(edge.get("type"))}</data>')
        if edge.get("kind") is not None:
            lines.append(f'  <data key="e_kind">{esc(edge.get("kind"))}</data>')
        if edge.get("weight") is not None:
            lines.append(f'  <data key="e_weight">{esc(edge.get("weight"))}</data>')
        if edge.get("from_line") is not None:
            lines.append(f'  <data key="e_from_line">{esc(edge.get("from_line"))}</data>')
        if edge.get("to_line") is not None:
            lines.append(f'  <data key="e_to_line">{esc(edge.get("to_line"))}</data>')
        if edge.get("provenance") is not None:
            lines.append(f'  <data key="e_provenance">{esc(edge.get("provenance"))}</data>')
        if edge.get("module") is not None:
            lines.append(f'  <data key="e_module">{esc(edge.get("module"))}</data>')
        lines.append("</edge>")

    lines.append("</graph>")
    lines.append("</graphml>")
    return "\n".join(lines)
