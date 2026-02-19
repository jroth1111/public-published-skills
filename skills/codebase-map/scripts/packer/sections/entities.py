from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..digest_core import file_id, short_path


def entity_section_lines(
    entities: List[Dict[str, object]],
    *,
    entity_nodes: Optional[Dict[str, Dict[str, object]]] = None,
    entity_edges: Optional[List[Dict[str, object]]] = None,
    prefix: str,
    file_alias: Dict[str, str],
    max_entities: int = 8,
    max_fields: int = 4,
    max_rels: int = 3,
) -> Tuple[List[str], Dict[str, str]]:
    alias_map: Dict[str, str] = {}
    if entity_nodes:
        entity_list = []
        for ent_id, node in entity_nodes.items():
            entry = dict(node)
            entry["id"] = ent_id
            entity_list.append(entry)
        entities = entity_list
    if not entities:
        return [], alias_map
    sorted_entities = sorted(
        entities,
        key=lambda ent: (len(ent.get("relations", [])), len(ent.get("fields", []))),
        reverse=True,
    )
    usage: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"defines": [], "uses": []})
    if entity_edges:
        for edge in entity_edges:
            src = edge.get("from")
            dst = edge.get("to")
            kind = edge.get("kind")
            if not src or not dst or kind not in {"defines", "uses"}:
                continue
            usage[str(dst)][kind].append(str(src))
    lines: List[str] = ["[ENTITIES]"]
    lines.append(f"count={len(entities)}")
    entity_names = {str(ent.get("name") or "") for ent in entities if ent.get("name")}
    for idx, entity in enumerate(sorted_entities[:max_entities], start=1):
        ent_id = str(entity.get("id") or "")
        alias = f"E{idx}"
        if ent_id:
            alias_map[ent_id] = alias
        name = str(entity.get("name") or "")
        path = str(entity.get("path") or "")
        file_ref = file_alias.get(file_id(path)) or short_path(path, prefix, max_segments=3)
        fields: List[str] = []
        rels: List[str] = []
        raw_fields = entity.get("fields", [])
        if isinstance(raw_fields, list):
            for field in raw_fields[:max_fields]:
                if not isinstance(field, dict):
                    continue
                fname = str(field.get("name") or "")
                ftype = str(field.get("type") or "")
                optional = bool(field.get("optional"))
                if not fname:
                    continue
                pk = fname.lower() in {"id", "uuid"} or ftype.lower() in {"uuid", "id"} or "uuid" in ftype.lower()
                label = fname + ("?" if optional else "")
                if ftype:
                    label = f"{label}:{ftype}"
                if pk:
                    label = f"{label}(pk)"
                fields.append(label)
                # infer FK relation
                if fname.lower().endswith("id") and len(fname) > 2:
                    base = fname[:-2]
                    for ent_name in entity_names:
                        if ent_name and ent_name.lower() == base.lower():
                            rels.append(f"{ent_name}.id")
        for rel in entity.get("relations", [])[:max_rels]:
            if isinstance(rel, str) and rel:
                rels.append(rel)
        rels = list(dict.fromkeys(rels))[:max_rels]
        line = f"{alias} {name}"
        if file_ref:
            line += f" loc={file_ref}"
        if fields:
            line += " fields=" + ",".join(fields)
        if rels:
            line += " rels=" + ",".join(rels)
        if ent_id:
            defines = usage.get(ent_id, {}).get("defines", [])
            uses = usage.get(ent_id, {}).get("uses", [])
            if defines:
                define_refs = [file_alias.get(fid, fid) for fid in defines][:3]
                line += " defined_in=" + ",".join(define_refs)
            if uses:
                use_refs = [file_alias.get(fid, fid) for fid in uses][:3]
                line += " uses=" + ",".join(use_refs)
        lines.append(line)
    return lines, alias_map
