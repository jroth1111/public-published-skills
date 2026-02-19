from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ir import file_id


def is_noise_path(path: str) -> bool:
    lower = path.lower()
    if any(
        token in lower
        for token in (
            "/dist/",
            "/build/",
            "/out/",
            "/.output/",
            "/.tanstack/",
            "/.tanstack-start/",
            "/.nitro/",
            "/.vercel/",
            "/.vinxi/",
            "/.vite/",
            "/.svelte-kit/",
            "/.astro/",
            "/.serverless/",
            "/.wrangler/",
            "/.terraform/",
            "/.nx/",
            "/target/",
            "/coverage/",
            "/htmlcov/",
            "/.next/",
            "/.nuxt/",
            "/node_modules/",
            "/__pycache__/",
            "/.dart_tool/",
            "/.pytest_cache/",
            "/.mypy_cache/",
            "/.ruff_cache/",
            "/.tox/",
            "/.nox/",
            "/.code-state/",
            "/.beads/",
        )
    ):
        return True
    if any(token in lower for token in ("/__tests__/", "/tests/", "/test_", ".test.", ".spec.")):
        return True
    return False


def extract_ts_fields(body: str, *, max_fields: int) -> List[Dict[str, str]]:
    fields: List[Dict[str, str]] = []
    for name, opt, field_type in re.findall(r"\b(\w+)\s*(\??)\s*:\s*([^;]+);", body):
        if not name:
            continue
        if name.startswith("_"):
            continue
        field_type = field_type.strip()
        if field_type.endswith(","):
            field_type = field_type[:-1].strip()
        fields.append({"name": name, "type": field_type})
        if len(fields) >= max_fields:
            break
    return fields


def extract_braced_block(content: str, start_idx: int) -> Optional[str]:
    depth = 0
    for idx in range(start_idx, len(content)):
        if content[idx] == "{":
            depth += 1
            if depth == 1:
                start_idx = idx + 1
        elif content[idx] == "}":
            depth -= 1
            if depth == 0:
                return content[start_idx:idx]
    return None


def looks_like_entity(name: str, path: str, fields: List[Dict[str, str]], *, hint: bool) -> bool:
    if hint:
        return True
    if not name or name[0].islower():
        return False
    if len(fields) < 2:
        return False
    if any(token in path.lower() for token in ("/test", "/spec", "mock", "fixture")):
        return False
    return True


def extract_drizzle_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    for match in re.finditer(r"export\s+const\s+(\w+)\s*=\s*pgTable\(\s*['\"]([^'\"]+)['\"]", content):
        name = match.group(1)
        table_name = match.group(2)
        brace_idx = content.find("{", match.end())
        if brace_idx == -1:
            continue
        body = extract_braced_block(content, brace_idx)
        if body is None:
            continue
        fields: List[Dict[str, str]] = []
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped:
                continue
            field_match = re.match(r"(\w+)\s*:\s*([A-Za-z0-9_\.]+)", stripped)
            if field_match:
                fields.append({"name": field_match.group(1), "type": field_match.group(2)})
            if len(fields) >= max_fields:
                break
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "drizzle",
                "table": table_name,
                "fields": fields,
                "relations": [],
            }
        )
    return entities


def extract_typeorm_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    decorator_markers = ["@Column", "@PrimaryGeneratedColumn", "@PrimaryColumn"]
    for match in re.finditer(r"@Entity\(\)\s*\n\s*export\s+class\s+(\w+)", content):
        name = match.group(1)
        brace_idx = content.find("{", match.end())
        if brace_idx == -1:
            continue
        body = extract_braced_block(content, brace_idx)
        if body is None:
            continue
        fields: List[Dict[str, str]] = []
        decorator_active = False
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                if any(marker in stripped for marker in decorator_markers):
                    decorator_active = True
                continue
            if not decorator_active:
                continue
            match_field = re.match(r"(\w+)\s*(?:!|\?)?\s*:\s*([^=;]+)", stripped)
            if match_field:
                fields.append({"name": match_field.group(1), "type": match_field.group(2).strip()})
                decorator_active = False
            else:
                assign_match = re.match(r"(\w+)\s*=", stripped)
                if assign_match:
                    fields.append({"name": assign_match.group(1), "type": ""})
                    decorator_active = False
            if len(fields) >= max_fields:
                break
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "typeorm",
                "fields": fields,
                "relations": [],
            }
        )
    return entities


def extract_ts_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    seen: Set[str] = set()
    for entity in extract_drizzle_entities(content, path, max_fields=max_fields):
        if entity["name"] in seen:
            continue
        seen.add(entity["name"])
        entities.append(entity)
    for entity in extract_typeorm_entities(content, path, max_fields=max_fields):
        if entity["name"] in seen:
            continue
        seen.add(entity["name"])
        entities.append(entity)
    for match in re.finditer(r"(?:export\s+)?interface\s+(\w+)\s*\{", content):
        name = match.group(1)
        if name in seen:
            continue
        body = extract_braced_block(content, match.end() - 1)
        if body is None:
            continue
        fields = extract_ts_fields(body, max_fields=max_fields)
        if not looks_like_entity(name, path, fields, hint=False):
            continue
        seen.add(name)
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "interface",
                "fields": fields,
                "relations": [],
            }
        )
    for match in re.finditer(r"(?:export\s+)?type\s+(\w+)\s*=\s*\{", content):
        name = match.group(1)
        if name in seen:
            continue
        body = extract_braced_block(content, match.end() - 1)
        if body is None:
            continue
        fields = extract_ts_fields(body, max_fields=max_fields)
        if not looks_like_entity(name, path, fields, hint=False):
            continue
        seen.add(name)
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "type",
                "fields": fields,
                "relations": [],
            }
        )

    for match in re.finditer(r"(?:export\s+)?const\s+(\w+)\s*=\s*z\.object\s*\(", content):
        name = match.group(1)
        if name in seen:
            continue
        brace_idx = content.find("{", match.end())
        if brace_idx == -1:
            continue
        body = extract_braced_block(content, brace_idx)
        if body is None:
            continue
        fields = []
        for field_name in re.findall(r"\b(\w+)\s*:", body):
            fields.append({"name": field_name, "type": "z"})
            if len(fields) >= max_fields:
                break
        if not looks_like_entity(name, path, fields, hint=True):
            continue
        seen.add(name)
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "zod",
                "fields": fields,
                "relations": [],
            }
        )
    for entity in extract_graphql_template_entities(content, path, max_fields=max_fields):
        if entity["name"] in seen:
            continue
        seen.add(entity["name"])
        entities.append(entity)
    return entities


def extract_py_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    content_lower = content.lower()
    has_model_hint = any(token in content_lower for token in ("pydantic", "basemodel", "django.db", "sqlalchemy", "mongoengine"))

    lines = content.splitlines()
    for idx, line in enumerate(lines):
        match = re.match(r"\s*class\s+(\w+)\s*(\(([^)]*)\))?:", line)
        if not match:
            continue
        name = match.group(1)
        bases = match.group(3) or ""
        hint = has_model_hint or any(token in bases for token in ("BaseModel", "models.Model", "Model", "Document"))
        if not hint and not looks_like_entity(name, path, [], hint=False):
            continue

        indent = len(line) - len(line.lstrip())
        body_lines: List[str] = []
        for jdx in range(idx + 1, len(lines)):
            next_line = lines[jdx]
            if not next_line.strip():
                continue
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent <= indent:
                break
            body_lines.append(next_line)

        fields: List[Dict[str, str]] = []
        for body_line in body_lines:
            stripped = body_line.strip()
            if stripped.startswith("@"):
                continue
            ann_match = re.match(r"(\w+)\s*:\s*([^=]+)", stripped)
            if ann_match:
                field_name = ann_match.group(1)
                field_type = ann_match.group(2).strip()
                fields.append({"name": field_name, "type": field_type})
            else:
                assign_match = re.match(r"(\w+)\s*=\s*([\w\.]+)", stripped)
                if assign_match:
                    fields.append({"name": assign_match.group(1), "type": assign_match.group(2)})
            if len(fields) >= max_fields:
                break

        if not looks_like_entity(name, path, fields, hint=hint):
            continue
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "class",
                "fields": fields,
                "relations": [],
            }
        )
    return entities


def extract_prisma_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    for match in re.finditer(r"\bmodel\s+(\w+)\s*\{", content):
        name = match.group(1)
        body = extract_braced_block(content, match.end() - 1)
        if body is None:
            continue
        fields: List[Dict[str, str]] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("@"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            field_name = parts[0]
            field_type = parts[1]
            if field_name.startswith("_"):
                continue
            fields.append({"name": field_name, "type": field_type})
            if len(fields) >= max_fields:
                break
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "prisma",
                "fields": fields,
                "relations": [],
            }
        )
    return entities


def extract_graphql_entities(content: str, path: str, *, max_fields: int) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    for match in re.finditer(r"\b(type|input|interface)\s+(\w+)\s*\{", content):
        name = match.group(2)
        body = extract_braced_block(content, match.end() - 1)
        if body is None:
            continue
        fields: List[Dict[str, str]] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            field_name, field_type = stripped.split(":", 1)
            field_name = field_name.strip()
            field_type = field_type.strip().split()[0]
            if not field_name:
                continue
            fields.append({"name": field_name, "type": field_type})
            if len(fields) >= max_fields:
                break
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "graphql",
                "fields": fields,
                "relations": [],
            }
        )
    return entities


def extract_graphql_template_entities(
    content: str, path: str, *, max_fields: int
) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    for match in re.finditer(r"\b(?:gql|graphql)\s*`([\s\S]*?)`", content):
        chunk = match.group(1)
        if not chunk:
            continue
        entities.extend(extract_graphql_entities(chunk, path, max_fields=max_fields))
    return entities


def extract_sql_entities(content: str, path: str, *, max_fields: int, max_tables: int = 4) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    table_matches = re.finditer(
        r"create\s+table\s+([^\s(]+)\s*\((.*?)\);",
        content,
        re.IGNORECASE | re.DOTALL,
    )
    for match in table_matches:
        name = match.group(1).strip("`\"")
        body = match.group(2)
        fields: List[Dict[str, str]] = []
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped or stripped.startswith("--"):
                continue
            first = stripped.split()[0].strip("`\"")
            if first.lower() in {"primary", "foreign", "constraint", "unique", "index"}:
                continue
            fields.append({"name": first, "type": ""})
            if len(fields) >= max_fields:
                break
        entities.append(
            {
                "name": name,
                "path": path,
                "kind": "sql_table",
                "fields": fields,
                "relations": [],
            }
        )
        if len(entities) >= max_tables:
            break
    return entities


def component_key_for_entity(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return "root"
    if parts[0] in {"packages", "apps"} and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] in {"src", "lib", "app", "python"} and len(parts) > 1:
        return parts[1]
    return parts[0]


def extract_entities(
    repo: Path,
    files: Sequence[str],
    *,
    max_entities: int = 50,
    max_fields: int = 8,
    exported_names_by_path: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict[str, object]]:
    exported_names_by_path = exported_names_by_path or {}
    candidates: List[Dict[str, object]] = []
    seen: Set[Tuple[str, str]] = set()
    max_candidates = max_entities * 4
    for path in files:
        if is_noise_path(path):
            continue
        if not path.endswith((".ts", ".tsx", ".py", ".prisma", ".graphql", ".gql", ".sql")):
            continue
        file_path = repo / path
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(content) > 200_000 and not path.endswith((".prisma", ".graphql", ".gql", ".sql")):
            continue
        if path.endswith(".py"):
            extracted = extract_py_entities(content, path, max_fields=max_fields)
        elif path.endswith((".ts", ".tsx")):
            extracted = extract_ts_entities(content, path, max_fields=max_fields)
        elif path.endswith(".prisma"):
            extracted = extract_prisma_entities(content, path, max_fields=max_fields)
        elif path.endswith((".graphql", ".gql")):
            extracted = extract_graphql_entities(content, path, max_fields=max_fields)
        else:
            extracted = extract_sql_entities(content, path, max_fields=max_fields)
        for entity in extracted:
            key = (entity["name"], entity["path"])
            if key in seen:
                continue
            seen.add(key)
            entity_name = str(entity.get("name") or "")
            entity["component"] = component_key_for_entity(path)
            entity["export_boost"] = 1 if entity_name in exported_names_by_path.get(path, set()) else 0
            candidates.append(entity)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    by_component: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entity in candidates:
        component = str(entity.get("component") or "root")
        by_component[component].append(entity)
    for entities in by_component.values():
        entities.sort(
            key=lambda ent: (
                int(ent.get("export_boost", 0)),
                len(ent.get("relations", [])),
                len(ent.get("fields", [])),
            ),
            reverse=True,
        )

    selected: List[Dict[str, object]] = []
    components_sorted = sorted(by_component.keys())
    while len(selected) < max_entities and any(by_component.values()):
        for component in components_sorted:
            if not by_component[component]:
                continue
            selected.append(by_component[component].pop(0))
            if len(selected) >= max_entities:
                break

    names = {entity["name"] for entity in selected}
    for entity in selected:
        relations: Set[str] = set()
        for field in entity.get("fields", []):
            field_type = str(field.get("type") or "")
            for token in re.findall(r"[A-Z][A-Za-z0-9_]+", field_type):
                if token in names and token != entity["name"]:
                    relations.add(token)
        entity["relations"] = sorted(relations)[:4]
        entity.pop("component", None)
        entity.pop("export_boost", None)
    return selected


def build_entity_graph(
    entities: List[Dict[str, object]],
    *,
    file_edges: List[Dict[str, object]],
) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    entity_nodes: Dict[str, Dict[str, object]] = {}
    entity_edges: List[Dict[str, object]] = []
    if not entities:
        return entity_nodes, entity_edges
    used_ids: Counter[str] = Counter()
    entities_by_file: Dict[str, List[str]] = defaultdict(list)
    by_name: Dict[str, List[str]] = defaultdict(list)

    for entity in entities:
        name = str(entity.get("name") or "")
        path = str(entity.get("path") or "")
        if not name or not path:
            continue
        base_id = f"entity:{name}"
        used_ids[base_id] += 1
        entity_id = base_id if used_ids[base_id] == 1 else f"{base_id}#{used_ids[base_id]}"
        node = {
            "id": entity_id,
            "name": name,
            "kind": entity.get("kind"),
            "path": path,
            "fields": entity.get("fields", []),
            "relations": entity.get("relations", []),
            "source": "heuristic",
        }
        entity_nodes[entity_id] = node
        entities_by_file[file_id(path)].append(entity_id)
        by_name[name.lower()].append(entity_id)

    # Defines edges
    for fid, ent_ids in entities_by_file.items():
        for ent_id in ent_ids:
            entity_edges.append({"from": fid, "to": ent_id, "kind": "defines"})

    # Uses edges via file dependencies
    for edge in file_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        for ent_id in entities_by_file.get(dst, []):
            entity_edges.append({"from": src, "to": ent_id, "kind": "uses"})

    # Relation edges between entities (when unique)
    for ent_id, node in entity_nodes.items():
        for rel in node.get("relations", []) or []:
            rel_name = str(rel).lower()
            targets = by_name.get(rel_name, [])
            if len(targets) == 1:
                entity_edges.append({"from": ent_id, "to": targets[0], "kind": "relates"})

    return entity_nodes, entity_edges
