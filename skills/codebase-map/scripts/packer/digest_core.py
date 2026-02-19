from __future__ import annotations

import fnmatch
import json
import math
import posixpath
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ir import file_id

from .budget import SECTION_BUDGETS, configure_tokenizer, estimate_tokens


CAPABILITY_RULES: List[Dict[str, object]] = [
    {
        "name": "retry_backoff",
        "patterns": [r"\bretry\b", r"\bbackoff\b", r"exponential"],
        "path_tokens": ["retry", "backoff"],
    },
    {
        "name": "rate_limiting",
        "patterns": [r"rate.?limit", r"\bthrottl", r"\brate[_-]?limit"],
        "path_tokens": ["rate-limit", "ratelimit", "throttle"],
    },
    {
        "name": "caching",
        "patterns": [r"\bcache\b", r"\bmemoiz", r"\blru\b"],
        "path_tokens": ["cache", "caching"],
    },
    {
        "name": "queues_workers",
        "patterns": [r"\bqueue\b", r"\bworker\b", r"\bjob\b", r"\bcelery\b", r"\bbullmq\b"],
        "path_tokens": ["queue", "worker", "jobs"],
    },
    {
        "name": "schedulers_cron",
        "patterns": [r"\bcron\b", r"\bschedul", r"\binterval\b"],
        "path_tokens": ["cron", "scheduler", "schedule"],
    },
    {
        "name": "metrics_telemetry",
        "patterns": [r"\bmetrics?\b", r"\btelemetry\b", r"\botel\b", r"\bprometheus\b", r"\bstatsd\b"],
        "path_tokens": ["metrics", "telemetry", "observability"],
    },
    {
        "name": "feature_flags",
        "patterns": [r"feature.?flag", r"\blaunchdarkly\b", r"\bunleash\b"],
        "path_tokens": ["feature-flag", "feature_flag", "flags"],
    },
    {
        "name": "file_uploads",
        "patterns": [r"\bupload\b", r"\bmultipart\b", r"\bmulter\b", r"\bpresign"],
        "path_tokens": ["upload", "uploads"],
    },
    {
        "name": "email_delivery",
        "patterns": [r"\bemail\b", r"\bsmtp\b", r"\bsendgrid\b", r"\bmailgun\b", r"\bses\b"],
        "path_tokens": ["mail", "email", "mailer"],
    },
    {
        "name": "search_indexing",
        "patterns": [r"\bsearch\b", r"\bindex\b", r"\belasticsearch\b", r"\bmeilisearch\b"],
        "path_tokens": ["search", "index"],
    },
    {
        "name": "request_validation",
        "patterns": [r"\bvalidate\b", r"\bvalidator\b", r"\bzod\b", r"\bjoi\b", r"\bpydantic\b"],
        "path_tokens": ["validation", "validators"],
    },
    {
        "name": "structured_logging",
        "patterns": [r"\blogger\b", r"\blogging\b", r"\bpino\b", r"\bwinston\b", r"\bloguru\b"],
        "path_tokens": ["log", "logging", "logger"],
    },
]


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def auto_limits(budget_tokens: int) -> Tuple[int, int, int, int]:
    max_files = clamp(budget_tokens // 200, 8, 120)
    max_symbols = clamp(budget_tokens // max(1, max_files * 12), 1, 6)
    min_edges = 0 if budget_tokens < 800 else 20
    max_edges = clamp(budget_tokens // 120, min_edges, 240)
    if budget_tokens <= 400:
        max_sig_len = 40
    elif budget_tokens <= 800:
        max_sig_len = 60
    else:
        max_sig_len = 80
    return max_files, max_symbols, max_edges, max_sig_len


def compact_signature(signature: str, max_len: int) -> str:
    if max_len <= 0:
        return signature
    sig = signature.strip()
    if len(sig) <= max_len:
        return sig
    if "->" in sig:
        left = sig.split("->", 1)[0].rstrip()
        if len(left) <= max_len:
            return left
        sig = left
    if len(sig) <= max_len:
        return sig
    if max_len <= 3:
        return sig[:max_len]
    return sig[: max_len - 3] + "..."


def compact_doc(text: str, max_len: int = 80) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_len:
        return cleaned
    if max_len <= 3:
        return cleaned[:max_len]
    return cleaned[: max_len - 3] + "..."


def resolve_file_id(ir: Dict, value: str) -> Optional[str]:
    if value.startswith("file:"):
        return value if value in ir.get("files", {}) else None
    for fid, node in ir.get("files", {}).items():
        if node.get("path") == value:
            return fid
    return None


def resolve_symbol_id(ir: Dict, value: str) -> Optional[str]:
    if value.startswith("sym:"):
        return value if value in ir.get("symbols", {}) else None
    candidates = [
        sid
        for sid, node in ir.get("symbols", {}).items()
        if node.get("name") == value
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def top_symbols_for_file(ir: Dict, fid: str, limit: int) -> List[str]:
    file_node = ir.get("files", {}).get(fid, {})
    symbols = file_node.get("exports") or file_node.get("defines", [])
    symbols = list(dict.fromkeys(symbols))
    symbols.sort(key=lambda s: ir["symbols"].get(s, {}).get("score", 0.0), reverse=True)
    return symbols[:limit]


def is_barrel_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in {
        "index.ts",
        "index.tsx",
        "index.js",
        "index.jsx",
        "index.mjs",
        "index.cjs",
        "index.d.ts",
        "__init__.py",
        "__init__.pyi",
    }


def public_api_lines(
    files: Dict[str, Dict],
    *,
    prefix: str,
    max_items: int = 6,
) -> List[str]:
    candidates: List[Tuple[int, float, str]] = []
    for node in files.values():
        path = node.get("path", "")
        if not path:
            continue
        exports = node.get("exports") or []
        re_exports = node.get("re_exports") or []
        export_count = len(exports) + len(re_exports)
        if export_count <= 0:
            continue
        if is_barrel_path(path) or node.get("role") == "entrypoint":
            candidates.append((export_count, float(node.get("score", 0.0)), path))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    lines = ["[PUBLIC_API]", f"count={len(candidates)}"]
    for idx, (exports_count, _score, path) in enumerate(candidates[:max_items], start=1):
        short = short_path(path, prefix, max_segments=3)
        lines.append(f"P{idx} {short} exports={exports_count}")
    return lines


def focus_candidates(ir: Dict, focus: str, allowed_files: Optional[Set[str]] = None) -> Set[str]:
    focus = focus.lower()
    candidates: Set[str] = set()
    for fid, node in ir.get("files", {}).items():
        if allowed_files is not None and fid not in allowed_files:
            continue
        if focus in node.get("path", "").lower():
            candidates.add(fid)
    for sid, node in ir.get("symbols", {}).items():
        if focus in node.get("name", "").lower():
            file_path = node.get("defined_in", {}).get("path")
            if file_path:
                fid = file_id(file_path)
                if allowed_files is None or fid in allowed_files:
                    candidates.add(fid)
    return candidates


def neighbor_files(
    ir: Dict, file_ids: Iterable[str], allowed_files: Optional[Set[str]] = None
) -> Set[str]:
    neighbors: Set[str] = set()
    for edge in ir.get("edges", {}).get("file_dep", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src in file_ids:
            if allowed_files is None or dst in allowed_files:
                neighbors.add(dst)
        if dst in file_ids:
            if allowed_files is None or src in allowed_files:
                neighbors.add(src)
    return neighbors


def expand_focus_neighbors(
    ir: Dict,
    focus_ids: Set[str],
    *,
    depth: int = 1,
    allowed_files: Optional[Set[str]] = None,
) -> Set[str]:
    if depth <= 1:
        return set(focus_ids)
    current = set(focus_ids)
    frontier = set(focus_ids)
    for _ in range(depth - 1):
        nxt = neighbor_files(ir, frontier, allowed_files=allowed_files)
        nxt -= current
        if not nxt:
            break
        current |= nxt
        frontier = nxt
    return current


def dataflow_for_path(ir: Dict, path: str) -> Tuple[List[Dict], List[Dict]]:
    inbound: List[Dict] = []
    outbound: List[Dict] = []
    for edge in ir.get("edges", {}).get("dataflow", []):
        if edge.get("from_path") == path:
            outbound.append(edge)
        if edge.get("to_path") == path:
            inbound.append(edge)
    return inbound, outbound


def common_path_prefix(paths: List[str]) -> str:
    if len(paths) < 2:
        return ""
    parts = [path.split("/") for path in paths]
    prefix: List[str] = []
    for chunk in zip(*parts):
        if len(set(chunk)) == 1:
            prefix.append(chunk[0])
        else:
            break
    if not prefix:
        return ""
    prefix_str = "/".join(prefix)
    if prefix_str in {"", "."}:
        return ""
    return prefix_str + "/"


def short_path(path: str, prefix: str, max_segments: int = 3) -> str:
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    parts = path.split("/")
    if len(parts) > max_segments:
        return "/".join(parts[-max_segments:])
    return path


def component_key(path: str, prefix: str, depth: int) -> str:
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    dir_path = path.rsplit("/", 1)[0] if "/" in path else "."
    if dir_path in {"", "."}:
        return "."
    parts = [part for part in dir_path.split("/") if part]
    if not parts:
        return "."
    depth = max(1, depth)
    return "/".join(parts[:depth])


def component_for_path(path: str, prefix: str) -> str:
    if prefix and path.startswith(prefix):
        path = path[len(prefix) :]
    parts = [part for part in path.split("/") if part]
    if not parts:
        return "root"
    if parts[0] in {"packages", "apps"} and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] in {"src", "lib", "app", "python"}:
        return parts[1] if len(parts) > 1 else "root"
    return parts[0]


def build_component_index(files: Dict[str, Dict], prefix: str) -> Dict[str, List[str]]:
    comp_files: Dict[str, List[str]] = defaultdict(list)
    for fid, node in files.items():
        if node.get("role") in {"test", "config"}:
            continue
        path = node.get("path", "")
        if not path:
            continue
        comp = component_for_path(path, prefix)
        comp_files[comp].append(fid)
    return comp_files


def component_edge_counts(
    edges: List[Dict], comp_files: Dict[str, List[str]]
) -> Tuple[Counter, Counter, Counter]:
    comp_for_fid: Dict[str, str] = {}
    for comp, fids in comp_files.items():
        for fid in fids:
            comp_for_fid[fid] = comp
    counts: Counter[Tuple[str, str]] = Counter()
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in comp_for_fid or dst not in comp_for_fid:
            continue
        src_comp = comp_for_fid[src]
        dst_comp = comp_for_fid[dst]
        if src_comp == dst_comp:
            continue
        weight = edge.get("weight", 1)
        try:
            weight_value = int(weight) if weight is not None else 1
        except (TypeError, ValueError):
            weight_value = 1
        counts[(src_comp, dst_comp)] += weight_value
        out_degree[src_comp] += weight_value
        in_degree[dst_comp] += weight_value
    return counts, in_degree, out_degree


def median(values: List[int]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2.0


def component_risk_flags(
    comp_counts: Dict[str, int], in_degree: Counter, out_degree: Counter
) -> List[str]:
    flags: List[str] = []
    if not comp_counts:
        return flags
    total_files = sum(comp_counts.values())
    top_comp, top_count = max(comp_counts.items(), key=lambda item: item[1])
    if top_count >= max(20, int(total_files * 0.35)):
        flags.append(f"god_component={top_comp}")

    out_values = [count for count in out_degree.values() if count > 0]
    if out_values:
        top_out_comp = max(out_degree.items(), key=lambda item: item[1])[0]
        top_out = out_degree[top_out_comp]
        if top_out >= max(5, 2 * median(out_values)):
            flags.append(f"orchestrator={top_out_comp}")

    in_values = [count for count in in_degree.values() if count > 0]
    if in_values:
        top_in_comp = max(in_degree.items(), key=lambda item: item[1])[0]
        top_in = in_degree[top_in_comp]
        if top_in >= max(5, 2 * median(in_values)):
            flags.append(f"core_component={top_in_comp}")
    return flags


def component_cycles(
    comp_edges: Counter, limit: int = 3, max_depth: int = 4
) -> List[List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for (src, dst), _weight in comp_edges.items():
        adj[src].append(dst)

    cycles: Set[Tuple[str, ...]] = set()

    def record_cycle(path: List[str]) -> None:
        if len(path) < 2:
            return
        cycle = path[:-1]
        if not cycle:
            return
        rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
        cycles.add(min(rotations))

    def dfs(start: str, node: str, path: List[str]) -> None:
        if len(path) > max_depth:
            return
        for nxt in adj.get(node, []):
            if nxt == start and len(path) > 1:
                record_cycle(path + [nxt])
                if len(cycles) >= limit:
                    return
            elif nxt not in path:
                dfs(start, nxt, path + [nxt])
                if len(cycles) >= limit:
                    return

    for node in list(adj.keys()):
        dfs(node, node, [node])
        if len(cycles) >= limit:
            break
    return [list(cycle) for cycle in list(cycles)[:limit]]


def component_section_lines(
    comp_files: Dict[str, List[str]],
    files: Dict[str, Dict],
    *,
    max_components: int = 8,
    component_order: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    if not comp_files:
        return [], {}
    ordered: List[Tuple[str, List[str]]] = []
    if component_order:
        ordered = [(comp, comp_files[comp]) for comp in component_order if comp in comp_files]
        if len(ordered) < max_components:
            selected = {comp for comp, _fids in ordered}
            remaining = [
                item
                for item in sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
                if item[0] not in selected
            ]
            ordered.extend(remaining)
    else:
        ordered = sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
    ordered = ordered[:max_components]
    lines = ["[COMPONENTS]", f"count={len(comp_files)}"]
    aliases: Dict[str, str] = {}
    for idx, (comp, fids) in enumerate(ordered[:max_components], start=1):
        alias = f"C{idx}"
        aliases[comp] = alias
        roles = Counter(
            classify_path_role(files.get(fid, {}).get("path", "")) for fid in fids
        )
        role_list = [role for role, _count in roles.most_common(2) if role and role != "other"]
        line = f"{alias} {comp} files={len(fids)}"
        if role_list:
            line += " roles=" + ",".join(role_list)
        lines.append(line)
    return lines, aliases


def split_identifier(value: str) -> List[str]:
    if not value:
        return []
    parts = re.sub(r"[^A-Za-z0-9]+", " ", value).split()
    tokens: List[str] = []
    for part in parts:
        camel = re.findall(r"[A-Z]?[a-z]+|[0-9]+|[A-Z]+(?=[A-Z]|$)", part)
        if camel:
            tokens.extend(camel)
        else:
            tokens.append(part)
    return [token.lower() for token in tokens if token]


SEMANTIC_STOPWORDS = {
    "src",
    "lib",
    "app",
    "apps",
    "package",
    "packages",
    "test",
    "tests",
    "spec",
    "specs",
    "config",
    "configs",
    "common",
    "shared",
    "core",
    "utils",
    "util",
    "helpers",
    "helper",
    "index",
    "main",
    "init",
    "internal",
    "public",
    "private",
    "types",
    "type",
    "schema",
    "schemas",
}


def normalize_terms(tokens: Iterable[str]) -> List[str]:
    results: List[str] = []
    for token in tokens:
        t = token.strip().lower()
        if not t or t in SEMANTIC_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        if t.isdigit():
            continue
        results.append(t)
    return results


def semantic_terms_for_file(
    node: Dict[str, Any],
    symbols: Dict[str, Dict[str, Any]],
    *,
    max_symbols: int = 8,
    max_terms: int = 80,
) -> List[str]:
    tokens: List[str] = []
    path = node.get("path", "") or ""
    for part in path.split("/"):
        tokens.extend(split_identifier(part))
    summary = node.get("summary_1l", "") or ""
    if summary:
        tokens.extend(split_identifier(summary))
    symbol_ids = node.get("exports") or node.get("defines") or []
    scored = []
    for sid in symbol_ids:
        sym = symbols.get(sid, {})
        scored.append((sym.get("score", 0.0), sid))
    for _score, sid in sorted(scored, reverse=True)[:max_symbols]:
        sym = symbols.get(sid, {})
        tokens.extend(split_identifier(sym.get("name", "") or ""))
        doc = sym.get("documentation") if isinstance(sym.get("documentation"), dict) else {}
        doc_summary = doc.get("summary") or sym.get("doc_1l") or ""
        if doc_summary:
            tokens.extend(split_identifier(str(doc_summary)))
    normalized = normalize_terms(tokens)
    return normalized[:max_terms]


def semantic_terms_name_only(
    node: Dict[str, Any],
    *,
    max_terms: int = 80,
) -> List[str]:
    tokens: List[str] = []
    path = node.get("path", "") or ""
    for part in path.split("/"):
        tokens.extend(split_identifier(part))
    summary = node.get("summary_1l", "") or ""
    if summary:
        tokens.extend(split_identifier(summary))
    normalized = normalize_terms(tokens)
    return normalized[:max_terms]


def build_tfidf_vectors(
    docs: Dict[str, List[str]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    df: Counter[str] = Counter()
    for tokens in docs.values():
        df.update(set(tokens))
    total_docs = max(1, len(docs))
    idf: Dict[str, float] = {}
    for term, count in df.items():
        idf[term] = math.log((total_docs + 1) / (count + 1)) + 1.0
    vectors: Dict[str, Dict[str, float]] = {}
    norms: Dict[str, float] = {}
    for fid, tokens in docs.items():
        tf = Counter(tokens)
        vec: Dict[str, float] = {}
        norm = 0.0
        for term, count in tf.items():
            weight = float(count) * idf.get(term, 1.0)
            vec[term] = weight
            norm += weight * weight
        norm = math.sqrt(norm) if norm > 0 else 0.0
        vectors[fid] = vec
        norms[fid] = norm
    return vectors, norms


def cosine_similarity(
    vec_a: Dict[str, float],
    norm_a: float,
    vec_b: Dict[str, float],
    norm_b: float,
) -> float:
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        norm_a, norm_b = norm_b, norm_a
    dot = 0.0
    for term, val in vec_a.items():
        other = vec_b.get(term)
        if other is not None:
            dot += val * other
    return dot / (norm_a * norm_b)


def build_file_call_stats(
    ir: Dict[str, Any],
    files: Dict[str, Dict[str, Any]],
) -> Tuple[Counter, Counter]:
    edges = ir.get("edges", {}).get("symbol_ref", [])
    if not edges:
        return Counter(), Counter()
    file_ids = set(files.keys())
    symbols = ir.get("symbols", {})
    codeql_edges = [edge for edge in edges if edge.get("provenance") == "codeql"]
    use_edges = codeql_edges if len(codeql_edges) >= max(8, len(file_ids)) else edges
    counts: Counter = Counter()
    totals: Counter = Counter()
    for edge in use_edges:
        kind = edge.get("kind")
        if kind and kind != "call":
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if src not in symbols or dst not in symbols:
            continue
        src_path = symbols.get(src, {}).get("defined_in", {}).get("path")
        dst_path = symbols.get(dst, {}).get("defined_in", {}).get("path")
        if not src_path or not dst_path:
            continue
        src_fid = file_id(src_path)
        dst_fid = file_id(dst_path)
        if src_fid not in file_ids or dst_fid not in file_ids:
            continue
        if src_fid == dst_fid:
            continue
        key = tuple(sorted((src_fid, dst_fid)))
        counts[key] += 1
        totals[src_fid] += 1
        totals[dst_fid] += 1
    return counts, totals


def semantic_label(
    fid: str,
    vectors: Dict[str, Dict[str, float]],
    files: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    used: Set[str],
) -> str:
    vec = vectors.get(fid, {})
    top_terms = sorted(vec.items(), key=lambda item: item[1], reverse=True)
    keywords = [term for term, _score in top_terms[:2] if term]
    if keywords:
        base = "sem:" + "-".join(keywords)
    else:
        base = "sem:" + short_path(files.get(fid, {}).get("path", ""), prefix, max_segments=2)
    label = base
    counter = 1
    while label in used:
        counter += 1
        label = f"{base}-{counter}"
    used.add(label)
    return label


def semantic_component_index(
    files: Dict[str, Dict[str, Any]],
    symbols: Dict[str, Dict[str, Any]],
    *,
    prefix: str,
    max_components: int = 10,
    similarity_threshold: float = 0.12,
    call_counts: Optional[Counter] = None,
    call_totals: Optional[Counter] = None,
    call_weight: float = 0.3,
    mode: str = "tfidf",
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, object]]:
    meta: Dict[str, object] = {
        "fallback_reason": None,
        "call_weight_used": None,
        "call_edges": 0,
        "mode": mode,
    }
    docs: Dict[str, List[str]] = {}
    for fid, node in files.items():
        if node.get("role") in {"test", "config"}:
            continue
        if mode == "name_only":
            terms = semantic_terms_name_only(node)
        else:
            terms = semantic_terms_for_file(node, symbols)
        if terms:
            docs[fid] = terms
    if len(docs) < 3:
        meta["fallback_reason"] = "too_few_semantic_docs"
        return build_component_index(files, prefix), [], meta
    if len(docs) > 2000:
        meta["fallback_reason"] = "too_many_semantic_docs"
        return build_component_index(files, prefix), [], meta

    vectors, norms = build_tfidf_vectors(docs)
    if call_weight < 0:
        call_weight = 0.0
    if call_weight > 0.9:
        call_weight = 0.9
    meta["call_edges"] = int(sum(call_counts.values())) if call_counts else 0
    if not call_counts or not call_totals or meta["call_edges"] == 0:
        call_weight = 0.0
    meta["call_weight_used"] = call_weight
    ranked = sorted(
        docs.keys(),
        key=lambda fid: files.get(fid, {}).get("score", 0.0),
        reverse=True,
    )
    anchor_count = min(max_components, max(3, int(math.sqrt(len(ranked)))))
    anchors = ranked[:anchor_count]
    used_labels: Set[str] = set()
    anchor_labels: Dict[str, str] = {}
    for fid in anchors:
        anchor_labels[fid] = semantic_label(fid, vectors, files, prefix=prefix, used=used_labels)

    comp_files: Dict[str, List[str]] = defaultdict(list)
    for fid in docs:
        best_anchor = None
        best_score = 0.0
        for anchor in anchors:
            semantic_score = cosine_similarity(
                vectors.get(fid, {}),
                norms.get(fid, 0.0),
                vectors.get(anchor, {}),
                norms.get(anchor, 0.0),
            )
            call_score = 0.0
            if call_weight and call_counts and call_totals:
                key = tuple(sorted((fid, anchor)))
                shared = call_counts.get(key, 0)
                if shared:
                    denom_f = call_totals.get(fid, 0) or 0
                    denom_a = call_totals.get(anchor, 0) or 0
                    if denom_f:
                        call_score += shared / denom_f
                    if denom_a:
                        call_score += shared / denom_a
                    if denom_f and denom_a:
                        call_score *= 0.5
            score = (1.0 - call_weight) * semantic_score + call_weight * call_score
            if score > best_score:
                best_score = score
                best_anchor = anchor
        if best_anchor and best_score >= similarity_threshold:
            comp = anchor_labels[best_anchor]
        else:
            path_comp = component_for_path(files.get(fid, {}).get("path", ""), prefix)
            comp = f"path:{path_comp}"
        comp_files[comp].append(fid)

    component_order = [anchor_labels[fid] for fid in anchors if anchor_labels.get(fid) in comp_files]
    if comp_files:
        ordered_by_size = sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
        for comp, _fids in ordered_by_size:
            if comp not in component_order:
                component_order.append(comp)
    return comp_files, component_order, meta


def traceability_lines(
    links: List[Dict[str, object]],
    file_alias: Dict[str, str],
    entity_alias: Dict[str, str],
    *,
    prefix: str,
    max_items: int = 6,
) -> List[str]:
    if not links:
        return []
    lines: List[str] = ["[TRACEABILITY]"]
    for link in links[:max_items]:
        doc_path = str(link.get("doc_path") or "")
        target_id = str(link.get("target_id") or "")
        score = link.get("score")
        reason = str(link.get("reason") or "")
        doc_short = short_path(doc_path, prefix, max_segments=3)
        target_label = entity_alias.get(target_id) or file_alias.get(target_id) or target_id
        line = f"{doc_short} -> {target_label}"
        if score is not None:
            line += f" score={score}"
        if reason:
            line += f" reason={reason}"
        lines.append(line)
    return lines


def component_purpose_lines(
    comp_files: Dict[str, List[str]],
    files: Dict[str, Dict],
    symbols: Dict[str, Dict],
    *,
    prefix: str,
    max_components: int = 6,
    component_order: Optional[List[str]] = None,
) -> List[str]:
    if not comp_files:
        return []
    ordered: List[Tuple[str, List[str]]] = []
    if component_order:
        ordered = [(comp, comp_files[comp]) for comp in component_order if comp in comp_files]
        if len(ordered) < max_components:
            selected = {comp for comp, _fids in ordered}
            remaining = [
                item
                for item in sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
                if item[0] not in selected
            ]
            ordered.extend(remaining)
    else:
        ordered = sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
    ordered = ordered[:max_components]

    stopwords = {
        "get",
        "list",
        "create",
        "update",
        "delete",
        "set",
        "fetch",
        "find",
        "load",
        "save",
        "use",
        "build",
        "make",
        "init",
        "handle",
        "resolve",
        "compute",
        "read",
        "write",
        "add",
        "remove",
        "start",
        "end",
        "process",
        "register",
        "route",
        "routes",
        "controller",
        "handler",
        "service",
        "services",
        "utils",
        "helper",
        "helpers",
        "index",
        "main",
        "core",
        "lib",
        "api",
    }
    keyword_map = {
        "auth": "authentication",
        "login": "authentication",
        "session": "session management",
        "user": "user accounts",
        "account": "accounts",
        "billing": "billing",
        "payment": "payments",
        "invoice": "billing",
        "subscription": "subscriptions",
        "order": "orders",
        "product": "products/catalog",
        "catalog": "products/catalog",
        "inventory": "inventory",
        "report": "reporting",
        "analytics": "analytics",
        "metric": "analytics",
        "admin": "admin",
        "team": "teams/orgs",
        "org": "organizations",
        "tenant": "multi-tenant",
        "workspace": "workspaces",
        "notification": "notifications",
        "email": "notifications",
        "sms": "notifications",
        "webhook": "webhooks",
        "file": "file storage",
        "upload": "file uploads",
        "media": "media handling",
        "image": "media handling",
        "search": "search",
        "config": "configuration",
        "settings": "configuration",
    }
    role_phrases = {
        "routes": "HTTP/API routes",
        "edge": "request handlers",
        "services": "service logic",
        "domain": "domain models",
        "data": "data access",
        "worker": "background jobs",
        "ui": "UI components",
    }

    lines: List[str] = ["[PURPOSE]", f"count={len(comp_files)}"]
    for idx, (comp, fids) in enumerate(ordered, start=1):
        tokens: Counter[str] = Counter()
        roles: Counter[str] = Counter()
        for fid in fids:
            node = files.get(fid, {})
            path = node.get("path", "")
            roles[classify_path_role(path)] += 1
            for token in split_identifier(path):
                if token and token not in stopwords:
                    tokens[token] += 1
            for sid in node.get("exports", []):
                name = symbols.get(sid, {}).get("name")
                if not isinstance(name, str):
                    continue
                for token in split_identifier(name):
                    if token and token not in stopwords:
                        tokens[token] += 1
        summary = ""
        if tokens:
            for token, _count in tokens.most_common(4):
                if token in keyword_map:
                    summary = keyword_map[token]
                    break
            if not summary:
                top = tokens.most_common(1)[0][0]
                summary = top
        role = None
        for role_name, _count in roles.most_common():
            if role_name in role_phrases:
                role = role_phrases[role_name]
                break
        parts = []
        if summary:
            parts.append(summary)
        if role:
            parts.append(role)
        line = f"P{idx} {comp}"
        if parts:
            line += " :: " + " + ".join(parts)
        lines.append(line)
    return lines


def detect_architecture(
    files: Dict[str, Dict],
    *,
    config_paths: List[str],
    schema_paths: List[str],
    monorepo: Dict[str, object],
) -> Dict[str, object]:
    paths = [node.get("path", "") for node in files.values() if node.get("path")]
    lower_paths = [path.lower() for path in paths]
    config_names = {Path(path).name.lower() for path in config_paths}
    schema_names = {Path(path).name.lower() for path in schema_paths}

    def has_config(names: Iterable[str]) -> bool:
        return any(name in config_names for name in names)

    def has_path_token(token: str) -> bool:
        token = token.lower().strip("/")
        return any(f"/{token}/" in path or path.startswith(f"{token}/") for path in lower_paths)

    frameworks: Set[str] = set()
    if has_config({"next.config.js", "next.config.mjs", "next.config.ts", "next.config.cjs"}):
        frameworks.add("nextjs")
    if has_config({"nuxt.config.js", "nuxt.config.mjs", "nuxt.config.ts"}):
        frameworks.add("nuxt")
    if has_config({"remix.config.js", "remix.config.mjs", "remix.config.ts"}):
        frameworks.add("remix")
    if has_config({"svelte.config.js", "svelte.config.cjs", "svelte.config.mjs", "svelte.config.ts"}):
        frameworks.add("sveltekit")
    if has_config({"astro.config.js", "astro.config.mjs", "astro.config.ts"}):
        frameworks.add("astro")
    if has_config({"vite.config.js", "vite.config.mjs", "vite.config.ts"}):
        frameworks.add("vite")
    if has_config({"angular.json"}):
        frameworks.add("angular")
    if has_config({"gatsby-config.js", "gatsby-config.ts"}):
        frameworks.add("gatsby")
    if has_config({"nest-cli.json"}):
        frameworks.add("nestjs")
    if "manage.py" in lower_paths:
        frameworks.add("django")

    orms: Set[str] = set()
    if any(name.endswith(".prisma") for name in schema_names):
        orms.add("prisma")
    if has_config({"drizzle.config.ts", "drizzle.config.js", "drizzle.config.mjs", "drizzle.config.cjs"}):
        orms.add("drizzle")
    if has_config({"ormconfig.json", "ormconfig.js", "ormconfig.ts", "typeorm.config.js", "typeorm.config.ts"}):
        orms.add("typeorm")
    if has_config({"alembic.ini"}) or has_path_token("alembic"):
        orms.add("sqlalchemy")

    layout = ""
    layout_hints: List[str] = []
    has_domain = has_path_token("domain")
    has_usecases = has_path_token("usecase") or has_path_token("usecases")
    has_application = has_path_token("application") or has_path_token("app")
    has_infra = has_path_token("infra") or has_path_token("infrastructure")
    has_adapters = has_path_token("adapter") or has_path_token("adapters")
    has_controllers = has_path_token("controller") or has_path_token("controllers")
    has_services = has_path_token("service") or has_path_token("services")
    has_repos = has_path_token("repository") or has_path_token("repositories")
    has_modules = has_path_token("modules")

    if has_domain and (has_usecases or has_infra or has_adapters):
        layout = "clean-architecture"
        layout_hints = [hint for hint, flag in [
            ("domain", has_domain),
            ("usecases", has_usecases),
            ("infra", has_infra),
            ("adapters", has_adapters),
        ] if flag]
    elif has_controllers and has_services and has_repos:
        layout = "layered"
        layout_hints = ["controllers", "services", "repositories"]
    elif has_modules and (has_controllers or has_services):
        layout = "modular"
        layout_hints = ["modules"]
    elif has_application and (has_domain or has_infra):
        layout = "hexagonal"
        layout_hints = [hint for hint, flag in [
            ("application", has_application),
            ("domain", has_domain),
            ("infra", has_infra),
        ] if flag]

    mono_info: Dict[str, object] = {}
    if monorepo:
        packages = monorepo.get("packages", [])
        if isinstance(packages, list):
            pkg_count = len([pkg for pkg in packages if pkg.get("path") not in {".", ""}]) or len(packages)
        else:
            pkg_count = 0
        mono_info = {
            "type": monorepo.get("type", "workspace"),
            "packages": pkg_count,
            "markers": monorepo.get("markers", []),
        }

    return {
        "monorepo": mono_info,
        "frameworks": sorted(frameworks),
        "orms": sorted(orms),
        "layout": layout,
        "layout_hints": layout_hints,
    }


def architecture_overview_line(arch: Dict[str, object]) -> Optional[str]:
    if not arch:
        return None
    parts: List[str] = []
    monorepo = arch.get("monorepo", {})
    if isinstance(monorepo, dict) and monorepo:
        mono_type = monorepo.get("type", "workspace")
        pkg_count = monorepo.get("packages", 0)
        if pkg_count:
            parts.append(f"{mono_type} monorepo ({pkg_count} packages)")
        else:
            parts.append(f"{mono_type} monorepo")
    frameworks = arch.get("frameworks", [])
    if isinstance(frameworks, list) and frameworks:
        parts.append("frameworks " + ", ".join(frameworks[:4]))
    orms = arch.get("orms", [])
    if isinstance(orms, list) and orms:
        parts.append("orms " + ", ".join(orms[:4]))
    layout = arch.get("layout")
    if isinstance(layout, str) and layout:
        parts.append(f"{layout} layout")
    if not parts:
        return None
    return "overview=" + "; ".join(parts)


def architecture_section_lines(arch: Dict[str, object]) -> List[str]:
    if not arch:
        return []
    lines = ["[ARCHITECTURE]"]
    monorepo = arch.get("monorepo", {})
    if isinstance(monorepo, dict) and monorepo:
        mono_type = monorepo.get("type", "workspace")
        pkg_count = monorepo.get("packages", 0)
        line = f"monorepo={mono_type}"
        if pkg_count:
            line += f" packages={pkg_count}"
        markers = monorepo.get("markers", [])
        if markers:
            line += " markers=" + ",".join(str(m) for m in markers)
        lines.append(line)
    frameworks = arch.get("frameworks", [])
    if isinstance(frameworks, list) and frameworks:
        lines.append("frameworks=" + ",".join(frameworks))
    orms = arch.get("orms", [])
    if isinstance(orms, list) and orms:
        lines.append("orms=" + ",".join(orms))
    layout = arch.get("layout")
    if isinstance(layout, str) and layout:
        line = f"layout={layout}"
        hints = arch.get("layout_hints", [])
        if isinstance(hints, list) and hints:
            line += " hints=" + ",".join(str(hint) for hint in hints)
        lines.append(line)
    return lines if len(lines) > 1 else []


def pattern_section_lines(
    files: Dict[str, Dict],
    symbols: Optional[Dict[str, Dict]] = None,
    *,
    prefix: str,
) -> List[str]:
    if not files:
        return []
    layer_paths: Dict[str, Set[str]] = {
        "ui": set(),
        "business": set(),
        "data": set(),
        "infra": set(),
    }
    has_controller = False
    has_view = False
    has_model = False
    has_repository = False
    has_service = False
    repo_name_hits = 0
    service_name_hits = 0
    factory_name_hits = 0
    observer_name_hits = 0

    for node in files.values():
        path = node.get("path", "") or ""
        if not path:
            continue
        lower = path.lower()
        short = short_path(path, prefix, max_segments=3)
        if any(token in lower for token in ("/components/", "/pages/", "/views/", "/ui/", "/frontend/")):
            layer_paths["ui"].add(short)
            has_view = True
        if any(token in lower for token in ("/services/", "/service/", "/domain/", "/usecase/", "/usecases/")):
            layer_paths["business"].add(short)
            has_service = True
        if any(token in lower for token in ("/db/", "/database/", "/repo/", "/repository/", "/repositories/", "/models/", "/entities/", "/schema/")):
            layer_paths["data"].add(short)
            has_repository = True
            has_model = True
        if any(token in lower for token in ("/infra/", "/infrastructure/", "/adapters/", "/adapter/", "/gateway/", "/integrations/")):
            layer_paths["infra"].add(short)
        if any(token in lower for token in ("/controllers/", "/controller/")):
            has_controller = True
        if "repository" in lower:
            repo_name_hits += 1
        if "service" in lower:
            service_name_hits += 1
        if "factory" in lower:
            factory_name_hits += 1
        if any(token in lower for token in ("observer", "listener", "subscriber")):
            observer_name_hits += 1

    if symbols:
        for sym in symbols.values():
            name = str(sym.get("name") or "")
            if not name:
                continue
            lower = name.lower()
            if "repository" in lower:
                repo_name_hits += 1
            if "service" in lower:
                service_name_hits += 1
            if "factory" in lower:
                factory_name_hits += 1
            if any(token in lower for token in ("observer", "listener", "subscriber", "emitter")):
                observer_name_hits += 1

    lines = ["[PATTERNS]"]
    for layer in ("ui", "business", "data", "infra"):
        entries = sorted(layer_paths[layer])
        if entries:
            lines.append(f"layer_{layer}=" + ",".join(entries[:4]))

    def pattern_confidence(count: int, path_hint: bool) -> str:
        if count >= 2 and path_hint:
            return "high"
        if count >= 1 or path_hint:
            return "medium"
        return "low"

    if has_repository or repo_name_hits:
        lines.append(f"pattern_repository={pattern_confidence(repo_name_hits, has_repository)} n={repo_name_hits}")
    if has_service or service_name_hits:
        lines.append(f"pattern_service={pattern_confidence(service_name_hits, has_service)} n={service_name_hits}")
    if factory_name_hits:
        lines.append(f"pattern_factory={pattern_confidence(factory_name_hits, False)} n={factory_name_hits}")
    if observer_name_hits:
        lines.append(f"pattern_observer={pattern_confidence(observer_name_hits, False)} n={observer_name_hits}")

    if has_controller and has_view and has_model:
        lines.append("pattern_mvc=true")
    return lines if len(lines) > 1 else []


def layer_for_path(path: str) -> str:
    lower = path.lower()
    if any(token in lower for token in ("/components/", "/pages/", "/views/", "/ui/", "/frontend/")):
        return "ui"
    if any(token in lower for token in ("/services/", "/service/", "/domain/", "/usecase/", "/usecases/")):
        return "business"
    if any(token in lower for token in ("/db/", "/database/", "/repo/", "/repository/", "/repositories/", "/models/", "/entities/", "/schema/")):
        return "data"
    if any(token in lower for token in ("/infra/", "/infrastructure/", "/adapters/", "/adapter/", "/gateway/", "/integrations/")):
        return "infra"
    return ""


def layer_section_lines(
    files: Dict[str, Dict],
    edges: List[Dict],
    *,
    prefix: str,
) -> List[str]:
    if not files:
        return []
    layer_paths: Dict[str, Set[str]] = {"ui": set(), "business": set(), "data": set(), "infra": set()}
    layer_for_fid: Dict[str, str] = {}
    for fid, node in files.items():
        path = node.get("path", "") or ""
        if not path:
            continue
        layer = layer_for_path(path)
        if not layer:
            continue
        layer_for_fid[fid] = layer
        layer_paths[layer].add(short_path(path, prefix, max_segments=3))

    if not layer_for_fid:
        return []

    edge_counts: Counter[Tuple[str, str]] = Counter()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in layer_for_fid and dst in layer_for_fid:
            src_layer = layer_for_fid[src]
            dst_layer = layer_for_fid[dst]
            if src_layer and dst_layer and src_layer != dst_layer:
                edge_counts[(src_layer, dst_layer)] += 1

    lines = ["[LAYERS]"]
    for layer in ("ui", "business", "data", "infra"):
        entries = sorted(layer_paths[layer])
        if entries:
            lines.append(f"{layer}=" + ",".join(entries[:4]))

    def edge_count(src: str, dst: str) -> int:
        return edge_counts.get((src, dst), 0)

    graph: List[str] = []
    if edge_count("ui", "business") > 0:
        graph = ["ui", "business"]
        if edge_count("business", "data") > 0:
            graph.append("data")
            if edge_count("data", "infra") > 0:
                graph.append("infra")
    elif edge_count("business", "data") > 0:
        graph = ["business", "data"]
        if edge_count("data", "infra") > 0:
            graph.append("infra")
    if graph:
        lines.append("graph=" + "->".join(graph))
    return lines if len(lines) > 1 else []


def arch_style_lines(
    files: Dict[str, Dict],
    edges: List[Dict],
    *,
    prefix: str,
) -> List[str]:
    if not files:
        return []
    layer_for_fid: Dict[str, str] = {}
    for fid, node in files.items():
        path = node.get("path", "") or ""
        if not path:
            continue
        layer = layer_for_path(path)
        if layer:
            layer_for_fid[fid] = layer

    layer_order = {"ui": 0, "business": 1, "data": 2, "infra": 3}
    down_edges = 0
    up_edges = 0
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in layer_for_fid and dst in layer_for_fid:
            src_layer = layer_for_fid[src]
            dst_layer = layer_for_fid[dst]
            if src_layer == dst_layer:
                continue
            if layer_order.get(src_layer, 0) < layer_order.get(dst_layer, 0):
                down_edges += 1
            else:
                up_edges += 1

    paths = [node.get("path", "") or "" for node in files.values()]
    has_ports = any("/ports/" in path.lower() for path in paths)
    has_adapters = any("/adapters/" in path.lower() or "/adapter/" in path.lower() for path in paths)
    has_domain = any("/domain/" in path.lower() for path in paths)
    has_app = any("/application/" in path.lower() for path in paths)
    has_infra = any("/infrastructure/" in path.lower() for path in paths)
    has_interfaces = any("/interfaces/" in path.lower() for path in paths)

    lines = ["[ARCH_STYLE]"]
    if down_edges > 0:
        if up_edges == 0:
            lines.append("layered=high")
        elif down_edges >= up_edges * 2:
            lines.append("layered=medium")
        else:
            lines.append("layered=low")
    if (has_ports and has_adapters) or any("/hexagonal/" in path.lower() for path in paths):
        lines.append("hexagonal=probable")
    clean_score = sum([has_domain, has_app, has_infra, has_interfaces])
    if clean_score >= 3:
        lines.append("clean_architecture=probable")
    elif clean_score == 2:
        lines.append("clean_architecture=possible")
    return lines if len(lines) > 1 else []


def test_mapping_lines(
    test_mapping: List[Dict[str, object]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 12,
    summary: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    if not test_mapping:
        return []
    path_to_alias = {fid.replace("file:", ""): alias for fid, alias in file_alias.items() if fid.startswith("file:")}
    lines = ["[TEST_MAPPING]"]
    for entry in test_mapping[:max_items]:
        test_path = str(entry.get("test", ""))
        targets = entry.get("targets", [])
        symbols = entry.get("symbols", [])
        if not test_path or not isinstance(targets, list) or not targets:
            targets = []
        if not test_path:
            continue
        test_label = path_to_alias.get(test_path) or short_path(test_path, prefix, max_segments=3)
        target_labels = []
        for target in targets[:6]:
            label = path_to_alias.get(target) or short_path(target, prefix, max_segments=3)
            target_labels.append(label)
        line = f"{test_label} -> {','.join(target_labels)}" if target_labels else f"{test_label}"
        if isinstance(symbols, list) and symbols:
            line += " symbols=" + ",".join(str(sym) for sym in symbols[:4])
        lines.append(line)
    if summary:
        orphan_tests = summary.get("orphan_tests", [])
        if isinstance(orphan_tests, list) and orphan_tests:
            labels = [
                path_to_alias.get(path) or short_path(path, prefix, max_segments=3)
                for path in orphan_tests[:6]
            ]
            lines.append("orphan_tests=" + ",".join(labels))
        untested_entrypoints = summary.get("untested_entrypoints", [])
        if isinstance(untested_entrypoints, list) and untested_entrypoints:
            labels = [
                path_to_alias.get(path) or short_path(path, prefix, max_segments=3)
                for path in untested_entrypoints[:6]
            ]
            lines.append("untested_entrypoints=" + ",".join(labels))
    return lines if len(lines) > 1 else []


def provenance_label(
    path: str,
    line: int,
    *,
    prefix: str,
    file_alias: Dict[str, str],
) -> str:
    fid = file_id(path) if path else ""
    alias = file_alias.get(fid, "")
    short = short_path(path, prefix, max_segments=4) if path else "unknown"
    loc = f"{short}:{line}" if line > 0 else short
    return f"{alias}@{loc}" if alias else loc


def capability_insight_lines(
    symbols: Dict[str, Dict[str, object]],
    files: Dict[str, Dict[str, object]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 8,
) -> Tuple[List[str], List[str]]:
    hits_by_capability: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    path_seen: Set[Tuple[str, str]] = set()

    for sid, symbol in symbols.items():
        defined_in = symbol.get("defined_in", {}) if isinstance(symbol.get("defined_in"), dict) else {}
        path = str(defined_in.get("path") or "")
        line = int(defined_in.get("line") or 0)
        if not path:
            continue
        name = str(symbol.get("name") or "")
        signature = str(symbol.get("signature") or "")
        doc = str(symbol.get("doc_1l") or "")
        text = " ".join((path, name, signature, doc)).lower()
        for rule in CAPABILITY_RULES:
            cap_name = str(rule.get("name") or "")
            patterns = rule.get("patterns") or []
            score = 0
            for pattern in patterns:
                if isinstance(pattern, str) and pattern and re.search(pattern, text):
                    score += 1
            if score <= 0:
                continue
            hits_by_capability[cap_name].append(
                {
                    "score": score,
                    "path": path,
                    "line": line,
                    "source": name or sid,
                }
            )

    for node in files.values():
        path = str(node.get("path") or "")
        if not path:
            continue
        lower = path.lower()
        for rule in CAPABILITY_RULES:
            cap_name = str(rule.get("name") or "")
            path_tokens = rule.get("path_tokens") or []
            matched = False
            for token in path_tokens:
                if isinstance(token, str) and token and token in lower:
                    matched = True
                    break
            if not matched:
                continue
            dedupe_key = (cap_name, path)
            if dedupe_key in path_seen:
                continue
            path_seen.add(dedupe_key)
            hits_by_capability[cap_name].append(
                {
                    "score": 1,
                    "path": path,
                    "line": 0,
                    "source": "path_hint",
                }
            )

    statuses: Dict[str, Dict[str, object]] = {}
    for rule in CAPABILITY_RULES:
        cap_name = str(rule.get("name") or "")
        hits = hits_by_capability.get(cap_name, [])
        if not hits:
            statuses[cap_name] = {"status": "missing", "confidence": "medium", "hits": []}
            continue
        ordered = sorted(
            hits,
            key=lambda item: (
                -int(item.get("score") or 0),
                str(item.get("path") or ""),
                int(item.get("line") or 0),
            ),
        )
        top_score = int(ordered[0].get("score") or 0)
        if top_score >= 2:
            status = "exists"
            confidence = "high"
        elif len(ordered) >= 2:
            status = "exists"
            confidence = "medium"
        else:
            status = "uncertain"
            confidence = "low"
        statuses[cap_name] = {"status": status, "confidence": confidence, "hits": ordered}

    existing_or_uncertain = [
        (cap_name, data)
        for cap_name, data in statuses.items()
        if str(data.get("status")) in {"exists", "uncertain"}
    ]
    if not existing_or_uncertain:
        return [], []

    capability_lines: List[str] = ["[CAPABILITIES]"]
    for idx, (cap_name, data) in enumerate(existing_or_uncertain[:max_items], start=1):
        hits = data.get("hits", [])
        if not isinstance(hits, list):
            continue
        evidence_parts: List[str] = []
        seen_evidence: Set[str] = set()
        for hit in hits[:3]:
            path = str(hit.get("path") or "")
            line = int(hit.get("line") or 0)
            source = str(hit.get("source") or "")
            evidence = provenance_label(path, line, prefix=prefix, file_alias=file_alias)
            if source and source != "path_hint":
                evidence += f"({source})"
            if evidence in seen_evidence:
                continue
            seen_evidence.add(evidence)
            evidence_parts.append(evidence)
        confidence = str(data.get("confidence") or "medium")
        status = str(data.get("status") or "uncertain")
        evidence_text = ",".join(evidence_parts) if evidence_parts else "none"
        capability_lines.append(
            f"C{idx} {cap_name} status={status} conf={confidence} evidence={evidence_text}"
        )

    missing = [cap_name for cap_name, data in statuses.items() if data.get("status") == "missing"]
    uncertain = [cap_name for cap_name, data in statuses.items() if data.get("status") == "uncertain"]
    gap_lines: List[str] = ["[CAPABILITY_GAPS]"]
    gap_idx = 1
    for cap_name in missing[:max_items]:
        gap_lines.append(f"G{gap_idx} {cap_name} status=missing")
        gap_idx += 1
    for cap_name in uncertain[: max(0, max_items - len(missing))]:
        gap_lines.append(f"G{gap_idx} {cap_name} status=uncertain")
        gap_idx += 1
    if len(gap_lines) == 1:
        gap_lines.append("none_detected")
    return capability_lines, gap_lines


def test_confidence_lines(
    test_mapping: List[Dict[str, object]],
    test_summary: Dict[str, List[str]],
    files: Dict[str, Dict[str, object]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 8,
) -> List[str]:
    if not test_mapping and not test_summary:
        return []
    path_to_alias = {fid.replace("file:", ""): alias for fid, alias in file_alias.items() if fid.startswith("file:")}
    role_by_path = {
        str(node.get("path") or ""): str(node.get("role") or "")
        for node in files.values()
        if node.get("path")
    }
    entrypoint_paths = {path for path, role in role_by_path.items() if role == "entrypoint"}
    tests_by_path: Dict[str, Dict[str, object]] = {}
    for entry in test_mapping:
        test_path = str(entry.get("test") or "")
        if not test_path:
            continue
        targets = entry.get("targets", [])
        symbols = entry.get("symbols", [])
        existing = tests_by_path.setdefault(test_path, {"targets": set(), "symbols": set()})
        if isinstance(targets, list):
            existing["targets"].update(str(target) for target in targets if target)
        if isinstance(symbols, list):
            existing["symbols"].update(str(symbol) for symbol in symbols if symbol)

    strong_tests: List[str] = []
    weak_tests: List[str] = []
    theatrical_tests: List[str] = []
    for test_path, data in tests_by_path.items():
        targets = set(data.get("targets", set()))
        symbols = set(data.get("symbols", set()))
        if not targets:
            theatrical_tests.append(test_path)
            continue
        has_entrypoint_target = any(target in entrypoint_paths for target in targets)
        if symbols or has_entrypoint_target:
            strong_tests.append(test_path)
        else:
            weak_tests.append(test_path)

    orphan_tests = test_summary.get("orphan_tests", []) if isinstance(test_summary, dict) else []
    if isinstance(orphan_tests, list):
        for path in orphan_tests:
            if path and path not in theatrical_tests:
                theatrical_tests.append(path)
    untested_entrypoints = test_summary.get("untested_entrypoints", []) if isinstance(test_summary, dict) else []
    if not strong_tests and not weak_tests and not theatrical_tests and not untested_entrypoints:
        return []

    total_tests = len(strong_tests) + len(weak_tests) + len(theatrical_tests)
    lines = ["[TEST_CONFIDENCE]"]
    lines.append(
        f"summary=strong:{len(strong_tests)} weak:{len(weak_tests)} theatrical:{len(theatrical_tests)} total:{total_tests}"
    )

    if weak_tests:
        labels = [path_to_alias.get(path) or short_path(path, prefix, max_segments=4) for path in weak_tests[:max_items]]
        lines.append("weak_tests=" + ",".join(labels))
    if theatrical_tests:
        labels = [
            path_to_alias.get(path) or short_path(path, prefix, max_segments=4)
            for path in theatrical_tests[:max_items]
        ]
        lines.append("theatrical_tests=" + ",".join(labels))
    if isinstance(untested_entrypoints, list) and untested_entrypoints:
        labels = [
            path_to_alias.get(path) or short_path(path, prefix, max_segments=4)
            for path in untested_entrypoints[:max_items]
        ]
        lines.append("untested_entrypoints=" + ",".join(labels))
    return lines if len(lines) > 1 else []


def invariants_hc_lines(
    entry_details: List[str],
    *,
    max_items: int = 8,
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not entry_details:
        return [], []
    keyword_map = {
        "auth_required": ("auth", "login", "session", "token"),
        "authorization_required": ("permission", "access", "role", "admin", "owner"),
        "tenant_scope_required": ("tenant", "org", "workspace"),
        "rate_limit_guard": ("rate", "throttle"),
        "feature_flag_guard": ("feature", "flag"),
        "billing_guard": ("billing", "payment", "subscription"),
        "csrf_guard": ("csrf",),
    }
    facts: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for line in entry_details:
        pre_match = re.search(r"\bpre=([^\s]+)", line)
        if not pre_match:
            continue
        loc_match = re.search(r"([A-Za-z0-9_./-]+\.(?:ts|tsx|js|jsx|mjs|cjs|py):\d+)", line)
        loc = loc_match.group(1) if loc_match else ""
        raw_pre = pre_match.group(1)
        for precondition in raw_pre.split(","):
            pre = precondition.strip()
            if not pre:
                continue
            lower = pre.lower()
            for invariant_name, keywords in keyword_map.items():
                if not any(keyword in lower for keyword in keywords):
                    continue
                key = (invariant_name, loc, pre)
                if key in seen:
                    continue
                seen.add(key)
                facts.append(
                    {
                        "invariant": invariant_name,
                        "loc": loc,
                        "source": pre,
                    }
                )
                break

    if not facts:
        return ["[INVARIANTS_HC]", "none_detected"], []

    lines = ["[INVARIANTS_HC]"]
    for idx, fact in enumerate(facts[:max_items], start=1):
        invariant_name = fact.get("invariant", "guard")
        loc = fact.get("loc", "")
        source = fact.get("source", "")
        loc_part = f" evidence={loc}" if loc else ""
        source_part = f" source={source}" if source else ""
        lines.append(f"IHC{idx} {invariant_name} conf=high{loc_part}{source_part}")
    return lines, facts


def change_risk_lines(
    files: Dict[str, Dict[str, object]],
    quality: Dict[str, object],
    test_summary: Dict[str, List[str]],
    invariants_hc: List[Dict[str, str]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 8,
) -> List[str]:
    risk_scores: Dict[str, float] = defaultdict(float)
    risk_reasons: Dict[str, List[str]] = defaultdict(list)
    path_to_fid = {
        str(node.get("path") or ""): fid
        for fid, node in files.items()
        if node.get("path")
    }

    def resolve_fid(path_hint: str) -> Optional[str]:
        if not path_hint:
            return None
        if path_hint in path_to_fid:
            return path_to_fid[path_hint]
        for path, fid in path_to_fid.items():
            if path.endswith(path_hint):
                return fid
        return None

    coupling = quality.get("coupling", {}) if isinstance(quality, dict) else {}
    fan_in = coupling.get("fan_in", []) if isinstance(coupling, dict) else []
    fan_out = coupling.get("fan_out", []) if isinstance(coupling, dict) else []
    instability = coupling.get("instability", []) if isinstance(coupling, dict) else []

    if isinstance(fan_in, list):
        for item in fan_in:
            if not isinstance(item, dict):
                continue
            fid = item.get("id")
            count = int(item.get("count") or 0)
            if not isinstance(fid, str) or fid not in files or count <= 0:
                continue
            risk_scores[fid] += min(6.0, 1.0 + count * 0.5)
            risk_reasons[fid].append(f"fan_in:{count}")
    if isinstance(fan_out, list):
        for item in fan_out:
            if not isinstance(item, dict):
                continue
            fid = item.get("id")
            count = int(item.get("count") or 0)
            if not isinstance(fid, str) or fid not in files or count <= 0:
                continue
            risk_scores[fid] += min(4.0, 0.5 + count * 0.25)
            risk_reasons[fid].append(f"fan_out:{count}")
    if isinstance(instability, list):
        for item in instability:
            if not isinstance(item, dict):
                continue
            fid = item.get("id")
            value = float(item.get("value") or 0.0)
            if not isinstance(fid, str) or fid not in files or value < 0.7:
                continue
            risk_scores[fid] += min(2.5, value * 2.5)
            risk_reasons[fid].append(f"instability:{value:.2f}")

    untested = test_summary.get("untested_entrypoints", []) if isinstance(test_summary, dict) else []
    if isinstance(untested, list):
        for path in untested:
            if not isinstance(path, str):
                continue
            fid = resolve_fid(path)
            if not fid:
                continue
            risk_scores[fid] += 3.0
            risk_reasons[fid].append("untested_entrypoint")

    for fact in invariants_hc:
        loc = str(fact.get("loc") or "")
        path_hint = loc.rsplit(":", 1)[0] if ":" in loc else loc
        fid = resolve_fid(path_hint)
        if not fid:
            continue
        invariant_name = str(fact.get("invariant") or "invariant")
        risk_scores[fid] += 2.0
        risk_reasons[fid].append(f"invariant:{invariant_name}")

    ranked = sorted(risk_scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return []
    lines = ["[CHANGE_RISK]"]
    for idx, (fid, score) in enumerate(ranked[:max_items], start=1):
        path = str(files.get(fid, {}).get("path") or "")
        loc = provenance_label(path, 0, prefix=prefix, file_alias=file_alias)
        dedup_reasons = list(dict.fromkeys(risk_reasons.get(fid, [])))
        reasons = ",".join(dedup_reasons[:4])
        lines.append(f"R{idx} {loc} score={score:.1f} reasons={reasons}")
    return lines if len(lines) > 1 else []


def dead_code_lines(
    symbols: Dict[str, Dict[str, object]],
    files: Dict[str, Dict[str, object]],
    edges: List[Dict[str, object]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 8,
) -> List[str]:
    if not symbols or not files:
        return []
    inbound: Counter[str] = Counter()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in files and dst in files:
            inbound[dst] += 1
    candidates: List[Tuple[float, Dict[str, object]]] = []
    for sid, sym in symbols.items():
        if sym.get("visibility") != "public":
            continue
        if sym.get("refs_in_count", 0) != 0:
            continue
        defined_in = sym.get("defined_in", {}) if isinstance(sym.get("defined_in"), dict) else {}
        path = str(defined_in.get("path") or "")
        if not path:
            continue
        fid = file_id(path)
        role = files.get(fid, {}).get("role")
        if role in {"test", "config"} or role == "entrypoint":
            continue
        score = float(sym.get("score", 0.0))
        candidates.append((score, {"name": sym.get("name", ""), "path": path}))
    candidates.sort(key=lambda item: item[0], reverse=True)

    reexport_candidates = [
        fid
        for fid, node in files.items()
        if node.get("re_exports") and inbound.get(fid, 0) == 0
    ]

    lines = ["[DEAD_CODE]"]
    for _score, sym in candidates[:max_items]:
        path = sym.get("path", "")
        alias = file_alias.get(file_id(path)) or short_path(path, prefix, max_segments=3)
        name = sym.get("name", "")
        if name:
            lines.append(f"unreferenced_export={alias}:{name}")
    for fid in reexport_candidates[:max_items]:
        alias = file_alias.get(fid) or short_path(files.get(fid, {}).get("path", ""), prefix, max_segments=3)
        lines.append(f"unused_reexport_file={alias}")
    return lines if len(lines) > 1 else []


def api_contract_lines(
    api_contracts: List[Dict[str, object]],
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 12,
) -> List[str]:
    from .sections.routes import format_loc

    if not api_contracts:
        return []
    path_to_alias: Dict[str, str] = {}
    for fid, alias in file_alias.items():
        path = fid.replace("file:", "")
        if path:
            path_to_alias[path] = alias
    lines = ["[API_CONTRACTS]"]
    for entry in api_contracts[:max_items]:
        method = str(entry.get("method", "")).lower()
        route = str(entry.get("route", ""))
        path = str(entry.get("path", ""))
        line = int(entry.get("line", 0) or 0)
        if not route:
            continue
        loc = path_to_alias.get(path) or format_loc(path, line, prefix)
        label = f"{method}:{route}" if method else route
        notes: List[str] = []
        response = entry.get("response")
        auth = entry.get("auth")
        status = entry.get("status")
        handler = entry.get("handler")
        params = entry.get("params")
        if response:
            notes.append(f"resp={response}")
        if auth:
            notes.append(f"auth={auth}")
        if status:
            notes.append(f"status={status}")
        if handler:
            notes.append(f"handler={handler}")
        if isinstance(params, list) and params:
            notes.append("params=" + ",".join(str(param) for param in params[:3]))
        note_text = f" {' '.join(notes)}" if notes else ""
        lines.append(f"{label} :: {loc}{note_text}")
    return lines if len(lines) > 1 else []


def type_hierarchy_lines(
    ir: Dict,
    *,
    prefix: str,
    file_alias: Dict[str, str],
    max_items: int = 12,
) -> List[str]:
    edges = ir.get("edges", {}).get("type_ref", [])
    symbols = ir.get("symbols", {})
    if not edges:
        return []
    lines = ["[TYPE_HIERARCHY]"]
    for edge in edges[:max_items]:
        from_id = edge.get("from")
        to_id = edge.get("to")
        from_name = edge.get("from_name") or ""
        if not from_name and isinstance(from_id, str):
            from_name = symbols.get(from_id, {}).get("name") or from_id
        to_name = edge.get("to_name") or ""
        if not to_name and isinstance(to_id, str):
            to_name = symbols.get(to_id, {}).get("name") or to_id
        if not from_name or not to_name:
            continue
        kind = edge.get("kind", "")
        path = edge.get("from_path", "")
        loc = file_alias.get(file_id(path)) or short_path(path, prefix, max_segments=3)
        note = " ambiguous" if edge.get("ambiguous") else ""
        lines.append(f"{from_name} -{kind}-> {to_name}{note} :: {loc}")
    return lines if len(lines) > 1 else []


def cycle_section_lines(
    cycles: List[List[str]],
    comp_counts: Dict[str, int],
    comp_in: Counter,
    comp_out: Counter,
) -> List[str]:
    if not cycles:
        return []
    total_files = sum(comp_counts.values()) or 1
    lines = ["[CYCLES]"]
    for idx, cycle in enumerate(cycles, start=1):
        size = sum(comp_counts.get(comp, 0) for comp in cycle)
        ratio = size / total_files
        severity = "low"
        if ratio >= 0.3 or len(cycle) >= 4:
            severity = "high"
        elif ratio >= 0.15 or len(cycle) >= 3:
            severity = "medium"
        break_hint = max(cycle, key=lambda comp: comp_out.get(comp, 0) + comp_in.get(comp, 0))
        lines.append(f"C{idx} severity={severity} components=" + ">".join(cycle) + f" break_hint={break_hint}")
    return lines


def component_edges_lines(
    comp_edges: Counter,
    aliases: Dict[str, str],
    *,
    max_edges: int = 8,
) -> List[str]:
    if not comp_edges or not aliases:
        return []
    ordered = sorted(comp_edges.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    lines = ["[COMPONENT_EDGES]", f"count={len(comp_edges)}"]
    listed = 0
    for (src, dst), weight in ordered:
        if src not in aliases or dst not in aliases:
            continue
        lines.append(f"{aliases[src]}->{aliases[dst]}:{weight}")
        listed += 1
        if listed >= max_edges:
            break
    return lines

def classify_path_role(path: str) -> str:
    if not path:
        return "other"
    parts = [part for part in path.lower().split("/") if part]
    if "routes" in parts:
        return "routes"
    if any(part in {"controllers", "handlers", "resolvers"} for part in parts):
        return "edge"
    if any(part in {"services", "use-cases", "usecases"} for part in parts):
        return "services"
    if any(part in {"domain", "entities", "models"} for part in parts):
        return "domain"
    if any(part in {"db", "repository", "repositories", "migrations", "schema"} for part in parts):
        return "data"
    if any(part in {"worker", "workers", "jobs", "queues"} for part in parts):
        return "worker"
    if any(part in {"components", "ui", "views"} for part in parts):
        return "ui"
    if any(part in {"lib", "utils", "shared"} for part in parts):
        return "lib"
    return "other"


def role_summary_line(files: Dict[str, Dict], max_roles: int = 6) -> str:
    counts: Counter[str] = Counter()
    for node in files.values():
        if node.get("role") in {"test", "config"}:
            continue
        path = node.get("path", "")
        if not path:
            continue
        counts[classify_path_role(path)] += 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_roles]
    return "roles=" + ",".join(f"{role}({count})" for role, count in ordered)


def role_edges_line(files: Dict[str, Dict], edges: List[Dict], max_items: int = 6) -> str:
    counts: Counter[Tuple[str, str]] = Counter()
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in files or dst not in files:
            continue
        if files[src].get("role") in {"test", "config"}:
            continue
        if files[dst].get("role") in {"test", "config"}:
            continue
        src_role = classify_path_role(files[src].get("path", ""))
        dst_role = classify_path_role(files[dst].get("path", ""))
        counts[(src_role, dst_role)] += 1
    if not counts:
        return ""
    non_other = [
        (roles, count)
        for roles, count in counts.items()
        if roles != ("other", "other")
    ]
    ordered = non_other or list(counts.items())
    ordered.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    ordered = ordered[:max_items]
    return "role_edges=" + ",".join(f"{src}->{dst}({count})" for (src, dst), count in ordered)


def selection_penalty(node: Dict, path: str) -> float:
    role = node.get("role")
    if role == "test":
        return 3.0
    if role == "config":
        return 1.0
    lower = path.lower()
    parts = [part for part in lower.split("/") if part]
    test_markers = {"test", "tests", "__tests__", "spec", "specs", "__mocks__", "fixtures"}
    build_markers = {
        "dist",
        "build",
        "coverage",
        "out",
        "generated",
        "gen",
        "vendor",
        "node_modules",
        ".next",
        ".nuxt",
        "target",
        "__pycache__",
    }
    if any(part in test_markers for part in parts):
        return 2.0
    if any(part in build_markers for part in parts):
        return 2.5
    name = Path(lower).name
    if name.startswith(("test_", "tests_", "spec_", "specs_")):
        return 1.5
    if name.endswith((".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx", ".test.js", ".spec.js", "_test.py")):
        return 1.5
    return 0.0


def module_tokens(module: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", module.lower()) if token]


def categorize_external(module: str) -> str:
    tokens = module_tokens(module)
    if any(token in tokens for token in ("react", "vue", "angular", "svelte", "next", "nextjs")):
        return "frontend"
    if any(token in tokens for token in ("express", "fastify", "koa", "hapi", "flask", "django", "fastapi")):
        return "web"
    if any(token in tokens for token in ("prisma", "sequelize", "typeorm", "sqlalchemy", "knex", "orm")):
        return "orm"
    if any(token in tokens for token in ("postgres", "mysql", "sqlite", "mongodb", "redis", "asyncpg", "psycopg")):
        return "db"
    if any(token in tokens for token in ("pytest", "jest", "vitest", "mocha", "unittest", "playwright", "cypress")):
        return "test"
    if any(token in tokens for token in ("winston", "pino", "loguru", "structlog", "logging")):
        return "log"
    if any(token in tokens for token in ("zod", "joi", "yup", "pydantic", "validator")):
        return "validation"
    if any(token in tokens for token in ("webpack", "rollup", "vite", "esbuild", "babel", "ts", "ts-node", "typescript")):
        return "build"
    if any(token in tokens for token in ("axios", "requests", "httpx", "fetch")):
        return "http"
    return "other"


def external_summary_line(ext_counts: Dict[str, int], max_groups: int = 5, max_items: int = 2) -> str:
    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for module, count in ext_counts.items():
        grouped[categorize_external(module)].append((module, count))
    ordered_groups = sorted(grouped.items(), key=lambda item: -sum(c for _, c in item[1]))
    parts: List[str] = []
    for group, items in ordered_groups:
        if group == "other":
            continue
        items.sort(key=lambda item: (-item[1], item[0]))
        mods = ",".join(name for name, _ in items[:max_items])
        if mods:
            parts.append(f"{group}:{mods}")
        if len(parts) >= max_groups:
            break
    if not parts:
        return ""
    return "external=" + ";".join(parts)


def is_config_path(path: str) -> bool:
    lower = path.lower()
    name = Path(lower).name
    if name in {
        "package.json",
        "tsconfig.json",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "dockerfile",
        "docker-compose.yml",
        "makefile",
        "pnpm-workspace.yaml",
        "lerna.json",
        "turbo.json",
    }:
        return True
    if name.startswith(".env"):
        return True
    if ".config." in name:
        return True
    if name.endswith((".toml", ".yaml", ".yml", ".ini", ".cfg", ".conf")):
        return True
    return False


def is_schema_path(path: str) -> bool:
    lower = path.lower()
    name = Path(lower).name
    if name.endswith((".prisma", ".graphql", ".gql", ".sql")):
        return True
    if "/schema" in lower or "/schemas/" in lower:
        return True
    if "/migrations/" in lower:
        return True
    return False


def collect_configs_and_schemas(files: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    configs: List[str] = []
    schemas: List[str] = []
    for node in files.values():
        path = node.get("path", "")
        if not path:
            continue
        if is_config_path(path):
            configs.append(path)
        if is_schema_path(path):
            schemas.append(path)
    return sorted(set(configs)), sorted(set(schemas))

def build_dir_skeleton(
    paths: Iterable[str],
    *,
    depth: int,
    max_files_per_dir: int = 12,
) -> List[str]:
    if depth <= 0:
        return []
    tree: Dict[str, Dict] = {}
    for path in paths:
        parts = [part for part in path.split("/") if part]
        if not parts:
            continue
        node = tree
        for idx, part in enumerate(parts):
            if idx == len(parts) - 1:
                node.setdefault("__files__", []).append(part)
            else:
                node = node.setdefault(part, {})

    lines: List[str] = []

    def walk(node: Dict, prefix: str, level: int) -> None:
        if level > depth:
            return
        dirs = sorted(key for key in node.keys() if key != "__files__")
        files = sorted(node.get("__files__", []))
        for name in dirs:
            lines.append(f"{prefix}{name}/")
            walk(node[name], prefix + "  ", level + 1)
        if files:
            for filename in files[:max_files_per_dir]:
                lines.append(f"{prefix}- {filename}")
            if len(files) > max_files_per_dir:
                lines.append(f"{prefix}- ... ({len(files) - max_files_per_dir} more files)")

    walk(tree, "", 1)
    return lines


def edge_counts(edges: List[Dict], selected: Set[str]) -> Tuple[Dict[str, int], Dict[str, int], int]:
    inbound: Dict[str, int] = defaultdict(int)
    outbound: Dict[str, int] = defaultdict(int)
    total = 0
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in selected and dst in selected:
            outbound[src] += 1
            inbound[dst] += 1
            total += 1
    return inbound, outbound, total


def file_call_counts(ir: Dict) -> Tuple[Dict[str, int], Dict[str, int]]:
    inbound: Dict[str, int] = defaultdict(int)
    outbound: Dict[str, int] = defaultdict(int)
    symbols = ir.get("symbols", {})
    for edge in ir.get("edges", {}).get("symbol_ref", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src in symbols:
            src_path = symbols[src].get("defined_in", {}).get("path")
            if src_path:
                outbound[file_id(src_path)] += 1
        if dst in symbols:
            dst_path = symbols[dst].get("defined_in", {}).get("path")
            if dst_path:
                inbound[file_id(dst_path)] += 1
    return inbound, outbound


def package_stats(
    files: Dict[str, Dict],
    symbols: Dict[str, Dict],
    packages: List[Dict[str, object]],
) -> List[Tuple[str, int, int]]:
    package_roots = [pkg.get("path", "") for pkg in packages if pkg.get("path") not in {".", ""}]
    if not package_roots:
        package_roots = [pkg.get("path", "") for pkg in packages if pkg.get("path")]
    package_roots = [root for root in package_roots if root]
    if not package_roots:
        return []
    roots_sorted = sorted(package_roots, key=len, reverse=True)
    stats: Dict[str, Dict[str, int]] = {root: {"files": 0, "symbols": 0} for root in roots_sorted}

    def root_for_path(path: str) -> Optional[str]:
        for root in roots_sorted:
            if path == root or path.startswith(root + "/"):
                return root
        return None

    for node in files.values():
        path = node.get("path", "")
        root = root_for_path(path)
        if root:
            stats[root]["files"] += 1
    for sym in symbols.values():
        path = sym.get("defined_in", {}).get("path", "")
        root = root_for_path(path)
        if root:
            stats[root]["symbols"] += 1

    ranked = sorted(
        [(root, data["files"], data["symbols"]) for root, data in stats.items()],
        key=lambda item: (-item[1], -item[2], item[0]),
    )
    return ranked


_TARGET_STACK_ALIAS = {
    "py": {"py", "python"},
    "ts": {"ts", "tsx", "js", "jsx", "typescript", "javascript"},
    "rs": {"rs", "rust"},
}


def _normalize_target_stack(target_stack: Optional[Iterable[str]]) -> List[str]:
    raw = list(target_stack) if target_stack else ["py", "ts", "rs"]
    normalized: List[str] = []
    seen: Set[str] = set()
    for item in raw:
        token = str(item).strip().lower()
        canonical = token
        for candidate, aliases in _TARGET_STACK_ALIAS.items():
            if token == candidate or token in aliases:
                canonical = candidate
                break
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


def evaluate_confidence_gate(
    *,
    quality: Dict[str, object],
    truncated: bool,
    unsupported_langs: Dict[str, object],
    codeql_meta: Dict[str, object],
    target_stack: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    targets = _normalize_target_stack(target_stack)
    ast_conf = str(quality.get("ast_edge_confidence") or "n/a").strip().lower()
    codeql_mode = str(quality.get("codeql_mode") or "").strip().lower()
    codeql_ran = bool(quality.get("codeql_ran"))
    codeql_status = str(codeql_meta.get("status") or "").strip().lower()
    if not codeql_status:
        codeql_status = "ok" if codeql_ran else "disabled"

    unsupported_keys = {
        str(name).strip().lower()
        for name, value in unsupported_langs.items()
        if value
    }
    target_aliases: Set[str] = set()
    for target in targets:
        target_aliases |= _TARGET_STACK_ALIAS.get(target, {target})
    unsupported_hits = sorted(unsupported_keys & target_aliases)

    codeql_on_ok = codeql_mode == "on" and codeql_status == "ok"
    rule1 = ast_conf in {"high", "medium"} or (ast_conf == "low" and codeql_on_ok)
    rule2 = not truncated
    rule3 = not unsupported_hits
    codeql_required = ast_conf in {"medium", "low"}
    rule4 = (not codeql_required) or codeql_on_ok

    checks = {
        "ast_confidence_not_low": rule1,
        "not_truncated": rule2,
        "target_stack_supported": rule3,
        "codeql_rerun_if_needed": rule4,
    }
    status = "pass" if all(checks.values()) else "fail"

    actions: List[str] = []
    if not rule2:
        actions.append(
            "increase --budget and/or --max-files/--max-edges, or narrow --focus until [LIMITS] truncated=no"
        )
    if not rule3:
        actions.append("ensure unsupported_langs excludes target stack")
    if codeql_required and not rule4:
        actions.append("rerun with --codeql on")
    if not rule1 and not actions:
        actions.append("improve AST edge confidence before relying on this digest")

    return {
        "status": status,
        "target_stack": targets,
        "ast_edge_confidence": ast_conf,
        "codeql_required": codeql_required,
        "codeql_mode": codeql_mode or "off",
        "codeql_status": codeql_status,
        "checks": checks,
        "unsupported_hits": unsupported_hits,
        "actions": actions,
    }


def confidence_gate_lines(gate: Dict[str, object]) -> List[str]:
    checks = gate.get("checks", {}) if isinstance(gate.get("checks"), dict) else {}
    unsupported_hits = gate.get("unsupported_hits", [])
    actions = gate.get("actions", [])
    target_stack = gate.get("target_stack", [])

    def check_line(name: str) -> str:
        return "pass" if checks.get(name) else "fail"

    return [
        "[CONFIDENCE_GATE]",
        f"status={gate.get('status', 'fail')}",
        f"target_stack={','.join(str(item) for item in target_stack) or 'py,ts,rs'}",
        f"ast_edge_confidence={gate.get('ast_edge_confidence', 'n/a')}",
        f"rule1_ast_confidence_not_low={check_line('ast_confidence_not_low')}",
        f"rule2_not_truncated={check_line('not_truncated')}",
        f"rule3_target_stack_supported={check_line('target_stack_supported')}",
        f"rule4_codeql_rerun_if_needed={check_line('codeql_rerun_if_needed')}",
        f"codeql_required={'yes' if gate.get('codeql_required') else 'no'}",
        f"codeql_mode={gate.get('codeql_mode', 'off')}",
        f"codeql_status={gate.get('codeql_status', 'disabled')}",
        (
            "unsupported_hits=none"
            if not isinstance(unsupported_hits, list) or not unsupported_hits
            else "unsupported_hits=" + ",".join(str(item) for item in unsupported_hits)
        ),
        (
            "action=none"
            if not isinstance(actions, list) or not actions
            else "action=" + "; ".join(str(item) for item in actions)
        ),
    ]


def digest(
    ir: Dict,
    budget_tokens: int,
    *,
    focus: Optional[str] = None,
    focus_depth: int = 2,
    flow_symbols: bool = False,
    precise_tokens: bool = False,
    semantic_cluster_mode: str = "tfidf",
    semantic_cluster: bool = False,
    doc_quality: bool = False,
    doc_quality_strict: bool = False,
    semantic_call_weight: float = 0.3,
    dead_code: bool = False,
    call_chains: bool = False,
    static_traces: bool = False,
    trace_depth: int = 4,
    trace_max: int = 10,
    trace_direction: str = "both",
    trace_start: Optional[str] = None,
    trace_end: Optional[str] = None,
    graph_cache: Optional[Dict[str, object]] = None,
    graph_cache_hit: bool = False,
    graph_cache_signature: Optional[str] = None,
    section_budgets: Optional[Dict[str, float]] = None,
    entity_graph: bool = True,
    traceability: bool = False,
    traceability_targets: str = "all",
    max_files: Optional[int] = None,
    max_symbols_per_file: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_sig_len: Optional[int] = None,
    entry_details_limit: Optional[int] = None,
    entry_details_per_kind: Optional[int] = None,
    code_only: bool = True,
    compress_paths: bool = True,
    include_routes: bool = False,
    routes_mode: str = "auto",
    routes_limit: int = 8,
    routes_format: str = "compact",
    explain_scores: bool = False,
    dense: bool = False,
    dir_depth: Optional[int] = None,
    dir_file_cap: int = 12,
    budget_margin: float = 1.0,
    diagram_mode: str = "compact+sequence",
    diagram_depth: int = 2,
    diagram_nodes: int = 8,
    diagram_edges: int = 12,
    stream_output: bool = False,
) -> Dict[str, object]:
    from .sections.diagrams import diagram_lines, sequence_lines_from_flows
    from .sections.docs import doc_quality_lines, doc_section_lines
    from .sections.entities import entity_section_lines
    from .sections.flows import flow_edges_from_symbol_refs, flow_lines, flow_symbol_pairs
    from .sections.routes import (
        entrypoint_group_lines,
        entrypoint_detail_lines,
        entrypoint_inventory_lines,
        group_route_entries,
        invariants_section_lines,
        middleware_lines,
        route_archetypes,
        route_candidates,
        route_entries_for_file,
    )
    from .sections.traces import call_chain_lines, is_test_path, static_trace_lines
    from .trace_graph import trace_graph_lines

    configure_tokenizer(precise_tokens)
    if dense and dir_depth is None:
        dir_depth = 3
    if dir_depth is None:
        dir_depth = 0
    budget_margin = max(0.5, min(1.0, budget_margin))
    budget_cap = max(1, int(budget_tokens * budget_margin))
    budget_sources = dict(SECTION_BUDGETS)
    budget_warnings: List[str] = []
    if isinstance(section_budgets, dict):
        for key, value in section_budgets.items():
            key_name = str(key).strip().upper()
            if key_name not in SECTION_BUDGETS:
                budget_warnings.append(f"Unknown section budget key: {key}")
                continue
            try:
                ratio = float(value)
            except (TypeError, ValueError):
                budget_warnings.append(f"Invalid budget ratio for {key}: {value}")
                continue
            if ratio <= 0:
                budget_warnings.append(f"Invalid budget ratio for {key}: {value}")
                continue
            budget_sources[key_name] = min(1.0, ratio)
    section_caps = {
        name: max(1, int(budget_cap * ratio))
        for name, ratio in budget_sources.items()
    }
    section_usage: Dict[str, int] = {}

    def section_allow(section: str, line: str) -> bool:
        cap = section_caps.get(section)
        if not cap:
            return True
        used = section_usage.get(section, 0)
        projected = used + estimate_tokens(line)
        if projected > cap:
            return False
        section_usage[section] = projected
        return True

    used_tokens = 0

    def append_line(section: str, line: str) -> bool:
        nonlocal used_tokens
        if not section_allow(section, line):
            return False
        projected = estimate_tokens("\n".join(lines + [line]))
        if projected > budget_cap:
            return False
        lines.append(line)
        used_tokens = projected
        return True

    meta = ir.get("meta", {})
    repo_root_value = meta.get("repo")
    repo_root = Path(repo_root_value) if isinstance(repo_root_value, str) and repo_root_value else None
    config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    unsupported_langs = meta.get("unsupported_languages", {}) if isinstance(meta.get("unsupported_languages"), dict) else {}
    monorepo = meta.get("monorepo", {}) if isinstance(meta.get("monorepo"), dict) else {}
    codeql_meta = meta.get("codeql", {}) if isinstance(meta.get("codeql"), dict) else {}
    configured_mode = config.get("routes_mode")
    if routes_mode == "auto":
        routes_mode = configured_mode if isinstance(configured_mode, str) else "heuristic"
    route_mode_active = routes_mode
    if include_routes:
        if budget_cap >= 30_000:
            routes_limit = max(routes_limit, 128)
        elif budget_cap >= 18_000:
            routes_limit = max(routes_limit, 64)
        elif budget_cap >= 12_000:
            routes_limit = max(routes_limit, 40)
    if routes_format not in {"files", "compact"}:
        routes_format = "files"
    diagram_sequence = diagram_mode == "compact+sequence"
    if diagram_mode not in {"off", "compact", "mermaid", "compact+sequence"}:
        diagram_mode = "off"
        diagram_sequence = False
    if diagram_mode != "off":
        diagram_depth = max(1, diagram_depth)
    route_globs = config.get("routes_globs", [])
    route_names = config.get("routes_names", [])
    route_exclude_globs = config.get("routes_exclude_globs", [])
    entry_config = config.get("entrypoints") if isinstance(config.get("entrypoints"), dict) else {}
    entry_frameworks = entry_config.get("frameworks") if isinstance(entry_config.get("frameworks"), list) else None
    entry_max_items = entry_config.get("max_items") if isinstance(entry_config.get("max_items"), int) else None
    entry_max_per_group = entry_config.get("max_per_group") if isinstance(entry_config.get("max_per_group"), int) else None
    entry_include_middleware = bool(entry_config.get("include_middleware")) if "include_middleware" in entry_config else False
    config_details_limit = entry_config.get("details_limit") if isinstance(entry_config.get("details_limit"), int) else None
    config_details_per_kind = entry_config.get("details_per_kind") if isinstance(entry_config.get("details_per_kind"), int) else None
    warnings = meta.get("warnings", [])
    entity_nodes = ir.get("entities", {}) if entity_graph else {}
    entities = meta.get("entities", []) if isinstance(meta.get("entities"), list) else []
    entity_edges = ir.get("edges", {}).get("entity_use", []) if entity_graph else []
    test_mapping = meta.get("test_mapping", []) if isinstance(meta.get("test_mapping"), list) else []
    test_summary = meta.get("test_summary", {}) if isinstance(meta.get("test_summary"), dict) else {}
    api_contracts = meta.get("api_contracts", []) if isinstance(meta.get("api_contracts"), list) else []
    trace_links = meta.get("trace_links", []) if isinstance(meta.get("trace_links"), list) else []
    files_all = ir.get("files", {})
    if trace_links and traceability and traceability_targets != "all":
        docs_list = meta.get("docs", []) if isinstance(meta.get("docs"), list) else []
        docs_set = {str(path) for path in docs_list}

        def is_doc_target(target_id: str) -> bool:
            node = files_all.get(target_id, {})
            path = str(node.get("path") or "")
            return bool(path and path in docs_set)

        filtered_links: List[Dict[str, object]] = []
        if traceability_targets == "entities":
            for link in trace_links:
                if link.get("target_type") == "entity":
                    filtered_links.append(link)
        elif traceability_targets == "code":
            for link in trace_links:
                if link.get("target_type") != "file":
                    continue
                target_id = str(link.get("target_id") or "")
                if target_id and not is_doc_target(target_id):
                    filtered_links.append(link)
        else:
            filtered_links = trace_links
        trace_links = filtered_links
    if code_only:
        files = {
            fid: node
            for fid, node in files_all.items()
            if node.get("language") in {"ts", "tsx", "py", "js", "jsx", "go", "rs"}
        }
        if not files:
            files = files_all
    else:
        files = files_all

    config_paths, schema_paths = collect_configs_and_schemas(files_all)
    arch_info = detect_architecture(
        files_all,
        config_paths=config_paths,
        schema_paths=schema_paths,
        monorepo=monorepo,
    )
    if include_routes and route_mode_active == "heuristic":
        frameworks = {
            str(item).lower()
            for item in (arch_info.get("frameworks", []) if isinstance(arch_info.get("frameworks"), list) else [])
            if item
        }
        if "nextjs" in frameworks:
            route_mode_active = "nextjs"
        elif "django" in frameworks:
            route_mode_active = "django"
    overview_line = architecture_overview_line(arch_info)
    file_ids = set(files.keys())
    symbols_all = ir.get("symbols", {})
    if code_only:
        symbols = {
            sid: node
            for sid, node in symbols_all.items()
            if file_id(node.get("defined_in", {}).get("path", "")) in file_ids
        }
    else:
        symbols = symbols_all

    call_in: Dict[str, int] = {}
    call_out: Dict[str, int] = {}
    if explain_scores:
        call_in, call_out = file_call_counts(ir)

    sorted_files = sorted(files.values(), key=lambda f: f.get("score", 0.0), reverse=True)
    files_available = len(files)
    if dense and budget_tokens <= 4000 and files_available > 2000:
        dense = False
        dir_depth = 0
    selected: List[str] = []
    path_to_fid = {node.get("path", ""): fid for fid, node in files.items() if node.get("path")}

    entrypoints = [fid for fid, node in files.items() if node.get("role") == "entrypoint"]
    selected.extend(entrypoints)

    focus_set: Set[str] = set()
    neighbor_set: Set[str] = set()
    if focus:
        focus_set = focus_candidates(ir, focus, allowed_files=file_ids)
        selected.extend(focus_set)
        depth = max(1, min(3, int(focus_depth))) if focus_depth else 1
        neighbor_set = expand_focus_neighbors(
            ir, focus_set, depth=depth, allowed_files=file_ids
        )
        selected.extend(neighbor_set)

    route_set: Set[str] = set()
    route_files: List[str] = []
    if include_routes:
        route_files = route_candidates(
            files,
            route_globs=route_globs,
            route_names=route_names,
            exclude_globs=route_exclude_globs,
            routes_mode=route_mode_active,
            include_entrypoint_routes=True,
            repo_root=repo_root,
        )
        route_set = set(route_files[: max(routes_limit * 2, 8)])
        selected.extend(route_set)

    file_edges = ir.get("edges", {}).get("file_dep", [])

    for node in sorted_files:
        fid = node["id"]
        if fid not in selected:
            selected.append(fid)

    tests_for_selected: Set[str] = set()
    if test_mapping:
        target_to_tests: Dict[str, List[str]] = defaultdict(list)
        for entry in test_mapping:
            targets = entry.get("targets", [])
            test_path = entry.get("test", "")
            if not isinstance(targets, list) or not isinstance(test_path, str):
                continue
            for target_path in targets:
                if isinstance(target_path, str):
                    target_to_tests[target_path].append(test_path)
        for fid in list(selected):
            path = files.get(fid, {}).get("path", "")
            if not path:
                continue
            for test_path in target_to_tests.get(path, []):
                test_fid = path_to_fid.get(test_path)
                if test_fid and test_fid in files:
                    tests_for_selected.add(test_fid)
        max_tests = max(4, int((max_files or 0) * 0.2)) if max_files else 8
        if len(tests_for_selected) > max_tests:
            tests_for_selected = set(list(tests_for_selected)[:max_tests])
        selected.extend(tests_for_selected)

    if max_files is None or max_symbols_per_file is None or max_edges is None or max_sig_len is None:
        auto_files, auto_symbols, auto_edges, auto_sig_len = auto_limits(budget_cap)
        max_files = auto_files if max_files is None else max_files
        max_symbols_per_file = auto_symbols if max_symbols_per_file is None else max_symbols_per_file
        max_edges = auto_edges if max_edges is None else max_edges
        max_sig_len = auto_sig_len if max_sig_len is None else max_sig_len

    if files_available <= 200:
        if max_files is not None:
            max_files = min(files_available, max(max_files, int(max_files * 1.5)))
        if max_edges is not None and file_edges:
            max_edges = min(len(file_edges), max(max_edges, int(max_edges * 1.5)))
        if max_symbols_per_file is not None:
            max_symbols_per_file = max(max_symbols_per_file, min(12, max_symbols_per_file + 2))
        if routes_limit is not None and route_files:
            routes_limit = max(routes_limit, min(len(route_files), routes_limit + 4))
        trace_max = max(trace_max, min(12, trace_max + 2))

    priorities: Dict[str, float] = {}
    for fid, node in files.items():
        score = float(node.get("score", 0.0))
        bonus = 0.0
        penalty = selection_penalty(node, node.get("path", ""))
        if fid in entrypoints:
            bonus += 2.0
        if fid in focus_set:
            bonus += 1.5
        elif fid in neighbor_set:
            bonus += 0.5
        if fid in route_set:
            bonus += 1.0
        if fid in tests_for_selected:
            bonus += 0.8
        priorities[fid] = score + bonus - penalty

    hub_scores: Counter[str] = Counter()
    for edge in file_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in files:
            hub_scores[src] += 1
        if dst in files:
            hub_scores[dst] += 1
    top_hubs = [fid for fid, _count in hub_scores.most_common(6)]

    entry_bucket_cap = 6
    if budget_cap >= 30_000:
        entry_bucket_cap = 12
    elif budget_cap >= 18_000:
        entry_bucket_cap = 8
    bucket_targets: List[str] = []
    bucket_targets.extend(entrypoints[:entry_bucket_cap])
    if focus_set:
        bucket_targets.extend(list(focus_set)[:4])
    bucket_targets.extend(list(route_set)[:6])
    bucket_targets.extend(top_hubs[:4])

    selected = list(dict.fromkeys(selected))
    selected.sort(key=lambda fid: priorities.get(fid, 0.0), reverse=True)
    must_keep = [fid for fid in bucket_targets if fid in files]
    must_keep = list(dict.fromkeys(must_keep))
    if must_keep:
        must_keep.sort(key=lambda fid: priorities.get(fid, 0.0), reverse=True)
    if max_files:
        if len(must_keep) > max_files:
            must_keep = must_keep[:max_files]
            selected = must_keep
        else:
            remaining = [fid for fid in selected if fid not in must_keep]
            selected = must_keep + remaining[: max_files - len(must_keep)]
    else:
        selected = selected[: max_files or 0]
    files_selected = len(selected)

    prefix = ""
    if compress_paths:
        prefix = common_path_prefix([files[fid]["path"] for fid in selected if fid in files])

    aliases: Dict[str, str] = {}
    file_alias: Dict[str, str] = {}
    symbol_alias: Dict[str, str] = {}
    lines: List[str] = []

    append_line("REPO", "[REPO]")
    meta = ir.get("meta", {})
    append_line("REPO", f"name={Path(meta.get('repo', '')).name}")
    langs = ",".join(meta.get("languages", []))
    append_line("REPO", f"langs={langs}")
    append_line("REPO", f"files={len(files)} symbols={len(symbols)}")
    if prefix:
        append_line("REPO", f"path_prefix={prefix}")
    append_line("REPO", "")
    append_line("SUMMARY", "[SUMMARY]")

    top_dirs: Dict[str, int] = defaultdict(int)
    for node in files.values():
        path = node.get("path", "")
        top = path.split("/")[0] if "/" in path else "."
        top_dirs[top] += 1
    dir_line = ",".join(f"{name}={count}" for name, count in sorted(top_dirs.items(), key=lambda i: -i[1])[:6])
    if dir_line:
        append_line("SUMMARY", f"top_dirs={dir_line}")

    ext_counts: Dict[str, int] = defaultdict(int)
    for edge in ir.get("edges", {}).get("external_dep", []):
        module = edge.get("module")
        if module:
            ext_counts[module] += 1
    ext_line = ",".join(f"{name}:{count}" for name, count in sorted(ext_counts.items(), key=lambda i: -i[1])[:8])
    if ext_line:
        append_line("SUMMARY", f"top_external={ext_line}")
    external_line = external_summary_line(ext_counts)
    if external_line:
        append_line("SUMMARY", external_line)
    call_counts: Optional[Counter] = None
    call_totals: Optional[Counter] = None
    semantic_meta: Optional[Dict[str, object]] = None
    if semantic_cluster:
        call_counts, call_totals = build_file_call_stats(ir, files)
        comp_files, component_order, semantic_meta = semantic_component_index(
            files,
            symbols,
            prefix=prefix,
            max_components=12 if dense else 10,
            call_counts=call_counts,
            call_totals=call_totals,
            call_weight=semantic_call_weight,
            mode=semantic_cluster_mode,
        )
    else:
        comp_files = build_component_index(files, prefix)
        component_order = []
    comp_counts = {comp: len(fids) for comp, fids in comp_files.items()}
    comp_edges, comp_in, comp_out = component_edge_counts(file_edges, comp_files)
    comp_cycles = component_cycles(comp_edges)
    risk_flags = component_risk_flags(comp_counts, comp_in, comp_out)
    max_components = 12 if dense else 10
    if comp_files and not component_order:
        ordered_by_size = sorted(comp_files.items(), key=lambda item: (-len(item[1]), item[0]))
        component_order = [comp for comp, _fids in ordered_by_size[:max_components]]
        if comp_edges:
            comp_degree: Counter[str] = Counter()
            for (src, dst), weight in comp_edges.items():
                comp_degree[src] += weight
                comp_degree[dst] += weight
            for comp, _count in comp_degree.most_common(max_components):
                if comp not in component_order:
                    component_order.append(comp)
    comp_lines, comp_aliases = component_section_lines(
        comp_files,
        files,
        max_components=max_components,
        component_order=component_order,
    )
    comp_edge_lines = component_edges_lines(comp_edges, comp_aliases)
    purpose_lines = component_purpose_lines(
        comp_files,
        files,
        symbols,
        prefix=prefix,
        max_components=min(6, max_components),
        component_order=component_order,
    )
    pattern_lines = pattern_section_lines(files, symbols, prefix=prefix)
    layer_lines = layer_section_lines(files, file_edges, prefix=prefix)
    arch_style_section = arch_style_lines(files, file_edges, prefix=prefix)
    cycle_lines = cycle_section_lines(comp_cycles, comp_counts, comp_in, comp_out)
    roles_line = role_summary_line(files)
    if roles_line:
        append_line("SUMMARY", roles_line)
    role_edges = role_edges_line(files, file_edges)
    if role_edges:
        append_line("SUMMARY", role_edges)

    entry_summary_limit = 6
    entry_summary_segments = 3
    if budget_cap >= 30_000:
        entry_summary_limit = min(96, max(48, len(entrypoints)))
        entry_summary_segments = 8
    elif budget_cap >= 18_000:
        entry_summary_limit = min(48, max(24, len(entrypoints)))
        entry_summary_segments = 6
    elif budget_cap >= 12_000:
        entry_summary_limit = min(32, max(16, len(entrypoints)))
        entry_summary_segments = 5
    entry_paths = []
    for fid in entrypoints[:entry_summary_limit]:
        node = files.get(fid)
        if not node:
            continue
        entry_path = str(node.get("path", "")).lstrip("./")
        if not entry_path:
            continue
        role = node.get("entrypoint_role")
        if role:
            entry_paths.append(f"{role}:{entry_path}")
        else:
            entry_paths.append(entry_path)
    if entry_paths:
        append_line("SUMMARY", "entrypoints=" + ",".join(entry_paths))
        append_line("SUMMARY", f"entrypoints_total={len(entrypoints)}")

    if comp_counts:
        append_line("SUMMARY", f"components={len(comp_counts)}")
    if config_paths:
        samples = ",".join(
            short_path(path, prefix, max_segments=3) for path in config_paths[:3]
        )
        append_line("SUMMARY", f"configs={len(config_paths)} samples={samples}")
    if schema_paths:
        samples = ",".join(
            short_path(path, prefix, max_segments=3) for path in schema_paths[:3]
        )
        append_line("SUMMARY", f"schemas={len(schema_paths)} samples={samples}")
    if entity_nodes:
        append_line("SUMMARY", f"entities={len(entity_nodes)}")
    elif entities:
        append_line("SUMMARY", f"entities={len(entities)}")
    if test_mapping:
        append_line("SUMMARY", f"tests_mapped={len(test_mapping)}")
    if api_contracts:
        append_line("SUMMARY", f"api_contracts={len(api_contracts)}")
    if semantic_cluster:
        fallback_reason = semantic_meta.get("fallback_reason") if semantic_meta else None
        if fallback_reason:
            append_line("SUMMARY", f"semantic_cluster=fallback:{fallback_reason}")
        else:
            append_line("SUMMARY", "semantic_cluster=on")
            call_weight_used = (
                semantic_meta.get("call_weight_used") if semantic_meta else None
            )
            if semantic_call_weight > 0 and call_weight_used == 0:
                append_line("SUMMARY", "semantic_cluster_call_affinity=off (no_call_edges)")
        if semantic_meta and semantic_meta.get("mode"):
            append_line("SUMMARY", f"semantic_cluster_mode={semantic_meta.get('mode')}")
    if risk_flags:
        append_line("SUMMARY", "risk_flags=" + ",".join(risk_flags))
    if comp_cycles:
        cycle_list = ";".join(">".join(cycle) for cycle in comp_cycles)
        append_line("SUMMARY", f"cycles={cycle_list}")

    if unsupported_langs:
        unsupported_line = ",".join(
            f"{name}:{count}" for name, count in list(unsupported_langs.items())[:6]
        )
        append_line("SUMMARY", f"unsupported_langs={unsupported_line}")

    if monorepo:
        packages = monorepo.get("packages", [])
        package_count = len([pkg for pkg in packages if pkg.get("path") not in {".", ""}]) or len(packages)
        monorepo_type = monorepo.get("type", "workspace")
        append_line("SUMMARY", f"monorepo={monorepo_type} packages={package_count}")

    if overview_line:
        append_line("SUMMARY", overview_line)

    if include_routes:
        append_line("SUMMARY", f"routes_mode={route_mode_active}")
        if routes_format != "files":
            append_line("SUMMARY", f"routes_format={routes_format}")

    if dense:
        inbound_all: Dict[str, int] = defaultdict(int)
        outbound_all: Dict[str, int] = defaultdict(int)
        if isinstance(graph_cache, dict):
            for fid, count in (graph_cache.get("inbound") or {}).items():
                if fid in files:
                    inbound_all[fid] = int(count)
            for fid, count in (graph_cache.get("outbound") or {}).items():
                if fid in files:
                    outbound_all[fid] = int(count)
        if not inbound_all and not outbound_all:
            for edge in file_edges:
                src = edge.get("from")
                dst = edge.get("to")
                if src in files and dst in files:
                    outbound_all[src] += 1
                    inbound_all[dst] += 1
        top_in = sorted(
            inbound_all.items(),
            key=lambda item: (-item[1], files.get(item[0], {}).get("path", "")),
        )[:6]
        top_out = sorted(
            outbound_all.items(),
            key=lambda item: (-item[1], files.get(item[0], {}).get("path", "")),
        )[:6]
        if top_in or top_out:
            append_line("TOP_HUBS", "")
            append_line("TOP_HUBS", "[TOP_HUBS]")
            if top_in:
                in_line = ",".join(
                    f"{short_path(files[fid]['path'], prefix, max_segments=3)}:{count}"
                    for fid, count in top_in
                )
                append_line("TOP_HUBS", f"in={in_line}")
            if top_out:
                out_line = ",".join(
                    f"{short_path(files[fid]['path'], prefix, max_segments=3)}:{count}"
                    for fid, count in top_out
                )
                append_line("TOP_HUBS", f"out={out_line}")

    public_api_section = public_api_lines(files, prefix=prefix)
    if public_api_section:
        append_line("PUBLIC_API", "")
        for line in public_api_section:
            if not append_line("PUBLIC_API", line):
                break

    entrypoint_section, entry_items, middleware_items = entrypoint_inventory_lines(
        files,
        prefix=prefix,
        repo_root=repo_root,
        max_items=entry_max_items or 120,
        max_per_group=entry_max_per_group or 24,
        frameworks=set(entry_frameworks) if entry_frameworks else None,
        include_middleware=entry_include_middleware,
    )
    if entrypoint_section:
        append_line("ENTRYPOINTS", "")
        for line in entrypoint_section:
            if not append_line("ENTRYPOINTS", line):
                break

    entrypoint_groups = entrypoint_group_lines(entry_items, prefix=prefix)
    if entrypoint_groups:
        append_line("ENTRYPOINT_GROUPS", "")
        for line in entrypoint_groups:
            if not append_line("ENTRYPOINT_GROUPS", line):
                break

    middleware_section = middleware_lines(middleware_items, prefix=prefix)
    if middleware_section:
        append_line("MIDDLEWARE", "")
        for line in middleware_section:
            if not append_line("MIDDLEWARE", line):
                break

    details_limit = entry_details_limit if entry_details_limit is not None else config_details_limit or 12
    details_per_kind = entry_details_per_kind if entry_details_per_kind is not None else config_details_per_kind or 4
    entry_details = entrypoint_detail_lines(
        entry_items,
        repo_root=repo_root,
        prefix=prefix,
        max_entries=max(1, details_limit),
        max_per_kind=max(1, details_per_kind),
    )
    if entry_details:
        append_line("ENTRY_DETAILS", "")
        for line in entry_details:
            if not append_line("ENTRY_DETAILS", line):
                break

    invariants_section = invariants_section_lines(entry_details)
    if invariants_section:
        append_line("INVARIANTS", "")
        for line in invariants_section:
            if not append_line("INVARIANTS", line):
                break

    if dir_depth > 0:
        dir_paths = [node.get("path", "") for node in files.values() if node.get("path")]
        if prefix:
            dir_paths = [path[len(prefix) :] if path.startswith(prefix) else path for path in dir_paths]
        dir_lines = build_dir_skeleton(
            dir_paths,
            depth=dir_depth,
            max_files_per_dir=dir_file_cap,
        )
        if dir_lines:
            append_line("DIR", "")
            append_line("DIR", "[DIR]")
            for line in dir_lines:
                if not append_line("DIR", line):
                    break

    if comp_lines:
        append_line("COMPONENTS", "")
        for line in comp_lines:
            if not append_line("COMPONENTS", line):
                break
    if comp_edge_lines:
        append_line("COMPONENT_EDGES", "")
        for line in comp_edge_lines:
            if not append_line("COMPONENT_EDGES", line):
                break

    if cycle_lines:
        append_line("CYCLES", "")
        for line in cycle_lines:
            if not append_line("CYCLES", line):
                break

    warnings_list: List[str] = []
    for warning in budget_warnings:
        if warning:
            warnings_list.append(str(warning))
        if len(warnings_list) >= 6:
            break
    if isinstance(warnings, list) and len(warnings_list) < 6:
        for warning in warnings:
            if warning:
                warnings_list.append(str(warning))
            if len(warnings_list) >= 6:
                break
    warning_lines = [f"- {warning}" for warning in warnings_list]
    if warning_lines:
        append_line("WARNINGS", "")
        append_line("WARNINGS", "[WARNINGS]")
        for warning in warning_lines:
            if not append_line("WARNINGS", warning):
                break

    has_codeql_langs = bool(set(meta.get("languages", [])) & {"ts", "tsx", "py", "js", "jsx"})
    if has_codeql_langs:
        append_line("CODEQL", "")
        append_line("CODEQL", "[CODEQL]")
        status = codeql_meta.get("status", "disabled")
        append_line("CODEQL", f"status={status}")
        if status == "ok":
            counts = codeql_meta.get("counts", {}) if isinstance(codeql_meta, dict) else {}
            if counts:
                count_line = ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))
                append_line("CODEQL", f"counts={count_line}")
        else:
            append_line("CODEQL", "hint=enable with --codeql on for call graph and dataflow edges (slower)")

    packages_meta = monorepo.get("packages", []) if isinstance(monorepo, dict) else []
    if packages_meta:
        stats = package_stats(files, symbols, packages_meta)
        if stats:
            append_line("PACKAGES", "")
            append_line("PACKAGES", "[PACKAGES]")
            monorepo_type = monorepo.get("type", "workspace") if isinstance(monorepo, dict) else "workspace"
            append_line("PACKAGES", f"type={monorepo_type} count={len(stats)}")
            for idx, (root, file_count, symbol_count) in enumerate(stats[:8], start=1):
                label = "root" if root in {".", ""} else short_path(root, prefix, max_segments=3)
                if not append_line("PACKAGES", f"P{idx} {label} files={file_count} symbols={symbol_count}"):
                    break

    limits_lines = [
        "[LIMITS]",
        "budget=",
        "files=",
        "symbols=",
        "edges=",
    ]
    if include_routes:
        limits_lines.append("routes=")
    limits_lines.append("truncated=")

    confidence_placeholders = [
        "[CONFIDENCE_GATE]",
        "status=",
        "target_stack=",
        "ast_edge_confidence=",
        "rule1_ast_confidence_not_low=",
        "rule2_not_truncated=",
        "rule3_target_stack_supported=",
        "rule4_codeql_rerun_if_needed=",
        "codeql_required=",
        "codeql_mode=",
        "codeql_status=",
        "unsupported_hits=",
        "action=",
    ]

    append_line("CONFIDENCE_GATE", "")
    confidence_start = len(lines)
    confidence_added = 0
    for line in confidence_placeholders:
        if not append_line("CONFIDENCE_GATE", line):
            break
        confidence_added += 1

    append_line("LIMITS", "")
    limits_start = len(lines)
    limits_added = 0
    for line in limits_lines:
        if not append_line("LIMITS", line):
            break
        limits_added += 1

    append_line("FILES", "")
    skip_files_section = not append_line("FILES", "[FILES]")

    used_tokens = estimate_tokens("\n".join(lines))
    reserve_tail = clamp(int(budget_cap * 0.25), 200, 1800)
    if include_routes:
        reserve_tail += 120
    if entities:
        reserve_tail += 120
    if dense:
        reserve_tail += 80
    reserve_tail = min(reserve_tail, max(0, budget_cap - 100))
    file_budget = max(100, budget_cap - reserve_tail)
    file_index = 1
    symbol_index = 1
    files_included = 0
    symbols_selected = 0

    for fid in selected:
        if skip_files_section:
            break
        node = files.get(fid)
        if not node:
            continue
        alias = f"F{file_index}"
        file_alias[fid] = alias
        aliases[alias] = fid
        summary = node.get("summary_1l") or node.get("role", "")
        path = short_path(node.get("path", "") or "", prefix, max_segments=4)
        score_note = ""
        counts_note = ""
        if dense:
            imports_count = len(node.get("imports", []))
            exports_count = len(node.get("exports", []))
            defines_count = len(node.get("defines", []))
            language = node.get("language", "")
            counts_note = (
                f" lang={language} i={imports_count} e={exports_count} s={defines_count}"
                if language
                else f" i={imports_count} e={exports_count} s={defines_count}"
            )
        if explain_scores:
            exports_count = len(node.get("exports", []))
            calls = call_in.get(fid, 0) + call_out.get(fid, 0)
            score_note = f" score={node.get('score', 0.0):.2f} exports={exports_count} calls={calls}"
            if node.get("role") == "entrypoint":
                score_note += " entry"
        line = f"{alias} {path} :: {summary}{counts_note}{score_note}"
        if not section_allow("FILES", line):
            break
        projected = estimate_tokens("\n".join(lines + [line]))
        if projected > file_budget:
            break
        lines.append(line)
        used_tokens = projected
        files_included += 1

        symbols_for_file = top_symbols_for_file(ir, fid, max_symbols_per_file or 0)
        symbols_selected += len(symbols_for_file)
        for sid in symbols_for_file:
            sym = symbols.get(sid)
            if not sym:
                continue
            salias = f"S{symbol_index}"
            symbol_alias[sid] = salias
            aliases[salias] = sid
            signature = sym.get("signature") or sym.get("name")
            signature = compact_signature(signature, max_sig_len or 0)
            sym_note = ""
            if explain_scores:
                refs_in = sym.get("refs_in_count", 0)
                visibility = sym.get("visibility", "")
                kind = sym.get("kind", "")
                sym_note = f" score={sym.get('score', 0.0):.2f} refs_in={refs_in}"
                if visibility:
                    sym_note += f" {visibility}"
                if kind:
                    sym_note += f" kind={kind}"
            doc_note = compact_doc(str(sym.get("doc_1l") or ""), 80)
            doc_text = f" :: {doc_note}" if doc_note else ""
            sym_line = f"  + {salias} {signature}{sym_note}{doc_text}"
            if not section_allow("FILES", sym_line):
                break
            projected = estimate_tokens("\n".join(lines + [sym_line]))
            if projected > file_budget:
                break
            lines.append(sym_line)
            used_tokens = projected
            symbol_index += 1
        file_index += 1

    listed = 0
    route_entries: List[Dict[str, object]] = []
    if include_routes and route_files:
        route_label_cap = min(128, max(12, routes_limit))
        route_files_to_scan = list(route_files[: max(routes_limit * 3, 20)])
        ui_candidates = [
            fid
            for fid, node in files.items()
            if "/routes/" in (node.get("path") or "")
            and not is_test_path(str(node.get("path") or ""))
        ]
        for fid in ui_candidates:
            if fid in route_files_to_scan:
                continue
            route_files_to_scan.append(fid)
            if len(route_files_to_scan) >= max(routes_limit * 5, 40):
                break
        for fid in route_files_to_scan:
            node = files.get(fid, {})
            path = node.get("path", "") or ""
            if not path or is_test_path(path):
                continue
            for entry in route_entries_for_file(
                path=path,
                prefix=prefix,
                routes_mode=route_mode_active,
                repo_root=repo_root,
                entrypoint_role=str(node.get("entrypoint_role") or ""),
                max_labels=route_label_cap,
            ):
                entry["fid"] = fid
                entry["path"] = path
                route_entries.append(entry)

    entity_limit = 8
    if budget_cap >= 18_000:
        entity_limit = 24
    elif budget_cap >= 12_000:
        entity_limit = 16
    entity_lines, entity_alias = entity_section_lines(
        entities,
        entity_nodes=entity_nodes if entity_graph else None,
        entity_edges=entity_edges if entity_graph else None,
        prefix=prefix,
        file_alias=file_alias,
        max_entities=entity_limit,
    )
    for ent_id, alias in entity_alias.items():
        aliases[alias] = ent_id

    test_mapping_section = test_mapping_lines(
        test_mapping,
        prefix=prefix,
        file_alias=file_alias,
        max_items=12,
        summary=test_summary,
    )
    capability_section, capability_gap_section = capability_insight_lines(
        symbols,
        files,
        prefix=prefix,
        file_alias=file_alias,
        max_items=8,
    )
    test_confidence_section = test_confidence_lines(
        test_mapping,
        test_summary,
        files,
        prefix=prefix,
        file_alias=file_alias,
        max_items=8,
    )
    invariants_hc_section, invariants_hc_facts = invariants_hc_lines(
        entry_details,
        max_items=8,
    )
    quality_meta = meta.get("quality", {}) if isinstance(meta.get("quality"), dict) else {}
    change_risk_section = change_risk_lines(
        files,
        quality_meta,
        test_summary,
        invariants_hc_facts,
        prefix=prefix,
        file_alias=file_alias,
        max_items=8,
    )
    dead_code_section = (
        dead_code_lines(
            symbols,
            files,
            file_edges,
            prefix=prefix,
            file_alias=file_alias,
            max_items=8,
        )
        if dead_code
        else []
    )
    static_trace_section: List[str] = []
    trace_graph_section: List[str] = []
    if static_traces:
        static_trace_section, trace_edges = static_trace_lines(
            ir=ir,
            files=files,
            entities=entity_nodes,
            entity_edges=entity_edges,
            entrypoints=entrypoints,
            route_entries=route_entries,
            repo_root=Path(meta.get("repo", "")) if meta.get("repo") else None,
            api_contracts=api_contracts,
            file_alias=file_alias,
            entity_alias=entity_alias,
            prefix=prefix,
            max_depth=trace_depth,
            max_items=trace_max,
            trace_direction=trace_direction,
            trace_start=trace_start,
            trace_end=trace_end,
        )
        if trace_edges:
            trace_graph_section = trace_graph_lines(
                files=files,
                entities=entity_nodes,
                trace_edges=trace_edges,
                entrypoints=entrypoints,
                route_entries=route_entries,
                file_alias=file_alias,
                entity_alias=entity_alias,
                prefix=prefix,
                max_depth=trace_depth,
                max_items=trace_max * 2,
            )
    api_contract_section = api_contract_lines(
        api_contracts,
        prefix=prefix,
        file_alias=file_alias,
        max_items=12,
    )
    type_hierarchy_section = type_hierarchy_lines(
        ir,
        prefix=prefix,
        file_alias=file_alias,
        max_items=12,
    )
    doc_section = doc_section_lines(
        symbols,
        symbol_alias,
        max_items=8,
    )
    doc_quality_section = (
        doc_quality_lines(symbols, files, prefix=prefix, max_items=6, strict=doc_quality_strict)
        if doc_quality
        else []
    )
    if entity_lines:
        lines.append("")
        for line in entity_lines:
            if not section_allow("ENTITIES", line):
                break
            projected = estimate_tokens("\n".join(lines + [line]))
            if projected > budget_cap:
                break
            lines.append(line)
            used_tokens = projected

    traceability_section = (
        traceability_lines(trace_links, file_alias, entity_alias, prefix=prefix)
        if traceability and trace_links
        else []
    )

    skip_routes_section = False
    if include_routes and route_entries:
        def route_file_ref(fid: str, raw_path: str) -> str:
            existing = file_alias.get(fid)
            if existing:
                return existing
            alias = f"F{len(file_alias) + 1}"
            file_alias[fid] = alias
            aliases[alias] = raw_path
            return alias

        seen_routes: Set[str] = set()
        lines.append("")
        header = "[ROUTES]" if routes_format == "compact" else "[POTENTIAL_ROUTE_FILES]"
        if not section_allow("ROUTES", header):
            skip_routes_section = True
        if not skip_routes_section:
            lines.append(header)
            if section_allow("ROUTES", f"mode={route_mode_active}"):
                lines.append(f"mode={route_mode_active}")

            if routes_format == "compact":
                ui_count = sum(1 for entry in route_entries if entry.get("kind") == "ui")
                api_count = sum(1 for entry in route_entries if entry.get("kind") == "api")
                count_line = f"count={len(route_entries)} ui={ui_count} api={api_count}"
                projected = estimate_tokens("\n".join(lines + [count_line]))
                if projected <= budget_cap and section_allow("ROUTES", count_line):
                    lines.append(count_line)
                    used_tokens = projected

                ui_groups = group_route_entries(route_entries, kind="ui")
                if ui_groups:
                    group_line = "ui_groups=" + ",".join(ui_groups)
                    projected = estimate_tokens("\n".join(lines + [group_line]))
                    if projected <= budget_cap and section_allow("ROUTES", group_line):
                        lines.append(group_line)
                        used_tokens = projected

                api_groups = group_route_entries(route_entries, kind="api")
                if api_groups:
                    group_line = "api_groups=" + ",".join(api_groups)
                    projected = estimate_tokens("\n".join(lines + [group_line]))
                    if projected <= budget_cap and section_allow("ROUTES", group_line):
                        lines.append(group_line)
                        used_tokens = projected

                archetypes = route_archetypes(route_entries, files, file_edges)
                if archetypes:
                    archetype_line = "archetypes=" + ",".join(archetypes)
                    projected = estimate_tokens("\n".join(lines + [archetype_line]))
                    if projected <= budget_cap and section_allow("ROUTES", archetype_line):
                        lines.append(archetype_line)
                        used_tokens = projected

            ordered_routes = sorted(
                route_entries,
                key=lambda entry: (
                    0 if str(entry.get("label") or "").startswith("file:") else 1,
                    0 if entry.get("kind") == "ui" else 1,
                    str(entry.get("path") or ""),
                    str(entry.get("label") or ""),
                ),
            )
            for entry in ordered_routes:
                if listed >= routes_limit:
                    break
                fid = entry.get("fid")
                if not isinstance(fid, str):
                    continue
                node = files.get(fid, {})
                raw_path = entry.get("path") or node.get("path") or ""
                path = str(raw_path).lstrip("./")
                if not path:
                    continue
                file_ref = route_file_ref(fid, path)
                if routes_format == "compact":
                    label = str(entry.get("label") or "")
                    if not label:
                        continue
                    route_key = f"{label}|{fid}"
                    if route_key in seen_routes:
                        continue
                    seen_routes.add(route_key)
                    if label.startswith("file:"):
                        line = f"R{listed + 1} {label}"
                    else:
                        line = f"R{listed + 1} {label} file:{path}"
                else:
                    line = f"{file_ref} {path}"
                if not section_allow("ROUTES", line):
                    break
                projected = estimate_tokens("\n".join(lines + [line]))
                if projected > budget_cap:
                    break
                lines.append(line)
                used_tokens = projected
                listed += 1

    if api_contract_section:
        lines.append("")
        for line in api_contract_section:
            if not section_allow("API_CONTRACTS", line):
                break
            projected = estimate_tokens("\n".join(lines + [line]))
            if projected > budget_cap:
                break
            lines.append(line)
            used_tokens = projected

    flow_edges = flow_edges_from_symbol_refs(ir, set(file_alias.keys()))
    if not flow_edges:
        flow_edges = file_edges
    symbol_pairs = (
        flow_symbol_pairs(ir, set(file_alias.keys())) if flow_symbols else None
    )
    flow_section = flow_lines(
        files=files,
        edges=flow_edges,
        file_alias=file_alias,
        entrypoints=entrypoints,
        route_entries=route_entries,
        prefix=prefix,
        flow_symbols=flow_symbols,
        symbol_pairs=symbol_pairs,
        adj_cache=(graph_cache or {}).get("file_adj") if isinstance(graph_cache, dict) else None,
        out_degree_cache=(graph_cache or {}).get("outbound") if isinstance(graph_cache, dict) else None,
    )
    call_chain_section = (
        call_chain_lines(
            ir,
            file_alias=file_alias,
            symbol_alias=symbol_alias,
            entrypoints=entrypoints,
            prefix=prefix,
            max_items=6,
            max_depth=3,
        )
        if call_chains
        else []
    )
    if call_chains and not call_chain_section:
        call_chain_section = ["[CALL_CHAINS]", "none_detected"]
    diagram_seq_lines: List[str] = []
    diagram_section: List[str] = []
    if diagram_sequence:
        diagram_seq_lines = sequence_lines_from_flows(flow_section)
    if diagram_mode != "off":
        diagram_section = diagram_lines(
            files,
            ir.get("edges", {}).get("file_dep", []),
            prefix=prefix,
            depth=diagram_depth,
            max_nodes=diagram_nodes,
            max_edges=diagram_edges,
            mode=diagram_mode,
        )

    quality_section: List[str] = []
    quality = quality_meta
    if isinstance(quality, dict) and quality:
        for key in [
            "code_files_total",
            "ast_enabled",
            "ast_coverage",
            "ast_files_scanned",
            "ast_files_cached",
            "import_capture_rate",
            "import_estimate",
            "alias_resolved",
            "rg_fallback",
            "codeql_enabled",
            "codeql_ran",
        ]:
            if key in quality:
                quality_section.append(f"{key}={quality.get(key)}")
        if graph_cache_signature:
            quality_section.append(f"graph_cache_hit={bool(graph_cache_hit)}")
        coupling = quality.get("coupling") if isinstance(quality.get("coupling"), dict) else {}
        if coupling:
            def format_coupling(entries: List[Dict[str, object]], *, key: str) -> str:
                parts: List[str] = []
                for item in entries[:6]:
                    fid = item.get("id")
                    if not isinstance(fid, str):
                        continue
                    alias = file_alias.get(fid) or short_path(files.get(fid, {}).get("path", ""), prefix, max_segments=3)
                    value = item.get(key)
                    if value is None:
                        continue
                    parts.append(f"{alias}:{value}")
                return ",".join(parts)

            fan_in = coupling.get("fan_in", [])
            fan_out = coupling.get("fan_out", [])
            instability = coupling.get("instability", [])
            if isinstance(fan_in, list) and fan_in:
                line = format_coupling(fan_in, key="count")
                if line:
                    quality_section.append(f"coupling_fan_in={line}")
            if isinstance(fan_out, list) and fan_out:
                line = format_coupling(fan_out, key="count")
                if line:
                    quality_section.append(f"coupling_fan_out={line}")
            if isinstance(instability, list) and instability:
                line = format_coupling(instability, key="value")
                if line:
                    quality_section.append(f"coupling_instability={line}")

    def section_size(name: str, items: List[str]) -> int:
        if not items:
            return 0
        header = f"[{name}]"
        return estimate_tokens("\n".join([header] + items))

    diagram_block = diagram_seq_lines + diagram_section
    section_sizes = {
        "DIAGRAM": section_size("DIAGRAM", diagram_block),
        "CALL_CHAINS": section_size("CALL_CHAINS", call_chain_section),
        "TRACE_GRAPH": section_size("TRACE_GRAPH", trace_graph_section),
        "DOCS_QUALITY": section_size("DOCS_QUALITY", doc_quality_section),
        "QUALITY": section_size("QUALITY", quality_section),
        "CAPABILITIES": section_size("CAPABILITIES", capability_section),
        "CAPABILITY_GAPS": section_size("CAPABILITY_GAPS", capability_gap_section),
        "TEST_CONFIDENCE": section_size("TEST_CONFIDENCE", test_confidence_section),
        "INVARIANTS_HC": section_size("INVARIANTS_HC", invariants_hc_section),
        "CHANGE_RISK": section_size("CHANGE_RISK", change_risk_section),
        "PATTERNS": section_size("PATTERNS", pattern_lines),
        "ARCH_STYLE": section_size("ARCH_STYLE", arch_style_section),
        "LAYERS": section_size("LAYERS", layer_lines),
        "DOCS": section_size("DOCS", doc_section),
        "FLOWS": section_size("FLOWS", flow_section),
    }
    projected_total = used_tokens + sum(section_sizes.values())
    drop_sections: Set[str] = set()
    drop_order = [
        "DIAGRAM",
        "CALL_CHAINS",
        "TRACE_GRAPH",
        "DOCS_QUALITY",
        "CAPABILITY_GAPS",
        "CAPABILITIES",
        "TEST_CONFIDENCE",
        "INVARIANTS_HC",
        "CHANGE_RISK",
        "QUALITY",
        "PATTERNS",
        "ARCH_STYLE",
        "LAYERS",
        "DOCS",
        "FLOWS",
    ]
    for section in drop_order:
        if projected_total <= budget_cap:
            break
        if section == "TRACE_GRAPH" and not static_trace_section:
            continue
        if section == "FLOWS" and not static_trace_section:
            continue
        size = section_sizes.get(section, 0)
        if size <= 0:
            continue
        drop_sections.add(section)
        projected_total -= size

    if static_trace_section:
        append_line("STATIC_TRACES", "")
        for line in static_trace_section:
            if not append_line("STATIC_TRACES", line):
                break

    if flow_section and "FLOWS" not in drop_sections:
        append_line("FLOWS", "")
        for line in flow_section:
            if not append_line("FLOWS", line):
                break

    if trace_graph_section and "TRACE_GRAPH" not in drop_sections:
        append_line("TRACE_GRAPH", "")
        for line in trace_graph_section:
            if not append_line("TRACE_GRAPH", line):
                break

    if doc_section and "DOCS" not in drop_sections:
        append_line("DOCS", "")
        for line in doc_section:
            if not append_line("DOCS", line):
                break

    arch_lines = architecture_section_lines(arch_info)
    if arch_lines:
        append_line("ARCHITECTURE", "")
        for line in arch_lines:
            if not append_line("ARCHITECTURE", line):
                break

    if arch_style_section and "ARCH_STYLE" not in drop_sections:
        append_line("ARCH_STYLE", "")
        for line in arch_style_section:
            if not append_line("ARCH_STYLE", line):
                break

    if layer_lines and "LAYERS" not in drop_sections:
        append_line("LAYERS", "")
        for line in layer_lines:
            if not append_line("LAYERS", line):
                break

    if pattern_lines and "PATTERNS" not in drop_sections:
        append_line("PATTERNS", "")
        for line in pattern_lines:
            if not append_line("PATTERNS", line):
                break

    if purpose_lines:
        append_line("PURPOSE", "")
        for line in purpose_lines:
            if not append_line("PURPOSE", line):
                break

    if capability_section and "CAPABILITIES" not in drop_sections:
        append_line("CAPABILITIES", "")
        for line in capability_section:
            if not append_line("CAPABILITIES", line):
                break

    if capability_gap_section and "CAPABILITY_GAPS" not in drop_sections:
        append_line("CAPABILITY_GAPS", "")
        for line in capability_gap_section:
            if not append_line("CAPABILITY_GAPS", line):
                break

    if invariants_hc_section and "INVARIANTS_HC" not in drop_sections:
        append_line("INVARIANTS_HC", "")
        for line in invariants_hc_section:
            if not append_line("INVARIANTS_HC", line):
                break

    if quality_section and "QUALITY" not in drop_sections:
        append_line("QUALITY", "")
        append_line("QUALITY", "[QUALITY]")
        for line in quality_section:
            if not append_line("QUALITY", line):
                break

    if doc_quality_section and "DOCS_QUALITY" not in drop_sections:
        append_line("DOCS_QUALITY", "")
        for line in doc_quality_section:
            if not append_line("DOCS_QUALITY", line):
                break

    if test_confidence_section and "TEST_CONFIDENCE" not in drop_sections:
        append_line("TEST_CONFIDENCE", "")
        for line in test_confidence_section:
            if not append_line("TEST_CONFIDENCE", line):
                break

    if traceability_section:
        append_line("TRACEABILITY", "")
        for line in traceability_section:
            if not append_line("TRACEABILITY", line):
                break

    if test_mapping_section:
        append_line("TEST_MAPPING", "")
        for line in test_mapping_section:
            if not append_line("TEST_MAPPING", line):
                break

    if change_risk_section and "CHANGE_RISK" not in drop_sections:
        append_line("CHANGE_RISK", "")
        for line in change_risk_section:
            if not append_line("CHANGE_RISK", line):
                break

    if dead_code_section:
        append_line("DEAD_CODE", "")
        for line in dead_code_section:
            if not append_line("DEAD_CODE", line):
                break

    if type_hierarchy_section:
        append_line("TYPE_HIERARCHY", "")
        for line in type_hierarchy_section:
            if not append_line("TYPE_HIERARCHY", line):
                break

    append_line("EDGES", "")
    skip_edges_section = not append_line("EDGES", "[EDGES]")
    edges = file_edges
    inbound, outbound, total = edge_counts(edges, set(file_alias.keys()))
    summary_line = f"edges_total={total}"
    if not skip_edges_section and section_allow("EDGES", summary_line):
        projected = estimate_tokens("\n".join(lines + [summary_line]))
        if projected <= budget_cap:
            lines.append(summary_line)
            used_tokens = projected
    for fid in selected[: min(len(selected), 12)]:
        if fid not in file_alias:
            continue
        in_count = inbound.get(fid, 0)
        out_count = outbound.get(fid, 0)
        if in_count == 0 and out_count == 0:
            continue
        count_line = f"{file_alias[fid]} in={in_count} out={out_count}"
        if skip_edges_section or not section_allow("EDGES", count_line):
            break
        projected = estimate_tokens("\n".join(lines + [count_line]))
        if projected > budget_cap:
            break
        lines.append(count_line)
        used_tokens = projected

    eligible_edges = [
        edge
        for edge in edges
        if edge.get("from") in file_alias and edge.get("to") in file_alias
    ]
    edges_eligible = len(eligible_edges)
    edges_cap = edges_eligible if max_edges is None else min(edges_eligible, max_edges)
    listed_edges = 0
    for edge in eligible_edges:
        if max_edges is not None and listed_edges >= max_edges:
            break
        src = edge.get("from")
        dst = edge.get("to")
        if src in file_alias and dst in file_alias:
            edge_line = f"{file_alias[src]} -> {file_alias[dst]}"
            if skip_edges_section or not section_allow("EDGES", edge_line):
                break
            projected = estimate_tokens("\n".join(lines + [edge_line]))
            if projected > budget_cap:
                break
            lines.append(edge_line)
            used_tokens = projected
            listed_edges += 1

    if call_chain_section and "CALL_CHAINS" not in drop_sections:
        append_line("CALL_CHAINS", "")
        for line in call_chain_section:
            if not append_line("CALL_CHAINS", line):
                break

    if diagram_block and "DIAGRAM" not in drop_sections:
        append_line("DIAGRAM", "")
        for line in diagram_block:
            if not append_line("DIAGRAM", line):
                break

    append_line("LEGEND", "")
    append_line("LEGEND", "[LEGEND]")
    append_line("LEGEND", "expand: use F# or S# aliases via repomap.expand")

    files_omit_limit = max(0, files_available - files_selected)
    files_omit_budget = max(0, files_selected - files_included)
    symbols_included = len(symbol_alias)
    symbols_omit_budget = max(0, symbols_selected - symbols_included)
    edges_omit_limit = max(0, edges_eligible - edges_cap)
    edges_omit_budget = max(0, edges_cap - listed_edges)
    routes_available = len(route_files) if include_routes else 0
    routes_cap = min(routes_limit, routes_available) if include_routes else 0
    routes_omit_limit = max(0, routes_available - routes_cap) if include_routes else 0
    routes_omit_budget = max(0, routes_cap - listed) if include_routes else 0

    truncated = any(
        value > 0
        for value in (
            files_omit_limit,
            files_omit_budget,
            symbols_omit_budget,
            edges_omit_limit,
            edges_omit_budget,
        )
    )
    if include_routes:
        truncated = truncated or routes_omit_limit > 0 or routes_omit_budget > 0

    limits_block = {
        "budget": {
            "tokens": budget_tokens,
            "cap": budget_cap,
            "margin": budget_margin,
            "used": used_tokens,
            "remaining": max(0, budget_cap - used_tokens),
        },
        "files": {
            "available": files_available,
            "selected": files_selected,
            "included": files_included,
            "omitted_by_limit": files_omit_limit,
            "omitted_by_budget": files_omit_budget,
        },
        "symbols": {
            "selected": symbols_selected,
            "included": symbols_included,
            "omitted_by_budget": symbols_omit_budget,
        },
        "edges": {
            "eligible": edges_eligible,
            "listed": listed_edges,
            "omitted_by_limit": edges_omit_limit,
            "omitted_by_budget": edges_omit_budget,
        },
        "truncated": truncated,
    }
    if section_budgets:
        limits_block["section_caps"] = section_caps
    if include_routes:
        limits_block["routes"] = {
            "available": routes_available,
            "listed": listed,
            "omitted_by_limit": routes_omit_limit,
            "omitted_by_budget": routes_omit_budget,
        }

    limits_lines = [
        "[LIMITS]",
        (
            f"budget={budget_tokens} cap={budget_cap} margin={budget_margin:.2f} "
            f"used={used_tokens} remaining={max(0, budget_cap - used_tokens)}"
        ),
        (
            "files="
            f"{files_available} selected={files_selected} included={files_included} "
            f"omit_limit={files_omit_limit} omit_budget={files_omit_budget}"
        ),
        (
            "symbols="
            f"{symbols_selected} included={symbols_included} "
            f"omit_budget={symbols_omit_budget}"
        ),
        (
            "edges="
            f"{edges_eligible} listed={listed_edges} "
            f"omit_limit={edges_omit_limit} omit_budget={edges_omit_budget}"
        ),
    ]
    if include_routes:
        limits_lines.append(
            "routes="
            f"{routes_available} listed={listed} "
            f"omit_limit={routes_omit_limit} omit_budget={routes_omit_budget}"
        )
    if section_budgets:
        caps_line = ",".join(
            f"{name}:{cap}" for name, cap in sorted(section_caps.items())
        )
        limits_lines.append(f"section_caps={caps_line}")
    limits_lines.append(f"truncated={'yes' if truncated else 'no'}")
    for offset, line in enumerate(limits_lines[:limits_added]):
        lines[limits_start + offset] = line

    confidence_gate = evaluate_confidence_gate(
        quality=quality if isinstance(quality, dict) else {},
        truncated=truncated,
        unsupported_langs=unsupported_langs,
        codeql_meta=codeql_meta,
    )
    confidence_lines = confidence_gate_lines(confidence_gate)
    for offset, line in enumerate(confidence_lines[:confidence_added]):
        lines[confidence_start + offset] = line

    digest_text = "" if stream_output else "\n".join(lines)
    result = {
        "digest": digest_text,
        "aliases": aliases,
        "included": {
            "files": list(file_alias.keys()),
            "symbols": list(symbol_alias.keys()),
            "entities": list(entity_alias.keys()),
        },
        "budget": {
            "tokens": budget_tokens,
            "used": used_tokens,
            "cap": budget_cap,
            "margin": budget_margin,
            "remaining": max(0, budget_cap - used_tokens),
        },
        "limits": limits_block,
        "graph_cache": {
            "hit": bool(graph_cache_hit),
            "signature": graph_cache_signature,
        },
        "warnings": warnings_list,
        "confidence_gate": confidence_gate,
    }
    if stream_output:
        result["lines"] = lines
    return result


def find_symbol(ir: Dict, query: str, limit: int = 20) -> List[Dict[str, str]]:
    query_lower = query.lower()
    results: List[Dict[str, str]] = []
    for sid, node in ir.get("symbols", {}).items():
        name = node.get("name", "")
        signature = node.get("signature", "")
        if query_lower in name.lower() or query_lower in signature.lower():
            results.append(
                {
                    "id": sid,
                    "name": name,
                    "signature": signature,
                    "defined_in": node.get("defined_in", {}).get("path", ""),
                }
            )
    results.sort(key=lambda r: r.get("name", ""))
    return results[:limit]


def format_symbol_ref(ir: Dict, sid: str, prefix: str) -> str:
    node = ir.get("symbols", {}).get(sid, {})
    name = node.get("name", sid)
    defined_in = node.get("defined_in", {}) if isinstance(node, dict) else {}
    path = defined_in.get("path", "")
    line = defined_in.get("line", 0)
    if path:
        return f"{name}@{short_path(path, prefix, max_segments=3)}:{line}"
    return str(name)


def query_symbol(
    ir: Dict,
    query: str,
    *,
    depth: int = 1,
    limit: int = 6,
) -> Dict[str, object]:
    matches: List[Dict[str, object]] = []
    direct_id = resolve_symbol_id(ir, query)
    if direct_id:
        node = ir.get("symbols", {}).get(direct_id, {})
        matches = [
            {
                "id": direct_id,
                "name": node.get("name", query),
                "signature": node.get("signature", ""),
                "defined_in": node.get("defined_in", {}),
            }
        ]
    else:
        matches = find_symbol(ir, query, limit=limit)
    if not matches:
        return {"error": "symbol not found", "query": query}

    all_files = [node.get("path", "") for node in ir.get("files", {}).values() if node.get("path")]
    prefix = common_path_prefix(all_files) if all_files else ""
    edges = ir.get("edges", {}).get("symbol_ref", [])
    has_edges = bool(edges)
    lines: List[str] = ["[QUERY]", f"query={query} matches={len(matches)}"]
    if not has_edges:
        lines.append("note=call_refs_missing (run with codeql for call graph)")
    for idx, match in enumerate(matches, start=1):
        sid = match.get("id", "")
        sym = ir.get("symbols", {}).get(sid, {})
        name = sym.get("name", match.get("name", ""))
        signature = sym.get("signature", match.get("signature", ""))
        defined_in = sym.get("defined_in", {}) if isinstance(sym, dict) else {}
        path = defined_in.get("path", "")
        line = defined_in.get("line", 0)
        lines.append(f"S{idx} {name} signature={signature}")
        if path:
            lines.append(f"  defined_in={short_path(path, prefix, max_segments=3)}:{line}")
        callers: List[str] = []
        callees: List[str] = []
        if has_edges:
            for edge in edges:
                if edge.get("to") == sid:
                    callers.append(format_symbol_ref(ir, edge.get("from"), prefix))
                if edge.get("from") == sid:
                    callees.append(format_symbol_ref(ir, edge.get("to"), prefix))
            if callers:
                lines.append("  callers=" + ",".join(callers[:10]))
            if callees:
                lines.append("  callees=" + ",".join(callees[:10]))
        if path:
            fid = file_id(path)
            dep_info = deps(ir, fid, depth=max(1, min(3, depth)))
            inbound = [
                short_path(ir.get("files", {}).get(f, {}).get("path", ""), prefix, max_segments=3)
                for f in dep_info.get("inbound", [])
            ]
            outbound = [
                short_path(ir.get("files", {}).get(f, {}).get("path", ""), prefix, max_segments=3)
                for f in dep_info.get("outbound", [])
            ]
            if inbound:
                lines.append("  neighbors_in=" + ",".join(inbound[:10]))
            if outbound:
                lines.append("  neighbors_out=" + ",".join(outbound[:10]))
    return {"text": "\n".join(lines)}


def bfs_file_neighbors(
    edges: List[Dict[str, object]],
    start: str,
    depth: int,
    limit: int,
    direction: str = "out",
) -> List[str]:
    if depth <= 0:
        return []
    adj: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        if direction in {"out", "both"}:
            adj[src].append(dst)
        if direction in {"in", "both"}:
            adj[dst].append(src)
    seen = {start}
    frontier = [start]
    results: List[str] = []
    for _ in range(depth):
        next_frontier: List[str] = []
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                results.append(neighbor)
                next_frontier.append(neighbor)
                if limit and len(results) >= limit:
                    return results
        frontier = next_frontier
        if not frontier:
            break
    return results


def search_repo(
    ir: Dict,
    query: str,
    *,
    depth: int = 1,
    limit: int = 6,
    direction: str = "both",
) -> Dict[str, object]:
    files = ir.get("files", {})
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {})
    query_lower = query.lower()

    direct_symbol = resolve_symbol_id(ir, query)
    direct_file = resolve_file_id(ir, query)
    symbol_matches: List[Dict[str, object]] = []
    if direct_symbol:
        symbol_matches = [{"id": direct_symbol}]
    else:
        symbol_matches = find_symbol(ir, query, limit=limit)
    file_matches: List[str] = []
    if direct_file:
        file_matches = [direct_file]
    else:
        for fid, node in files.items():
            path = str(node.get("path") or "")
            summary = str(node.get("summary_1l") or "")
            if query_lower in path.lower() or (summary and query_lower in summary.lower()):
                file_matches.append(fid)

    center_type = None
    center_id = None
    if symbol_matches:
        symbol_matches.sort(
            key=lambda match: symbols.get(match.get("id", ""), {}).get("score", 0.0),
            reverse=True,
        )
        center_id = symbol_matches[0].get("id")
        center_type = "symbol"
    elif file_matches:
        file_matches.sort(
            key=lambda fid: files.get(fid, {}).get("score", 0.0), reverse=True
        )
        center_id = file_matches[0]
        center_type = "file"

    defs: List[Dict[str, object]] = []
    refs: List[Dict[str, object]] = []
    neighbors: List[str] = []
    if center_type == "symbol" and center_id:
        sym = symbols.get(center_id, {})
        defined_in = sym.get("defined_in", {}) if isinstance(sym, dict) else {}
        defs.append(
            {
                "id": center_id,
                "name": sym.get("name", ""),
                "path": defined_in.get("path", ""),
                "line": defined_in.get("line", 0),
            }
        )
        for edge in edges.get("symbol_ref", []):
            src = edge.get("from")
            dst = edge.get("to")
            if dst == center_id and src in symbols:
                ref_sym = symbols.get(src, {})
                ref_def = ref_sym.get("defined_in", {}) if isinstance(ref_sym, dict) else {}
                refs.append(
                    {
                        "id": src,
                        "name": ref_sym.get("name", ""),
                        "path": ref_def.get("path", ""),
                        "line": ref_def.get("line", 0),
                    }
                )
        refs = refs[:limit]
        path = defined_in.get("path")
        if path:
            center_fid = file_id(path)
            neighbors = bfs_file_neighbors(
                edges.get("file_dep", []), center_fid, depth, limit, direction=direction
            )
    elif center_type == "file" and center_id:
        for sid in files.get(center_id, {}).get("exports", [])[:limit]:
            sym = symbols.get(sid, {})
            defined_in = sym.get("defined_in", {}) if isinstance(sym, dict) else {}
            defs.append(
                {
                    "id": sid,
                    "name": sym.get("name", ""),
                    "path": defined_in.get("path", ""),
                    "line": defined_in.get("line", 0),
                }
            )
        neighbors = bfs_file_neighbors(
            edges.get("file_dep", []), center_id, depth, limit, direction=direction
        )

    return {
        "query": query,
        "center": {"id": center_id, "type": center_type},
        "defs": defs,
        "refs": refs,
        "neighbors": neighbors,
    }


def deps(ir: Dict, fid: str, depth: int = 1) -> Dict[str, List[str]]:
    current = {fid}
    inbound: Set[str] = set()
    outbound: Set[str] = set()
    for _ in range(depth):
        next_set: Set[str] = set()
        for edge in ir.get("edges", {}).get("file_dep", []):
            src = edge.get("from")
            dst = edge.get("to")
            if src in current and dst:
                outbound.add(dst)
                next_set.add(dst)
            if dst in current and src:
                inbound.add(src)
                next_set.add(src)
        current = next_set
    return {
        "inbound": sorted(inbound),
        "outbound": sorted(outbound),
    }


def snippet_for_line(path: Path, line_no: int, radius: int = 2) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if line_no <= 0 or line_no > len(lines):
        return ""
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    snippet_lines = []
    for i in range(start, end + 1):
        snippet_lines.append(f"{i}: {lines[i - 1]}")
    return "\n".join(snippet_lines)


def expand(
    ir: Dict,
    node_id: str,
    *,
    budget_tokens: int,
    include: List[str],
    repo_root: Optional[Path] = None,
    flows_limit: int = 5,
) -> Dict[str, object]:
    fid = resolve_file_id(ir, node_id)
    if fid:
        file_node = ir["files"].get(fid, {})
        lines = [f"[FILE] {file_node.get('path')}"]
        summary = file_node.get("summary_1l")
        if summary:
            lines.append(f"summary={summary}")
        lines.append(f"role={file_node.get('role')}")
        used = estimate_tokens("\n".join(lines))

        if "symbols" in include:
            for sid in top_symbols_for_file(ir, fid, 12):
                sym = ir["symbols"].get(sid, {})
                signature = sym.get("signature") or sym.get("name")
                line = f"+ {signature}"
                projected = estimate_tokens("\n".join(lines + [line]))
                if projected > budget_tokens:
                    break
                lines.append(line)
                used = projected

        if "deps" in include:
            deps_info = deps(ir, fid, depth=1)
            for label, items in deps_info.items():
                if not items:
                    continue
                line = f"{label}={','.join(items[:10])}"
                projected = estimate_tokens("\n".join(lines + [line]))
                if projected > budget_tokens:
                    break
                lines.append(line)
                used = projected

        if "flows" in include or "dataflow" in include:
            in_edges, out_edges = dataflow_for_path(ir, file_node.get("path", ""))
            line = f"dataflow_in={len(in_edges)} dataflow_out={len(out_edges)}"
            projected = estimate_tokens("\n".join(lines + [line]))
            if projected <= budget_tokens:
                lines.append(line)
                used = projected
            for edge in out_edges[:flows_limit]:
                flow_line = f"flow_out L{edge.get('from_line', 0)}->{edge.get('to_path')}:{edge.get('to_line', 0)}"
                projected = estimate_tokens("\n".join(lines + [flow_line]))
                if projected > budget_tokens:
                    break
                lines.append(flow_line)
                used = projected
            for edge in in_edges[:flows_limit]:
                flow_line = (
                    f"flow_in {edge.get('from_path')}:{edge.get('from_line', 0)}"
                    f"->L{edge.get('to_line', 0)}"
                )
                projected = estimate_tokens("\n".join(lines + [flow_line]))
                if projected > budget_tokens:
                    break
                lines.append(flow_line)
                used = projected

        if "snippets" in include and repo_root:
            for sid in top_symbols_for_file(ir, fid, 3):
                sym = ir["symbols"].get(sid, {})
                line_no = sym.get("defined_in", {}).get("line", 0)
                snippet = snippet_for_line(repo_root / file_node.get("path", ""), line_no)
                if snippet:
                    block = "[SNIP]\n" + snippet
                    projected = estimate_tokens("\n".join(lines + [block]))
                    if projected > budget_tokens:
                        break
                    lines.append(block)
                    used = projected

        text = "\n".join(lines)
        return {"id": fid, "text": text, "tokens_used": used}

    sid = resolve_symbol_id(ir, node_id)
    if sid:
        sym = ir["symbols"].get(sid, {})
        lines = [f"[SYMBOL] {sym.get('name')}"]
        signature = sym.get("signature")
        if signature:
            lines.append(f"signature={signature}")
        lines.append(f"defined_in={sym.get('defined_in', {}).get('path')}:{sym.get('defined_in', {}).get('line')}")
        lines.append(f"visibility={sym.get('visibility')}")
        used = estimate_tokens("\n".join(lines))

        if "callers" in include or "callees" in include:
            callers = []
            callees = []
            for edge in ir.get("edges", {}).get("symbol_ref", []):
                if edge.get("to") == sid:
                    callers.append(edge.get("from"))
                if edge.get("from") == sid:
                    callees.append(edge.get("to"))
            if "callers" in include:
                line = f"callers={','.join(callers[:10])}"
                projected = estimate_tokens("\n".join(lines + [line]))
                if projected <= budget_tokens:
                    lines.append(line)
                    used = projected
            if "callees" in include:
                line = f"callees={','.join(callees[:10])}"
                projected = estimate_tokens("\n".join(lines + [line]))
                if projected <= budget_tokens:
                    lines.append(line)
                    used = projected
        if "flows" in include or "dataflow" in include:
            def_path = sym.get("defined_in", {}).get("path", "")
            def_line = sym.get("defined_in", {}).get("line", 0)
            in_edges, out_edges = dataflow_for_path(ir, def_path)
            if def_line:
                in_edges = [edge for edge in in_edges if edge.get("to_line") == def_line]
                out_edges = [edge for edge in out_edges if edge.get("from_line") == def_line]
            line = f"dataflow_in={len(in_edges)} dataflow_out={len(out_edges)}"
            projected = estimate_tokens("\n".join(lines + [line]))
            if projected <= budget_tokens:
                lines.append(line)
                used = projected
        text = "\n".join(lines)
        return {"id": sid, "text": text, "tokens_used": used}

    return {"error": "node_id not found", "id": node_id}
