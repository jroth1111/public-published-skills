from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import hashlib


def pagerank(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
    *,
    damping: float = 0.85,
    iterations: int = 20,
) -> Dict[str, float]:
    if not nodes:
        return {}
    scores = {node: 1.0 / len(nodes) for node in nodes}
    outgoing: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for src, dst, weight in edges:
        outgoing[src].append((dst, weight))
        incoming[dst].append((src, weight))
    for _ in range(iterations):
        new_scores = {}
        for node in nodes:
            rank_sum = 0.0
            for src, weight in incoming.get(node, []):
                out_weight = sum(w for _, w in outgoing.get(src, [])) or 1.0
                rank_sum += scores.get(src, 0.0) * (weight / out_weight)
            new_scores[node] = (1.0 - damping) / len(nodes) + damping * rank_sum
        scores = new_scores
    return scores


def score_files(ir: Dict) -> None:
    files = ir.get("files", {})
    edges = ir.get("edges", {}).get("file_dep", [])
    nodes = list(files.keys())
    edge_list = [(e["from"], e["to"], float(e.get("weight", 1.0))) for e in edges]
    signature = hashlib.sha1()
    signature.update(str(len(nodes)).encode("utf-8"))
    for src, dst, weight in edge_list:
        signature.update(f"{src}->{dst}:{weight}\n".encode("utf-8"))
    signature_value = signature.hexdigest()
    cached = ir.get("meta", {}).get("rank_cache", {}) if isinstance(ir.get("meta"), dict) else {}
    pr: Dict[str, float] = {}
    if isinstance(cached, dict) and cached.get("signature") == signature_value:
        cached_scores = cached.get("scores", {})
        if isinstance(cached_scores, dict):
            pr = {node: float(score) for node, score in cached_scores.items()}
    if not pr:
        pr = pagerank(nodes, edge_list)
        ir.setdefault("meta", {})["rank_cache"] = {
            "signature": signature_value,
            "scores": pr,
        }

    symbols = ir.get("symbols", {})
    call_edges = ir.get("edges", {}).get("symbol_ref", [])
    call_in: Dict[str, int] = defaultdict(int)
    call_out: Dict[str, int] = defaultdict(int)
    for edge in call_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in symbols:
            src_path = symbols[src].get("defined_in", {}).get("path")
            if src_path:
                call_out[f"file:{src_path}"] += 1
        if dst in symbols:
            dst_path = symbols[dst].get("defined_in", {}).get("path")
            if dst_path:
                call_in[f"file:{dst_path}"] += 1
    call_values = list(call_in.values()) + list(call_out.values()) + [0]
    max_calls = max(call_values) or 1

    max_exports = max((len(f.get("exports", [])) for f in files.values()), default=1)
    for fid, node in files.items():
        export_bonus = len(node.get("exports", [])) / max_exports if max_exports else 0.0
        entry_bonus = 0.2 if node.get("role") == "entrypoint" else 0.0
        call_bonus = (call_in.get(fid, 0) + call_out.get(fid, 0)) / max_calls
        base = pr.get(fid, 0.0)
        node["score"] = 0.6 * base + 0.2 * export_bonus + 0.1 * entry_bonus + 0.1 * call_bonus


def score_symbols(ir: Dict) -> None:
    symbols = ir.get("symbols", {})
    edges = ir.get("edges", {}).get("symbol_ref", [])
    in_count: Dict[str, int] = defaultdict(int)
    out_count: Dict[str, int] = defaultdict(int)
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src:
            out_count[src] += 1
        if dst:
            in_count[dst] += 1

    kind_weight = {
        "class": 1.2,
        "component": 1.2,
        "function": 1.0,
        "interface": 0.9,
        "type": 0.9,
        "enum": 0.9,
        "const": 0.7,
    }

    max_in = max(in_count.values(), default=1)
    for sid, node in symbols.items():
        weight = kind_weight.get(node.get("kind", ""), 0.8)
        exported = node.get("visibility") == "public"
        export_bonus = 0.3 if exported else 0.0
        refs_in = in_count.get(sid, 0)
        node["refs_in_count"] = refs_in
        node["refs_out_count"] = out_count.get(sid, 0)
        ref_bonus = (refs_in / max_in) if max_in else 0.0
        node["score"] = weight + export_bonus + 0.2 * ref_bonus
