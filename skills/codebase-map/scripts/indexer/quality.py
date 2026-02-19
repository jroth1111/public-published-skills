from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple


def build_test_mapping(
    ir: Dict[str, Any],
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    files = ir.get("files", {})
    edges = ir.get("edges", {}).get("file_dep", [])
    symbol_edges = ir.get("edges", {}).get("symbol_ref", [])
    symbols = ir.get("symbols", {})
    path_by_fid = {fid: node.get("path", "") for fid, node in files.items()}
    fid_by_path = {node.get("path", ""): fid for fid, node in files.items() if node.get("path")}

    test_fids = {fid for fid, node in files.items() if node.get("role") == "test"}
    targets_by_test: Dict[str, Set[str]] = defaultdict(set)
    symbol_targets_by_test: Dict[str, Set[str]] = defaultdict(set)
    tested_symbols: Dict[str, Set[str]] = defaultdict(set)

    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in test_fids and dst in files:
            targets_by_test[src].add(dst)

    test_paths = {path_by_fid.get(fid, "") for fid in test_fids}
    test_symbol_ids = {
        sid
        for sid, sym in symbols.items()
        if sym.get("defined_in", {}).get("path") in test_paths
    }
    for edge in symbol_edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src not in test_symbol_ids or dst not in symbols:
            continue
        dst_path = symbols.get(dst, {}).get("defined_in", {}).get("path", "")
        if dst_path in test_paths:
            continue
        test_path = symbols.get(src, {}).get("defined_in", {}).get("path", "")
        test_fid = fid_by_path.get(test_path)
        if test_fid:
            symbol_targets_by_test[test_fid].add(dst)
            tested_symbols[dst].add(test_path)

    for test_fid, sym_ids in symbol_targets_by_test.items():
        for sym_id in sym_ids:
            dst_path = symbols.get(sym_id, {}).get("defined_in", {}).get("path", "")
            if dst_path:
                dst_fid = fid_by_path.get(dst_path)
                if dst_fid:
                    targets_by_test[test_fid].add(dst_fid)

    def candidate_targets(test_path: str) -> List[str]:
        candidates: List[str] = []
        if not test_path:
            return candidates
        base = re.sub(r"(\\.test|\\.spec)(?=\\.)", "", test_path)
        base = base.replace("__tests__/", "")
        candidates.append(base)
        if base.startswith("tests/"):
            candidates.append(base[len("tests/") :])
        if base.startswith("test/"):
            candidates.append(base[len("test/") :])
        candidates.append(base.replace("/tests/", "/src/").replace("/test/", "/src/"))
        filename = Path(base).name
        for path in fid_by_path.keys():
            if path.endswith("/" + filename):
                candidates.append(path)
        return candidates

    for fid in test_fids:
        if targets_by_test.get(fid):
            continue
        test_path = path_by_fid.get(fid, "")
        for candidate in candidate_targets(test_path):
            target_fid = fid_by_path.get(candidate)
            if target_fid:
                targets_by_test[fid].add(target_fid)
        if not targets_by_test.get(fid):
            continue

    test_mapping: List[Dict[str, object]] = []
    tested_by: Dict[str, Set[str]] = defaultdict(set)
    for test_fid, targets in targets_by_test.items():
        if not targets:
            continue
        test_path = path_by_fid.get(test_fid, "")
        target_paths = [path_by_fid.get(t, "") for t in targets if path_by_fid.get(t)]
        if not test_path or not target_paths:
            continue
        for target_fid in targets:
            tested_by[target_fid].add(test_fid)
        symbols_for_test = []
        for sym_id in sorted(symbol_targets_by_test.get(test_fid, [])):
            sym_name = symbols.get(sym_id, {}).get("name")
            if sym_name:
                symbols_for_test.append(sym_name)
        entry: Dict[str, object] = {
            "test": test_path,
            "targets": sorted(set(target_paths)),
        }
        if symbols_for_test:
            entry["symbols"] = sorted(set(symbols_for_test))[:6]
        test_mapping.append(entry)

    for target_fid, tests in tested_by.items():
        node = files.get(target_fid)
        if not node:
            continue
        node["tested_by"] = sorted(tests)

    for sym_id, tests in tested_symbols.items():
        sym = symbols.get(sym_id)
        if not sym:
            continue
        sym["tested_by"] = sorted(set(tests))

    orphan_tests = [
        path_by_fid.get(fid, "")
        for fid in sorted(test_fids)
        if not targets_by_test.get(fid)
    ]
    entrypoint_fids = {fid for fid, node in files.items() if node.get("role") == "entrypoint"}
    tested_entrypoints = {
        fid for fid, tests in tested_by.items() if fid in entrypoint_fids and tests
    }
    untested_entrypoints = [
        path_by_fid.get(fid, "")
        for fid in sorted(entrypoint_fids)
        if fid not in tested_entrypoints
    ]
    summary = {
        "orphan_tests": [path for path in orphan_tests if path],
        "untested_entrypoints": [path for path in untested_entrypoints if path],
    }

    return sorted(test_mapping, key=lambda item: item.get("test", "")), summary


def compute_coupling_metrics(
    ir: Dict[str, Any],
    *,
    top_n: int = 6,
    min_edges: int = 2,
) -> Dict[str, List[Dict[str, object]]]:
    files = ir.get("files", {})
    edges = ir.get("edges", {}).get("file_dep", [])
    inbound: Dict[str, int] = defaultdict(int)
    outbound: Dict[str, int] = defaultdict(int)
    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in files:
            outbound[src] += 1
        if dst in files:
            inbound[dst] += 1
    fan_in = sorted(inbound.items(), key=lambda item: item[1], reverse=True)[:top_n]
    fan_out = sorted(outbound.items(), key=lambda item: item[1], reverse=True)[:top_n]
    instability: List[Tuple[str, float, int]] = []
    for fid in files.keys():
        total = inbound.get(fid, 0) + outbound.get(fid, 0)
        if total < min_edges:
            continue
        value = outbound.get(fid, 0) / total if total else 0.0
        instability.append((fid, value, total))
    instability.sort(key=lambda item: item[1], reverse=True)
    instability = instability[:top_n]
    return {
        "fan_in": [{"id": fid, "count": count} for fid, count in fan_in if count > 0],
        "fan_out": [{"id": fid, "count": count} for fid, count in fan_out if count > 0],
        "instability": [
            {"id": fid, "value": round(value, 3), "total": total}
            for fid, value, total in instability
        ],
    }


def build_trace_links(
    docs: Sequence[str],
    entities: Dict[str, Dict[str, object]],
    files: Dict[str, Dict[str, object]],
    *,
    max_links_per_doc: int = 2,
) -> List[Dict[str, object]]:
    links: List[Dict[str, object]] = []
    if not docs:
        return links

    def tokenize(text: str) -> Set[str]:
        return {
            token
            for token in re.split(r"[^A-Za-z0-9]+", text.lower())
            if len(token) >= 3
        }

    entity_tokens: Dict[str, Set[str]] = {}
    for ent_id, node in entities.items():
        name = str(node.get("name") or "")
        if not name:
            continue
        entity_tokens[ent_id] = tokenize(name)

    file_tokens: Dict[str, Set[str]] = {}
    for fid, node in files.items():
        path = str(node.get("path") or "")
        if not path:
            continue
        base = Path(path).stem
        file_tokens[fid] = tokenize(base)

    for doc in docs:
        doc_tokens = tokenize(Path(doc).stem)
        if not doc_tokens:
            continue
        scored: List[Tuple[float, str, str]] = []
        for ent_id, tokens in entity_tokens.items():
            overlap = doc_tokens & tokens
            if overlap:
                score = len(overlap) / max(1, len(doc_tokens))
                scored.append((score, ent_id, "entity"))
        for fid, tokens in file_tokens.items():
            overlap = doc_tokens & tokens
            if overlap:
                score = len(overlap) / max(1, len(doc_tokens))
                scored.append((score, fid, "file"))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for score, target_id, target_type in scored[:max_links_per_doc]:
            if score < 0.2:
                continue
            links.append(
                {
                    "doc_path": doc,
                    "target_id": target_id,
                    "target_type": target_type,
                    "score": round(score, 3),
                    "reason": "name_overlap",
                }
            )
    return links
