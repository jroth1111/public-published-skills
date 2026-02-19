#!/usr/bin/env python3
"""Repo mapper CLI: build IR, rank, and pack into token budgets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from _fs import safe_preview_text, write_json, write_lines, write_text, workspace_root
from exporter import (
    build_graph,
    export_graph_json,
    export_graphml,
    normalize_edge_types,
)
from indexer import IndexOptions, index_repo, detect_framework, diff_ir, deep_copy_ir
from ir import load_ir, save_ir
from packer import deps as deps_view
from packer import digest as digest_view
from packer import expand as expand_view
from packer import find_symbol as find_symbol_view
from packer import query_symbol as query_symbol_view
from packer import search_repo as search_repo_view
from ranker import score_files, score_symbols
from utils import ToolState
from .profiles import apply_profile, apply_quick_preset
from .config import auto_precise_tokens, parse_languages, resolve_out_dir


DIGEST_CACHE_VERSION = 4


def top_counts(values: Sequence[str], max_items: int) -> List[str]:
    counts = Counter(values)
    return [f"{name}: {count}" for name, count in counts.most_common(max_items)]


def write_tsv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\t".join(row))
            handle.write("\n")


def file_sha1(path: Path) -> str:
    import hashlib

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def packer_package_hash() -> Optional[str]:
    packer_dir = Path(__file__).with_name("packer")
    if packer_dir.is_dir():
        hashes: List[str] = []
        for path in sorted(packer_dir.rglob("*.py")):
            try:
                hashes.append(file_sha1(path))
            except OSError:
                continue
        if hashes:
            import hashlib

            combined = "\n".join(hashes).encode("utf-8")
            return hashlib.sha1(combined).hexdigest()
    packer_path = Path(__file__).with_name("packer.py")
    if packer_path.exists():
        return file_sha1(packer_path)
    return None


def build_summary(
    *,
    files: List[str],
    docs: List[str],
    configs: List[str],
    tests: List[str],
    ir: Dict,
    warnings: List[str],
    tool_versions: Dict[str, str],
    max_sample: int,
    explain_ranking: bool = False,
) -> str:
    summary_lines: List[str] = []
    summary_lines.append(f"FILE_COUNT: {len(files)}")

    top_dirs = top_counts([p.split("/")[0] if "/" in p else "." for p in files], 12)
    summary_lines.append("TOP_DIRS:")
    summary_lines.extend(f"  {line}" for line in top_dirs)

    extensions = []
    for path in files:
        if "." in Path(path).name:
            extensions.append(Path(path).suffix.lstrip("."))
        else:
            extensions.append("(none)")
    top_exts = top_counts(extensions, 12)
    summary_lines.append("FILE_TYPES:")
    summary_lines.extend(f"  {line}" for line in top_exts)

    summary_lines.append("DOCS_SAMPLE:")
    summary_lines.extend(f"  {p}" for p in docs[:max_sample])
    summary_lines.append("CONFIGS_SAMPLE:")
    summary_lines.extend(f"  {p}" for p in configs[:max_sample])
    summary_lines.append("TESTS_SAMPLE:")
    summary_lines.extend(f"  {p}" for p in tests[:max_sample])

    unsupported_langs = ir.get("meta", {}).get("unsupported_languages", {})
    if isinstance(unsupported_langs, dict) and unsupported_langs:
        summary_lines.append("UNSUPPORTED_LANGS:")
        for name, count in list(unsupported_langs.items())[:max_sample]:
            summary_lines.append(f"  {name}: {count}")

    monorepo = ir.get("meta", {}).get("monorepo", {})
    if isinstance(monorepo, dict) and monorepo:
        packages = monorepo.get("packages", [])
        package_count = len(
            [pkg for pkg in packages if pkg.get("path") not in {".", ""}]
        ) or len(packages)
        summary_lines.append(
            f"MONOREPO: type={monorepo.get('type', 'workspace')} packages={package_count}"
        )

    file_edges = ir.get("edges", {}).get("file_dep", [])
    symbol_edges = ir.get("edges", {}).get("symbol_ref", [])
    flow_edges = ir.get("edges", {}).get("dataflow", [])
    summary_lines.append(f"IMPORT_EDGES: {len(file_edges)}")
    summary_lines.append(f"CALL_EDGES: {len(symbol_edges)}")
    summary_lines.append(f"DATAFLOW_EDGES: {len(flow_edges)}")
    summary_lines.append(
        f"EXPORT_SYMBOLS: {sum(len(f.get('exports', [])) for f in ir.get('files', {}).values())}"
    )

    codeql_meta = ir.get("meta", {}).get("codeql", {})
    if codeql_meta:
        summary_lines.append("CODEQL:")
        for key, value in codeql_meta.get("counts", {}).items():
            summary_lines.append(f"  {key}: {value}")

    if tool_versions:
        summary_lines.append("TOOL_VERSIONS:")
        for key, value in sorted(tool_versions.items()):
            summary_lines.append(f"  {key}: {value}")

    if warnings:
        summary_lines.append("WARNINGS:")
        for warning in warnings[:max_sample]:
            summary_lines.append(f"  {warning}")

    if explain_ranking:
        summary_lines.append("RANKING_FACTORS:")
        summary_lines.append(
            "  Files: PageRank (60%) + exports (20%) + entrypoint bonus (10%) + call activity (10%)"
        )
        summary_lines.append(
            "  Symbols: kind weight + export bonus (0.3) + reference bonus (20% of inbound refs)"
        )
        summary_lines.append("  Top 10 files by score:")
        files_sorted = sorted(
            [
                (fid, node.get("score", 0), node.get("path", "unknown"))
                for fid, node in ir.get("files", {}).items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for fid, score, path in files_sorted:
            summary_lines.append(f"    {score:.3f} {path}")
        symbols_sorted = sorted(
            [
                (sid, node.get("score", 0), node.get("name", "unknown"))
                for sid, node in ir.get("symbols", {}).items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        if symbols_sorted:
            summary_lines.append("  Top 10 symbols by score:")
            for sid, score, name in symbols_sorted:
                sym_node = ir.get("symbols", {}).get(sid, {})
                kind = sym_node.get("kind", "")
                summary_lines.append(f"    {score:.3f} {name} ({kind})")

    quality = ir.get("meta", {}).get("quality", {})
    if isinstance(quality, dict) and quality:
        summary_lines.append("QUALITY:")
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
                summary_lines.append(f"  {key}: {quality.get(key)}")

    return "\n".join(summary_lines)


def write_artifacts(
    out_dir: Path,
    *,
    files: List[str],
    docs: List[str],
    configs: List[str],
    tests: List[str],
    packages: List[List[str]],
    ir: Dict,
    mode: str = "standard",
) -> None:
    if mode == "minimal":
        return

    write_lines(str(out_dir / "files.txt"), files)
    write_lines(str(out_dir / "docs.txt"), docs)
    write_lines(str(out_dir / "configs.txt"), configs)
    write_lines(str(out_dir / "tests.txt"), tests)
    if packages and mode in {"standard", "full"}:
        write_tsv(out_dir / "packages.tsv", packages)

    imports_rows: List[List[str]] = []
    for fnode in ir.get("files", {}).values():
        for module in fnode.get("imports", []):
            imports_rows.append([fnode.get("path", ""), module])
    if imports_rows and mode in {"standard", "full"}:
        write_tsv(out_dir / "imports.tsv", imports_rows)

    reexport_rows: List[List[str]] = []
    for fnode in ir.get("files", {}).values():
        for module in fnode.get("re_exports", []):
            reexport_rows.append([fnode.get("path", ""), module])
    if reexport_rows and mode in {"standard", "full"}:
        write_tsv(out_dir / "reexports.tsv", reexport_rows)

    export_rows: List[List[str]] = []
    symbols = ir.get("symbols", {})
    for fnode in ir.get("files", {}).values():
        for sid in fnode.get("exports", []):
            sym = symbols.get(sid, {})
            export_rows.append(
                [
                    fnode.get("path", ""),
                    sym.get("kind", ""),
                    sym.get("name", ""),
                    sym.get("signature", ""),
                ]
            )
    if export_rows and mode in {"standard", "full"}:
        write_tsv(out_dir / "exports.tsv", export_rows)

    py_rows: List[List[str]] = []
    ts_rows: List[List[str]] = []
    for sym in symbols.values():
        defined_in = sym.get("defined_in", {}).get("path", "")
        if defined_in.endswith(".py"):
            py_rows.append([defined_in, sym.get("kind", ""), sym.get("name", "")])
        elif defined_in.endswith(".ts") or defined_in.endswith(".tsx"):
            ts_rows.append([defined_in, sym.get("kind", ""), sym.get("name", "")])
    if mode == "full":
        if py_rows:
            write_tsv(out_dir / "py_symbols.tsv", py_rows)
        if ts_rows:
            write_tsv(out_dir / "ts_symbols.tsv", ts_rows)

    flow_rows: List[List[str]] = []
    for edge in ir.get("edges", {}).get("dataflow", []):
        flow_rows.append(
            [
                edge.get("from_path", ""),
                str(edge.get("from_line", "")),
                edge.get("to_path", ""),
                str(edge.get("to_line", "")),
            ]
        )
    if flow_rows and mode == "full":
        write_tsv(out_dir / "dataflow.tsv", flow_rows)


def check_ir_stale(repo: Path, out_dir: Path) -> Tuple[bool, int]:
    """Check if IR is stale by comparing file hashes. Returns (is_stale, num_changed_files)."""
    ir_path = out_dir / "ir.json"
    if not ir_path.exists():
        return True, 0

    try:
        prev_ir = load_ir(ir_path)
        if not prev_ir:
            return True, 0

        prev_hashes = prev_ir.get("meta", {}).get("file_hashes", {})

        from indexer import list_repo_files, ToolState

        files = list_repo_files(repo, [], ToolState())
        current_hashes = {path: hash_file(repo / path) for path in files}

        changed_files = [
            path
            for path, digest in current_hashes.items()
            if prev_hashes.get(path) != digest
        ]

        return len(changed_files) > 0, len(changed_files)
    except Exception:
        return True, 0


def timing_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "timing", False))


def run_index(
    args: argparse.Namespace,
    *,
    emit_summary: bool = True,
    explain_ranking: bool = False,
    auto_refresh: bool = False,
) -> Dict[str, object]:
    import time

    repo = Path(args.repo).resolve()
    out_dir = resolve_out_dir(repo, args.out, workspace_root=workspace_root())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Timing tracking
    timing = {}
    if timing_enabled(args):
        start_total = time.time()

    if auto_refresh:
        is_stale, num_changed = check_ir_stale(repo, out_dir)
        if is_stale:
            if num_changed > 0:
                print(
                    f"[auto-refresh] Detected {num_changed} changed files; re-indexing...",
                    file=sys.stderr,
                )
            else:
                print("[auto-refresh] IR is stale; re-indexing...", file=sys.stderr)
        else:
            print(
                "[auto-refresh] IR is up to date; skipping re-index.", file=sys.stderr
            )
            if (out_dir / "ir.json").exists():
                ir = load_ir(out_dir / "ir.json")
                if emit_summary:
                    summary_path = out_dir / "summary.txt"
                    if summary_path.exists():
                        summary = summary_path.read_text(encoding="utf-8")
                        preview = safe_preview_text(summary, max_bytes=900)
                        print(preview)
                return {"ir": ir, "out_dir": out_dir}

    options = IndexOptions(
        languages=parse_languages(args.languages),
        ast=args.ast != "off",
        codeql=args.codeql != "off",
        ast_mode=args.ast,
        codeql_mode=args.codeql,
        ast_auto_max_files=args.ast_auto_max_files,
        codeql_auto_max_files=args.codeql_auto_max_files,
        codeql_auto_max_mb=args.codeql_auto_max_mb,
        include_internal=args.include_internal,
        incremental=args.incremental,
        reuse_codeql_db=args.reuse_codeql_db,
        keep_codeql_db=args.keep_codeql_db,
        codeql_db=Path(args.codeql_db) if args.codeql_db else None,
        max_sample=args.max_sample,
        workers=max(1, int(getattr(args, "workers", 1))),
        semantic_hash_cache=bool(getattr(args, "semantic_hash_cache", True)),
        api_contracts=bool(getattr(args, "api_contracts", False)),
    )

    tools = ToolState()
    prev_ir = load_ir(out_dir / "ir.json") if args.incremental else None

    if timing_enabled(args):
        start_index = time.time()

    result = index_repo(repo, out_dir, options, tools, prev_ir=prev_ir)
    ir = result["ir"]

    if timing_enabled(args):
        timing["index_repo"] = time.time() - start_index
        start_scoring = time.time()

    score_files(ir)
    score_symbols(ir)

    if timing_enabled(args):
        timing["scoring"] = time.time() - start_scoring
        start_save = time.time()

    save_ir(out_dir / "ir.json", ir)

    if timing_enabled(args):
        timing["save_ir"] = time.time() - start_save
        start_artifacts = time.time()

    write_artifacts(
        out_dir,
        files=result["files"],
        docs=result["docs"],
        configs=result["configs"],
        tests=result["tests"],
        packages=result["packages"],
        ir=ir,
        mode=args.artifacts,
    )

    if timing_enabled(args):
        timing["write_artifacts"] = time.time() - start_artifacts
        start_summary = time.time()

    summary_text = build_summary(
        files=result["files"],
        docs=result["docs"],
        configs=result["configs"],
        tests=result["tests"],
        ir=ir,
        warnings=result["warnings"],
        tool_versions=result["tool_versions"],
        max_sample=args.max_sample,
        explain_ranking=explain_ranking,
    )
    write_text(str(out_dir / "summary.txt"), summary_text)
    write_json(
        str(out_dir / "summary.json"),
        {
            "counts": {
                "files": len(result["files"]),
                "imports": len(ir.get("edges", {}).get("file_dep", [])),
                "exports": sum(
                    len(f.get("exports", [])) for f in ir.get("files", {}).values()
                ),
                "dataflow": len(ir.get("edges", {}).get("dataflow", [])),
            },
            "unsupported_languages": ir.get("meta", {}).get(
                "unsupported_languages", {}
            ),
            "monorepo": ir.get("meta", {}).get("monorepo", {}),
            "warnings": result["warnings"],
            "tool_versions": result["tool_versions"],
            "quality": ir.get("meta", {}).get("quality", {}),
            "out_dir": out_dir.as_posix(),
        },
    )

    if timing_enabled(args):
        timing["build_summary"] = time.time() - start_summary
        timing["total"] = time.time() - start_total

    preview = safe_preview_text(summary_text, max_bytes=900)
    if emit_summary:
        print(preview)
        print(f"Artifacts: {out_dir}")

    if timing_enabled(args):
        print("\n[TIMING]", file=sys.stderr)
        total_time = timing.get("total", 0.0)
        print(f"  Total: {total_time:.2f}s", file=sys.stderr)
        for stage, stage_time in timing.items():
            if stage != "total":
                percent = (stage_time / total_time * 100) if total_time > 0 else 0.0
                print(f"  {stage}: {stage_time:.2f}s ({percent:.1f}%)", file=sys.stderr)

    return {"ir": ir, "out_dir": out_dir, "summary": summary_text}


def run_digest(
    args: argparse.Namespace, *, ir: Dict[str, object], out_dir: Path
) -> Dict[str, object]:
    if args.dense and args.dir_depth is None:
        args.dir_depth = 3
    precise_tokens = auto_precise_tokens(getattr(args, "precise_tokens", False))
    cache_key = None
    cache_path = out_dir / "digest-cache.json"
    ir_path = out_dir / "ir.json"
    graph_cache_hit = False
    graph_cache: Optional[Dict[str, object]] = None
    graph_cache_signature = None
    if ir_path.exists():
        try:
            packer_hash = packer_package_hash()
            payload = {
                "cache_version": DIGEST_CACHE_VERSION,
                "packer_hash": packer_hash,
                "ir": file_sha1(ir_path),
                "budget": args.budget,
                "focus": args.focus,
                "focus_depth": getattr(args, "focus_depth", 2),
                "max_files": args.max_files,
                "max_symbols": args.max_symbols,
                "max_edges": args.max_edges,
                "max_sig_len": args.max_sig_len,
                "entry_details_limit": args.entry_details_limit,
                "entry_details_per_kind": args.entry_details_per_kind,
                "code_only": args.code_only,
                "compress_paths": args.compress_paths,
                "include_routes": args.include_routes,
                "routes_mode": args.routes_mode,
                "routes_limit": args.routes_limit,
                "routes_format": args.routes_format,
                "explain_ranking": args.explain_ranking,
                "dense": args.dense,
                "dir_depth": args.dir_depth,
                "dir_file_cap": args.dir_file_cap,
                "budget_margin": args.budget_margin,
                "diagrams": args.diagrams,
                "diagram_depth": args.diagram_depth,
                "diagram_nodes": args.diagram_nodes,
                "diagram_edges": args.diagram_edges,
                "flow_symbols": getattr(args, "flow_symbols", False),
                "precise_tokens": precise_tokens,
                "semantic_cluster_mode": getattr(args, "semantic_cluster_mode", "tfidf"),
                "semantic_cluster": getattr(args, "semantic_cluster", False),
                "doc_quality": getattr(args, "doc_quality", False),
                "doc_quality_strict": getattr(args, "doc_quality_strict", False),
                "semantic_call_weight": getattr(args, "semantic_call_weight", 0.3),
                "dead_code": getattr(args, "dead_code", False),
                "call_chains": getattr(args, "call_chains", False),
                "static_traces": getattr(args, "static_traces", False),
                "trace_depth": getattr(args, "trace_depth", 4),
                "trace_max": getattr(args, "trace_max", 10),
                "trace_direction": getattr(args, "trace_direction", "both"),
                "trace_start": getattr(args, "trace_start", None),
                "trace_end": getattr(args, "trace_end", None),
                "graph_cache": getattr(args, "graph_cache", True),
                "section_budgets": getattr(args, "section_budgets", None),
                "entity_graph": getattr(args, "entity_graph", True),
                "traceability": getattr(args, "traceability", False),
                "traceability_targets": getattr(args, "traceability_targets", "all"),
            }
            cache_key = json.dumps(payload, sort_keys=True)
        except OSError:
            cache_key = None
    section_budgets = None
    raw_section_budgets = getattr(args, "section_budgets", None)
    if isinstance(raw_section_budgets, dict):
        section_budgets = raw_section_budgets
    elif raw_section_budgets:
        try:
            value = str(raw_section_budgets)
            if Path(value).exists():
                section_budgets = json.loads(Path(value).read_text(encoding="utf-8"))
            else:
                section_budgets = json.loads(value)
        except (json.JSONDecodeError, OSError, ValueError):
            section_budgets = None
    if getattr(args, "graph_cache", True) and ir_path.exists():
        graph_cache_signature = file_sha1(ir_path)
        graph_cache_path = out_dir / "graph-cache.json"
        if graph_cache_path.exists():
            try:
                cached = json.loads(graph_cache_path.read_text(encoding="utf-8"))
                if cached.get("signature") == graph_cache_signature:
                    graph_cache = cached.get("graph")
                    graph_cache_hit = True
            except json.JSONDecodeError:
                graph_cache = None
        if not graph_cache:
            file_edges = ir.get("edges", {}).get("file_dep", [])
            adj: Dict[str, List[str]] = {}
            inbound: Counter[str] = Counter()
            outbound: Counter[str] = Counter()
            for edge in file_edges:
                src = edge.get("from")
                dst = edge.get("to")
                if not src or not dst:
                    continue
                adj.setdefault(src, []).append(dst)
                outbound[src] += 1
                inbound[dst] += 1
            graph_cache = {
                "file_adj": adj,
                "inbound": dict(inbound),
                "outbound": dict(outbound),
            }
            graph_cache_path.write_text(
                json.dumps(
                    {"signature": graph_cache_signature, "graph": graph_cache},
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
    if cache_key and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = None
        if isinstance(cache, dict) and cache.get("key") == cache_key:
            digest = cache.get("digest")
            if isinstance(digest, dict) and digest.get("digest"):
                digest_json_path = out_dir / "digest.json"
                digest_text_path = out_dir / "digest.txt"
                if not digest_json_path.exists():
                    write_json(str(digest_json_path), digest)
                if not digest_text_path.exists():
                    write_text(str(digest_text_path), str(digest.get("digest", "")))
                if args.format == "text":
                    print(digest["digest"])
                else:
                    print(json.dumps(digest, ensure_ascii=True, indent=2))
                return digest
    stream_output = bool(getattr(args, "stream_digest", False)) and args.format == "text"
    digest = digest_view(
        ir,
        args.budget,
        focus=args.focus,
        focus_depth=getattr(args, "focus_depth", 2),
        flow_symbols=getattr(args, "flow_symbols", False),
        precise_tokens=precise_tokens,
        semantic_cluster_mode=getattr(args, "semantic_cluster_mode", "tfidf"),
        semantic_cluster=getattr(args, "semantic_cluster", False),
        doc_quality=getattr(args, "doc_quality", False),
        doc_quality_strict=getattr(args, "doc_quality_strict", False),
        semantic_call_weight=getattr(args, "semantic_call_weight", 0.3),
        dead_code=getattr(args, "dead_code", False),
        call_chains=getattr(args, "call_chains", False),
        static_traces=getattr(args, "static_traces", False),
        trace_depth=getattr(args, "trace_depth", 4),
        trace_max=getattr(args, "trace_max", 10),
        trace_direction=getattr(args, "trace_direction", "both"),
        trace_start=getattr(args, "trace_start", None),
        trace_end=getattr(args, "trace_end", None),
        graph_cache=graph_cache,
        graph_cache_hit=graph_cache_hit,
        graph_cache_signature=graph_cache_signature,
        section_budgets=section_budgets,
        entity_graph=getattr(args, "entity_graph", True),
        traceability=getattr(args, "traceability", False),
        traceability_targets=getattr(args, "traceability_targets", "all"),
        max_files=args.max_files,
        max_symbols_per_file=args.max_symbols,
        max_edges=args.max_edges,
        max_sig_len=args.max_sig_len,
        entry_details_limit=args.entry_details_limit,
        entry_details_per_kind=args.entry_details_per_kind,
        code_only=args.code_only,
        compress_paths=args.compress_paths,
        include_routes=args.include_routes,
        routes_mode=args.routes_mode,
        routes_limit=args.routes_limit,
        routes_format=args.routes_format,
        explain_scores=args.explain_ranking,
        dense=args.dense,
        dir_depth=args.dir_depth,
        dir_file_cap=args.dir_file_cap,
        budget_margin=args.budget_margin,
        diagram_mode=args.diagrams,
        diagram_depth=args.diagram_depth,
        diagram_nodes=args.diagram_nodes,
        diagram_edges=args.diagram_edges,
        stream_output=stream_output,
    )
    digest_lines = digest.pop("lines", None)
    if digest_lines:
        digest_text = "\n".join(digest_lines)
        digest["digest"] = digest_text
        write_lines(str(out_dir / "digest.txt"), digest_lines)
        write_json(str(out_dir / "digest.json"), digest)
        if args.format == "text":
            print(digest_text)
        else:
            print(json.dumps(digest, ensure_ascii=True, indent=2))
    else:
        write_json(str(out_dir / "digest.json"), digest)
        write_text(str(out_dir / "digest.txt"), digest["digest"])
        if args.format == "text":
            print(digest["digest"])
        else:
            print(json.dumps(digest, ensure_ascii=True, indent=2))
    if cache_key:
        try:
            cache_path.write_text(
                json.dumps({"key": cache_key, "digest": digest}, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
    return digest


def apply_limit(items: List, limit: int) -> Tuple[List, bool]:
    if limit is None or limit <= 0:
        return items, False
    if len(items) <= limit:
        return items, False
    return items[:limit], True


def format_symbol_location(symbol: Dict[str, object]) -> str:
    name = str(symbol.get("name", "") or "")
    path = str(symbol.get("path", "") or "")
    line = symbol.get("line", 0) or 0
    location = f"{path}:{line}" if path else ""
    return f"{name} {location}".strip()




def load_or_index(
    args: argparse.Namespace,
    *,
    emit_summary: bool = True,
    explain_ranking: bool = False,
) -> Dict[str, object]:
    repo = Path(args.repo).resolve()
    out_dir = resolve_out_dir(repo, args.out, workspace_root=workspace_root())
    ir_path = out_dir / "ir.json"
    if ir_path.exists():
        ir = load_ir(ir_path)
        if ir:
            return {"ir": ir, "out_dir": out_dir}
    if args.auto_index:
        return run_index(
            args, emit_summary=emit_summary, explain_ranking=explain_ranking
        )
    raise FileNotFoundError(f"IR not found at {ir_path}")


def check_tool(name: str, version_cmd: Sequence[str] = None) -> Dict[str, object]:
    """Check if a tool is available and return its status."""
    import platform

    result = {"name": name, "available": False, "version": None}

    try:
        if version_cmd:
            proc = subprocess.run(
                version_cmd, capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0:
                result["available"] = True
                result["version"] = proc.stdout.splitlines()[0].strip()
        else:
            proc = subprocess.run(
                [name, "--version"], capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0:
                result["available"] = True
                result["version"] = proc.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return result


def run_install_guide(args: argparse.Namespace) -> None:
    """Check tool availability and print installation guidance."""
    import platform

    tools = [
        {
            "name": "fd",
            "cmd": ["fd", "--version"],
            "desc": "Fast file discovery (recommended)",
        },
        {
            "name": "rg",
            "cmd": ["rg", "--version"],
            "desc": "Ripgrep for file search (fallback)",
        },
        {
            "name": "ast-grep",
            "cmd": ["ast-grep", "--version"],
            "desc": "AST mapping for imports/exports",
        },
        {
            "name": "codeql",
            "cmd": ["codeql", "version"],
            "desc": "CodeQL for call/dataflow analysis (optional)",
        },
    ]

    results = []
    missing = []

    for tool_info in tools:
        result = check_tool(tool_info["name"], tool_info["cmd"])
        result["description"] = tool_info["desc"]
        results.append(result)
        if not result["available"]:
            missing.append(tool_info["name"])

    system = platform.system().lower()
    install_commands = []

    if system == "darwin":
        if "fd" in missing:
            install_commands.append("brew install fd")
        if "rg" in missing:
            install_commands.append("brew install ripgrep")
        if "ast-grep" in missing:
            install_commands.append("brew install ast-grep")
        if "codeql" in missing:
            install_commands.append("brew install codeql")
    elif system == "linux":
        if "fd" in missing:
            install_commands.append("cargo install fd-find  # or apt install fd-find")
        if "rg" in missing:
            install_commands.append("cargo install ripgrep  # or apt install ripgrep")
        if "ast-grep" in missing:
            install_commands.append("npm install -g ast-grep")
        if "codeql" in missing:
            install_commands.append(
                "# Download from: https://codeql.github.com/docs/codeql-cli/obtaining-codeql-cli"
            )
    elif system == "windows":
        if "fd" in missing:
            install_commands.append("winget install sharkdp.fd")
        if "rg" in missing:
            install_commands.append("winget install BurntSushi.ripgrep.MSVC")
        if "ast-grep" in missing:
            install_commands.append("npm install -g ast-grep")
        if "codeql" in missing:
            install_commands.append(
                "# Download from: https://codeql.github.com/docs/codeql-cli/obtaining-codeql-cli"
            )

    if args.format == "json":
        json_output = {
            "system": system,
            "tools": results,
            "missing": missing,
            "install_commands": install_commands,
        }
        print(json.dumps(json_output, indent=2))
        return

    # Print text output
    print("Tool Availability Check")
    print("=" * 50)
    for result in results:
        status = "✓" if result["available"] else "✗"
        version = f" ({result['version']})" if result["version"] else ""
        print(f"{status} {result['name']:12s}{version} - {result['description']}")
    print()

    if missing:
        print(f"Missing tools: {', '.join(missing)}")
        print()
        print("Installation commands for " + system.upper() + ":")
        for cmd in install_commands:
            print(f"  {cmd}")
    else:
        print("All tools are installed!")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Token-efficient repo mapper")
    parser.add_argument("--repo", default=".", help="Repo root (default: .)")
    parser.add_argument(
        "--out", default=None, help="Output dir (relative to workspace/ or absolute)"
    )
    parser.add_argument(
        "--languages", default="auto", help="Comma list: ts,tsx,js,jsx,py,go,rs or auto"
    )
    parser.add_argument("--ast", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--codeql", choices=["auto", "on", "off"], default="off")
    parser.add_argument(
        "--ast-auto-max-files",
        type=int,
        default=20000,
        help="Auto AST cutoff (max code files before disabling ast-grep)",
    )
    parser.add_argument(
        "--codeql-auto-max-files",
        type=int,
        default=6000,
        help="Auto CodeQL cutoff (max code files before disabling)",
    )
    parser.add_argument(
        "--codeql-auto-max-mb",
        type=int,
        default=200,
        help="Auto CodeQL cutoff in MB (0 disables size check)",
    )
    parser.add_argument(
        "--include-internal", action="store_true", help="Include internal symbols"
    )
    parser.add_argument(
        "--api-contracts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable API contract extraction during index",
    )
    parser.add_argument(
        "--incremental", action="store_true", help="Reuse previous IR when possible"
    )
    parser.add_argument(
        "--reuse-codeql-db",
        action="store_true",
        help="Reuse existing CodeQL DB if present",
    )
    parser.add_argument(
        "--keep-codeql-db", action="store_true", help="Keep CodeQL DBs in output dir"
    )
    parser.add_argument(
        "--codeql-db", default=None, help="Base directory for CodeQL DBs"
    )
    parser.add_argument("--max-sample", type=int, default=30, help="Summary sample cap")
    parser.add_argument(
        "--artifacts",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Artifact emission level (minimal, standard, full)",
    )
    parser.add_argument(
        "--budget-margin",
        type=float,
        default=1.0,
        help="Fraction of budget to cap at (use <1.0 for safety margin)",
    )
    default_workers = max(1, min(8, os.cpu_count() or 2))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel workers for AST extraction (default: CPU count capped at 8)",
    )
    parser.add_argument(
        "--semantic-hash-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache semantic hashes to speed incremental runs",
    )

    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser(
        "index", help="Build or refresh the repo index"
    )
    index_parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help="Automatically refresh if files have changed",
    )
    index_parser.add_argument(
        "--timing", action="store_true", help="Show timing breakdown of operations"
    )
    index_parser.add_argument(
        "--profile",
        action="store_true",
        dest="timing",
        help="Deprecated: use --timing for timing breakdown",
    )

    map_parser = subparsers.add_parser(
        "map", help="Index then emit a digest (one step)"
    )
    map_parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help="Automatically refresh if files have changed",
    )
    map_parser.add_argument(
        "--quick", action="store_true", help="Deprecated: use --profile fast"
    )
    map_parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "deep"],
        default=None,
        help="Preset tuning for speed/accuracy; overrides individual flags",
    )
    map_parser.add_argument("--budget", type=int, default=20000, help="Token budget")
    map_parser.add_argument("--focus", default=None, help="Focus query")
    map_parser.add_argument(
        "--focus-depth",
        type=int,
        default=2,
        help="Focus expansion depth (1=only focus, 2=neighbors, 3=2-hop)",
    )
    map_parser.add_argument("--format", choices=["json", "text"], default="text")
    map_parser.add_argument(
        "--gate-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when confidence gate fails",
    )
    map_parser.add_argument(
        "--flow-symbols",
        action="store_true",
        help="Include symbol names in flow traces (when symbol refs are available)",
    )
    map_parser.add_argument(
        "--precise-tokens",
        action="store_true",
        help="Use tiktoken for token estimation when available",
    )
    map_parser.add_argument(
        "--semantic-cluster",
        action="store_true",
        help="Use semantic-weighted component clustering in the digest",
    )
    map_parser.add_argument(
        "--semantic-cluster-mode",
        choices=["tfidf", "name_only"],
        default="tfidf",
        help="Semantic clustering mode (tfidf uses docs+symbols; name_only uses paths/names)",
    )
    map_parser.add_argument(
        "--doc-quality",
        action="store_true",
        help="Include docstring coverage/quality section in the digest",
    )
    map_parser.add_argument(
        "--doc-quality-strict",
        action="store_true",
        help="Enable stricter doc/signature consistency checks",
    )
    map_parser.add_argument(
        "--semantic-call-weight",
        type=float,
        default=0.3,
        help="Weight for call-graph affinity when semantic clustering (0-0.9)",
    )
    map_parser.add_argument(
        "--dead-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include dead-code heuristics section in the digest",
    )
    map_parser.add_argument(
        "--call-chains",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include multi-hop symbol call chains for entrypoints",
    )
    map_parser.add_argument(
        "--static-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include static trace paths and trace graph",
    )
    map_parser.add_argument(
        "--trace-depth",
        type=int,
        default=4,
        help="Max depth for static traces",
    )
    map_parser.add_argument(
        "--trace-max",
        type=int,
        default=10,
        help="Max number of static trace paths",
    )
    map_parser.add_argument(
        "--trace-direction",
        choices=["forward", "reverse", "both"],
        default="both",
        help="Direction for static traces",
    )
    map_parser.add_argument(
        "--trace-start",
        default=None,
        help="Trace start node (file path, alias, or entity name/id; comma list)",
    )
    map_parser.add_argument(
        "--trace-end",
        default=None,
        help="Trace end node (file path, alias, or entity name/id; comma list)",
    )
    map_parser.add_argument(
        "--graph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache derived graph data for faster digest reruns",
    )
    map_parser.add_argument(
        "--section-budgets",
        default=None,
        help="JSON section budget overrides (e.g. '{\"FILES\":0.4,\"EDGES\":0.1}')",
    )
    map_parser.add_argument(
        "--entity-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include entity graph-backed details when entities are available",
    )
    map_parser.add_argument(
        "--traceability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include doc-to-code traceability hints",
    )
    map_parser.add_argument(
        "--traceability-targets",
        choices=["all", "code", "entities"],
        default="all",
        help="Filter traceability targets (all|code|entities)",
    )
    map_parser.add_argument(
        "--stream-digest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream digest output (text only) to reduce peak memory",
    )
    map_parser.add_argument(
        "--max-files", type=int, default=None, help="Cap files in digest"
    )
    map_parser.add_argument(
        "--max-symbols", type=int, default=None, help="Cap symbols per file"
    )
    map_parser.add_argument(
        "--max-edges", type=int, default=None, help="Cap edges in digest"
    )
    map_parser.add_argument(
        "--max-sig-len", type=int, default=None, help="Max signature length"
    )
    map_parser.add_argument(
        "--entry-details-limit",
        type=int,
        default=None,
        help="Max entry details to list",
    )
    map_parser.add_argument(
        "--entry-details-per-kind",
        type=int,
        default=None,
        help="Max entry details per kind",
    )
    map_parser.add_argument(
        "--code-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include only code files by default",
    )
    map_parser.add_argument(
        "--compress-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip common path prefix in digest",
    )
    map_parser.add_argument(
        "--dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit dense layout extras (directory skeleton, hub summaries, file counts)",
    )
    map_parser.add_argument(
        "--dir-depth",
        type=int,
        default=None,
        help="Directory skeleton depth (0 disables; default auto when dense)",
    )
    map_parser.add_argument(
        "--dir-file-cap", type=int, default=12, help="Max files per dir in skeleton"
    )
    map_parser.add_argument(
        "--include-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a routes/API section in the digest",
    )
    map_parser.add_argument(
        "--routes-mode",
        choices=["auto", "heuristic", "nextjs", "fastapi", "django", "express", "rust", "go"],
        default="auto",
        help="Route detection mode (framework-aware patterns)",
    )
    map_parser.add_argument(
        "--routes-limit", type=int, default=24, help="Max routes to list"
    )
    map_parser.add_argument(
        "--routes-format",
        choices=["files", "compact"],
        default="compact",
        help="Routes section format (files or compact paths)",
    )
    map_parser.add_argument(
        "--diagrams",
        choices=["off", "compact", "compact+sequence", "mermaid"],
        default="compact+sequence",
        help="Emit a compact component diagram (token-efficient; compact+sequence adds flow lines)",
    )
    map_parser.add_argument(
        "--diagram-depth", type=int, default=2, help="Component depth for diagrams"
    )
    map_parser.add_argument(
        "--diagram-nodes", type=int, default=8, help="Max components in diagram"
    )
    map_parser.add_argument(
        "--diagram-edges", type=int, default=12, help="Max edges in diagram"
    )
    map_parser.add_argument(
        "--explain-ranking",
        action="store_true",
        help="Include ranking factor details in summaries and digests",
    )

    digest_parser = subparsers.add_parser("digest", help="Emit a token-budgeted digest")
    digest_parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "deep"],
        default=None,
        help="Preset tuning for speed/accuracy; overrides individual flags",
    )
    digest_parser.add_argument("--budget", type=int, default=20000, help="Token budget")
    digest_parser.add_argument("--focus", default=None, help="Focus query")
    digest_parser.add_argument(
        "--focus-depth",
        type=int,
        default=2,
        help="Focus expansion depth (1=only focus, 2=neighbors, 3=2-hop)",
    )
    digest_parser.add_argument("--format", choices=["json", "text"], default="text")
    digest_parser.add_argument(
        "--gate-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when confidence gate fails",
    )
    digest_parser.add_argument(
        "--flow-symbols",
        action="store_true",
        help="Include symbol names in flow traces (when symbol refs are available)",
    )
    digest_parser.add_argument(
        "--precise-tokens",
        action="store_true",
        help="Use tiktoken for token estimation when available",
    )
    digest_parser.add_argument(
        "--semantic-cluster",
        action="store_true",
        help="Use semantic-weighted component clustering in the digest",
    )
    digest_parser.add_argument(
        "--semantic-cluster-mode",
        choices=["tfidf", "name_only"],
        default="tfidf",
        help="Semantic clustering mode (tfidf uses docs+symbols; name_only uses paths/names)",
    )
    digest_parser.add_argument(
        "--doc-quality",
        action="store_true",
        help="Include docstring coverage/quality section in the digest",
    )
    digest_parser.add_argument(
        "--doc-quality-strict",
        action="store_true",
        help="Enable stricter doc/signature consistency checks",
    )
    digest_parser.add_argument(
        "--semantic-call-weight",
        type=float,
        default=0.3,
        help="Weight for call-graph affinity when semantic clustering (0-0.9)",
    )
    digest_parser.add_argument(
        "--dead-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include dead-code heuristics section in the digest",
    )
    digest_parser.add_argument(
        "--call-chains",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include multi-hop symbol call chains for entrypoints",
    )
    digest_parser.add_argument(
        "--static-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include static trace paths and trace graph",
    )
    digest_parser.add_argument(
        "--trace-depth",
        type=int,
        default=4,
        help="Max depth for static traces",
    )
    digest_parser.add_argument(
        "--trace-max",
        type=int,
        default=10,
        help="Max number of static trace paths",
    )
    digest_parser.add_argument(
        "--trace-direction",
        choices=["forward", "reverse", "both"],
        default="both",
        help="Direction for static traces",
    )
    digest_parser.add_argument(
        "--trace-start",
        default=None,
        help="Trace start node (file path, alias, or entity name/id; comma list)",
    )
    digest_parser.add_argument(
        "--trace-end",
        default=None,
        help="Trace end node (file path, alias, or entity name/id; comma list)",
    )
    digest_parser.add_argument(
        "--graph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache derived graph data for faster digest reruns",
    )
    digest_parser.add_argument(
        "--section-budgets",
        default=None,
        help="JSON section budget overrides (e.g. '{\"FILES\":0.4,\"EDGES\":0.1}')",
    )
    digest_parser.add_argument(
        "--entity-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include entity graph-backed details when entities are available",
    )
    digest_parser.add_argument(
        "--traceability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include doc-to-code traceability hints",
    )
    digest_parser.add_argument(
        "--traceability-targets",
        choices=["all", "code", "entities"],
        default="all",
        help="Filter traceability targets (all|code|entities)",
    )
    digest_parser.add_argument(
        "--stream-digest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream digest output (text only) to reduce peak memory",
    )
    digest_parser.add_argument(
        "--max-files", type=int, default=None, help="Cap files in digest"
    )
    digest_parser.add_argument(
        "--max-symbols", type=int, default=None, help="Cap symbols per file"
    )
    digest_parser.add_argument(
        "--max-edges", type=int, default=None, help="Cap edges in digest"
    )
    digest_parser.add_argument(
        "--max-sig-len", type=int, default=None, help="Max signature length"
    )
    digest_parser.add_argument(
        "--entry-details-limit",
        type=int,
        default=None,
        help="Max entry details to list",
    )
    digest_parser.add_argument(
        "--entry-details-per-kind",
        type=int,
        default=None,
        help="Max entry details per kind",
    )
    digest_parser.add_argument(
        "--code-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include only code files by default",
    )
    digest_parser.add_argument(
        "--compress-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip common path prefix in digest",
    )
    digest_parser.add_argument(
        "--dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit dense layout extras (directory skeleton, hub summaries, file counts)",
    )
    digest_parser.add_argument(
        "--dir-depth",
        type=int,
        default=None,
        help="Directory skeleton depth (0 disables; default auto when dense)",
    )
    digest_parser.add_argument(
        "--dir-file-cap", type=int, default=12, help="Max files per dir in skeleton"
    )
    digest_parser.add_argument(
        "--include-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a routes/API section in the digest",
    )
    digest_parser.add_argument(
        "--routes-mode",
        choices=["auto", "heuristic", "nextjs", "fastapi", "django", "express", "rust", "go"],
        default="auto",
        help="Route detection mode (framework-aware patterns)",
    )
    digest_parser.add_argument(
        "--routes-limit", type=int, default=24, help="Max routes to list"
    )
    digest_parser.add_argument(
        "--routes-format",
        choices=["files", "compact"],
        default="compact",
        help="Routes section format (files or compact paths)",
    )
    digest_parser.add_argument(
        "--diagrams",
        choices=["off", "compact", "compact+sequence", "mermaid"],
        default="compact+sequence",
        help="Emit a compact component diagram (token-efficient; compact+sequence adds flow lines)",
    )
    digest_parser.add_argument(
        "--diagram-depth", type=int, default=2, help="Component depth for diagrams"
    )
    digest_parser.add_argument(
        "--diagram-nodes", type=int, default=8, help="Max components in diagram"
    )
    digest_parser.add_argument(
        "--diagram-edges", type=int, default=12, help="Max edges in diagram"
    )
    digest_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )
    digest_parser.add_argument(
        "--auto-refresh",
        action="store_true",
        help="Automatically refresh if files have changed",
    )
    digest_parser.add_argument(
        "--explain-ranking",
        action="store_true",
        help="Include ranking factor details in summaries and digests",
    )

    diff_parser = subparsers.add_parser("diff", help="Show changes since last index")
    diff_parser.add_argument("--format", choices=["json", "text"], default="text")
    diff_parser.add_argument(
        "--limit", type=int, default=50, help="Limit entries per section"
    )
    diff_parser.add_argument(
        "--auto-index",
        action="store_true",
        help="Allow diff to proceed when ir.json is missing (treat baseline as empty)",
    )
    diff_parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not overwrite ir.json (use a temporary output directory)",
    )

    expand_parser = subparsers.add_parser("expand", help="Expand a file or symbol")
    expand_parser.add_argument(
        "--id", required=True, help="Node id, path, or symbol name"
    )
    expand_parser.add_argument("--budget", type=int, default=800, help="Token budget")
    expand_parser.add_argument(
        "--include",
        default="symbols,deps",
        help="Comma list: symbols,deps,callers,callees,flows,dataflow,snippets",
    )
    expand_parser.add_argument(
        "--flows-limit", type=int, default=5, help="Max flow edges to print"
    )
    expand_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    find_parser = subparsers.add_parser("find", help="Find symbol by name or signature")
    find_parser.add_argument("--query", required=True)
    find_parser.add_argument("--limit", type=int, default=20)
    find_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    query_parser = subparsers.add_parser(
        "query", help="Query symbol definitions, references, and neighbors"
    )
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--depth", type=int, default=1)
    query_parser.add_argument("--limit", type=int, default=6)
    query_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search the repo graph and return a ranked neighborhood"
    )
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--depth", type=int, default=1)
    search_parser.add_argument("--limit", type=int, default=6)
    search_parser.add_argument("--format", choices=["json", "text"], default="json")
    search_parser.add_argument(
        "--direction",
        choices=["in", "out", "both"],
        default="both",
        help="Neighbor traversal direction for file graph",
    )
    search_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    deps_parser = subparsers.add_parser("deps", help="Show file dependencies")
    deps_parser.add_argument("--path", required=True, help="File path or file id")
    deps_parser.add_argument("--depth", type=int, default=1)
    deps_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    export_parser = subparsers.add_parser(
        "export", help="Export graph in a standard format"
    )
    export_parser.add_argument("--format", choices=["json", "graphml"], default="json")
    export_parser.add_argument(
        "--edges",
        default="file_dep,symbol_ref",
        help="Comma list: file_dep,symbol_ref,dataflow,external_dep,entity_use,all",
    )
    export_parser.add_argument(
        "--include-all-nodes",
        action="store_true",
        help="Include all nodes for selected edge types",
    )
    export_parser.add_argument(
        "--auto-index", action="store_true", help="Index if IR is missing"
    )

    install_parser = subparsers.add_parser(
        "install-guide", help="Check tool availability and print installation guidance"
    )
    install_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    return parser


def resolve_arg_order(parser: argparse.ArgumentParser, argv: List[str]) -> argparse.Namespace:
    if not argv:
        return parser.parse_args(argv)
    subparser_action = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if not subparser_action:
        return parser.parse_args(argv)
    commands = set(subparser_action.choices.keys())
    cmd_index = next((idx for idx, token in enumerate(argv) if token in commands), None)
    if cmd_index is None:
        return parser.parse_args(argv)
    pre = argv[:cmd_index]
    command = argv[cmd_index]
    post = argv[cmd_index + 1 :]
    global_actions = parser._option_string_actions
    moved: List[str] = []
    remaining: List[str] = []
    i = 0
    while i < len(post):
        token = post[i]
        option_key = token
        if token.startswith("-") and "=" in token:
            option_key = token.split("=", 1)[0]
        action = global_actions.get(option_key)
        if action:
            moved.append(token)
            nargs = action.nargs
            if nargs == 0:
                i += 1
                continue
            count = 1
            if isinstance(nargs, int):
                count = nargs
            elif nargs in ("?",):
                count = 1
            for _ in range(count):
                if i + 1 < len(post):
                    i += 1
                    moved.append(post[i])
            i += 1
            continue
        remaining.append(token)
        i += 1
    if not moved:
        return parser.parse_args(argv)
    reordered = pre + moved + [command] + remaining
    return parser.parse_args(reordered)


def resolve_digest_alias(out_dir: Path, token: str) -> Optional[str]:
    if not token or len(token) < 2:
        return None
    if token[0] not in {"F", "S"} or not token[1:].isdigit():
        return None
    digest_path = out_dir / "digest.json"
    if not digest_path.exists():
        return None
    try:
        digest_data = json.loads(digest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return digest_data.get("aliases", {}).get(token)


def option_dest_map(parser: argparse.ArgumentParser) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    stack: List[argparse.ArgumentParser] = [parser]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        for action in current._actions:
            for option in getattr(action, "option_strings", []) or []:
                if option:
                    mapping[option] = action.dest
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    stack.append(sub)
    return mapping


def explicit_option_dests(parser: argparse.ArgumentParser, argv: List[str]) -> set[str]:
    actions = option_dest_map(parser)
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = actions.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def confidence_gate_failed(digest: Dict[str, object]) -> Tuple[bool, str]:
    gate = digest.get("confidence_gate")
    if not isinstance(gate, dict):
        return True, "confidence gate metadata missing"
    if gate.get("status") == "pass":
        return False, ""
    actions = gate.get("actions", [])
    if isinstance(actions, list) and actions:
        action_text = "; ".join(str(item) for item in actions)
        return True, action_text
    return True, "confidence gate failed"


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    explicit_dests = explicit_option_dests(parser, argv)
    args = resolve_arg_order(parser, argv)
    parsed_args = argparse.Namespace(**vars(args))
    explicit_codeql = "--codeql" in sys.argv[1:]
    requested_codeql = getattr(args, "codeql", None)
    if not args.command:
        args.command = "index"

    if args.command == "index":
        run_index(args, auto_refresh=getattr(args, "auto_refresh", False))
        return 0

    if args.command == "map":
        profile = getattr(args, "profile", None)
        if getattr(args, "quick", False):
            profile = "fast"
        if profile:
            apply_profile(args, profile)
            for dest in explicit_dests:
                if hasattr(parsed_args, dest):
                    setattr(args, dest, getattr(parsed_args, dest))
            if explicit_codeql and requested_codeql is not None:
                args.codeql = requested_codeql
        indexed = run_index(
            args,
            emit_summary=False,
            explain_ranking=getattr(args, "explain_ranking", False),
            auto_refresh=getattr(args, "auto_refresh", False),
        )
        ir = indexed["ir"]
        out_dir = Path(indexed["out_dir"])

        # Auto-detect framework and routes_mode if needed
        repo = Path(args.repo).resolve()
        config = ir.get("meta", {}).get("config", {})
        if args.routes_mode == "auto":
            framework = detect_framework(repo, config)
            if framework == "nextjs":
                args.routes_mode = "nextjs"
            elif framework == "fastapi":
                args.routes_mode = "fastapi"
            elif framework == "django":
                args.routes_mode = "django"
            elif framework == "express":
                args.routes_mode = "express"
            elif framework == "rust":
                args.routes_mode = "rust"
            elif framework == "go":
                args.routes_mode = "go"

        digest_result = run_digest(args, ir=ir, out_dir=out_dir)
        if getattr(args, "gate_confidence", False):
            failed, reason = confidence_gate_failed(digest_result)
            if failed:
                message = "Confidence gate failed"
                if reason:
                    message += f": {reason}"
                print(message, file=sys.stderr)
                return 2
        print(f"Artifacts: {indexed['out_dir']}", file=sys.stderr)
        return 0

    if args.command == "digest":
        profile = getattr(args, "profile", None)
        if profile:
            apply_profile(args, profile)
            for dest in explicit_dests:
                if hasattr(parsed_args, dest):
                    setattr(args, dest, getattr(parsed_args, dest))
            if explicit_codeql and requested_codeql is not None:
                args.codeql = requested_codeql
        args.auto_index = args.auto_index or False
        data = load_or_index(
            args,
            emit_summary=False,
            explain_ranking=getattr(args, "explain_ranking", False),
        )
        ir = data["ir"]
        out_dir = data["out_dir"]

        # Auto-detect framework and routes_mode if needed
        repo = Path(args.repo).resolve()
        config = ir.get("meta", {}).get("config", {})
        if args.routes_mode == "auto":
            framework = detect_framework(repo, config)
            if framework == "nextjs":
                args.routes_mode = "nextjs"
            elif framework == "fastapi":
                args.routes_mode = "fastapi"
            elif framework == "django":
                args.routes_mode = "django"
            elif framework == "express":
                args.routes_mode = "express"
            elif framework == "rust":
                args.routes_mode = "rust"
            elif framework == "go":
                args.routes_mode = "go"

        digest_result = run_digest(args, ir=ir, out_dir=out_dir)
        if getattr(args, "gate_confidence", False):
            failed, reason = confidence_gate_failed(digest_result)
            if failed:
                message = "Confidence gate failed"
                if reason:
                    message += f": {reason}"
                print(message, file=sys.stderr)
                return 2
        return 0

    if args.command == "diff":
        repo = Path(args.repo).resolve()
        out_dir = resolve_out_dir(repo, args.out, workspace_root=workspace_root())
        ir_path = out_dir / "ir.json"
        baseline_missing = False
        if not ir_path.exists():
            if not getattr(args, "auto_index", False):
                print(
                    json.dumps({"error": f"IR not found at {ir_path}"}), file=sys.stderr
                )
                return 2
            baseline_missing = True
            baseline_ir = {
                "meta": {"file_hashes": {}},
                "files": {},
                "symbols": {},
                "edges": {},
            }
        else:
            prev_ir = load_ir(ir_path)
            if not prev_ir:
                print(
                    json.dumps({"error": "Failed to load baseline IR"}),
                    file=sys.stderr,
                )
                return 2
            baseline_ir = deep_copy_ir(prev_ir)

        diff_args = args
        tmp_dir: Optional[Path] = None
        if getattr(args, "no_write", False):
            import copy
            import shutil

            tmp_dir = out_dir / "_diff_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            diff_args = copy.copy(args)
            diff_args.out = tmp_dir.as_posix()
        indexed = run_index(
            diff_args,
            emit_summary=False,
            explain_ranking=getattr(args, "explain_ranking", False),
        )
        new_ir = indexed["ir"]
        if tmp_dir and tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

        diff = diff_ir(baseline_ir, new_ir)
        files_added, add_trunc = apply_limit(diff["files"]["added"], args.limit)
        files_removed, rem_trunc = apply_limit(diff["files"]["removed"], args.limit)
        files_changed, chg_trunc = apply_limit(diff["files"]["changed"], args.limit)
        sym_added, sym_add_trunc = apply_limit(diff["symbols"]["added"], args.limit)
        sym_removed, sym_rem_trunc = apply_limit(diff["symbols"]["removed"], args.limit)
        sym_modified, sym_mod_trunc = apply_limit(
            diff["symbols"]["modified"], args.limit
        )

        truncated = any(
            [
                add_trunc,
                rem_trunc,
                chg_trunc,
                sym_add_trunc,
                sym_rem_trunc,
                sym_mod_trunc,
            ]
        )

        if args.format == "json":
            output = {
                "baseline_missing": baseline_missing,
                "files": {
                    "added": files_added,
                    "removed": files_removed,
                    "changed": files_changed,
                    "counts": {
                        "added": len(diff["files"]["added"]),
                        "removed": len(diff["files"]["removed"]),
                        "changed": len(diff["files"]["changed"]),
                    },
                },
                "symbols": {
                    "added": sym_added,
                    "removed": sym_removed,
                    "modified": sym_modified,
                    "counts": {
                        "added": len(diff["symbols"]["added"]),
                        "removed": len(diff["symbols"]["removed"]),
                        "modified": len(diff["symbols"]["modified"]),
                    },
                },
                "limit": args.limit,
                "truncated": truncated,
            }
            print(json.dumps(output, ensure_ascii=True, indent=2))
            return 0

        lines: List[str] = []
        if baseline_missing:
            lines.append("note=baseline_missing (treating baseline as empty)")
        lines.append("[FILES]")
        lines.append(
            f"added={len(diff['files']['added'])} removed={len(diff['files']['removed'])} "
            f"changed={len(diff['files']['changed'])}"
        )
        lines.extend(f"+ {path}" for path in files_added)
        lines.extend(f"- {path}" for path in files_removed)
        lines.extend(f"~ {path}" for path in files_changed)

        lines.append("")
        lines.append("[SYMBOLS]")
        lines.append(
            f"added={len(diff['symbols']['added'])} removed={len(diff['symbols']['removed'])} "
            f"modified={len(diff['symbols']['modified'])}"
        )
        lines.extend(f"+ {format_symbol_location(sym)}" for sym in sym_added)
        lines.extend(f"- {format_symbol_location(sym)}" for sym in sym_removed)
        for sym in sym_modified:
            before = sym.get("before", {}).get("signature", "")
            after = sym.get("after", {}).get("signature", "")
            lines.append(f"~ {format_symbol_location(sym)} :: {before} -> {after}")

        if truncated:
            lines.append("")
            lines.append("note=output truncated; use --limit to adjust")

        print("\n".join(lines))
        return 0

    if args.command == "expand":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        ir = data["ir"]
        out_dir = data["out_dir"]
        node_id = args.id
        alias = resolve_digest_alias(out_dir, node_id)
        if alias:
            node_id = alias
        include = [part.strip() for part in args.include.split(",") if part.strip()]
        expanded = expand_view(
            ir,
            node_id,
            budget_tokens=args.budget,
            include=include,
            repo_root=Path(args.repo).resolve(),
            flows_limit=args.flows_limit,
        )
        print(json.dumps(expanded, ensure_ascii=True, indent=2))
        return 0

    if args.command == "find":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        results = find_symbol_view(data["ir"], args.query, limit=args.limit)
        print(json.dumps({"results": results}, ensure_ascii=True, indent=2))
        return 0

    if args.command == "query":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        query_value = args.query
        alias = resolve_digest_alias(data["out_dir"], query_value)
        if alias and alias in data["ir"].get("symbols", {}):
            query_value = alias
        elif alias and alias in data["ir"].get("files", {}):
            print(
                json.dumps(
                    {"error": "query expects symbol alias; use deps/expand for files"},
                    ensure_ascii=True,
                ),
                file=sys.stderr,
            )
            return 2
        result = query_symbol_view(
            data["ir"], query_value, depth=args.depth, limit=args.limit
        )
        if isinstance(result, dict) and "text" in result:
            print(result["text"])
            return 0
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0

    if args.command == "search":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        query_value = args.query
        alias = resolve_digest_alias(data["out_dir"], query_value)
        alias_warning = False
        if not alias and len(query_value) > 1 and query_value[0] in {"F", "S"} and query_value[1:].isdigit():
            alias_warning = True
        if alias:
            query_value = alias
        result = search_repo_view(
            data["ir"],
            query_value,
            depth=args.depth,
            limit=args.limit,
            direction=args.direction,
        )
        if args.format == "text":
            center = result.get("center", {})
            lines = [
                "[SEARCH]",
                f"query={result.get('query')}",
                f"center={center.get('type')}:{center.get('id')}",
            ]
            if alias_warning:
                lines.append("warning=alias not resolved; run digest to generate aliases")
            defs = result.get("defs", [])
            refs = result.get("refs", [])
            neighbors = result.get("neighbors", [])
            if defs:
                lines.append("defs=" + ",".join(format_symbol_location(d) for d in defs))
            if refs:
                lines.append("refs=" + ",".join(format_symbol_location(r) for r in refs))
            if neighbors:
                lines.append("neighbors=" + ",".join(neighbors))
            print("\n".join(lines))
        else:
            if alias_warning:
                result = dict(result)
                result["warnings"] = ["alias not resolved; run digest to generate aliases"]
            print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0

    if args.command == "deps":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        fid = None
        alias = resolve_digest_alias(data["out_dir"], args.path)
        if alias and alias in data["ir"].get("files", {}):
            fid = alias
        elif alias and alias in data["ir"].get("symbols", {}):
            print(
                json.dumps(
                    {"error": "deps expects file alias; use query/expand for symbols"},
                    ensure_ascii=True,
                ),
                file=sys.stderr,
            )
            return 2
        elif args.path.startswith("file:"):
            fid = args.path
        else:
            for file_id_key, node in data["ir"].get("files", {}).items():
                if node.get("path") == args.path:
                    fid = file_id_key
                    break
        if not fid:
            print(
                json.dumps({"error": "file not found"}, ensure_ascii=True),
                file=sys.stderr,
            )
            return 2
        deps = deps_view(data["ir"], fid, depth=args.depth)
        print(json.dumps(deps, ensure_ascii=True, indent=2))
        return 0

    if args.command == "export":
        args.auto_index = args.auto_index or False
        data = load_or_index(args, emit_summary=False)
        edge_types = normalize_edge_types(args.edges)
        if not edge_types:
            print(
                json.dumps({"error": "no edge types selected"}, ensure_ascii=True),
                file=sys.stderr,
            )
            return 2
        graph = build_graph(
            data["ir"],
            edge_types=edge_types,
            include_all_nodes=args.include_all_nodes,
        )
        if args.format == "graphml":
            print(export_graphml(graph))
        else:
            print(export_graph_json(graph))
        return 0

    if args.command == "install-guide":
        run_install_guide(args)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
