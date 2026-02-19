from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ir import IR_VERSION, file_id, new_ir, now_iso, symbol_id
from utils import ToolState, progress, run_cmd, tool_version
from .ast_extract import run_ast_grep, run_ast_grep_scan
from .ast_edges import build_ast_primary_edges
from .ast_rules import ast_rules_for_lang, normalize_match_path, rule_to_yaml, rules_to_yaml
from .cache import (
    build_ast_cache,
    build_file_hashes,
    build_semantic_hashes,
    hash_file,
    language_for_path,
    load_ast_cache,
    load_semantic_hash_cache,
    save_semantic_hash_cache,
)
from .codeql import run_codeql
from .constants import (
    CONFIG_FILES,
    ENTITY_AUTO_MAX_FILES,
    ENTRYPOINT_NAME_ROLES,
    ENTRYPOINT_NAMES,
    EXCLUDE_DIRS,
    FD_EXCLUDES,
    INCREMENTAL_DEP_DEPTH,
    INCREMENTAL_DEP_LIMIT,
    JS_GLOBS,
    JSX_GLOBS,
    MONOREPO_MARKERS,
    PACKAGE_MANIFESTS,
    PY_GLOBS,
    QUALITY_MAX_FILES,
    REPOMAP_CONFIG_FILES,
    RG_EXCLUDES,
    TS_GLOBS,
    TSX_GLOBS,
    UNSUPPORTED_LANG_EXTS,
    ast_globs_for,
)
from .discovery import (
    code_files_for_languages,
    code_files_size_mb,
    detect_languages,
    list_repo_files,
    list_repo_files_fallback,
    unsupported_language_counts,
)
from .docs import (
    attach_symbol_docs,
    extract_py_docstring,
    jsdoc_blocks,
    jsdoc_for_symbol_line,
    parse_jsdoc_block,
    parse_jsdoc_param,
    parse_jsdoc_raises,
    parse_jsdoc_returns,
    parse_py_docstring,
    truncate_doc,
)
from .entrypoints import (
    detect_entrypoints,
    entrypoint_role_for_content,
    entrypoint_role_for_path,
    entrypoints_from_dockerfile,
    entrypoints_from_package_json,
    entrypoints_from_pyproject,
    resolve_entrypoint_candidate,
    summarize_file,
)
from .imports import (
    match_globs,
    resolve_alias_candidates,
    resolve_go_module,
    resolve_internal_fallback_module,
    resolve_js_module,
    resolve_module_candidates,
    resolve_py_module,
    resolve_rust_module,
    resolve_ts_module,
)
from .entities import (
    build_entity_graph,
    extract_entities,
    is_noise_path,
)
from .api_contracts import extract_api_contracts
from .filters import filter_configs, filter_docs, filter_tests, role_for_file
from .quality import build_test_mapping, build_trace_links, compute_coupling_metrics
from .repo_config import (
    collect_package_roots,
    config_fingerprint,
    detect_framework,
    detect_monorepo,
    index_signature,
    load_package_jsons,
    load_repo_config,
    load_tsconfig_paths,
    normalize_globs,
    normalize_str_list,
    probe_tool_versions,
    resolve_auto_flags,
    strip_json_comments,
    tsconfig_fingerprint,
)
from .rg_helpers import (
    estimate_import_count_rg,
    rg_collect_imports,
    rg_collect_py_imports,
    rg_collect_reexports,
    rg_collect_symbols,
    rg_count_matches,
    rg_json_matches,
)
from .symbols import (
    clean_type_name,
    collect_imports,
    collect_symbols,
    ensure_file_data,
    extract_type_edges,
    is_pascal,
    line_number_for_offset,
    match_line,
    maybe_component,
    meta_multi,
    meta_single,
    normalize_tokens,
    regex_js_imports,
    regex_js_symbols,
    signature_from_match,
    symbol_visibility,
)


@dataclass
class IndexOptions:
    languages: Set[str]
    ast: bool
    codeql: bool
    ast_mode: str
    codeql_mode: str
    ast_auto_max_files: int
    codeql_auto_max_files: int
    codeql_auto_max_mb: int
    include_internal: bool
    incremental: bool
    reuse_codeql_db: bool
    keep_codeql_db: bool
    codeql_db: Optional[Path]
    max_sample: int
    workers: int
    semantic_hash_cache: bool
    api_contracts: bool


def load_extra_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != 1:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    return payload


def save_extra_cache(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except OSError:
        pass


def deep_copy_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(ir))


def collect_exported_names(files_map: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
    exported_names: Dict[str, Set[str]] = {}
    for node in files_map.values():
        path = node.get("path")
        exports = node.get("exports")
        if not isinstance(path, str) or not isinstance(exports, list):
            continue
        names: Set[str] = set()
        for raw in exports:
            if not isinstance(raw, str):
                continue
            symbol = raw.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", symbol):
                names.add(symbol)
        if names:
            exported_names[path] = names
    return exported_names


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def diff_ir(prev_ir: Dict[str, Any], next_ir: Dict[str, Any]) -> Dict[str, object]:
    prev_hashes = prev_ir.get("meta", {}).get("file_hashes", {})
    next_hashes = next_ir.get("meta", {}).get("file_hashes", {})

    prev_files = set(prev_hashes)
    next_files = set(next_hashes)

    added_files = sorted(next_files - prev_files)
    removed_files = sorted(prev_files - next_files)
    changed_files = sorted(
        path for path in next_files if path in prev_files and prev_hashes.get(path) != next_hashes.get(path)
    )

    prev_symbols = prev_ir.get("symbols", {})
    next_symbols = next_ir.get("symbols", {})

    def symbol_record(node: Dict[str, Any], sid: str) -> Dict[str, object]:
        defined_in = node.get("defined_in", {})
        return {
            "id": sid,
            "name": node.get("name", ""),
            "path": defined_in.get("path", ""),
            "line": defined_in.get("line", 0),
            "signature": node.get("signature") or node.get("name", ""),
            "kind": node.get("kind", ""),
            "visibility": node.get("visibility", ""),
        }

    added_symbols = [
        symbol_record(next_symbols[sid], sid) for sid in sorted(set(next_symbols) - set(prev_symbols))
    ]
    removed_symbols = [
        symbol_record(prev_symbols[sid], sid) for sid in sorted(set(prev_symbols) - set(next_symbols))
    ]
    modified_symbols: List[Dict[str, object]] = []
    for sid in sorted(set(prev_symbols) & set(next_symbols)):
        prev_node = prev_symbols.get(sid, {})
        next_node = next_symbols.get(sid, {})
        prev_sig = prev_node.get("signature") or prev_node.get("name", "")
        next_sig = next_node.get("signature") or next_node.get("name", "")
        if (
            prev_sig != next_sig
            or prev_node.get("kind") != next_node.get("kind")
            or prev_node.get("visibility") != next_node.get("visibility")
        ):
            modified_symbols.append(
                {
                    "id": sid,
                    "name": next_node.get("name", prev_node.get("name", "")),
                    "path": next_node.get("defined_in", {}).get("path", ""),
                    "line": next_node.get("defined_in", {}).get("line", 0),
                    "before": {
                        "signature": prev_sig,
                        "kind": prev_node.get("kind", ""),
                        "visibility": prev_node.get("visibility", ""),
                    },
                    "after": {
                        "signature": next_sig,
                        "kind": next_node.get("kind", ""),
                        "visibility": next_node.get("visibility", ""),
                    },
                }
            )

    added_symbols.sort(key=lambda item: (item.get("name", ""), item.get("path", "")))
    removed_symbols.sort(key=lambda item: (item.get("name", ""), item.get("path", "")))
    modified_symbols.sort(key=lambda item: (item.get("name", ""), item.get("path", "")))

    return {
        "files": {
            "added": added_files,
            "removed": removed_files,
            "changed": changed_files,
        },
        "symbols": {
            "added": added_symbols,
            "removed": removed_symbols,
            "modified": modified_symbols,
        },
    }


def prune_ir(
    ir: Dict[str, Any],
    *,
    changed_files: Set[str],
    removed_files: Set[str],
) -> None:
    remove_files = set(changed_files) | set(removed_files)
    remove_fids = {file_id(path) for path in remove_files}

    remove_symbols = {
        sid
        for sid, sym in ir.get("symbols", {}).items()
        if sym.get("defined_in", {}).get("path") in remove_files
    }

    for path in removed_files:
        ir.get("files", {}).pop(file_id(path), None)

    for sid in remove_symbols:
        ir.get("symbols", {}).pop(sid, None)

    for node in ir.get("files", {}).values():
        if node.get("path") in remove_files:
            continue
        node["defines"] = [sid for sid in node.get("defines", []) if sid in ir["symbols"]]
        node["exports"] = [sid for sid in node.get("exports", []) if sid in ir["symbols"]]

    ir["edges"]["file_dep"] = [
        edge
        for edge in ir.get("edges", {}).get("file_dep", [])
        if edge.get("from") not in remove_fids and edge.get("to") not in {file_id(p) for p in removed_files}
    ]
    ir["edges"]["external_dep"] = [
        edge
        for edge in ir.get("edges", {}).get("external_dep", [])
        if edge.get("from") not in remove_fids
    ]
    ir["edges"]["symbol_ref"] = [
        edge
        for edge in ir.get("edges", {}).get("symbol_ref", [])
        if edge.get("from") not in remove_symbols and edge.get("to") not in remove_symbols
    ]
    ir["edges"]["dataflow"] = [
        edge
        for edge in ir.get("edges", {}).get("dataflow", [])
        if edge.get("from_path") not in remove_files and edge.get("to_path") not in remove_files
    ]
    ir["edges"]["type_ref"] = [
        edge
        for edge in ir.get("edges", {}).get("type_ref", [])
        if edge.get("from_path") not in remove_files
    ]


def expand_changed_with_dependents(
    prev_ir: Dict[str, Any],
    changed_files: Set[str],
    *,
    max_depth: int = INCREMENTAL_DEP_DEPTH,
    max_nodes: int = INCREMENTAL_DEP_LIMIT,
    warnings: Optional[List[str]] = None,
) -> Set[str]:
    if not prev_ir or not changed_files:
        return changed_files
    id_to_path = {fid: node.get("path", "") for fid, node in prev_ir.get("files", {}).items()}
    file_scores = {fid: float(node.get("score", 0.0)) for fid, node in prev_ir.get("files", {}).items()}
    base_fids = {file_id(path) for path in changed_files if file_id(path) in id_to_path}
    if not base_fids:
        return changed_files
    reverse: Dict[str, Set[str]] = defaultdict(set)
    for edge in prev_ir.get("edges", {}).get("file_dep", []):
        src = edge.get("from")
        dst = edge.get("to")
        if isinstance(src, str) and isinstance(dst, str):
            reverse[dst].add(src)
    if len(base_fids) >= max_nodes:
        if warnings is not None:
            warnings.append("Incremental dependents skipped: changed files exceed limit")
        return set(changed_files)
    frontier = [fid for fid in base_fids if fid in reverse]
    visited: Set[str] = set(base_fids)
    depth = 0
    truncated = False
    while frontier and depth < max_depth:
        next_frontier: List[str] = []
        for fid in frontier:
            deps = sorted(
                reverse.get(fid, []),
                key=lambda dep: file_scores.get(dep, 0.0),
                reverse=True,
            )
            for dep in deps:
                if dep in visited:
                    continue
                visited.add(dep)
                next_frontier.append(dep)
                if len(visited) >= max_nodes:
                    truncated = True
                    frontier = []
                    break
            if len(visited) >= max_nodes:
                break
        frontier = next_frontier
        depth += 1
    if len(visited) > max_nodes:
        truncated = True
        keep: Set[str] = set(base_fids)
        remaining = [fid for fid in visited if fid not in keep]
        remaining_sorted = sorted(
            remaining,
            key=lambda fid: file_scores.get(fid, 0.0),
            reverse=True,
        )
        for fid in remaining_sorted:
            if len(keep) >= max_nodes:
                break
            keep.add(fid)
        visited = keep
    if truncated and warnings is not None:
        warnings.append("Incremental dependents truncated to limit")
    dependent_paths = {id_to_path.get(fid, "") for fid in visited}
    dependent_paths.discard("")
    return set(changed_files) | dependent_paths


def index_repo(
    repo: Path,
    out_dir: Path,
    options: IndexOptions,
    tools: ToolState,
    prev_ir: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    config, config_source = load_repo_config(repo, warnings)
    files = list_repo_files(repo, warnings, tools)
    languages = detect_languages(files, options.languages)
    auto_stats = resolve_auto_flags(
        repo,
        options,
        files,
        languages,
        warnings=warnings,
        tools=tools,
    )
    code_file_count = int(auto_stats.get("code_file_count", 0)) if isinstance(auto_stats, dict) else 0
    entity_scan_ok = bool(options.ast) or code_file_count <= ENTITY_AUTO_MAX_FILES
    tsconfig = load_tsconfig_paths(repo, warnings, files=files)
    tool_versions = probe_tool_versions(repo, options, warnings=warnings, tools=tools)
    signature = index_signature(
        options, config, tool_versions, tsconfig_fp=tsconfig_fingerprint(tsconfig)
    )
    if options.incremental and prev_ir:
        prev_meta = prev_ir.get("meta", {})
        if prev_meta.get("version") != IR_VERSION:
            warnings.append("Incremental cache ignored: IR version changed")
            prev_ir = None
        elif prev_meta.get("index_signature") != signature:
            warnings.append("Incremental cache ignored: index options or config changed")
            prev_ir = None
    unsupported_langs = unsupported_language_counts(files)
    monorepo = detect_monorepo(repo, files)
    tests = filter_tests(
        files,
        extra_globs=config.get("tests_globs"),
        exclude_globs=config.get("tests_exclude_globs"),
    )
    configs = filter_configs(
        files,
        extra_globs=config.get("configs_globs"),
        exclude_globs=config.get("configs_exclude_globs"),
    )
    docs = filter_docs(
        files,
        extra_globs=config.get("docs_globs"),
        exclude_globs=config.get("docs_exclude_globs"),
    )
    file_hashes = build_file_hashes(repo, files)
    code_files = auto_stats.get("code_files", []) if isinstance(auto_stats, dict) else []
    semantic_cache = (
        load_semantic_hash_cache(out_dir / "semantic-hash-cache.json")
        if options.semantic_hash_cache
        else {}
    )
    semantic_hashes, semantic_cache_out = build_semantic_hashes(
        repo, code_files, file_hashes=file_hashes, cache=semantic_cache
    )
    if options.semantic_hash_cache and semantic_cache_out:
        save_semantic_hash_cache(out_dir / "semantic-hash-cache.json", semantic_cache_out)
    entrypoints = detect_entrypoints(repo, files, set(tests))
    extra_cache = load_extra_cache(out_dir / "extra-cache.json")

    changed_files: Set[str] = set(files)
    removed_files: Set[str] = set()
    if options.incremental and prev_ir:
        prev_hashes = prev_ir.get("meta", {}).get("file_hashes", {})
        changed_files = {
            path for path, digest in file_hashes.items() if prev_hashes.get(path) != digest
        }
        removed_files = set(prev_hashes) - set(file_hashes)
        prev_entrypoints: Dict[str, str] = {}
        for fnode in prev_ir.get("files", {}).values():
            path = fnode.get("path")
            if not isinstance(path, str):
                continue
            role = fnode.get("entrypoint_role")
            if fnode.get("role") == "entrypoint":
                prev_entrypoints[path] = role or "entrypoint"
        for path in set(prev_entrypoints) | set(entrypoints):
            if prev_entrypoints.get(path) != entrypoints.get(path):
                changed_files.add(path)
        prev_semantic = prev_ir.get("meta", {}).get("semantic_hashes", {})
        if not isinstance(prev_semantic, dict):
            prev_semantic = {}
        semantic_changed = {
            path
            for path in changed_files
            if semantic_hashes.get(path) != prev_semantic.get(path)
        }
        expanded = expand_changed_with_dependents(
            prev_ir,
            semantic_changed,
            warnings=warnings,
        )
        changed_files = set(changed_files) | expanded
        if not changed_files and not removed_files:
            prev_ir["meta"]["generated_at"] = now_iso()
            prev_ir["meta"]["file_hashes"] = file_hashes
            prev_ir["meta"]["languages"] = sorted(languages)
            prev_ir["meta"]["index_signature"] = signature
            prev_ir["meta"]["semantic_hashes"] = semantic_hashes
            if config:
                prev_ir["meta"]["config"] = config
            if config_source:
                prev_ir["meta"]["config_source"] = config_source
            prev_ir["meta"]["tool_versions"] = tool_versions
            prev_ir["meta"]["warnings"] = warnings
            if unsupported_langs:
                prev_ir["meta"]["unsupported_languages"] = unsupported_langs
            if monorepo:
                prev_ir["meta"]["monorepo"] = monorepo
            if entrypoints:
                prev_ir["meta"]["entrypoints"] = [
                    {"path": path, "role": role}
                    for path, role in sorted(entrypoints.items())
                ]
            if entity_scan_ok:
                existing_entities = prev_ir.get("meta", {}).get("entities")
                if not isinstance(existing_entities, list) or not existing_entities:
                    entities = extract_entities(
                        repo,
                        files,
                        exported_names_by_path=collect_exported_names(prev_ir.get("files", {})),
                    )
                    if entities:
                        prev_ir["meta"]["entities"] = entities
            existing_entity_nodes = prev_ir.get("entities")
            if not isinstance(existing_entity_nodes, dict) or not existing_entity_nodes:
                entities_list = prev_ir.get("meta", {}).get("entities")
                if isinstance(entities_list, list) and entities_list:
                    entity_nodes, entity_edges = build_entity_graph(
                        entities_list, file_edges=prev_ir.get("edges", {}).get("file_dep", [])
                    )
                    if entity_nodes:
                        prev_ir["entities"] = entity_nodes
                    if entity_edges:
                        prev_ir.setdefault("edges", {}).setdefault("entity_use", [])
                        prev_ir["edges"]["entity_use"] = entity_edges
            existing_tests = prev_ir.get("meta", {}).get("test_mapping")
            if not isinstance(existing_tests, list) or not existing_tests:
                test_mapping, test_summary = build_test_mapping(prev_ir)
                if test_mapping:
                    prev_ir["meta"]["test_mapping"] = test_mapping
                if test_summary:
                    prev_ir["meta"]["test_summary"] = test_summary
            existing_apis = prev_ir.get("meta", {}).get("api_contracts")
            if options.api_contracts and (not isinstance(existing_apis, list) or not existing_apis) and len(files) <= 2000:
                api_contracts, api_cache = extract_api_contracts(
                    repo,
                    files,
                    file_hashes=file_hashes,
                    cache=extra_cache,
                )
                if api_contracts:
                    prev_ir["meta"]["api_contracts"] = api_contracts
                if api_cache:
                    save_extra_cache(out_dir / "extra-cache.json", api_cache)
            existing_links = prev_ir.get("meta", {}).get("trace_links")
            if not isinstance(existing_links, list) or not existing_links:
                trace_links = build_trace_links(
                    docs,
                    prev_ir.get("entities", {}),
                    prev_ir.get("files", {}),
                )
                if trace_links:
                    prev_ir["meta"]["trace_links"] = trace_links
            return {
                "ir": prev_ir,
                "files": files,
                "docs": docs,
                "configs": configs,
                "tests": tests,
                "packages": load_package_jsons(repo, files, warnings),
                "warnings": warnings,
                "tool_versions": {},
            }
        ir = deep_copy_ir(prev_ir)
        prune_ir(ir, changed_files=changed_files, removed_files=removed_files)
        ir["meta"]["generated_at"] = now_iso()
        ir["meta"]["file_hashes"] = file_hashes
        ir["meta"]["languages"] = sorted(languages)
    else:
        ir = new_ir(repo, sorted(languages), file_hashes)
        changed_files = set(files)
    ir["meta"]["index_signature"] = signature
    if config:
        ir["meta"]["config"] = config
    if config_source:
        ir["meta"]["config_source"] = config_source
    if tsconfig:
        ir["meta"]["tsconfig"] = tsconfig
    ir["meta"]["tool_versions"] = tool_versions
    ir["meta"]["warnings"] = warnings
    ir["meta"]["semantic_hashes"] = semantic_hashes
    if unsupported_langs:
        ir["meta"]["unsupported_languages"] = unsupported_langs
    if monorepo:
        ir["meta"]["monorepo"] = monorepo
    if entrypoints:
        ir["meta"]["entrypoints"] = [
            {"path": path, "role": role} for path, role in sorted(entrypoints.items())
        ]
    ir.setdefault("edges", {})
    ir["edges"].setdefault("file_dep", [])
    ir["edges"].setdefault("external_dep", [])
    ir["edges"].setdefault("symbol_ref", [])
    ir["edges"].setdefault("dataflow", [])
    ir["edges"].setdefault("type_ref", [])

    file_data: Dict[str, Dict[str, Any]] = {}
    symbol_nodes: Dict[str, Dict[str, Any]] = {}

    def add_symbol(node: Dict[str, Any]) -> None:
        symbol_nodes[node["id"]] = node

    scan_files = list(changed_files)
    ast_files: Dict[str, List[str]] = {
        "ts": [f for f in scan_files if f.endswith(".ts") and not f.endswith(".d.ts")],
        "tsx": [f for f in scan_files if f.endswith(".tsx")],
        "javascript": [
            f
            for f in scan_files
            if f.endswith((".js", ".mjs", ".cjs")) and not f.endswith(".min.js")
        ],
        "jsx": [f for f in scan_files if f.endswith(".jsx")],
        "python": [f for f in scan_files if f.endswith(".py")],
        "rust": [f for f in scan_files if f.endswith(".rs")],
    }

    ast_cached_files: Set[str] = set()
    ast_scanned_files: Set[str] = set()
    rg_fallback_used = False

    if options.ast:
        progress("Extracting AST symbols (imports/exports)...")
        ast_cache_path = out_dir / "ast-cache.json"
        ast_cache = load_ast_cache(ast_cache_path, options.include_internal)
        cache_files = ast_cache.get("files", {}) if isinstance(ast_cache, dict) else {}
        if isinstance(cache_files, dict):
            for paths in ast_files.values():
                for file_path in list(paths):
                    cached = cache_files.get(file_path)
                    if not isinstance(cached, dict):
                        continue
                    if cached.get("hash") != file_hashes.get(file_path):
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    fdata["imports"].extend(cached.get("imports", []))
                    fdata["re_exports"].extend(cached.get("re_exports", []))
                    fdata["exports"].extend(cached.get("exports", []))
                    fdata["defines"].extend(cached.get("defines", []))
                    for sym in cached.get("symbols", []):
                        if isinstance(sym, dict) and sym.get("id"):
                            symbol_nodes[sym["id"]] = sym
                    ast_cached_files.add(file_path)
        if ast_cached_files:
            for key in ast_files:
                ast_files[key] = [p for p in ast_files[key] if p not in ast_cached_files]

        lang_map = {
            "ts": "ts",
            "tsx": "tsx",
            "js": "javascript",
            "jsx": "jsx",
            "py": "python",
            "rs": "rust",
        }
        for lang_key, ast_lang in lang_map.items():
            if lang_key not in languages:
                continue
            files_for_lang = ast_files.get(ast_lang, [])
            if not files_for_lang:
                continue
            rules_yaml, rule_meta = ast_rules_for_lang(ast_lang, options.include_internal)
            matches = run_ast_grep_scan(
                repo,
                rules_yaml,
                files=files_for_lang,
                globs=ast_globs_for(ast_lang),
                warnings=warnings,
                tools=tools,
                workers=options.workers,
            )
            if "ast-grep" in tools.missing:
                break
            ast_scanned_files.update(files_for_lang)
            for match in matches:
                file_path_raw = match.get("file", "")
                if not isinstance(file_path_raw, str):
                    continue
                file_path = normalize_match_path(repo, file_path_raw)
                if not file_path:
                    continue
                rule_id = match.get("ruleId")
                if not isinstance(rule_id, str):
                    continue
                info = rule_meta.get(rule_id)
                if not isinstance(info, dict):
                    continue
                category = info.get("category")
                if category in {"import", "reexport"}:
                    module = meta_single(match, "MOD")
                    if not module:
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    if category == "import":
                        fdata["imports"].append(module)
                    else:
                        fdata["re_exports"].append(module)
                    continue
                if category == "symbol":
                    name = meta_single(match, "NAME") or info.get("default_name") or "default"
                    if not name:
                        continue
                    line = match_line(match)
                    kind = str(info.get("kind", ""))
                    exported = bool(info.get("exported"))
                    signature = signature_from_match(ast_lang, kind, name, match)
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, line, name)
                    fdata["defines"].append(sym_id)
                    if ast_lang == "python":
                        exported = not name.startswith("_")
                    if exported:
                        fdata["exports"].append(sym_id)
                    kind = maybe_component(lang_key, kind, name)
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": name,
                            "kind": kind,
                            "signature": signature,
                            "doc_1l": "",
                            "visibility": symbol_visibility(name, exported),
                            "defined_in": {"path": file_path, "line": line or 0},
                            "score": 0.0,
                            "provenance": "ast-grep",
                        }
                    )

        if "ast-grep" in tools.missing:
            rg_fallback_used = True
            progress("ast-grep missing; falling back to rg patterns...")
            if "ts" in languages:
                for file_path, module in rg_collect_imports(
                    repo, globs=TS_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["imports"].append(module)
                for file_path, module in rg_collect_reexports(
                    repo, globs=TS_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["re_exports"].append(module)
                for symbol in rg_collect_symbols(
                    repo,
                    globs=TS_GLOBS,
                    include_internal=options.include_internal,
                    is_python=False,
                    warnings=warnings,
                    tools=tools,
                ):
                    file_path = normalize_match_path(repo, symbol.get("file", ""))
                    if not file_path:
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, symbol.get("line"), symbol.get("name"))
                    fdata["defines"].append(sym_id)
                    if symbol.get("exported"):
                        fdata["exports"].append(sym_id)
                    kind = maybe_component("ts", symbol.get("kind", ""), symbol.get("name", ""))
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": symbol.get("name", ""),
                            "kind": kind,
                            "signature": symbol.get("signature", ""),
                            "doc_1l": "",
                            "visibility": symbol_visibility(symbol.get("name", ""), bool(symbol.get("exported"))),
                            "defined_in": {"path": file_path, "line": symbol.get("line") or 0},
                            "score": 0.0,
                            "provenance": "rg",
                        }
                    )
            if "tsx" in languages:
                for file_path, module in rg_collect_imports(
                    repo, globs=TSX_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["imports"].append(module)
                for file_path, module in rg_collect_reexports(
                    repo, globs=TSX_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["re_exports"].append(module)
                for symbol in rg_collect_symbols(
                    repo,
                    globs=TSX_GLOBS,
                    include_internal=options.include_internal,
                    is_python=False,
                    warnings=warnings,
                    tools=tools,
                ):
                    file_path = normalize_match_path(repo, symbol.get("file", ""))
                    if not file_path:
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, symbol.get("line"), symbol.get("name"))
                    fdata["defines"].append(sym_id)
                    if symbol.get("exported"):
                        fdata["exports"].append(sym_id)
                    kind = maybe_component("tsx", symbol.get("kind", ""), symbol.get("name", ""))
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": symbol.get("name", ""),
                            "kind": kind,
                            "signature": symbol.get("signature", ""),
                            "doc_1l": "",
                            "visibility": symbol_visibility(symbol.get("name", ""), bool(symbol.get("exported"))),
                            "defined_in": {"path": file_path, "line": symbol.get("line") or 0},
                            "score": 0.0,
                            "provenance": "rg",
                        }
                    )
            if "js" in languages:
                for file_path, module in rg_collect_imports(
                    repo, globs=JS_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["imports"].append(module)
                for file_path, module in rg_collect_reexports(
                    repo, globs=JS_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["re_exports"].append(module)
                for symbol in rg_collect_symbols(
                    repo,
                    globs=JS_GLOBS,
                    include_internal=options.include_internal,
                    is_python=False,
                    warnings=warnings,
                    tools=tools,
                ):
                    file_path = normalize_match_path(repo, symbol.get("file", ""))
                    if not file_path:
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, symbol.get("line"), symbol.get("name"))
                    fdata["defines"].append(sym_id)
                    if symbol.get("exported"):
                        fdata["exports"].append(sym_id)
                    kind = maybe_component("js", symbol.get("kind", ""), symbol.get("name", ""))
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": symbol.get("name", ""),
                            "kind": kind,
                            "signature": symbol.get("signature", ""),
                            "doc_1l": "",
                            "visibility": symbol_visibility(symbol.get("name", ""), bool(symbol.get("exported"))),
                            "defined_in": {"path": file_path, "line": symbol.get("line") or 0},
                            "score": 0.0,
                            "provenance": "rg",
                        }
                    )
            if "jsx" in languages:
                for file_path, module in rg_collect_imports(
                    repo, globs=JSX_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["imports"].append(module)
                for file_path, module in rg_collect_reexports(
                    repo, globs=JSX_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["re_exports"].append(module)
                for symbol in rg_collect_symbols(
                    repo,
                    globs=JSX_GLOBS,
                    include_internal=options.include_internal,
                    is_python=False,
                    warnings=warnings,
                    tools=tools,
                ):
                    file_path = normalize_match_path(repo, symbol.get("file", ""))
                    if not file_path:
                        continue
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, symbol.get("line"), symbol.get("name"))
                    fdata["defines"].append(sym_id)
                    if symbol.get("exported"):
                        fdata["exports"].append(sym_id)
                    kind = maybe_component("jsx", symbol.get("kind", ""), symbol.get("name", ""))
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": symbol.get("name", ""),
                            "kind": kind,
                            "signature": symbol.get("signature", ""),
                            "doc_1l": "",
                            "visibility": symbol_visibility(symbol.get("name", ""), bool(symbol.get("exported"))),
                            "defined_in": {"path": file_path, "line": symbol.get("line") or 0},
                            "score": 0.0,
                            "provenance": "rg",
                        }
                    )
            if "py" in languages:
                for file_path, module in rg_collect_py_imports(
                    repo, globs=PY_GLOBS, warnings=warnings, tools=tools
                ):
                    ensure_file_data(file_data, normalize_match_path(repo, file_path))["imports"].append(module)
                for symbol in rg_collect_symbols(
                    repo,
                    globs=PY_GLOBS,
                    include_internal=options.include_internal,
                    is_python=True,
                    warnings=warnings,
                    tools=tools,
                ):
                    file_path = normalize_match_path(repo, symbol.get("file", ""))
                    if not file_path:
                        continue
                    is_exported = not str(symbol.get("name", "")).startswith("_")
                    fdata = ensure_file_data(file_data, file_path)
                    sym_id = symbol_id(file_path, symbol.get("line"), symbol.get("name"))
                    fdata["defines"].append(sym_id)
                    if is_exported:
                        fdata["exports"].append(sym_id)
                    add_symbol(
                        {
                            "id": sym_id,
                            "name": symbol.get("name", ""),
                            "kind": symbol.get("kind", ""),
                            "signature": symbol.get("signature", ""),
                            "doc_1l": "",
                            "visibility": symbol_visibility(symbol.get("name", ""), is_exported),
                            "defined_in": {"path": file_path, "line": symbol.get("line") or 0},
                            "score": 0.0,
                            "provenance": "rg",
                        }
                    )

            if "rg" in tools.missing and languages & {"js", "jsx"}:
                warnings.append("rg missing; falling back to regex parsing for JS/JSX")
                for file_path in ast_files.get("javascript", []):
                    try:
                        content = (repo / file_path).read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if len(content) > 200_000:
                        continue
                    for module in regex_js_imports(content):
                        ensure_file_data(file_data, file_path)["imports"].append(module)
                    for symbol in regex_js_symbols(content):
                        sym_id = symbol_id(file_path, symbol["line"], symbol["name"])
                        fdata = ensure_file_data(file_data, file_path)
                        fdata["defines"].append(sym_id)
                        if symbol["exported"]:
                            fdata["exports"].append(sym_id)
                        kind = maybe_component("js", symbol["kind"], symbol["name"])
                        add_symbol(
                            {
                                "id": sym_id,
                                "name": symbol["name"],
                                "kind": kind,
                                "signature": symbol["signature"],
                                "doc_1l": "",
                                "visibility": symbol_visibility(symbol["name"], symbol["exported"]),
                                "defined_in": {"path": file_path, "line": symbol["line"] or 0},
                                "score": 0.0,
                                "provenance": "regex",
                            }
                        )
                for file_path in ast_files.get("jsx", []):
                    try:
                        content = (repo / file_path).read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        continue
                    if len(content) > 200_000:
                        continue
                    for module in regex_js_imports(content):
                        ensure_file_data(file_data, file_path)["imports"].append(module)
                    for symbol in regex_js_symbols(content):
                        sym_id = symbol_id(file_path, symbol["line"], symbol["name"])
                        fdata = ensure_file_data(file_data, file_path)
                        fdata["defines"].append(sym_id)
                        if symbol["exported"]:
                            fdata["exports"].append(sym_id)
                        kind = maybe_component("jsx", symbol["kind"], symbol["name"])
                        add_symbol(
                            {
                                "id": sym_id,
                                "name": symbol["name"],
                                "kind": kind,
                                "signature": symbol["signature"],
                                "doc_1l": "",
                                "visibility": symbol_visibility(symbol["name"], symbol["exported"]),
                                "defined_in": {"path": file_path, "line": symbol["line"] or 0},
                                "score": 0.0,
                                "provenance": "regex",
                            }
                        )

        progress(f"Extracted {len(symbol_nodes)} symbols", done=True)
        attach_symbol_docs(repo, symbol_nodes, limit_paths=set(changed_files))
    else:
        warnings.append("AST mapping disabled")

    if options.ast:
        ast_cache_out = build_ast_cache(
            file_data, symbol_nodes, file_hashes, include_internal=options.include_internal
        )
        try:
            (out_dir / "ast-cache.json").write_text(
                json.dumps(ast_cache_out, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            warnings.append("Failed to write ast-cache.json")

    files_set = set(files)
    repo_roots = {path.split("/", 1)[0] for path in files_set if path}
    alias_config = tsconfig if tsconfig else None
    alias_resolved_count = 0
    for path in files:
        fid = file_id(path)
        if path not in changed_files and fid in ir.get("files", {}):
            continue
        fdata = ensure_file_data(file_data, path)
        language = language_for_path(path)
        summary = summarize_file(repo / path)
        entrypoint_role = entrypoints.get(path)
        role = role_for_file(path, set(tests), set(configs), set(entrypoints))
        ir["files"][fid] = {
            "id": fid,
            "path": path,
            "language": language,
            "role": role,
            "entrypoint_role": entrypoint_role,
            "summary_1l": summary,
            "imports": list(dict.fromkeys(fdata["imports"])),
            "re_exports": list(dict.fromkeys(fdata.get("re_exports", []))),
            "exports": list(dict.fromkeys(fdata["exports"])),
            "defines": list(dict.fromkeys(fdata["defines"])),
            "score": 0.0,
        }

    for fid, fnode in ir["files"].items():
        if options.incremental and prev_ir and fnode.get("path") not in changed_files:
            continue
        importer = fnode["path"]
        for module in fnode.get("imports", []):
            target: Optional[str] = None
            provenance = "ast-grep"
            if fnode["language"] in {"ts", "tsx"}:
                target = resolve_ts_module(module, importer, files_set, alias_config=alias_config)
            elif fnode["language"] in {"js", "jsx"}:
                target = resolve_js_module(module, importer, files_set, alias_config=alias_config)
            elif fnode["language"] == "py":
                target = resolve_py_module(module, importer, files_set)
            elif fnode["language"] == "go":
                target = resolve_go_module(module, importer, files_set)
            elif fnode["language"] == "rs":
                target = resolve_rust_module(module, importer, files_set)
            if not target:
                fallback_target = resolve_internal_fallback_module(
                    module,
                    importer,
                    files_set,
                    repo_roots=repo_roots,
                )
                if fallback_target:
                    target = fallback_target
                    provenance = "import-fallback"
            if target:
                if provenance == "ast-grep" and not module.startswith("."):
                    alias_resolved_count += 1
                ir["edges"]["file_dep"].append(
                    {
                        "from": fid,
                        "to": file_id(target),
                        "weight": 1.0,
                        "kind": "import",
                        "provenance": provenance,
                    }
                )
            else:
                ir["edges"]["external_dep"].append(
                    {
                        "from": fid,
                        "module": module,
                        "kind": "external",
                        "provenance": "ast-grep",
                    }
                )
        for module in fnode.get("re_exports", []):
            target: Optional[str] = None
            provenance = "ast-grep"
            if fnode["language"] in {"ts", "tsx"}:
                target = resolve_ts_module(module, importer, files_set, alias_config=alias_config)
            elif fnode["language"] in {"js", "jsx"}:
                target = resolve_js_module(module, importer, files_set, alias_config=alias_config)
            elif fnode["language"] == "py":
                target = resolve_py_module(module, importer, files_set)
            if not target:
                fallback_target = resolve_internal_fallback_module(
                    module,
                    importer,
                    files_set,
                    repo_roots=repo_roots,
                )
                if fallback_target:
                    target = fallback_target
                    provenance = "reexport-fallback"
            if target:
                if provenance == "ast-grep" and not module.startswith("."):
                    alias_resolved_count += 1
                ir["edges"]["file_dep"].append(
                    {
                        "from": fid,
                        "to": file_id(target),
                        "weight": 1.0,
                        "kind": "reexport",
                        "provenance": provenance,
                    }
                )
            else:
                ir["edges"]["external_dep"].append(
                    {
                        "from": fid,
                        "module": module,
                        "kind": "reexport",
                        "provenance": "ast-grep",
                    }
                )

    exported_names_by_path = collect_exported_names(ir.get("files", {}))

    # Type relationships (extends/implements/alias), best-effort regex scan.
    ir["edges"]["type_ref"] = [
        edge
        for edge in ir["edges"]["type_ref"]
        if edge.get("from_path") not in changed_files
    ]
    symbol_by_path_name: Dict[Tuple[str, str], str] = {}
    symbol_by_name: Dict[str, str] = {}
    name_counts: Counter[str] = Counter()
    for sid, sym in symbol_nodes.items():
        name = sym.get("name")
        defined_in = sym.get("defined_in", {})
        path = defined_in.get("path")
        if isinstance(name, str) and isinstance(path, str):
            symbol_by_path_name[(path, name)] = sid
            name_counts[name] += 1
    for name, count in name_counts.items():
        if count == 1:
            for (path, sym_name), sid in symbol_by_path_name.items():
                if sym_name == name:
                    symbol_by_name[name] = sid
                    break
    seen_type_edges: Set[Tuple[object, object, object, object]] = set()
    for path in changed_files:
        if not path.endswith((".ts", ".tsx", ".js", ".jsx", ".py")):
            continue
        try:
            content = (repo / path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(content) > 200_000:
            continue
        for edge in extract_type_edges(content, path):
            from_name = str(edge.get("from_name") or "")
            to_name = str(edge.get("to_name") or "")
            from_sid = symbol_by_path_name.get((path, from_name)) if from_name else None
            to_sid = symbol_by_path_name.get((path, to_name)) or symbol_by_name.get(to_name)
            ambiguous = False
            if to_name and to_name in name_counts and name_counts[to_name] > 1 and to_sid is None:
                ambiguous = True
            edge["from"] = from_sid or from_name
            edge["to"] = to_sid or to_name
            edge["provenance"] = "regex"
            if ambiguous:
                edge["ambiguous"] = True
            edge_key = (
                edge.get("from"),
                edge.get("to"),
                edge.get("kind"),
                edge.get("from_path"),
            )
            if edge_key not in seen_type_edges:
                seen_type_edges.add(edge_key)
                ir["edges"]["type_ref"].append(edge)

    test_mapping, test_summary = build_test_mapping(ir)
    if test_mapping:
        ir["meta"]["test_mapping"] = test_mapping
    if test_summary:
        ir["meta"]["test_summary"] = test_summary

    if options.api_contracts and len(files) <= 2000:
        api_contracts, api_cache = extract_api_contracts(
            repo,
            files,
            file_hashes=file_hashes,
            cache=extra_cache,
        )
        if api_contracts:
            ir["meta"]["api_contracts"] = api_contracts
        if api_cache:
            save_extra_cache(out_dir / "extra-cache.json", api_cache)

    ir["symbols"].update(symbol_nodes)

    def integrate_call_edges(call_edges: List[Dict[str, Any]], default_provenance: str) -> None:
        for edge in call_edges:
            edge_provenance = str(edge.get("provenance") or default_provenance)
            caller_path = edge.get("caller_path", "")
            callee_path = edge.get("callee_path", "")
            if caller_path not in files_set or callee_path not in files_set:
                continue
            caller_line = int(edge.get("caller_line") or 0)
            callee_line = int(edge.get("callee_line") or 0)
            caller_name = str(edge.get("caller_name") or "")
            callee_name = str(edge.get("callee_name") or "")
            if not caller_name or not callee_name:
                continue
            caller_id = symbol_id(caller_path, caller_line, caller_name)
            callee_id = symbol_id(callee_path, callee_line, callee_name)
            if caller_id not in ir["symbols"]:
                ir["symbols"][caller_id] = {
                    "id": caller_id,
                    "name": caller_name,
                    "kind": "function",
                    "signature": caller_name,
                    "doc_1l": "",
                    "visibility": "internal",
                    "defined_in": {"path": caller_path, "line": caller_line},
                    "score": 0.0,
                    "provenance": edge_provenance,
                }
                file_node = ir["files"].get(file_id(caller_path))
                if file_node:
                    file_node["defines"].append(caller_id)
            if callee_id not in ir["symbols"]:
                ir["symbols"][callee_id] = {
                    "id": callee_id,
                    "name": callee_name,
                    "kind": "function",
                    "signature": callee_name,
                    "doc_1l": "",
                    "visibility": "internal",
                    "defined_in": {"path": callee_path, "line": callee_line},
                    "score": 0.0,
                    "provenance": edge_provenance,
                }
                file_node = ir["files"].get(file_id(callee_path))
                if file_node:
                    file_node["defines"].append(callee_id)
            ir["edges"]["symbol_ref"].append(
                {
                    "from": caller_id,
                    "to": callee_id,
                    "kind": "call",
                    "provenance": edge_provenance,
                }
            )

    def integrate_dataflow_edges(dataflow_edges: List[Dict[str, Any]], default_provenance: str) -> None:
        for edge in dataflow_edges:
            edge_provenance = str(edge.get("provenance") or default_provenance)
            from_path = edge.get("from_path", "")
            to_path = edge.get("to_path", "")
            if from_path not in files_set or to_path not in files_set:
                continue
            ir["edges"]["dataflow"].append(
                {
                    "from_path": from_path,
                    "from_line": int(edge.get("from_line") or 0),
                    "to_path": to_path,
                    "to_line": int(edge.get("to_line") or 0),
                    "kind": "dataflow",
                    "provenance": edge_provenance,
                }
            )

    ast_edge_info = build_ast_primary_edges(repo, files, languages)
    ir["edges"]["symbol_ref"] = [
        edge for edge in ir["edges"]["symbol_ref"] if edge.get("provenance") != "ast"
    ]
    ir["edges"]["dataflow"] = [
        edge for edge in ir["edges"]["dataflow"] if edge.get("provenance") != "ast"
    ]
    integrate_call_edges(list(ast_edge_info.get("call_edges", [])), "ast")
    integrate_dataflow_edges(list(ast_edge_info.get("dataflow_edges", [])), "ast")
    ir["meta"]["ast_edges"] = {
        "status": "ok",
        "counts": ast_edge_info.get("counts", {}),
        "confidence": ast_edge_info.get("confidence", "n/a"),
        "confidence_by_language": ast_edge_info.get("confidence_by_language", {}),
        "low_confidence_languages": ast_edge_info.get("low_confidence_languages", []),
    }

    codeql_info: Dict[str, Any] = {}
    codeql_supported = bool(languages & {"ts", "tsx", "js", "jsx", "py"})
    if options.codeql and codeql_supported:
        should_run_codeql = options.codeql_mode == "on"
        skip_reason = ""
        if options.codeql_mode == "auto":
            should_run_codeql = bool(ast_edge_info.get("low_confidence"))
            if not should_run_codeql:
                skip_reason = "ast_confidence_sufficient"
                warnings.append("CodeQL auto skipped: AST edge confidence sufficient")
        if should_run_codeql:
            code_files_set = set(auto_stats.get("code_files", [])) if isinstance(auto_stats, dict) else set()
            changed_code_files = [path for path in changed_files if path in code_files_set]
            if options.incremental and prev_ir and not changed_code_files:
                warnings.append("CodeQL skipped: no code changes detected")
                codeql_info = {"ran": False, "skipped": "no_code_changes"}
                progress("CodeQL skipped (no code changes)", done=True)
            else:
                progress("Running CodeQL analysis (high precision augmenter)...")
                codeql_info = run_codeql(
                    repo,
                    out_dir,
                    has_js=bool(languages & {"ts", "tsx", "js", "jsx"}),
                    has_py="py" in languages,
                    options=options,
                    warnings=warnings,
                    tools=tools,
                    python_fallback=False,
                )
                if codeql_info.get("ran"):
                    progress("CodeQL analysis complete", done=True)
                else:
                    progress("CodeQL skipped or unavailable", done=True)
        elif skip_reason:
            codeql_info = {"ran": False, "skipped": skip_reason}

    if codeql_info:
        if codeql_info.get("ran"):
            ir["edges"]["symbol_ref"] = [
                edge
                for edge in ir["edges"]["symbol_ref"]
                if edge.get("provenance") not in {"codeql", "codeql-fallback"}
            ]
            ir["edges"]["dataflow"] = [
                edge
                for edge in ir["edges"]["dataflow"]
                if edge.get("provenance") not in {"codeql", "codeql-fallback"}
            ]
        for edge in codeql_info.get("call_edges", []):
            edge_provenance = str(edge.get("provenance") or "codeql")
            edge["provenance"] = edge_provenance
        integrate_call_edges(list(codeql_info.get("call_edges", [])), "codeql")
        for edge in codeql_info.get("dataflow_edges", []):
            edge_provenance = str(edge.get("provenance") or "codeql")
            edge["provenance"] = edge_provenance
        integrate_dataflow_edges(list(codeql_info.get("dataflow_edges", [])), "codeql")
        ir["meta"]["codeql"] = {
            "status": "ok" if codeql_info.get("ran") else "skipped",
            "artifacts": codeql_info.get("artifacts", {}),
            "counts": codeql_info.get("counts", {}),
        }

    if entity_scan_ok:
        entities = extract_entities(
            repo,
            files,
            exported_names_by_path=exported_names_by_path,
        )
        if entities:
            ir["meta"]["entities"] = entities
            entity_nodes, entity_edges = build_entity_graph(
                entities, file_edges=ir.get("edges", {}).get("file_dep", [])
            )
            if entity_nodes:
                ir["entities"] = entity_nodes
            if entity_edges:
                ir.setdefault("edges", {}).setdefault("entity_use", [])
                ir["edges"]["entity_use"] = entity_edges

    code_file_count = int(auto_stats.get("code_file_count", 0)) if isinstance(auto_stats, dict) else 0
    ast_total = len(ast_scanned_files) + len(ast_cached_files)
    ast_coverage = (ast_total / code_file_count) if code_file_count > 0 else 0.0
    import_estimate = 0
    import_capture_rate = None
    if code_file_count and code_file_count <= QUALITY_MAX_FILES and options.ast:
        import_estimate = estimate_import_count_rg(repo, languages, warnings=warnings, tools=tools)
        imports_captured = sum(
            len(node.get("imports", [])) + len(node.get("re_exports", []))
            for node in ir.get("files", {}).values()
        )
        if import_estimate > 0:
            import_capture_rate = imports_captured / import_estimate
            if import_capture_rate < 0.6:
                warnings.append(
                    f"Low import coverage: captured={imports_captured} estimated={import_estimate}"
                )

    ir["meta"]["quality"] = {
        "code_files_total": code_file_count,
        "ast_enabled": bool(options.ast),
        "ast_files_scanned": len(ast_scanned_files),
        "ast_files_cached": len(ast_cached_files),
        "ast_coverage": round(ast_coverage, 3),
        "ast_call_edges": len(ast_edge_info.get("call_edges", [])),
        "ast_dataflow_edges": len(ast_edge_info.get("dataflow_edges", [])),
        "ast_edge_confidence": ast_edge_info.get("confidence", "n/a"),
        "rg_fallback": bool(rg_fallback_used),
        "alias_resolved": alias_resolved_count,
        "codeql_enabled": bool(options.codeql),
        "codeql_mode": options.codeql_mode,
        "codeql_ran": bool(codeql_info.get("ran")) if isinstance(codeql_info, dict) else False,
    }
    if import_estimate:
        ir["meta"]["quality"]["import_estimate"] = import_estimate
    if import_capture_rate is not None:
        ir["meta"]["quality"]["import_capture_rate"] = round(import_capture_rate, 3)
    coupling_metrics = compute_coupling_metrics(ir)
    if coupling_metrics:
        ir["meta"]["quality"]["coupling"] = coupling_metrics

    trace_links = build_trace_links(docs, ir.get("entities", {}), ir.get("files", {}))
    if trace_links:
        ir["meta"]["trace_links"] = trace_links

    return {
        "ir": ir,
        "files": files,
        "docs": docs,
        "configs": configs,
        "tests": tests,
        "packages": load_package_jsons(repo, files, warnings),
        "warnings": warnings,
        "tool_versions": tool_versions,
    }
