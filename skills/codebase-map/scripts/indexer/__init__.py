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
    resolve_js_module,
    resolve_module_candidates,
    resolve_py_module,
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


from .core import (
    IndexOptions,
    deep_copy_ir,
    diff_ir,
    expand_changed_with_dependents,
    index_repo,
    load_extra_cache,
    prune_ir,
    remove_path,
    save_extra_cache,
)
