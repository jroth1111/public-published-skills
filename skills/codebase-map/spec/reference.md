# Codebase Map (reference)

## Intent
- Goals:
  - Produce a compact, token-efficient map of a repo without dumping code
  - Capture structural edges (imports/exports/symbols) and optional flows (CodeQL)
- Non-goals:
  - Implement features or modify source code
  - Perform runtime tracing or execute tests/builds

## Inputs/Outputs
- Inputs:
  - repo path (default: current directory)
  - languages: auto or list (ts, tsx, js, jsx, py)
  - ast: auto/on/off
  - codeql: auto/on/off (default off; enable for deep flow analysis)
  - profile: fast/balanced/deep (optional preset; overrides individual flags)
  - out: output directory under workspace or absolute path (default: workspace/codebase-map/<repo-name>)
  - budget: token budget for digest/expand
  - budget-margin: fraction of budget to cap at (default: 1.0; use <1.0 for safety)
  - max-files/max-symbols/max-edges/max-sig-len (optional digest caps)
  - entry-details-limit/entry-details-per-kind (optional entry detail caps)
  - artifacts: minimal|standard|full (default: standard)
  - code-only (default on), compress-paths (default on)
  - dense, dir-depth, dir-file-cap (default dense on; layout extras)
  - include-routes, routes-mode, routes-limit (default 12), routes-format (default include on; routes/API section)
  - diagrams (default compact+sequence), diagram-depth, diagram-nodes, diagram-edges (compact or compact+sequence diagram)
  - repomap.json or .repomap.json (optional overrides for docs/tests/configs/routes)
- Outputs:
  - summary.txt, summary.json
  - ir.json (normalized index)
  - digest.txt, digest.json
  - artifacts minimal: summary + ir
  - artifacts standard (default): files.txt, docs.txt, configs.txt, tests.txt, imports.tsv, exports.tsv, packages.tsv (if any)
  - artifacts full: adds py_symbols.tsv, ts_symbols.tsv, dataflow.tsv when CodeQL flows are available
  - codeql/*.csv when CodeQL runs
  - graph export via `export` subcommand (JSON or GraphML)
  - digest summary includes `unsupported_langs` and `PACKAGES` when detected
  - digest includes `DIR` and `TOP_HUBS` sections when dense output is enabled
  - digest includes `DIAGRAM` when diagrams are enabled and `DIAGRAM_SEQ` for compact+sequence
  - digest includes `roles` and `role_edges` summary lines (path heuristics)
  - digest includes `ui_groups`, `api_groups`, and `archetypes` in `ROUTES` (when compact)
  - digest includes `COMPONENTS` and `COMPONENT_EDGES` (component clusters + dependencies)
  - digest includes `PUBLIC_API` (barrel/entrypoint exports), `ENTRYPOINTS` + `MIDDLEWARE` + `ENTRYPOINT_GROUPS` + `ENTRY_DETAILS` (entrypoint inventory + summaries with input/precondition hints), `ENTITIES` (heuristic models), and `FLOWS` (entrypoint/route flows)
  - digest summary may include `configs`, `schemas`, `external`, `risk_flags`, and `cycles`

## Archetype
- basic
- Entry point: scripts/main.py

## Filesystem pattern
- Save raw artifacts under workspace/
- Print only summaries + small previews

## Discovery
- File listing uses `fd` when available; falls back to `rg --files`, then Python walk.
- Routes section is heuristic based on path/name matches plus best-effort decorator/method extraction for FastAPI/Flask/Django/Express and TanStack file routes; framework-agnostic and best-effort for conventional layouts (Express, Next.js, Django, Flask, FastAPI), no full router graph. Digest labels this as `POTENTIAL_ROUTE_FILES` or `ROUTES` (compact format) and includes `routes_mode`. Route groups and archetypes are best-effort and rely on file path heuristics and file_dep edges. Override with `repomap.json` if needed.

## Repo Config
- docs_globs/docs_exclude_globs, tests_globs/tests_exclude_globs, configs_globs/configs_exclude_globs
- routes_globs/routes_names/routes_exclude_globs and routes_mode (auto/heuristic/nextjs/fastapi/django/express)
- entrypoints.frameworks/max_items/max_per_group/include_middleware/details_limit/details_per_kind

## One-step mapping
- `map` subcommand runs `index` then `digest` and prints the digest to stdout.
- `digest.json` includes a `limits` block that reports caps/budget omissions.
- `map --profile fast` applies a fast preset (low budget, CodeQL off); `--quick` remains as a deprecated alias.

## Warnings
- Digest output includes a `[WARNINGS]` section (and `warnings` in JSON) when tool or parsing warnings occur.

## Diff mode
- `diff` compares the current repo against the cached `ir.json` and reports added/removed/changed files and symbols.

## IR Schema
- See `spec/ir-schema.json` for the current JSON schema of `ir.json`.

## Module Map
- See `spec/modules.md` for module boundaries and responsibilities.

## Edge cases
- Missing tools: skip step and warn
- CodeQL database creation fails on unusual build setups
- Large repos: keep CodeQL off unless flows are needed
- No file changes with --incremental: returns cached IR

## Best practice (stateless + changing repos)
1. Run `index --incremental` before `digest`.
2. Keep a stable `--out` path per repo to persist `ir.json` and `digest.json`.
3. Default profile is balanced (CodeQL off); use `--profile deep` or `--codeql on` for flows.

## Version history
- v0.1.0: initial scaffold
- v0.2.0: implemented codebase mapping and CodeQL flows
- v0.3.0: added IR + digest/expand + ranking
- v0.4.0: added dataflow edges in IR, digest artifacts, alias expansion
- v0.5.0: tightened token budgeting, code-only digest, path compression
- v0.6.0: added summary/routes sections and callgraph-weighted ranking
- v0.7.0 (current): added --install-guide, --auto-refresh, --explain-ranking, --profile, framework auto-detection, edge case robustness (IR schema v4)
