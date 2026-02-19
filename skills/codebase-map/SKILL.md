---
name: codebase-map
description: Maps Rust, TypeScript, and Python repos into an evidence-backed, compact digest (modules, entrypoints, routes/APIs, flows, entities, confidence/warnings). Use when mapping or summarizing a codebase, including onboarding and architecture understanding. Not for modifying source code.
metadata:
  short-description: Codebase mapping
allowed-tools: Read Write Edit Grep Glob Bash
---

# Codebase Map Skill

Generate a compact, framework-aware codebase map for Rust/TypeScript/Python using adapter detection and confidence-tagged evidence.

## Quickstart
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
```

## Profiles (3)
- `fast`: quick structural scan, low token budget, no routes/diagrams/traces.
- `balanced` (default): best first map; entrypoints + routes + entities + static traces + API contracts.
- `deep`: max understanding; AST-first flows/call-chains, CodeQL auto augment (runs only when AST confidence is low), static traces deeper, flow symbols, call chains, doc quality, semantic clustering.

## 5 Common Workflows
1) **Repo onboarding**
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
```
2) **API + route surface**
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --api-contracts --include-routes --routes-mode auto
```
3) **Data model understanding**
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --entity-graph
```
4) **Flow/path tracing**
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --static-traces --trace-depth 3 --trace-max 6
```
5) **Targeted search / expansion**
```bash
python {baseDir}/scripts/main.py query --repo /path/to/repo --out codebase-map/my-repo --query auth --depth 2
python {baseDir}/scripts/main.py expand --repo /path/to/repo --out codebase-map/my-repo --focus auth --depth 2
```

## Key Flags
- `--profile` (fast|balanced|deep)
- `--out` (workspace output dir)
- `--focus` (narrow to a topic)
- `--budget` (token budget)
- `--static-traces` (infer end‑to‑end flows)
- `--include-routes` (include route/API section)
- `--routes-mode` (auto|heuristic|nextjs|fastapi|django|express|rust|go)
- `--api-contracts` (include API contract extraction)
- `--entity-graph` (include entity-graph details)
- `--gate-confidence` (enforce confidence gate with non-zero exit on fail)

## Adapter Policy (Required)
- Detect ecosystem/framework hints from repository manifests and entrypoint evidence.
- Rust: use `Cargo.toml` plus Rust entrypoint and web/router signals (for example `axum`, `actix`, `rocket`, `warp`).
- Python: use `pyproject.toml`/packaging entrypoints plus CLI/web signals (for example `argparse`, `click`, `typer`, `FastAPI`, `Django`, `Flask`).
- TypeScript/JavaScript: use `package.json` scripts/bin plus CLI/web signals (for example `commander`, `yargs`, `Next.js`, `Express`, `Fastify`).
- If adapter confidence is weak, surface uncertainty via confidence/warning output instead of asserting unsupported behavior.

## Output Contract (Mandatory)
Every map output should include, when detectable from the repo and budget:
- module/component responsibilities
- entrypoints and roles
- route/API or command-like surface
- key entities/schemas
- primary flows/traces
- confidence/warning signals and explicit unknowns

## Advanced Details
See `reference/` for deep usage and full flag lists:
- `reference/PROFILES.md`
- `reference/WORKFLOWS.md`
- `reference/SECTIONS.md`
- `reference/FLAGS.md`
- `reference/CONFIG.md`
- `reference/TRACE.md`
Do not emit capability claims without direct code evidence; tag weaker inference as low-confidence.

## Verification
Run these before finishing:
```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
cd {baseDir} && python -m unittest tests/test_indexer.py tests/test_packer.py
```
