# Profiles

_Generated from scripts/cli/profiles.py. Do not edit manually._

## fast
**Goal:** quick scan, minimal tokens.

Defaults:
- codeql=off
- budget=4000
- budget_margin=0.85
- max_files=20
- max_symbols=2
- max_edges=25
- max_sig_len=60
- dense=true
- dir_depth=0
- diagrams=off
- include_routes=false
- api_contracts=false
- static_traces=false
- trace_depth=2
- trace_max=3
- flow_symbols=false
- semantic_cluster=false
- doc_quality=false
- diagram_nodes=6
- diagram_edges=8
- artifacts=standard
- explain_ranking=false

Use when:
- very large repos
- you want a light overview

---

## balanced
**Goal:** best first map for LLMs under ~30k tokens.

Defaults:
- codeql=off
- budget=20000
- budget_margin=0.85
- max_files=60
- max_symbols=4
- max_edges=120
- dense=true
- dir_depth=3
- diagrams=compact+sequence
- include_routes=true
- api_contracts=true
- static_traces=true
- trace_depth=3
- trace_max=6
- diagram_nodes=8
- diagram_edges=12
- artifacts=standard
- explain_ranking=false
- section_budgets:
  - API_CONTRACTS: 0.06
  - DIAGRAM: 0.06
  - DOCS: 0.03
  - ENTITIES: 0.08
  - FILES: 0.3
  - FLOWS: 0.08
  - LIMITS: 0.02
  - QUALITY: 0.04
  - ROUTES: 0.08
  - STATIC_TRACES: 0.06
  - SUMMARY: 0.08
  - TRACE_GRAPH: 0.06
  - WARNINGS: 0.02

Use when:
- onboarding
- most Q&A tasks

---

## deep
**Goal:** maximum semantic understanding while staying bounded.

Defaults:
- codeql=auto
- budget=35000
- budget_margin=0.85
- max_files=120
- max_symbols=8
- max_edges=250
- max_sig_len=120
- dense=true
- dir_depth=3
- diagrams=compact+sequence
- include_routes=true
- api_contracts=true
- static_traces=true
- trace_depth=4
- trace_max=12
- flow_symbols=true
- call_chains=true
- semantic_cluster=true
- doc_quality=true
- diagram_nodes=12
- diagram_edges=18
- artifacts=full
- explain_ranking=true
- reuse_codeql_db=true
- keep_codeql_db=true
- section_budgets:
  - API_CONTRACTS: 0.07
  - CALL_CHAINS: 0.03
  - DIAGRAM: 0.06
  - DOCS: 0.05
  - DOCS_QUALITY: 0.03
  - ENTITIES: 0.09
  - FILES: 0.28
  - FLOWS: 0.09
  - LIMITS: 0.02
  - QUALITY: 0.05
  - ROUTES: 0.08
  - STATIC_TRACES: 0.08
  - SUMMARY: 0.06
  - TRACE_GRAPH: 0.07
  - WARNINGS: 0.02

Use when:
- deep architectural review
- call/dataflow analysis

Notes:
- AST edges are primary for Python/TS/Rust.
- CodeQL is an optional high-precision augmenter.
- Use `--codeql on` to force CodeQL even when AST confidence is high.

---
