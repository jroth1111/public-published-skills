# Digest Sections

## Core map (highest value)
- `[SUMMARY]` repo overview + top dirs + key counts
- `[FILES]` file anchors + top symbols
- `[ENTRYPOINTS]` runtime entrypoints (HTTP/CLI/jobs)
- `[ROUTES]` UI/API route inventory
- `[API_CONTRACTS]` structured endpoints (if enabled)
- `[ENTITIES]` schema/model inventory + uses

## Flow reasoning
- `[STATIC_TRACES]` inferred flow paths (forward + reverse)
- `[TRACE_GRAPH]` branching flow graph from starts
- `[FLOWS]` linear file‑level chains (fallback when static traces are thin)

## Semantic framing
- `[DOCS]` docstring/JSDoc excerpts
- `[ARCH_STYLE]` layered/clean/hex hints
- `[LAYERS]` layer assignments
- `[PATTERNS]` light pattern signals
- `[PURPOSE]` component summaries

## Diagnostics / hygiene
- `[QUALITY]` coverage + coupling summary
- `[DOCS_QUALITY]` missing/low‑quality docs
- `[DEAD_CODE]` low‑confidence unused exports
- `[CONFIDENCE_GATE]` pass/fail checks for confidence readiness

## Visual / auxiliary
- `[DIAGRAM]` compact component diagram
- `[DIAGRAM_SEQ]` flow sequence summary
- `[CALL_CHAINS]` bounded call chains

## Meta
- `[LIMITS]` budget + truncation info
- `[WARNINGS]` parse/heuristic warnings
- `[LEGEND]` alias usage
