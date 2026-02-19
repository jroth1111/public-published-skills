# Flags (advanced)

## Budgeting
- `--budget` token budget (map/digest)
- `--budget-margin` safety margin
- `--section-budgets` JSON override per section
- `--max-files`, `--max-symbols`, `--max-edges`, `--max-sig-len`

## Flows / traces
- `--static-traces`, `--trace-depth`, `--trace-max`, `--trace-direction`
- `--trace-start`, `--trace-end`
- `--flow-symbols`
- `--call-chains`

## Routes / APIs
- `--include-routes`
- `--routes-mode` (auto|heuristic|nextjs|fastapi|django|express)
- `--routes-limit`, `--routes-format`
- `--api-contracts`

## Entities
- `--entity-graph` (uses entity edges in digest)

## Semantics
- `--semantic-cluster`
- `--semantic-cluster-mode` (tfidf|name_only)
- `--semantic-call-weight`
- `--doc-quality`, `--doc-quality-strict`

## Output formatting
- `--code-only`, `--compress-paths`, `--dense`, `--dir-depth`, `--dir-file-cap`
- `--diagrams` (off|compact|compact+sequence|mermaid)
- `--diagram-depth`, `--diagram-nodes`, `--diagram-edges`
- `--stream-digest`
- `--gate-confidence` fail fast when confidence gate checks fail

## Indexing
- `--ast` (auto|on|off)
- `--codeql` (auto|on|off): `auto` runs only when AST edge confidence is low; `on` forces high-precision augmentation.
- `--workers`
- `--semantic-hash-cache`

## Querying
- `query --query <symbol> --depth <n>`
- `search --query <term> --depth <n> --direction both|in|out`
- `expand --focus <term> --depth <n>`
