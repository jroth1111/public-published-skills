# Static Traces

Static traces infer linear flows through the repo graph (file + symbol + dataflow + entity edges).

## Confidence
- **high**: CodeQL dataflow edges present
- **medium**: symbol_ref/call edges present
- **low**: file_dep only

## Edge types
- file_dep (imports)
- symbol_ref (calls/refs)
- dataflow (CodeQL)
- entity_use (schema/model usage)
- client_call (HTTP/RPC client call surfaces)

## Start/End
- auto‑start picks client call sources, route files, top fan‑out, then entrypoints
- auto‑end picks entities, db/models, or top fan‑in

## Controls
- `--trace-depth` (max hops)
- `--trace-max` (max paths)
- `--trace-direction` (forward|reverse|both)
- `--trace-start`, `--trace-end` to constrain
