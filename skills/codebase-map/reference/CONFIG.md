# repomap.json

Optional repo‑local config file (repo root): `repomap.json` or `.repomap.json`.

Common keys:
```json
{
  "docs_globs": ["docs/**"],
  "tests_globs": ["**/*.test.ts"],
  "configs_globs": ["**/*.config.*"],
  "routes_globs": ["src/routes/**"],
  "routes_names": ["routes"],
  "routes_exclude_globs": ["**/__tests__/**"],
  "routes_mode": "auto",
  "entrypoints": {
    "frameworks": ["nextjs", "fastapi"],
    "max_items": 120,
    "max_per_group": 24,
    "include_middleware": false,
    "details_limit": 12,
    "details_per_kind": 4
  }
}
```

Also supported:
- `docs_exclude_globs`, `tests_exclude_globs`, `configs_exclude_globs`, `routes_exclude_globs`
- `entrypoints` block to tune entrypoint detection
