# codebase-map

`codebase-map` maps Rust, TypeScript, and Python repositories into a compact, evidence-backed architecture digest.

## What It Does

- Identifies modules and responsibilities
- Extracts entrypoints
- Surfaces routes/APIs (when detectable)
- Surfaces entities/schemas (when detectable)
- Produces flow/path traces (when enabled)
- Emits confidence/warning signals for uncertain inference

## What It Does Not Do

- It does not modify source code
- It does not claim capabilities without direct code evidence

## Usage

Run from the skill folder or with `{baseDir}` expanded by your agent runtime.

### Quickstart

```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
```

### Common Workflows

```bash
# Onboarding map
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced

# API + route surface
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --api-contracts --include-routes --routes-mode auto

# Data model understanding
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --entity-graph

# Flow/path tracing
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced --static-traces --trace-depth 3 --trace-max 6

# Targeted exploration
python {baseDir}/scripts/main.py query --repo /path/to/repo --out codebase-map/my-repo --query auth --depth 2
python {baseDir}/scripts/main.py expand --repo /path/to/repo --out codebase-map/my-repo --focus auth --depth 2
```

## Profiles

- `fast`: quick structural scan, minimal depth
- `balanced`: default profile for first-pass architecture understanding
- `deep`: maximum detail and deeper traces

## Key Flags

- `--profile` (`fast|balanced|deep`)
- `--out` (output directory)
- `--focus` (topic boundary)
- `--budget` (token budget)
- `--static-traces` (enable flow tracing)
- `--include-routes` (include route/API section)
- `--routes-mode` (`auto|heuristic|nextjs|fastapi|django|express|rust|go`)
- `--api-contracts` (extract API contracts)
- `--entity-graph` (include entity graph)
- `--gate-confidence` (non-zero exit if confidence gate fails)

## Repository Contents

- `SKILL.md`: skill contract and trigger description
- `scripts/`: implementation
- `reference/` and `references/`: compact operational references
- `tests/`: validation suite

## Verification

```bash
python {baseDir}/scripts/main.py map --repo /path/to/repo --out codebase-map/my-repo --profile balanced
cd {baseDir} && python -m unittest tests/test_indexer.py tests/test_packer.py
```
