# Codebase Map

An agent skill that gives your AI assistant the ability to analyze and map Rust, TypeScript, and Python repositories into structured architecture digests.

---

## What This Skill Does

When installed, your agent can analyze a repository and produce a compact summary of its architecture: what the modules do, where the entrypoints are, what routes and APIs exist, how data flows through the system, and what entities matter - all without modifying any source code.

Think of it as giving your agent an "understand this codebase" capability.

---

## Install

From the repository root (`public-published-skills`), copy the skill directory into your agent's skills folder:

```bash
cp -r skills/codebase-map ~/.agent/skills/codebase-map
```

Or symlink it:

```bash
ln -s /path/to/public-published-skills/skills/codebase-map ~/.agent/skills/codebase-map
```

The agent will pick up `SKILL.md` automatically.

---

## What You Get

Once installed, you can ask your agent things like:

- *"Map out this repo for me"*
- *"What are the API endpoints in this project?"*
- *"Show me the data models and how they relate"*
- *"Trace the auth flow end to end"*
- *"I just joined this project - give me the lay of the land"*

The agent will run the appropriate analysis and return a structured map covering modules, entrypoints, routes, entities, flows, and confidence signals.

---

## Supported Languages

| Language | Frameworks Detected |
|---|---|
| Rust | axum, actix-web, rocket, warp |
| Python | FastAPI, Django, Flask, click, typer |
| TypeScript / JS | Next.js, Express, Fastify, commander, yargs |

Framework detection is automatic based on project manifests (`Cargo.toml`, `pyproject.toml`, `package.json`).

---

## Profiles

The skill supports three analysis depths. You can ask for a specific one or let the agent choose:

| Profile | When to Use |
|---|---|
| **fast** | Quick look, low token cost |
| **balanced** | Good default for most repos |
| **deep** | Full analysis with call chains, AST tracing, optional CodeQL |

---

## Prerequisites

- Python 3.10+
- The target repository must be locally available (cloned)

### Verify the install

```bash
cd /path/to/public-published-skills/skills/codebase-map
python -m unittest tests/test_indexer.py tests/test_packer.py
```

---

## Configuration

The skill works out of the box with sensible defaults. For advanced tuning (token budgets, trace depth, route detection modes, confidence gates), see the reference docs:

- [`reference/PROFILES.md`](reference/PROFILES.md) - profile comparison
- [`reference/FLAGS.md`](reference/FLAGS.md) - all available flags
- [`reference/CONFIG.md`](reference/CONFIG.md) - configuration options
- [`reference/WORKFLOWS.md`](reference/WORKFLOWS.md) - workflow examples
- [`reference/TRACE.md`](reference/TRACE.md) - static trace details

---

## How It Works

The agent reads `SKILL.md`, which teaches it how to:

1. Detect the project's language and framework from manifests
2. Run `scripts/main.py` with the appropriate profile and flags
3. Return a structured map with confidence tags on every claim

Everything is read-only and static - no code is executed from the target repo, and nothing is modified.

---

## Limitations

- **Rust, TypeScript, and Python only.** Other languages aren't supported yet.
- **Static analysis.** No runtime tracing. Dynamic routing or heavy metaprogramming may produce incomplete results (the output will say so).
- **Local repos only.** The repo needs to be on disk.

---

## License

This repository currently has no `LICENSE` file. Add one at the repo root if you want explicit open-source licensing.
