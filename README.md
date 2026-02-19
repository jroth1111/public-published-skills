# Public Published Skills

Portable Agent Skills repository designed to work across the `skills.sh` ecosystem.

## Install

Install all skills to all supported agents:

```bash
npx skills add jroth1111/public-published-skills --agent '*' --skill '*' -y
```

Install specific skills:

```bash
npx skills add jroth1111/public-published-skills --skill repo-hygiene --skill issue-triage -y
npx skills add jroth1111/public-published-skills --skill codebase-map -y
```

## Skill Docs

- `skills/codebase-map/README.md`

## Repo Layout

Each skill is a directory that contains a `SKILL.md` file:

```text
skills/
  repo-hygiene/
    SKILL.md
  issue-triage/
    SKILL.md
```

## Compatibility Rules

To maximize compatibility across agents:
- Keep frontmatter minimal: `name`, `description`.
- Avoid agent-specific features unless necessary (`context: fork`, hooks).
- Keep instructions tool-agnostic and filesystem-safe.

## Validate Quickly

```bash
find skills -name SKILL.md -maxdepth 3 -print
npx skills add . --list
```
