---
name: repo-hygiene
description: Audits repository hygiene and proposes safe cleanup steps. Use when checking repo health, structure, and stale artifacts. Not for feature implementation.
---

# Repo Hygiene

## Goal
Produce a safe, prioritized cleanup report without destructive actions.

## Steps
1. Inspect git status, ignored files, and obvious stale directories.
2. Identify low-risk cleanup opportunities first.
3. Propose exact commands and expected effects.
4. Ask before any destructive operation.

## Output Format
- Findings
- Recommended actions
- Commands
- Risks
