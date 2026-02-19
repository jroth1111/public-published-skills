# Test Scenarios for codebase-map

## Quick Test Checklist

Before deploying, verify each scenario:

### Trigger Tests (should activate)

| # | Prompt | Expected Behavior | Pass? |
|---|--------|-------------------|-------|
| 1 | "map the codebase" | Skill activates, follows instructions | [ ] |
| 2 | "generate a codebase map for this repo" | Skill activates, follows instructions | [ ] |
| 3 | "token-efficient repo summary" | Skill activates, follows instructions | [ ] |
| 4 | "extract call graph for this project" | Skill activates, follows instructions | [ ] |
| 5 | "run CodeQL flow analysis" | Skill activates, follows instructions | [ ] |

### Anti-Trigger Tests (should NOT activate)

| # | Prompt | Expected Behavior | Pass? |
|---|--------|-------------------|-------|
| 1 | "implement new functionality" | Skill does NOT activate | [ ] |
| 2 | "fix a bug" | Skill does NOT activate | [ ] |
| 3 | "refactor modules" | Skill does NOT activate | [ ] |

### Acceptance Test Verification

For each acceptance test in skill.spec.json, verify:

| # | Acceptance Test | Observed Behavior | Pass? |
|---|-----------------|-------------------|-------|
| 1 | "Creates a compact map output directory with summary, imports, and exports artifacts" | | [ ] |
| 2 | "Skips unavailable tools with a clear warning instead of failing hard" | | [ ] |
| 3 | "Generates CodeQL call/dataflow edge CSVs when CodeQL is available" | | [ ] |
| 4 | "Digest reports limits/truncation when caps or budgets omit data" | | [ ] |
| 5 | "repomap.json overrides docs/tests/configs/routes categorization" | | [ ] |

## Detailed Test Cases

### Test Case 1: Happy Path

**Setup:**
-- Input: repo_path=".", codeql="auto"
-- Context: TS/TSX or Python repo

**Steps:**
1. Provide the input to Claude
2. Verify skill activates
3. Check output matches expectations

**Expected Output:**
-- Files created: workspace/codebase-map/<repo>/*
-- Summary includes: file count, import count
- No errors or warnings

**Actual Result:**
- [ ] Pass / [ ] Fail
- Notes:

### Test Case 2: Edge Case - Empty Input

**Setup:**
- Input: (empty or minimal input)

**Expected Behavior:**
- Skill should ask for required information
- Should NOT crash or produce garbage output

**Actual Result:**
- [ ] Pass / [ ] Fail
- Notes:

### Test Case 3: Edge Case - Invalid Input

**Setup:**
- Input: (malformed or invalid input)

**Expected Behavior:**
- Clear error message
- Suggests how to fix

**Actual Result:**
- [ ] Pass / [ ] Fail
- Notes:

### Test Case 4: Digest Limits

**Setup:**
- Input: map with a small budget (e.g., 400-800)

**Expected Behavior:**
- Digest includes a `[LIMITS]` section
- `limits.truncated` is true when data is omitted

**Actual Result:**
- [ ] Pass / [ ] Fail
- Notes:

### Test Case 5: repomap.json Overrides

**Setup:**
- Add `repomap.json` with custom docs/tests/configs/routes globs

**Expected Behavior:**
- Output files reflect overrides (docs/tests/configs/routes)

**Actual Result:**
- [ ] Pass / [ ] Fail
- Notes:

## Automated Testing

Run trigger analysis:
```bash
python /path/to/skill-creator/scripts/analyze_triggers.py {baseDir}
```

Run effectiveness evaluation:
```bash
python /path/to/skill-creator/scripts/evaluate_skill.py {baseDir}
```

## Test Results Summary

| Category | Passed | Failed | Notes |
|----------|--------|--------|-------|
| Triggers | /5 | | |
| Anti-triggers | /3 | | |
| Acceptance | /5 | | |
| Edge cases | /4 | | |

**Overall Status:** [ ] Ready for use / [ ] Needs work

**Date Tested:** ____
**Tested By:** ____
