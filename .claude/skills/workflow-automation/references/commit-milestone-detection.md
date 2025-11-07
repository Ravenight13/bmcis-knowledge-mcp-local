# Commit Milestone Recognition

**Purpose**: Decision trees for when Claude recognizes and prompts for commits.

---

## How Claude Recognizes When to Prompt for Commits

### Decision Tree

```
Check working directory status
├─ 20-50 lines changed? → Prompt for commit
├─ 30+ minutes since last commit? → Prompt for commit
├─ Logical milestone reached?
│   ├─ Helper function complete → Prompt
│   ├─ Test file created → Prompt
│   ├─ Research artifact saved → Prompt
│   └─ Component implemented → Prompt
└─ None of above → Continue work
```

---

## Detection Commands

**Line changes**:
```bash
git diff --stat
# or
git status --porcelain | wc -l
```

**Time since commit**:
```bash
git log -1 --format='%cr'  # Relative time (e.g., "25 minutes ago")
git log -1 --format='%ct'  # Unix timestamp
```

**Milestone indicators**:
- Subagent reports "complete", "done", "working", "tested"
- User says "that's working" or "tests pass"
- File count increases (new file created)
- Quality gates pass after failing

---

## Prompting Examples

### 20-50 Lines Changed

**Prompt**:
```
I see 35 lines changed since last commit. This is a good checkpoint.
Let me commit this work before continuing.
```

**Action**: Run `checkall`, then commit with descriptive message

---

### 30 Minutes Elapsed

**Prompt**:
```
It's been 40 minutes since last commit (last: {timestamp}).
To prevent work loss, I'll commit current progress.
```

**Action**: Commit as WIP if work incomplete, or with full message if logical milestone

---

### Helper Function Complete

**Prompt**:
```
The calculate_unit_price() helper is working and tested.
Let me commit this before continuing to next component.
```

**Action**: Commit with feat/refactor type, clear scope

---

### Research Artifact Saved

**Prompt**:
```
Research report saved to docs/subagent-reports/.
Committing immediately to preserve research artifacts.
```

**Action**: Use `docs(research): {description}` commit message

---

### Test Created/Passing

**Prompt**:
```
Tests now passing after implementing {feature}.
This is a natural commit boundary.
```

**Action**: Commit with test(scope) or feat(scope) type

---

## Threshold Tuning

### Task-Specific Thresholds

**Small tasks** (bug fixes, documentation):
- Commit every 20 lines
- Commit at logical boundaries (no time threshold needed)

**Large refactors** (framework changes, vendor rewrites):
- Commit every 50 lines
- Strict 30-minute time threshold
- Commit at phase boundaries (research → plan → implement)

**Research tasks**:
- Commit immediately when artifact saved
- Use `/cc-checkpoint` for automatic commits
- No line threshold (preserve all research)

**Debugging**:
- Commit after each fix validated
- Use debug loop phases (TEST → EVALUATE → PLAN → FIX → COMMIT)
- No line threshold (track each iteration)

---

## When NOT to Prompt for Commit

**Don't prompt if**:
- Less than 10 lines changed (too small)
- Work is in broken state (tests failing, won't run)
- Mid-refactor (file half-edited)
- User explicitly said "don't commit yet"

**Wait for**:
- Quality gates pass (`checkall` succeeds)
- Logical boundary reached
- User signals readiness ("this is working")

---

## Prompting Pattern Summary

**Pattern**: {observation} → {recommendation} → {action}

**Example**:
```
Observation: "35 lines changed across 3 files"
Recommendation: "This is a good checkpoint for a commit"
Action: "Let me run checkall and commit with message: feat(coastal-source): implement brand mapping logic"
```
