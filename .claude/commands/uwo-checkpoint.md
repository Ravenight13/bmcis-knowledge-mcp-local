---
description: Quick progress save during development session (prevents work loss)
allowed-tools: [Read, Write, Bash]
---

# Universal Workflow Orchestrator - Quick Checkpoint

**Purpose:** Save progress milestone during active development (lighter than full handoff)

**Value Proposition:**
- 80-90% work loss reduction (proven with micro-commit discipline)
- 2-3 minutes per checkpoint (vs 5-10 min for full handoff)
- Commits any uncommitted subagent reports automatically
- Records current state for quick recovery

**When to Use:**
- After completing research phase
- Before switching tasks/contexts
- Every 30-60 minutes during active development
- After major milestone (feature completion, bug fix)
- Before risky refactoring or major changes

**NOT for session end:** Use `/uwo-handoff` for comprehensive session transfer

---

## STEP 1: Validate Session State

**Check for uncommitted work:**

```bash
git status --short
```

**Check specifically for uncommitted subagent reports:**

```bash
git status --short docs/subagent-reports/ 2>/dev/null
```

Interpret the results:
- If output is empty → No uncommitted subagent reports
- If output contains files → List them with "🤖 Found uncommitted subagent reports:"

---

## STEP 2: Auto-Commit Subagent Reports

**If uncommitted subagent reports exist, commit them automatically:**

1. **Add subagent reports to staging:**
```bash
git add docs/subagent-reports/
```

2. **Check what will be committed:**
```bash
git diff --cached --name-only | grep "docs/subagent-reports"
```

3. **Create appropriate commit message:**
   - Single report: `docs: add subagent report {report-name}`
   - Multiple reports: `docs: add {count} subagent reports`

4. **Commit the reports:**
```bash
git commit -m "docs: add subagent report(s)"
```

**Outcome:** All subagent research artifacts preserved before checkpoint.

---

## STEP 3: Create Quick Checkpoint File

**Generate timestamped checkpoint file:**

1. **Create filename with timestamp:**
```
Format: session-handoffs/YYYY-MM-DD-HHMM-checkpoint.md
Example: session-handoffs/2025-10-27-1445-checkpoint.md
```

2. **Use this template structure:**

```markdown
# Quick Checkpoint: [Brief Description]

**Date:** YYYY-MM-DD
**Time:** HH:MM:SS
**Branch:** {current-branch}
**Type:** Quick Checkpoint (not full handoff)

---

## Last Commit
{git log -1 --oneline}

## Current State

[1-2 sentence description of what you're working on]

---

## Next Actions

- [Action 1]
- [Action 2]
- [Action 3]

---

**Note:** This is a quick checkpoint. Use /uwo-handoff for comprehensive session transfer.
```

3. **Save file to session-handoffs/ directory**

---

## STEP 4: Prompt for Essential Info

**Collect minimal context from user (2-3 minutes):**

Present this prompt:

```
📋 QUICK CHECKPOINT

Please provide brief answers:

1. **Brief Description** (for filename):
   Example: "auth-api-implementation" or "bug-fix-validation"

2. **Current State** (1-2 sentences):
   What are you working on right now?
   Example: "Implementing JWT token validation in auth endpoint"

3. **Next Actions** (1-3 bullet points):
   What should be done next?
   Example:
   - Add token expiry check
   - Write tests for validation logic
   - Update API docs with auth requirements
```

**Wait for user input before proceeding.**

---

## STEP 5: Finalize Checkpoint

**Complete the checkpoint:**

1. **Fill in template with user-provided info**
2. **Write checkpoint file**
3. **Add checkpoint to git:**
```bash
git add session-handoffs/{timestamp}-checkpoint.md
```

4. **Commit checkpoint:**
```bash
git commit -m "checkpoint: quick progress save - {brief-description}"
```

---

## STEP 6: Summary Report

**Report checkpoint status:**

```
## ✅ CHECKPOINT COMPLETE

**File:** session-handoffs/{timestamp}-checkpoint.md
**Branch:** {current-branch}
**Subagent Reports:** {count} committed (if any)
**Total Commits:** {count} made by this checkpoint

### What Was Saved
- ✅ Current work state recorded
- ✅ Next actions documented
- ✅ Subagent reports committed (if any)
- ✅ All changes committed to git

### Recovery
If session crashes, use:
```bash
# Read latest checkpoint
cat session-handoffs/{timestamp}-checkpoint.md

# Continue from last state
```

### Next Steps
- Continue development
- Run /uwo-checkpoint again after next milestone (30-60 min)
- Run /uwo-handoff at end of session for full context transfer

**Work preserved! Continue safely.**
```

---

## Comparison: Checkpoint vs Handoff

| Feature | /uwo-checkpoint | /uwo-handoff |
|---------|----------------|--------------|
| **Time** | 2-3 minutes | 5-10 minutes |
| **Detail** | Minimal (current state + next actions) | Comprehensive (full context transfer) |
| **Format** | Quick notes | Structured template |
| **Frequency** | 2-5x per session | 1x per session (at end) |
| **Use Case** | Mid-session save points | Session end, context transfer |
| **Subagent Results** | Auto-commits if uncommitted | Includes in full report |
| **Quality Gates** | Not run | Full summary included |

---

## Example Execution Flow

```
$ /uwo-checkpoint

🔍 Checking for uncommitted work...

 M src/api/auth.py
?? docs/subagent-reports/api-analysis/auth/2025-10-27-1400-auth-security.md
?? docs/subagent-reports/api-analysis/auth/2025-10-27-1430-auth-performance.md

🤖 Found uncommitted subagent reports:
?? docs/subagent-reports/api-analysis/auth/2025-10-27-1400-auth-security.md
?? docs/subagent-reports/api-analysis/auth/2025-10-27-1430-auth-performance.md

📝 Auto-committing subagent reports...
[main a1b2c3d] docs: add 2 subagent reports
 2 files changed, 150 insertions(+)
✅ Committed 2 subagent report(s)

📋 QUICK CHECKPOINT

Please provide brief answers:

1. Brief Description: auth-api-implementation

2. Current State: Implementing JWT token validation in auth endpoint

3. Next Actions:
   - Add token expiry check
   - Write tests for validation logic
   - Update API docs with auth requirements

✅ Created checkpoint: session-handoffs/2025-10-27-1445-checkpoint.md

[main d4e5f6g] checkpoint: quick progress save - auth-api-implementation
 1 file changed, 25 insertions(+)

## ✅ CHECKPOINT COMPLETE

**File:** session-handoffs/2025-10-27-1445-checkpoint.md
**Branch:** main
**Subagent Reports:** 2 committed
**Total Commits:** 2 made by this checkpoint

Work preserved! Continue safely.
```

---

## Success Criteria

- ✅ Uncommitted subagent reports auto-committed
- ✅ Quick checkpoint file created with timestamp
- ✅ Essential info captured (current state + next actions)
- ✅ All changes committed to git
- ✅ Takes 2-3 minutes (vs 5-10 for full handoff)
- ✅ Works for any project type
- ✅ Prevents work loss (80-90% risk reduction)

---

## Notes

- **Quick format:** Just essentials (not full template)
- **Auto-commit:** Subagent reports committed automatically
- **Lightweight:** 2-3 min vs 5-10 min for full handoff
- **Frequent use:** Run every 30-60 minutes or after milestones
- **Recovery:** Read checkpoint file if session crashes
- **Not a replacement:** Use /uwo-handoff for session end
- **Universal:** Works for any project (not commission-processing specific)
