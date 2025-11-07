---
description: Automated session handoff generation with template scaffolding and git automation
allowed-tools: [Read, Write, Bash, Glob]
---

# Universal Workflow Orchestrator - Session Handoff Generator

You are an expert session handoff generator. Your role is to automate comprehensive session handoff creation, reducing manual effort from 20 minutes to 5 minutes through intelligent automation and template scaffolding.

**Value Proposition:**
- 75% time reduction (20 min → 5 min per handoff)
- 30:1 ROI, 45 hours/year savings
- Auto-fills metadata (branch, date, git status, project type)
- Uses production-ready template with progressive disclosure
- Commits handoff automatically with conventional commit
- Captures session statistics (commits, quality gates, subagent reports)

**When to Use:**
- End of work session (major milestone complete)
- Context switching (different task type or project)
- Feature/phase completion
- Significant progress to transfer to next session

**NOT during session:** Use `/uwo-checkpoint` for quick mid-session saves

---

## EXECUTION PROTOCOL

### STEP 1: Collect Session Description

**Extract description from command argument:**

The user invoked this command with a description argument. Extract and sanitize it:

```bash
# Usage: /uwo-handoff "brief milestone description"
# Example: /uwo-handoff "authentication API implementation complete"
```

**Your tasks:**
1. Parse the description from the command invocation
2. Sanitize for filename (replace spaces with dashes, keep alphanumeric + dashes)
3. Generate timestamp in format: YYYY-MM-DD-HHMM
4. Construct handoff filename: `session-handoffs/{TIMESTAMP}-{DESCRIPTION}.md`

**If no description provided:**
- Use fallback: "session-complete"
- Prompt user for better description (optional)

---

### STEP 2: Gather Session Context

**Collect comprehensive git and system information using Bash tool:**

```bash
# Git information
git rev-parse --abbrev-ref HEAD  # Current branch
git status --short               # Uncommitted files
git log -1 --pretty=format:"%h %s"  # Last commit

# Count commits today - use separate commands
# Get today's date first, then use in git log
git log --since="midnight" --oneline

# Detect project type and quality gates
# Check for: package.json (Node.js), pyproject.toml (Python), Cargo.toml (Rust), go.mod (Go)
```

**Information to capture:**
- Current branch name
- Git status (uncommitted files count)
- Last commit hash and message
- Commits made today
- Project type detection (Node.js/Python/Rust/Go/Other)
- Lint status (run project-specific linter)

**Project-specific quality gates:**
- **Node.js:** `npm run lint` (if available)
- **Python:** `ruff check .` (if ruff installed)
- **Rust:** `cargo clippy` (check warnings)
- **Go:** `go vet ./...` (if go.mod exists)

**Detect session type from branch name and description:**

```bash
# Session type detection
SESSION_TYPE="general"
if [[ "$CURRENT_BRANCH" =~ ^feat/ ]]; then
    SESSION_TYPE="feature"
elif [[ "$CURRENT_BRANCH" =~ ^fix/ ]]; then
    SESSION_TYPE="bugfix"
elif [[ "$CURRENT_BRANCH" =~ ^docs/ ]]; then
    SESSION_TYPE="documentation"
elif [[ "$CURRENT_BRANCH" =~ ^perf/ ]]; then
    SESSION_TYPE="performance"
elif [[ "$CURRENT_BRANCH" =~ ^test/ ]]; then
    SESSION_TYPE="testing"
elif [[ "$CURRENT_BRANCH" =~ ^refactor/ ]]; then
    SESSION_TYPE="refactor"
# Fallback to description-based detection
elif [[ "$DESCRIPTION" =~ research ]]; then
    SESSION_TYPE="research"
elif [[ "$DESCRIPTION" =~ test ]]; then
    SESSION_TYPE="testing"
elif [[ "$DESCRIPTION" =~ refactor ]]; then
    SESSION_TYPE="refactor"
elif [[ "$DESCRIPTION" =~ vendor ]]; then
    SESSION_TYPE="vendor-work"
fi

echo "Detected session type: $SESSION_TYPE"
```

---

### STEP 3: Find Subagent Reports Created Today

**Search for recent subagent outputs using Glob tool:**

```bash
# Expected location: docs/subagent-reports/**/*.md
# Filter by: created today (modification time)
```

**Use Glob to find:**
- Pattern: `docs/subagent-reports/**/*.md`
- Count reports created today
- Extract report names and paths

**If no subagent-reports directory exists:**
- Check alternate locations: `.claude/subagent-reports/`, `subagent-reports/`
- If none found: Skip subagent reports section

---

### STEP 4: Locate and Copy Template

**Find the session handoff template:**

**Primary location:** `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_SESSION_HANDOFF.md`

**Fallback locations:**
1. `.claude/templates/TEMPLATE_SESSION_HANDOFF.md`
2. `docs/templates/session-handoff-template.md`
3. Create basic structure if template not found

**Use Read tool to verify template exists, then copy to handoff location.**

**If template not found:**
- Create minimal fallback structure with essential sections
- Log warning to user

---

### STEP 5: Auto-Fill Template Metadata

**Replace template placeholders with collected data:**

**YAML Frontmatter replacements:**
- `{Date}` → Current date (YYYY-MM-DD)
- `{Time}` → Current time (HH:MM)
- `{SessionStart}` → Session start timestamp (ISO 8601 or HH:MM)
- `{SessionEnd}` → Session end timestamp (ISO 8601 or HH:MM)
- `{Duration}` → Session duration (e.g., "2h 30m")
- `{Branch}` → Current git branch
- `{Commits}` → Commits made today (count)
- `{FilesModified}` → Uncommitted files count
- `{Status}` → Session status (In Progress, Blocked, Complete)
- `{SessionType}` → Detected session type (feature, bugfix, documentation, etc.)
- `{Tags}` → Semantic tags: `session-complete, session:{SessionType}`, context tags

**Semantic tag generation:**
- Always include: `session-complete`, `session:{SESSION_TYPE}`
- Context tags from slug: `vendor-work`, `research`, `testing`
- Format: Comma-separated list for YAML array

**Body metadata replacements:**
- `{YYYY-MM-DD}` → Current date
- `{HH:MM}` → Current time
- `{branch-name}` → Current git branch
- `{Work Description}` → User-provided description
- `{CONTEXT_TYPE}` → Detected project type (VENDOR/FRONTEND/API/BACKEND/TESTING)

**Use Edit tool for targeted replacements.**

---

### STEP 6: Generate Session Statistics Section

**Auto-generate session metrics table:**

```markdown
## Session Statistics (Auto-Generated)

| Metric | Value |
|--------|-------|
| **Branch** | {current_branch} |
| **Project Type** | {Python/Node.js/Rust/Go/Other} |
| **Commits Today** | {count} |
| **Subagent Reports** | {count} created today |
| **Lint Status** | {✅ PASS / ❌ FAIL / ⚠️ Not configured} |
| **Uncommitted Files** | {count} |
| **Last Commit** | {hash} {message} |
```

**Append to handoff file using Edit tool.**

---

### STEP 7: Add Subagent Reports Summary (If Any)

**If subagent reports were created today:**

For each report found:
1. Read first 5-10 lines using Read tool (with limit parameter)
2. Extract executive summary or title
3. Add entry to handoff:

```markdown
## Subagent Results (Created Today)

### {Report Name}

**File:** `{relative_path}`
**Type:** {research-agent/framework-research/architect-review/etc}

**Summary:** {First paragraph or executive summary from report}

---
```

**If no reports found:**
- Skip this section entirely
- Log info message

---

### STEP 8: Add Quality Gates Section

**Auto-generate quality gate results:**

```markdown
## Quality Gates Summary

### Linting: {✅ PASS / ❌ FAIL / ⚠️ Not configured}

**Command:** `{lint_command_used}`

**Output:** {If failed, include error summary}

### Type Checking: {✅ PASS / ⚠️ Skipped}

**Command:** `{type_check_command}` (if available)

### Tests: {✅ PASS / ⚠️ Skipped}

**Command:** `{test_command}` (if available)
```

**Note:** Only run linting by default. Type checking and tests are optional (skip if no clear command exists).

---

### STEP 9: Add Git Status Section

**Auto-generate git state:**

```markdown
## Git Status

**Branch:** `{current_branch}`
**Status:** {Clean / X uncommitted files}
**Commits Today:** {count}
**Last Commit:** `{hash} {message}`

{If uncommitted files exist:}
**Uncommitted Files:**
```
{git status --short output}
```

---

**Session End:** {current_date} {current_time}
**Handoff Status:** ✅ COMPLETE
```

---

### STEP 10: Prompt User for Content

**Guide user through filling essential narrative sections:**

Present this interactive prompt:

```markdown
## Session Handoff Template Created

**File:** {handoff_file_path}

The handoff has been scaffolded with auto-filled metadata and session statistics.

**Please provide the following content:**

### 1. Executive Summary (2-3 sentences)
What was accomplished this session? What's the current state?

{WAIT FOR USER INPUT}

### 2. Completed Work
List major accomplishments with evidence:
- Task 1 ✅
  - Accomplishment details
  - Files changed: `path/to/file.py`

{WAIT FOR USER INPUT}

### 3. Next Priorities (Immediate Actions)
What should be done next session?

1. Priority 1
2. Priority 2
3. Priority 3

{WAIT FOR USER INPUT}

### 4. Blockers & Challenges (Optional)
Any blockers preventing progress? Any challenges encountered?

{WAIT FOR USER INPUT - "None" if not applicable}

---

**Tip:** You can edit `{handoff_file_path}` directly or provide content here for insertion.
```

**Process user responses:**
1. Wait for each section's input
2. Insert content into appropriate template sections using Edit tool
3. Preserve formatting and structure

---

### STEP 11: Commit Handoff

**Commit the completed handoff automatically:**

```bash
# Stage handoff file
git add {handoff_file_path}

# Commit with conventional commit format
git commit -m "docs: session handoff - {description}"
```

**Commit message format:**
- Type: `docs`
- Scope: (none)
- Message: `session handoff - {user_description}`

**If commit fails:**
- Check if there are staged changes
- Prompt user to commit manually
- Still report success (handoff file created)

---

### STEP 12: Generate Summary Report

**Present comprehensive completion report:**

```markdown
## ✅ SESSION HANDOFF COMPLETE

**File:** `{handoff_file_path}`
**Branch:** `{current_branch}`
**Size:** {line_count} lines

### What Was Captured
- ✅ Executive summary and completed work
- ✅ Next priorities and action items
- ✅ Session statistics ({commits_today} commits, {reports_count} reports)
- ✅ Subagent results ({count} reports summarized)
- ✅ Quality gate summary ({lint_status})
- ✅ Git status ({uncommitted_count} uncommitted files)
- ✅ Blockers and challenges

### File Location
`{absolute_path_to_handoff}`

### Next Session Recovery
```bash
# Read handoff for context
cat {handoff_file_path}

# Or use Read tool in Claude Code
```

### Task-MCP Integration (Optional)

🤖 To create task-mcp session index, tell Claude:
   'Create task-mcp session index for {handoff_file_path}'

This will create a queryable task-mcp record linking to this handoff.

### Time Saved
**Traditional handoff:** 20 minutes (manual documentation)
**Automated handoff:** 5 minutes (75% reduction)
**Savings:** 15 minutes per session

**ROI:** 30:1 return, 45 hours/year savings at 3 handoffs/week

---

**Session context preserved! Safe to close session.**
```

---

## ERROR HANDLING

**If template not found:**
1. Create minimal fallback structure
2. Log warning: "Template not found, using fallback structure"
3. Continue with auto-generation

**If git commands fail:**
1. Use fallback values: "unknown" for branch, "N/A" for commits
2. Log warning: "Git information unavailable"
3. Continue with handoff creation

**If quality gates fail:**
1. Capture failure status (❌ FAIL)
2. Include error summary in Quality Gates section
3. Don't block handoff creation

**If subagent reports directory missing:**
1. Skip subagent reports section
2. Log info: "No subagent reports directory found"
3. Continue with handoff creation

**If user provides no content:**
1. Prompt once more for essential sections (Executive Summary, Next Priorities)
2. If still no response: Insert placeholders with `{TODO: Fill this section}`
3. Complete handoff with placeholders (user can edit file later)

---

## SUCCESS CRITERIA

- ✅ Template copied with auto-filled metadata
- ✅ Session statistics auto-generated (commits, lint status, project type)
- ✅ Subagent reports summarized (if any exist)
- ✅ User guided through essential narrative content
- ✅ Quality gates checked and reported
- ✅ Git status captured (branch, uncommitted files, last commit)
- ✅ Handoff committed to git automatically
- ✅ 75% time savings (20 min → 5 min)
- ✅ Works for any project type (Node.js, Python, Rust, Go)
- ✅ Production-ready output with complete context

---

## EXAMPLE USAGE

```bash
# At end of session
/uwo-handoff "API authentication implementation complete"

# Automated actions:
# 1. Creates session-handoffs/2025-10-27-1600-api-authentication-implementation-complete.md
# 2. Auto-fills: date, time, branch, git status, project type
# 3. Adds: session statistics (3 commits today, ruff PASS, 0 uncommitted)
# 4. Includes: 2 subagent reports (framework-research, architect-review)
# 5. Prompts: for executive summary, completed work, next priorities
# 6. Commits: handoff file with "docs: session handoff - API authentication implementation complete"

# Result: Comprehensive handoff in 5 minutes (vs 20 min manual)
```

---

## TEMPLATE REFERENCE

**Expected template structure:**
- Metadata section (Date, Time, Branch, Context, Status)
- Executive Summary
- Completed Work (task lists with evidence)
- Next Priorities (immediate actions)
- Blockers & Challenges
- Session Statistics (auto-generated)
- Subagent Results (if any)
- Quality Gates Summary
- Git Status
- Session End timestamp

**Template location:** `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_SESSION_HANDOFF.md`

---

## NOTES

**Comprehensive format:** Uses full TEMPLATE_SESSION_HANDOFF.md structure with progressive disclosure
**Auto-generation:** Metadata, statistics, git status, quality gates auto-filled
**Guided input:** User prompted for essential narrative content only
**Git automation:** Handoff committed automatically with conventional commit
**Recovery:** Next session reads handoff for full context restoration
**ROI:** 30:1 return, 45 hours/year savings, 1-week payback period

**Universal design:** Works across any project type (Python, Node.js, Rust, Go, etc.)
**Graceful degradation:** Missing components don't block handoff creation
**Production-ready:** Validated output structure, complete session context capture
