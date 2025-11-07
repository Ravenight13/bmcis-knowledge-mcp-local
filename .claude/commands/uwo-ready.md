---
description: Universal session initialization with intelligent context detection and directory setup
allowed-tools: [Read, Write, Bash, Glob]
---

# Universal Workflow Orchestrator - Session Initialization

**Purpose:** Automate session start with context detection, health validation, and directory setup

**Value Proposition:**
- Eliminates 15-25 min of manual setup per session
- Creates required directories automatically (idempotent)
- Validates system health before development
- Generates structured session checklist
- Works for ANY project type

**When to Use:**
- Start of every development session
- Switching to new project or task context
- After git pull (revalidate environment)
- First time using workflow in a project (handles setup automatically)

---

## STEP 1: Invoke Universal Workflow Orchestrator Skill

**Automatically load universal-workflow-orchestrator skill:**

"Apply skill: universal-workflow-orchestrator"

**This activates:**
- Context detection (git branch, directory, task type)
- Workflow principle enforcement (parallel orchestration, micro-commits)
- Health validation preparation

---

## STEP 2: Create Directory Structure (Idempotent)

**Create required directories if they don't exist:**

```bash
# Create all required directories (idempotent)
mkdir -p session-handoffs docs/subagent-reports docs/analysis \
    docs/subagent-reports/{api-analysis,architecture-review,security-analysis,performance-analysis,code-review} &>/dev/null

echo "✅ Directory structure verified"
```

**Note:** This is idempotent (safe to run multiple times). If directories exist, no action taken.

---

## STEP 3: Context Detection

**Run these commands to gather context information:**

```bash
# Capture context (suppress errors)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
WORKING_DIR=$(pwd)

echo "🌿 Branch: $CURRENT_BRANCH"
echo "📁 Directory: $WORKING_DIR"
```

**Then analyze branch/directory:**
**Detect context based on branch name:**
```bash
# Analyze branch name for context
CONTEXT="GENERAL"
[[ "$CURRENT_BRANCH" =~ ^feat/ ]] && CONTEXT="DEVELOPMENT"
[[ "$CURRENT_BRANCH" =~ ^test/ ]] && CONTEXT="TESTING"
[[ "$CURRENT_BRANCH" =~ ^docs/ ]] && CONTEXT="DOCUMENTATION"
[[ "$CURRENT_BRANCH" =~ ^fix/ ]] && CONTEXT="BUGFIX"

echo "📍 Detected Context: $CONTEXT"
```

---

## STEP 4: System Health Validation

**Run health checks (token-optimized):**

```bash
echo "🔍 System Health Checks"
echo ""

# 4.1 Git Status
UNCOMMITTED_COUNT=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNCOMMITTED_COUNT" -eq 0 ]; then
    echo "✅ Git: Clean working tree"
else
    echo "⚠️ Git: $UNCOMMITTED_COUNT uncommitted files"
fi

# 4.2 Branch Sync (capture, summarize)
SYNC_STATUS=$(git rev-list --left-right --count @{u}...HEAD 2>/dev/null || echo "")
if [ -n "$SYNC_STATUS" ]; then
    BEHIND=$(echo "$SYNC_STATUS" | awk '{print $1}')
    AHEAD=$(echo "$SYNC_STATUS" | awk '{print $2}')
    [ "$BEHIND" -gt 0 ] && echo "   ⚠️ Behind remote by $BEHIND commits"
    [ "$AHEAD" -gt 0 ] && echo "   📤 Ahead of remote by $AHEAD commits"
    [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ] && echo "   ✅ Synced with remote"
else
    echo "   ℹ️ No remote tracking"
fi

# 4.3 Detect Project Type
PROJECT_TYPE="Unknown"
[ -f "package.json" ] && PROJECT_TYPE="Node.js"
[ -f "pyproject.toml" ] && PROJECT_TYPE="Python"
[ -f "Cargo.toml" ] && PROJECT_TYPE="Rust"
[ -f "go.mod" ] && PROJECT_TYPE="Go"
echo "✅ Project type: $PROJECT_TYPE"

# 4.4 Check Quality Tools (condensed)
TOOLS=""
command -v ruff &>/dev/null && TOOLS="$TOOLS ruff"
command -v mypy &>/dev/null && TOOLS="$TOOLS mypy"
command -v pytest &>/dev/null && TOOLS="$TOOLS pytest"
command -v npm &>/dev/null && TOOLS="$TOOLS npm"
[ -n "$TOOLS" ] && echo "   Quality tools:$TOOLS" || echo "   ⚠️ No quality tools detected"

# 4.5 Count Session Handoffs
HANDOFF_COUNT=$(ls -1 session-handoffs/*.md 2>/dev/null | wc -l | tr -d ' ')
LATEST_HANDOFF=$(ls -t session-handoffs/*.md 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "none")
echo "📋 Session handoffs: $HANDOFF_COUNT total"
[ "$HANDOFF_COUNT" -gt 0 ] && echo "   Latest: $LATEST_HANDOFF"

# 4.6 Count Subagent Reports
REPORT_COUNT=$(find docs/subagent-reports -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "🤖 Subagent reports: $REPORT_COUNT total"
echo ""
```

---

## STEP 5: Generate Session Checklist

**Present structured session start checklist:**

```markdown
## ✅ SESSION INITIALIZED - Universal Workflow Orchestrator

**Context**: {CONTEXT_HINT} (detected from branch/directory)
**Branch**: {CURRENT_BRANCH}
**Directory**: {WORKING_DIR}

### 📁 Directory Structure
- ✅ session-handoffs/ (ready)
- ✅ docs/subagent-reports/ (ready)
- ✅ docs/analysis/ (ready)

### 🔍 System Health
- Git: {status from checks above}
- Project type: {detected type}
- Quality tools: {available tools}
- Previous work: {handoff count} handoffs, {report count} subagent reports

### 🎯 Workflow Reminders
- [ ] Use parallel subagents for complex tasks (3+ independent analyses)
- [ ] Subagents MUST write findings to files and micro-commit
- [ ] Commit every 20-50 lines or ≤30 minutes
- [ ] Run quality gates before each commit
- [ ] Create session handoff at end (use /uwo-handoff)

### 📚 Templates Available
- Session handoff: `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_SESSION_HANDOFF.md`
- Research report: `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_RESEARCH_REPORT.md`

### 🚀 Recommended Next Actions
{Based on context:}
- DEVELOPMENT: Review active feature, plan implementation, run quality checks
- TESTING: Review test coverage, identify gaps, run test suite
- DOCUMENTATION: Review existing docs, identify missing sections, plan updates
- BUGFIX: Reproduce issue, analyze root cause, plan fix
- GENERAL: Review recent commits, check for open PRs, plan today's work

### 💡 Tips
- Read latest session handoff (if exists) for context
- Check for blocked work items or open questions
- Review any recent subagent reports for insights

---

**Ready to begin work!** What would you like to tackle first?
```

---

## STEP 6: Report Summary

**Provide concise summary:**

"Session initialized successfully for {CONTEXT_HINT} work.

System health: {summary of key checks}
Previous work: {handoff count} handoffs, {report count} subagent reports

Use /uwo-checkpoint during work to save progress.
Use /uwo-handoff at end of session for full context transfer.

Ready to begin!"

---

## Success Criteria

- ✅ Universal-workflow-orchestrator skill invoked
- ✅ All directories created (idempotent)
- ✅ Context detected from git/directory
- ✅ Health checks run and reported
- ✅ Session checklist generated
- ✅ Works for any project type (web, data, DevOps, etc.)
- ✅ Safe to run multiple times (no duplication)

---

## Notes

**First-time use:** Creates all directories automatically
**Subsequent use:** Validates directories exist, runs health checks
**Project-agnostic:** No hard-coded context types, detects from environment
**Extensible:** Users can customize quality checks for their tech stack
