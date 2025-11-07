---
name: workflow-automation
description: Guides Claude to systematically remind subagents about workflow discipline during implementation work. This skill should be used when orchestrating subagents for feature implementation, bug fixes, or refactoring where task tracking, commit discipline, and file organization are required. Provides prompting patterns for when to remind about task-mcp updates (in_progress/done), commit milestones (20-50 lines, 30 minutes, logical milestones), and file path validation before writes. Helps Claude remember workflow discipline so subagents don't need constant manual reminders.
---

# Workflow Automation

## Overview

Guide Claude's orchestration of subagents by providing systematic prompting patterns for workflow discipline.

### Core Philosophy

Subagents often forget workflow discipline without explicit reminders:
- Task tracking: 85% forget to mark in_progress/done without prompting
- Commit discipline: Miss 20-50 line milestones or 30-minute thresholds
- File organization: Write files to incorrect locations without validation

This skill equips Claude with prompting patterns to automatically remind subagents at the right moments.

### What This Skill Does

When orchestrating subagents, Claude uses this skill to:

1. **Recognize when to remind about tasks**
   - Starting multi-file work → Prompt to mark in_progress
   - Work complete → Prompt to mark done
   - Validate completion criteria before marking done

2. **Recognize when to prompt for commits**
   - 20-50 lines changed → Prompt for milestone commit
   - 30+ minutes elapsed → Prompt to prevent work loss
   - Logical milestone reached → Prompt for atomic commit

3. **Recognize when to validate file paths**
   - Before Write tool → Validate location per constitutional rules
   - Vendor code → Ensure {VENDOR}/ isolation
   - Root directory → Catch misplaced files

Claude learns *when* to prompt and *what* to say to maintain discipline.

## When Claude Should Use This Skill

### Primary Scenarios

**✅ Use when orchestrating subagents for**:
- Multi-file implementation work (2+ files)
- Work sessions lasting >30 minutes
- Vendor extractor implementations
- Framework refactors requiring task tracking
- Any development work where discipline matters

### Anti-Patterns (When NOT to Use)

**❌ Don't use for**:
- Simple file reads (<5 minutes, no tracking needed)
- Quick questions (no implementation work)
- Exploratory analysis (no commits required)
- Work already complete (prompts after-the-fact are unhelpful)

### Skill Activation

This skill activates when Claude detects:
- Keywords: "implement", "refactor", "feature", "vendor", "extractor"
- Context: Main chat orchestrating subagents for implementation
- Need: Subagents require workflow discipline reminders

## Core Capabilities

### 1. Task Tracking Guidance

**When Claude Prompts**:
- Starting work: "Let's mark task #{id} in_progress before beginning"
- Work complete: "Validate completion criteria, then mark task #{id} done"
- No task found: "Should we create a task in task-mcp for tracking?"

**Validation Checklist** (before marking done):
- [ ] All files written and saved?
- [ ] Quality gates passed (checkall)?
- [ ] Tests passing (if applicable)?

**Graceful Degradation**:
If task-mcp offline: "Note: task-mcp offline. Continue work, update manually later."

See: `references/` for detailed prompting patterns

### 2. Commit Discipline Guidance

**When Claude Prompts**:
- Line threshold: "35 lines changed - good commit milestone"
- Time threshold: "40 minutes elapsed - commit to prevent work loss"
- Logical milestone: "Helper function complete - commit atomic change"

**Detection Patterns**:
- Check: `git diff --stat` for line changes
- Check: `git log -1 --format='%cr'` for time elapsed
- Listen: Subagent says "complete", "done", "working", "tested"

**Commit Format**:
Always conventional commits: `type(scope): description`

See: `prompting-patterns/commit-discipline-prompts.md`

### 3. File Organization Guidance

**When Claude Validates**:
- Before Write tool: "Checking file path against constitutional rules..."
- Vendor code: "This goes in backend/src/extractors/{VENDOR}/"
- Root directory: "Only CLAUDE.md, README.md, pyproject.toml allowed in root"

**Decision Tree**:
```
About to write file
├─ Vendor code? → backend/src/extractors/{VENDOR}/
├─ Tests? → backend/tests/
├─ Docs? → docs/{category}/
└─ Root? → Only if constitutional-allowed
```

See: `prompting-patterns/file-organization-prompts.md`

## How Claude Uses This Skill

### Typical Orchestration Flow

1. **User requests implementation work**
   User: "Implement Sound United vendor extractor"

2. **Claude recognizes workflow discipline needed**
   Claude loads this skill (keywords: "implement", "vendor", "extractor")

3. **Claude prompts for task tracking**
   Claude: "Let me mark task #42 as in_progress before starting"

4. **Claude spawns subagent for implementation**
   Subagent focuses on code, Claude handles workflow discipline

5. **Claude monitors for commit milestones**
   - Checks: `git diff --stat` shows 35 lines changed
   - Prompts: "35 lines changed - let me commit this milestone"

6. **Claude validates file paths**
   - Subagent about to write: `backend/src/extractors/sound_united/extractor.py`
   - Claude: "✓ Path correct for vendor code"

7. **Claude prompts for completion**
   - Subagent: "Extractor complete and tested"
   - Claude: "Validating: files written? tests pass? checkall green?"
   - Claude: "Marking task #42 as done"

### Prompting Frequency

Claude should prompt:
- **Proactively**: At milestones (line/time thresholds)
- **Not excessively**: Not every single line change
- **Gracefully**: If tools offline, continue work without blocking

## References

### Prompting Patterns (When to Remind)
- `prompting-patterns/task-tracking-prompts.md` - When to prompt about task updates
- `prompting-patterns/commit-discipline-prompts.md` - When to prompt for commits
- `prompting-patterns/file-organization-prompts.md` - When to validate file paths

### Decision Trees (How to Recognize)
- `references/commit-milestone-detection.md` - Recognizing commit milestones
- `references/file-organization-validation.md` - Validating file paths
- `references/error-recovery-patterns.md` - Handling offline services
- `references/advanced-workflows.md` - Complex scenarios (multi-session, hierarchies)

### Constitutional References (File Organization Rules)
- `references/FILE_ORGANIZATION_POLICY.md` - Complete constitutional file rules
- `references/MICRO_COMMIT_DISCIPLINE.md` - Commit discipline best practices
