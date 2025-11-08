---
name: task-master-workflow
description: Guides Claude through Task Master-based development workflows by automating task state tracking, dependency validation, checkpoint commits, and structured progress logging. Should be used when implementing subtasks from a Task Master project to ensure disciplined task progression, prevent work loss through checkpoints, and maintain audit trail of accomplishments.
---

# Task Master Workflow

## Overview

This skill enables disciplined, tracked execution of Task Master subtasks by automating task state management, dependency validation, and milestone-based checkpoints. Rather than manually tracking work, Claude coordinates task progression through structured workflows while maintaining git discipline and detailed progress documentation.

**Core Philosophy**: Task Master provides the *structure* (10 tasks, 50 subtasks, clear dependencies). This skill provides the *discipline* (state tracking, validation, checkpoints, progress logging).

## When to Use

**✅ Activate this skill when:**
- Implementing subtasks from a Task Master project (e.g., task 1.1, 2.3, 7.2)
- Work involves clear task state transitions (pending → in_progress → done)
- Dependencies need validation (can't start task until prerequisite completes)
- Progress tracking matters (want full audit trail of accomplishments)
- Work may span multiple sessions (checkpoints needed every 5 subtasks)

**Keywords that trigger activation:**
- "task-master", "task-master set-status", "task 1.1", "subtask", "Task Master"
- "working on task", "implement subtask", "start phase 0"
- Any mention of `.taskmaster/` directory or task IDs like "1.1", "3.2"

**Activation patterns:**
- **Agent self-invokes**: "I'll implement tasks 1.1-1.6. Using task-master-workflow skill..."
- **Main chat recommends**: "Spawning implementation agent with task-master-workflow for disciplined task tracking"
- **User requests**: "Use task-master-workflow for this work"

## Core Workflow

Every Task Master subtask follows this 4-phase lifecycle:

```
Phase 1: INITIATE          Phase 2: WORK              Phase 3: VALIDATE          Phase 4: COMPLETE
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Start Task 1.1   │  →   │ Implement code   │  →   │ Verify tests     │  →   │ Mark task done   │
│ Mark in_progress │      │ Checkpoint @30m  │      │ Commit files     │      │ Move to 1.2      │
│ Validate deps    │      │ Log progress     │      │ Validate notes   │      │ Check blockers   │
└──────────────────┘      └──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Detailed Workflow Steps

### Phase 1: INITIATE (Before Starting Work)

**Goal**: Establish clean state, validate readiness, confirm task is unblocked

**Step 1.1 - Mark Task In Progress**
```bash
task-master set-status --id=1.1 --status=in-progress
```
- Transitions task from `pending` to `in_progress`
- Communicates to team that work is active
- Prerequisite for all subsequent progress tracking

**Step 1.2 - Validate Dependencies**
- Check: Are all dependency tasks marked `done`?
- If blocked: Report which task(s) must complete first
- If unblocked: Proceed with confidence
- Reference: `references/dependency-validation.md`

**Step 1.3 - Review Task Details**
```bash
task-master show 1.1
```
- Read task description and implementation details
- Note any constraints or quality targets
- Understand what "done" means for this task

### Phase 2: WORK (During Implementation)

**Goal**: Implement functionality with structured checkpoints and progress logging

**Step 2.1 - Implement Core Functionality**
- Write code/documentation to accomplish task
- Follow logical milestones (not arbitrary line counts)
- Examples: "PostgreSQL installed", "config system working", "tests passing"

**Step 2.2 - Commit at Logical Milestones** (Every ~30 min or milestone)
When a logical milestone is reached:
```bash
git add <files>
git commit -m "type(scope): description"
```
- Format: Conventional commits (`feat(scope):`, `fix(scope):`, etc.)
- Message: Clear, describes what was accomplished
- Timing: Every logical milestone or ≤30 minutes elapsed

**Step 2.3 - Log Progress at Milestones**
After each milestone commit:
```bash
task-master update-subtask --id=1.1 --prompt="[Structured Progress Template]"
```
- Template: See `references/progress-template.md`
- Timing: After logical milestone OR every 30 minutes
- Content: What was done, what's next, any blockers
- Benefit: Audit trail of work, easy session resumption

### Phase 3: VALIDATE (Before Marking Done)

**Goal**: Verify all completion criteria met, files committed, quality gates passed

**Validation Checklist** (All must pass):
- [ ] **Code Committed**: All changes pushed to git with clear messages
- [ ] **Tests Passing**: Relevant tests pass (pytest, npm test, etc.)
- [ ] **Quality Gates**: Code follows project standards (linting, types, etc.)
- [ ] **Progress Notes**: Subtask description updated with final summary
- [ ] **No Blockers**: No known issues or TODOs blocking next task

**Step 3.1 - Final Testing**
- Run project test suite if applicable
- Execute manual verification if needed
- Document any edge cases in progress notes

**Step 3.2 - Final Commit**
- If validation found issues, fix them
- Final commit message should indicate completion: "Complete task 1.1 - PostgreSQL setup"

**Step 3.3 - Update Final Progress Notes**
- Comprehensive summary of what was accomplished
- Clear description of next task's starting point
- Any lessons learned or edge cases to watch

### Phase 4: COMPLETE (Mark Task Done)

**Step 4.1 - Mark Task Done**
```bash
task-master set-status --id=1.1 --status=done
```
- Transitions task from `in_progress` to `done`
- Enables dependent tasks to proceed
- Milestone for session checkpoint checks

**Step 4.2 - Check for Checkpoint Trigger**
- Count completed subtasks this session
- Every 5 subtasks: Create session checkpoint
- Command: See `references/checkpoint-patterns.md`

**Step 4.3 - Move to Next Task**
```bash
task-master next
```
- Shows next available task (no blockers)
- Displays dependencies and complexity
- Ready for Phase 1 → Initiate on next task

## Practical Examples

### Example 1: PostgreSQL Installation (Task 1.1)

**Phase 1: INITIATE**
```bash
task-master set-status --id=1.1 --status=in-progress
task-master show 1.1
```
Output shows: PostgreSQL 16 installation with pgvector extension

**Phase 2: WORK**
```
Milestone 1 (15 min): PostgreSQL installed
  → commit: "feat(db): Install PostgreSQL 16 with pgvector"
  → update-subtask: "PostgreSQL installed, running tests..."

Milestone 2 (25 min): pgvector extension loaded
  → commit: "feat(db): Load pgvector extension"
  → update-subtask: "pgvector loaded, verified with CREATE TABLE test"
```

**Phase 3: VALIDATE**
```bash
# Run test to verify installation
psql -U postgres -c "CREATE EXTENSION pgvector;"
# ✓ Tests pass

# Final progress note update
task-master update-subtask --id=1.1 --prompt="PostgreSQL 16 + pgvector fully operational. Connection pooling next."
```

**Phase 4: COMPLETE**
```bash
task-master set-status --id=1.1 --status=done
task-master next  # Shows: Task 1.2 (Schema creation)
```

### Example 2: Schema Creation (Task 1.2) - Depends on 1.1

**Phase 1: INITIATE**
```bash
task-master set-status --id=1.2 --status=in-progress
```
✓ Dependency 1.1 marked done → Safe to proceed

**Phase 2: WORK**
```
Milestone 1 (30 min): sql/schema_768.sql executed
  → commit: "feat(db): Execute schema from sql/schema_768.sql"
  → update-subtask: "Schema created with all tables and indexes"
```

(No Milestone 2 - task complete)

**Phase 3: VALIDATE**
```bash
# Verify all tables created
psql -U postgres -c "\dt"
# ✓ All tables visible

# Final note
task-master update-subtask --id=1.2 --prompt="Schema created, all indexes in place. Ready for connection pooling."
```

**Phase 4: COMPLETE**
```bash
task-master set-status --id=1.2 --status=done
# No checkpoint yet (2/6 subtasks of Task 1 done)
```

## Integration with Existing Workflows

### Git Discipline
- **Micro-commits**: Every logical milestone (aligns with workflow-automation skill)
- **Conventional commits**: `type(scope): description` format
- **Commit frequency**: ≤30 minutes between commits
- **Benefit**: Full audit trail, easy rollback, work loss prevention

### Session Continuity
- **Progress notes**: Structured template enables easy session resumption
- **Checkpoints**: Every 5 subtasks completes (session breakpoint)
- **Handoffs**: Session notes show exact state for next session
- **Reference**: See `/uwo-checkpoint` for checkpoint commands

### Quality Gates
- **Before done**: Tests must pass, files committed, quality standards met
- **No auto-merge**: Quality gates are manual verification (not automated)
- **Exception handling**: If gates fail, fix → commit → re-validate

## Error Handling

**Critical Errors** (Fail Loudly):
- ❌ Invalid task ID (e.g., "1.11" when max is 1.6)
- ❌ Dependency not done (e.g., trying to start 1.2 before 1.1 done)
- ❌ Task-master offline (no recovery mode)
- ❌ Marking done without meeting validation checklist

**Recovery**:
- Report error with full context
- Do not proceed with work
- Require user intervention to resolve

## References

See bundled reference files for detailed guidance:

- `references/dependency-validation.md` - How to validate task dependencies
- `references/progress-template.md` - Structured progress note template
- `references/checkpoint-patterns.md` - Session checkpoint creation
- `references/task-lifecycle.md` - Complete task state machine
- `references/milestone-detection.md` - Recognizing logical milestones

## When NOT to Use

**❌ Skip this skill when:**
- Quick exploratory work (no task tracking needed)
- Single file changes not tied to Task Master
- Work outside a Task Master project
- Manual task tracking preferred

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-08
**Portable**: Yes (works with any task-master project)
**Activation**: Model-invoked via keywords (task-master, task IDs, etc.)
