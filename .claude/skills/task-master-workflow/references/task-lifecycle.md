# Task Lifecycle & State Machine

Complete reference for Task Master task states and transitions.

## Task States

Every Task Master task has one of these states:

| State | Meaning | Next State | Actions Available |
|-------|---------|-----------|------------------|
| `pending` | Not started, waiting | `in_progress` | Start work |
| `in_progress` | Currently being worked on | `done` or `blocked` | Continue work, mark done, or mark blocked |
| `done` | Complete and validated | (final) | Review, revert if needed |
| `blocked` | Cannot proceed, waiting on external | `in_progress` | Unblock when dependency resolved |
| `deferred` | Postponed to later | `pending` or `in_progress` | Resume work |
| `cancelled` | No longer needed | (final) | Archive only |

## State Transition Diagram

```
                    ┌─────────────┐
                    │   pending   │
                    └────┬────────┘
                         │ task-master set-status --id=X.Y --status=in-progress
                         ▼
                    ┌─────────────┐
            ┌──────→│ in_progress │◄──────┐
            │       └────┬────────┘       │
            │            │                │
   unblock  │            │ (work & validation done)
            │            ▼                │
            │       ┌─────────────┐       │
            │       │    done     │       │
            │       └─────────────┘       │
            │                             │
            │ mark blocked when          │
            │ dependency fails            │
            │                             │
            └─────────────────────────────┘
                    (revert to in_progress
                     when unblocked)
```

## Task Subtask Relationship

**Main Task**: 1 (Database Setup)
- Represents a major capability or phase
- Status: `done` only when ALL subtasks are `done`
- Cannot mark main task `done` if any subtask is not `done`

**Subtask**: 1.1, 1.2, 1.3, ... 1.6
- Individual units of work within the main task
- Each subtask follows full lifecycle independently
- Main task inherits status from subtasks (if any pending/in_progress → main task not done)

### Subtask Dependencies

Subtasks within same task often have dependencies:
```
Task 1: Database Setup
├─ 1.1 - PostgreSQL installation (no deps)
├─ 1.2 - Schema creation (depends on 1.1)
├─ 1.3 - Pydantic config (no deps) [can run parallel to 1.1]
├─ 1.4 - Connection pooling (depends on 1.2, 1.3)
├─ 1.5 - Logging config (depends on 1.3) [can run parallel to 1.4]
└─ 1.6 - Virtual env setup (no deps) [can run parallel to all]
```

Parallelizable subtasks: 1.1, 1.3, 1.6 can start simultaneously
Sequential dependencies: 1.2 → 1.4 (schema must exist before pooling)

## Task Lifecycle by Phase

### Phase 1: INITIATE

**Entry**: Subtask in `pending` state
**Exit**: Subtask in `in_progress` state

**Commands**:
```bash
# View current task
task-master show 1.1

# Validate dependencies
task-master show <dependent-task>

# Mark as in progress
task-master set-status --id=1.1 --status=in-progress
```

**Validation**:
- ✅ All dependencies are `done`
- ✅ Task ID is valid
- ✅ No other issues blocking start

### Phase 2: IN PROGRESS

**Entry**: First `set-status --status=in-progress`
**Exit**: Work complete and ready for validation

**Commands**:
```bash
# Update progress notes
task-master update-subtask --id=1.1 --prompt="progress..."

# Check other tasks (don't change status)
task-master show <other-task>

# View current task details
task-master show 1.1
```

**Normal Duration**: 30+ minutes (can be multiple sessions)

**Progress Tracking**:
- Update-subtask at logical milestones (~30 min or feature complete)
- Commit to git at same milestones
- No status changes during this phase

### Phase 3: VALIDATE

**Entry**: Work appears complete
**Exit**: All validation checks pass

**Pre-done Checklist**:
```
[ ] Code/docs committed to git
[ ] Tests passing (if applicable)
[ ] Quality standards met
[ ] Progress notes updated
[ ] No blockers for next task
```

**Validation Commands**:
```bash
# Run tests
pytest  # or npm test, or manual verification

# Check git status
git status

# Final progress update
task-master update-subtask --id=1.1 --prompt="Complete summary..."
```

**If Validation Fails**:
- Remain in `in_progress`
- Fix identified issues
- Re-run validation
- Once all checks pass → proceed to Phase 4

### Phase 4: COMPLETE

**Entry**: All validation checks passed
**Exit**: Task marked `done`

**Commands**:
```bash
# Mark task as done
task-master set-status --id=1.1 --status=done

# Check for checkpoint trigger (every 5 subtasks)
# Create checkpoint if needed

# Move to next task
task-master next
```

**After Marking Done**:
- Task transitions to `done` state
- Dependent tasks become unblocked
- Subtask is closed (no further changes)
- If revert needed → mark back to `in_progress`

## State Recovery & Troubleshooting

### Issue 1: Task Stuck in In-Progress

**Problem**: Subtask marked `in_progress` but work stalled

**Recovery**:
```bash
# Option 1: Continue work
task-master show 1.1     # Review what's pending
task-master update-subtask --id=1.1 --prompt="Resumed work on..."

# Option 2: Defer for later
task-master set-status --id=1.1 --status=deferred

# Option 3: Revert to pending (if work should restart from scratch)
task-master set-status --id=1.1 --status=pending
```

### Issue 2: Can't Mark Blocked Task as Done

**Problem**: Subtask depends on another task that failed

**Recovery**:
```bash
# Check dependent task status
task-master show 1.2    # What's blocking 1.1?

# If dependent task failed:
task-master show 1.1
# Status: blocked (expected)

# Fix the blocking task first
task-master set-status --id=1.1 --status=in-progress
# Fix issue
# Then mark done when ready
```

### Issue 3: Accidentally Marked Done When Not Complete

**Problem**: Subtask marked `done` but work incomplete

**Recovery**:
```bash
# Immediately revert to in_progress
task-master set-status --id=1.1 --status=in-progress

# Complete remaining work
# Update progress notes

# Re-validate and mark done properly
task-master set-status --id=1.1 --status=done
```

## Main Task Status Inference

Task Master automatically determines main task status:

**Task 1 (Main) Status** based on subtasks:
- Subtasks 1.1 - 1.6

| Subtask Status | Main Task Status |
|---|---|
| All `done` | ✅ `done` |
| Any `in_progress` | 🔄 `in_progress` |
| Any `pending` | ⏸️ `pending` |
| Any `blocked` | 🚫 `blocked` |

**Rule**: Main task cannot be `done` unless ALL subtasks are `done`

## Task Metrics & Progress

### Individual Subtask Progress

```bash
task-master show 1.1

# Output includes:
# - ID, Title, Status
# - Dependencies
# - Complexity score
# - Description & implementation details
```

### Overall Project Progress

```bash
task-master list

# Output includes:
# - Total tasks: 10
# - Total subtasks: 50
# - Subtasks complete: X/50
# - Tasks complete: Y/10
# - Breakdown by status (pending, in_progress, done, etc.)
```

### Per-Task Progress

```bash
task-master show 1

# Task 1 status inference from subtasks:
# - 1.1: done
# - 1.2: done
# - 1.3: done
# - 1.4: in_progress ← Blocks Task 1 completion
# - 1.5: pending
# - 1.6: pending

# Overall: Task 1 is in_progress (3/6 subtasks done, 50%)
```

## Lifecycle Timing Expectations

| Phase | Expected Duration | Actual Variation |
|-------|------------------|-----------------|
| INITIATE | < 5 minutes | 2-10 min |
| WORK | 20-60 minutes | 15-120+ min |
| VALIDATE | 5-10 minutes | 2-20 min |
| COMPLETE | < 2 minutes | 1-5 min |
| **Total** | **~45 min** | **~20-180 min** |

**Note**: Timing varies by subtask complexity. Simple tasks (config) 20 min. Complex tasks (algorithms) 120+ min.

## Checkpoint Trigger During Lifecycle

**Checkpoint Creation Trigger**: Every 5 subtasks marked `done`

Example:
```
1.1 → done (count: 1)
1.2 → done (count: 2)
1.3 → done (count: 3)
1.4 → done (count: 4)
1.5 → done (count: 5) ← TRIGGER: Create checkpoint
```

After checkpoint:
```
1.6 → done (count: 1)
2.1 → done (count: 2)
... (work continues)
2.5 → done (count: 5) ← TRIGGER: Create checkpoint #2
```

See `checkpoint-patterns.md` for checkpoint creation details.
