# Dependency Validation Workflow

Before starting any Task Master subtask, validate that all dependencies are satisfied.

## Quick Check

```bash
# View task details including dependencies
task-master show 1.2

# Output shows:
# ID: 1.2
# Dependencies: 1.1
# Status: pending
```

If status shows dependencies are not `done`, **DO NOT PROCEED**. Task is blocked.

## Full Validation Workflow

### Step 1: View Task Details
```bash
task-master show X.Y
```

**What to look for:**
- `Dependencies:` field shows task(s) that must be done first
- `Status:` of current task (should be `pending` or `in_progress`)
- `Complexity:` and description for scope understanding

### Step 2: Check Dependency Status

**Example: Starting Task 1.2 (depends on 1.1)**
```bash
task-master show 1.1
```

Check if status is:
- ✅ `done` - Safe to proceed on 1.2
- ❌ `in_progress` - Wait (1.1 in progress, not complete)
- ❌ `pending` - Cannot proceed (1.1 hasn't started)

### Step 3: For Multiple Dependencies

Some tasks have multiple dependencies (e.g., Task 8: Neon Validation depends on Tasks 6 AND 7).

```bash
# View task with multiple dependencies
task-master show 8

# Output example:
# ID: 8
# Title: Neon Production Validation System
# Dependencies: 6, 7
# Status: pending
```

**Check both dependencies:**
```bash
task-master show 6   # Cross-Encoder Reranking
task-master show 7   # Entity Extraction and Knowledge Graph
```

- ✅ Both show `done` - Safe to start Task 8
- ❌ Either shows not `done` - Task 8 is blocked

### Step 4: Handle Blocked Task

If task is blocked:

1. **Identify which dependencies are incomplete**
   ```
   Task 8 is blocked:
   - Task 6: done ✓
   - Task 7: in_progress ✗

   Must wait for Task 7 to complete
   ```

2. **Work on other unblocked tasks instead**
   ```bash
   task-master next  # Shows next available task with no blockers
   ```

3. **Set reminder to check later**
   - Document in progress notes: "Task 8 blocked pending Task 7"
   - Check at next session

## Dependency Chain for bmcis-knowledge-mcp Project

Full linear dependency chain:

```
Task 1: Database Setup (no deps)
  ↓
Task 2: Document Parsing (depends on 1)
  ↓
Task 3: Embedding Pipeline (depends on 2)
  ↓
Task 4: Vector/BM25 Search (depends on 3)
  ↓
Task 5: Hybrid Search RRF (depends on 4)
  ↓
Task 6: Cross-Encoder Reranking (depends on 5)
  ├─ Task 7: Knowledge Graph (depends on 3) [PARALLEL]
  ↓
Task 8: Neon Validation (depends on 6, 7)
  ↓
Task 9: Search Optimization (depends on 8)
  ↓
Task 10: FastMCP Integration (depends on 9)
```

### Key Parallel Opportunities:
- **Tasks 6 & 7 can run in parallel** after Task 3 completes (but Task 8 must wait for both)
- All other tasks are strictly sequential

## Common Dependency Errors

### ❌ Error 1: "Task is blocked by dependency"
```
Trying to start: Task 2
Error: Task 2 depends on Task 1
Status of Task 1: in_progress (not done)

Solution: Wait for Task 1 to complete
```

### ❌ Error 2: "Invalid task ID"
```
Trying to check: Task 1.7
Error: Task 1 has only 6 subtasks (max 1.6)

Solution: Use correct task ID (check task-master show 1)
```

### ❌ Error 3: "Task Master offline"
```
Error: Cannot connect to .taskmaster/tasks.json

Solution: Fail loudly - require user intervention to resolve
No graceful degradation for Task Master
```

## Dependency Validation Checklist

Before marking a task `in_progress`:

- [ ] Task ID is valid (e.g., 1.2 where Task 1 has 6 subtasks)
- [ ] All listed dependencies in `Dependencies:` field are marked `done`
- [ ] Task current status is `pending` or `in_progress`
- [ ] No errors from task-master command

Before marking a task `done`:

- [ ] All code changes committed to git
- [ ] All tests passing
- [ ] Progress notes updated with final summary
- [ ] No known blockers for next task

## Example: Multi-Dependency Task (Task 8)

### Scenario: Ready to start Task 8 (Neon Validation)

**Check dependencies:**
```bash
task-master show 6  # Cross-Encoder Reranking
# Status: done ✓

task-master show 7  # Entity Extraction
# Status: done ✓
```

**Both dependencies satisfied → Safe to proceed:**
```bash
task-master set-status --id=8 --status=in-progress
task-master show 8
```

### Scenario: Task 8 blocked (Task 7 still in progress)

```bash
task-master show 6  # Status: done ✓
task-master show 7  # Status: in_progress ✗

# Task 8 is blocked by Task 7
# Instead of waiting:
task-master next    # Shows next available unblocked task
```

## Recovery: Unblocking a Blocked Task

If you discover a dependency is broken (e.g., Task 1 marked done but actually has issues):

1. **Identify the problem:**
   - Task is marked `done` but functionality incomplete
   - New blocker discovered during work on dependent task

2. **Report the issue:**
   - Document in progress notes: "Found issue in Task 1.1 - PostgreSQL connection timing out"
   - Do NOT mark dependent task as done

3. **Revert dependency status:**
   ```bash
   task-master set-status --id=1.1 --status=in-progress
   ```

4. **Fix the issue:**
   - Implement fix in Task 1.1
   - Commit fix with clear message
   - Re-validate with tests

5. **Remark complete:**
   ```bash
   task-master set-status --id=1.1 --status=done
   ```

6. **Resume blocked task:**
   ```bash
   task-master set-status --id=X.Y --status=in-progress
   ```

## Dependency Validation Summary

| Scenario | Action | Result |
|----------|--------|--------|
| All dependencies `done` | Start task safely | ✅ Proceed |
| 1+ dependencies not `done` | Wait or work on other tasks | ⏸️ Task blocked |
| Invalid task ID | Report error | ❌ Fail loudly |
| Task Master offline | Report error | ❌ Fail loudly |
| Dependency broken (marked done but issues) | Revert + fix + remark done | 🔄 Recovery mode |
