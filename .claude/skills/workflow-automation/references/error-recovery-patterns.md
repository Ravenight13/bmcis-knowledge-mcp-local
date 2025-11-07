# Graceful Degradation Prompting

**Purpose**: Prompting patterns for when services are offline or operations fail.

---

## When Services Are Offline

### Task-MCP Offline

**Detection**: MCP call returns connection error or timeout

**Prompt Pattern**:
```
Note: task-mcp appears offline (error: {error_message}).

I'll continue work without task tracking. You can update task status
manually later using:

mcp__task-mcp__update_task(task_id={id}, status="in_progress", workspace_path="{path}")

Proceeding with implementation...
```

**Don't block progress** on task tracking - it's a tracking tool, not a blocker.

---

### Semantic Search Offline

**Detection**: codebase-mcp unavailable

**Prompt Pattern**:
```
Note: Semantic search (codebase-mcp) is unavailable.

Falling back to traditional search methods:
- Using Grep for text search
- Using Glob for file patterns
- Using Read for direct file access

This may be slower but will work correctly.
```

**Fallback**: Use Grep/Glob instead of semantic search

---

## Git Conflicts

### Merge Conflict Detected

**Detection**: `git commit` or `git merge` returns conflict

**Prompt Pattern**:
```
Git conflict detected during {operation}.

Conflicted files:
{list of files with conflicts}

To resolve:
1. Open conflicted files and look for markers:
   <<<<<<< HEAD (your changes)
   =======
   >>>>>>> branch (incoming changes)

2. Edit files to resolve conflicts (keep one version or merge both)

3. Stage resolved files:
   git add {file}

4. Complete the operation:
   git commit (or git rebase --continue)

Alternatively, abort with:
   git merge --abort (or git rebase --abort)

Pausing work until conflicts resolved...
```

**Never**: Auto-resolve conflicts without user approval

---

### Rebase Conflicts

**Detection**: `git rebase` stops with conflicts

**Prompt Pattern**:
```
Rebase stopped due to conflicts.

Current state: Rebasing {commit} onto {base}

To continue:
1. Resolve conflicts in {files}
2. Stage resolved files: git add {file}
3. Continue rebase: git rebase --continue

To abort rebase and return to previous state:
   git rebase --abort

Should I:
1. Help resolve conflicts interactively?
2. Wait for you to resolve manually?
3. Abort rebase?
```

---

## Quality Gates Failing

### Ruff/Mypy Errors

**Detection**: `checkall` (ruff/mypy) returns non-zero exit code

**Prompt Pattern**:
```
Quality gates failing:

Ruff (linting): {X} errors
- {error summary}

Mypy (type checking): {Y} errors
- {error summary}

Options:
1. Fix errors now (recommended)
2. Commit as WIP and fix later (use "wip" prefix)
3. View detailed errors and decide

What would you like to do?
```

**Never**: Silently commit code that fails quality gates

---

### Tests Failing (Optional)

**Detection**: `pytest` returns failures

**Prompt Pattern**:
```
Tests are failing:

Failed: {count} tests
Passed: {count} tests

Failed tests:
- {test_name}: {failure_reason}

Tests are optional in this project (Constitution v3.0.0), but failures
may indicate issues.

Should I:
1. Investigate and fix test failures?
2. Skip tests and continue (document in commit message)?
3. Review test output with you?
```

---

## File Path Issues

### Ambiguous File Path

**Detection**: Multiple files match pattern or path doesn't exist

**Prompt Pattern**:
```
Ambiguous file path: "{pattern}"

Found multiple matches:
1. {path1}
2. {path2}
3. {path3}

Which file did you mean? (Or provide full path)
```

**Fallback**: Use context clues (current vendor, working directory)

---

### File Not Found

**Detection**: Read/Edit tool returns file not found

**Prompt Pattern**:
```
File not found: {path}

Did you mean:
- {similar_path_1}
- {similar_path_2}

Or should I:
1. Search for files matching "{filename}"?
2. Create this file (if it's a new file)?
3. Use a different path?
```

---

## Recovery Checklist

### When Degraded, Claude Should:

1. **Inform user of limitation**
   - What service/operation failed
   - Why it failed (if known)
   - What functionality is affected

2. **Offer workaround**
   - Alternative approach (Grep vs semantic search)
   - Manual steps user can take
   - Reduced functionality mode

3. **Continue work (don't block)**
   - Task tracking offline? → Continue without tracking
   - Quality gate warning? → Offer to fix or commit as WIP
   - Single service down? → Use fallbacks

4. **Log issue for manual fix later**
   - Note in commit message
   - Document in session handoff
   - Create task (when task-mcp returns)

---

## Prompting Pattern Template

**Standard degradation prompt**:

```
[Service/Operation] {status}

Impact: {what's affected}

Workaround: {alternative approach}

Action: {what I'll do now}

You can: {what user can do to restore}
```

**Example**:
```
Task-MCP offline

Impact: Cannot update task status automatically

Workaround: Using git commits to track progress

Action: Continuing implementation without task tracking

You can: Manually update tasks later or restart MCP server
```

---

## Never Do

**Don't**:
- Block all work when one service is offline
- Silently fail and hide errors from user
- Auto-commit code that fails quality gates
- Auto-resolve git conflicts
- Guess at ambiguous paths without asking

**Always**:
- Be transparent about limitations
- Offer workarounds
- Ask user for decisions on ambiguous situations
- Document degraded state in commits/handoffs
- Attempt recovery when services return
