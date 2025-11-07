# Advanced Workflow Prompting

**Purpose**: Prompting patterns for complex multi-session, multi-file, and hierarchical task scenarios.

---

## Multi-Session Continuity

### Session Handoff Pattern

**When session ending with unfinished work**:

**Prompt**:
```
This session has unfinished work. Let me create a handoff document
before ending:

Creating: session-handoffs/{timestamp}-{description}.md

Contents:
- Completed work: {summary}
- In progress: {current tasks}
- Next priorities: {todo list}
- Blockers: {any issues}
- Context: {git status, quality gates, etc.}

This will help the next session (or next AI) resume seamlessly.
```

**Action**: Use `/cc-checkpoint` or create handoff manually

---

### Loading Previous Session

**When starting session with existing handoff**:

**Prompt**:
```
I see a session handoff from {date} ({time_ago}):

📋 Last session summary:
   Completed: {completed_count} tasks
   - {task_1}
   - {task_2}

   In progress: {in_progress_count} tasks
   - {task_3}

   Next priority: {next_task}

   Blockers: {blockers_summary}

Should I continue from this checkpoint?
```

**Action**: Load context from handoff, resume work at natural continuation point

---

## Complex Task Hierarchies

### Parent-Child Tasks

**When creating multi-phase work**:

**Prompt**:
```
This work has {N} phases. Should we create a task hierarchy?

Proposed structure:
- Parent task: "{overall_goal}"
  - Child task 1: "{phase_1}" (status: todo)
  - Child task 2: "{phase_2}" (status: todo, depends on #1)
  - Child task 3: "{phase_3}" (status: todo, depends on #2)

This helps track progress across multiple sessions and provides
clear milestones.

Create hierarchy? (y/n)
```

**Action**: Create parent task, then subtasks with dependencies

---

### Completing Task Hierarchies

**When finishing last subtask**:

**Prompt**:
```
All subtasks for parent task #{id} are complete:
✅ Subtask #1: {title}
✅ Subtask #2: {title}
✅ Subtask #3: {title}

Marking parent task #{id} as complete: "{parent_title}"
```

**Action**: Update parent task status to "done"

---

### Blocked Subtasks

**When dependency not satisfied**:

**Prompt**:
```
Task #{id} depends on task #{dep_id} which is still {status}.

Dependencies:
❌ Task #{dep_id}: {title} (status: {status})

Cannot start this task until dependencies complete.

Should I:
1. Start dependency task #{dep_id} first?
2. Skip this task for now?
3. Remove dependency (if no longer needed)?
```

---

## Dependency Management

### Creating Dependent Tasks

**When task requires other tasks first**:

**Prompt**:
```
This task depends on completing other work first:

Task: "{new_task}"
Depends on:
- Task #{id1}: {title1} ({status1})
- Task #{id2}: {title2} ({status2})

Creating task with dependencies. It will remain "todo" until
dependencies are complete.
```

**Action**: Create task with `depends_on` field populated

---

### Starting Task with Dependencies

**When attempting blocked task**:

**Prompt**:
```
Cannot start task #{id}: "{title}"

Incomplete dependencies:
- Task #{dep1}: {title1} (status: {status})
- Task #{dep2}: {title2} (status: {status})

Recommended: Complete dependencies first, or start with task #{dep1}.

Proceed anyway? (not recommended)
```

---

## Multi-File Coordination

### Atomic Multi-File Changes

**When modifying 3+ related files**:

**Prompt**:
```
This change spans {N} files that must be updated together (atomic change):
- {file1} (add new function)
- {file2} (update import)
- {file3} (add tests)

I'll modify all files, run quality gates, then commit as single logical change.

Proceeding with atomic update...
```

**Action**: Edit all files, validate together, commit once

---

### Progressive Multi-File Commits

**When large refactor spans many files**:

**Prompt**:
```
Large refactor affecting {N} files. I'll commit progressively to prevent work loss:

Phase 1: Core changes ({count} files)
- {files}
→ Commit: "refactor({scope}): {phase1_description}"

Phase 2: Dependent changes ({count} files)
- {files}
→ Commit: "refactor({scope}): {phase2_description}"

Phase 3: Test updates ({count} files)
- {files}
→ Commit: "test({scope}): {phase3_description}"

This follows micro-commit discipline (≤30 min intervals).

Proceeding with phase 1...
```

---

## Edge Cases

### Stale Working Directory

**When starting session with uncommitted changes**:

**Prompt**:
```
⚠️  Uncommitted changes detected from previous session:

{git status output}

Options:
1. Commit changes now (recommended)
   → Will create: "chore: recover uncommitted changes from previous session"

2. Review changes first
   → I'll show you what changed and you decide

3. Discard changes (DANGEROUS)
   → Permanently deletes uncommitted work

What should I do?
```

---

### Diverged Branches

**When local and remote have different commits**:

**Prompt**:
```
⚠️  Branch diverged from remote:
- Local: {ahead} commits ahead
- Remote: {behind} commits behind

This usually means parallel work happened (different machine or developer).

Options:
1. Rebase local on remote (recommended if no conflicts)
   → git rebase origin/{branch}

2. Merge remote into local (safe but creates merge commit)
   → git merge origin/{branch}

3. Review divergence first
   → I'll show you the conflicting commits

What should I do?
```

---

## Complex Scenario Patterns

### Research → Implementation → Testing

**When starting multi-phase vendor work**:

**Prompt**:
```
Vendor implementation requires 3 phases:

Phase 1: Research (expected: 20-45 min)
- Format analysis
- Pattern identification
- Test fixture review
→ Deliverable: Research report in docs/subagent-reports/

Phase 2: Implementation (expected: 45-90 min)
- Extractor code
- Configuration
- Brand mapping
→ Deliverable: Working extractor.py

Phase 3: Testing (expected: 30-60 min)
- Golden file tests
- Reconciliation validation
- Edge cases
→ Deliverable: Test file and passing results

Should I:
1. Execute all phases sequentially? (2-3 hours)
2. Start with phase 1 only? (pause for review)
3. Use research-orchestrator skill? (parallel agents)
```

---

### Debugging Loop Coordination

**When entering debug cycle**:

**Prompt**:
```
Entering debug loop for {issue}:

Loop structure (from debug-loop-enforcer):
1. TEST → Run tests, identify failure
2. EVALUATE → Analyze root cause
3. PLAN → Design fix strategy
4. FIX → Implement changes
5. COMMIT → Save progress
→ Repeat until resolved

Starting iteration 1...
```

**Action**: Follow 5-phase loop, commit after each fix attempt

---

## Prompting Pattern Summary

**Complex workflows follow pattern**:

1. **Recognize complexity** (multi-file, multi-phase, dependencies)
2. **Propose structure** (hierarchy, phases, atomic changes)
3. **Ask for confirmation** (user may want different approach)
4. **Execute systematically** (follow structure, commit at boundaries)
5. **Track progress** (update tasks, create handoffs, communicate status)

**Example**:
```
Recognition: "This refactor touches 12 files across 3 vendors"
Proposal: "I recommend 3-phase approach with progressive commits"
Confirmation: "Proceed with phased approach?"
Execution: "Phase 1 complete (4 files), committing..."
Progress: "Phase 1/3 done, next: dependent changes"
```
