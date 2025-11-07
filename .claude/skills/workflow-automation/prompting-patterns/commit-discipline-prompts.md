# Commit Discipline Prompting Patterns

## When Claude Should Prompt for Commits

### Pattern 1: Line Threshold (20-50 lines)
**Detection**: Count file changes in working directory
**Command**: `git diff --stat`
**Prompt Pattern**:
"I notice 35 lines changed across 2 files. This is a good commit milestone. Let me commit:
`git add <files> && git commit -m 'type(scope): message'`"

### Pattern 2: Time Threshold (30 minutes)
**Detection**: Check time since last commit
**Command**: `git log -1 --format='%cr'`
**Prompt Pattern**:
"It's been 40 minutes since the last commit. To prevent work loss, let me commit current progress:
`git add <files> && git commit -m 'wip: checkpoint at <milestone>'`"

### Pattern 3: Logical Milestone
**Triggers**:
- Helper function complete
- Test file created
- Research artifact saved
- Component fully implemented

**Prompt Pattern**:
"The helper function is complete and working. This is a logical commit milestone:
`git commit -m 'feat(utils): add helper function for X'`"

## Commit Message Format

Always use conventional commits:
```
type(scope): description

Valid types: feat, fix, docs, refactor, test, chore
Max 72 chars for first line
```

## When NOT to Prompt

Don't interrupt for:
- Whitespace-only changes
- Single comment additions
- Trivial formatting (unless 30 min elapsed)
