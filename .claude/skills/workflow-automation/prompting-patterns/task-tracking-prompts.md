# Task Tracking Prompting Patterns

## When Claude Should Prompt About Tasks

Claude should recognize these scenarios and prompt subagents:

### Scenario 1: Starting Multi-File Work
**Trigger**: Subagent about to modify 2+ files OR work >30 minutes
**Prompt Pattern**:
"Before starting, let's mark task #{id} as in_progress using task-mcp:
`mcp__task-mcp__update_task(task_id={id}, status='in_progress', workspace_path=...)`"

### Scenario 2: Work Complete
**Trigger**: Subagent says "done" or "complete" or "finished"
**Validation Checklist**:
- [ ] All files written?
- [ ] Tests passing (if applicable)?
- [ ] Quality gates passed (checkall)?
**Prompt Pattern**:
"Now let's mark task #{id} as done:
`mcp__task-mcp__update_task(task_id={id}, status='done', workspace_path=...)`"

### Scenario 3: No Active Task
**Trigger**: Starting work without task reference
**Prompt Pattern**:
"I don't see an active task for this work. Should we create one in task-mcp for tracking?"

## Graceful Degradation

### If task-mcp Offline
**Prompt Pattern**:
"Note: task-mcp appears offline. I'll continue work and you can update task status manually later."

Don't block progress on task tracking failures.
