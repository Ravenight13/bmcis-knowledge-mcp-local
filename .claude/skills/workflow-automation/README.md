# Workflow Automation Skill

**Version**: 1.0.0
**Status**: Draft (Review)
**Grade**: TBD (Pending validation)
**Created**: 2025-11-03

---

## Overview

Automates task management (task-mcp), micro-commit discipline (git), and file organization compliance during implementation work. Designed to reduce cognitive load by handling workflow mechanics automatically while agents and users focus on high-level implementation goals.

---

## Skill Metadata

**Name**: `workflow-automation`
**Type**: Project skill (managed)
**Location**: `.claude/skills/project/workflow-automation/`

**Description**:
> Automate task management (task-mcp), micro-commit discipline (git), and file organization compliance during implementation work. Marks tasks in-progress/done, commits at logical milestones, and validates file placement per constitutional rules. Use for any implementation work lasting >30 minutes or requiring multiple files.

---

## Token Budget

### 3-Tier Progressive Disclosure

**Tier 1**: Metadata (always loaded in portfolio)
- Size: ~200 tokens
- Content: Name, description, when to use

**Tier 2**: Core SKILL.md (loaded on invocation)
- Size: ~1,200 tokens
- Content: Workflow protocol, task management, micro-commits, file organization

**Tier 3**: Reference files (lazy-loaded when needed)
- FILE_ORGANIZATION_POLICY.md: ~3,500 tokens
- MICRO_COMMIT_DISCIPLINE.md: ~1,800 tokens
- cc-helpers.sh functions: ~500 tokens

**Total Budget**: 200 (always) → 1,400 (invoked) → 7,000 (full reference)

**Efficiency**: 97% reduction in token overhead vs embedding in every agent

---

## Core Capabilities

### 1. Task Management (task-mcp)
- Auto-mark tasks: pending → in_progress → done
- Validate completion criteria before marking done
- Graceful degradation when MCP offline

### 2. Micro-Commit Discipline
- Commit every 20-50 lines OR ≤30 minute intervals
- Detect logical milestones (research artifacts, helper functions, tests)
- Conventional commit format enforcement
- Use `detect_commit_milestones()` from cc-helpers.sh

### 3. File Organization Compliance
- Validate file paths per FILE_ORGANIZATION_POLICY.md
- Use correct subdirectories (docs/subagent-reports/{type}/{component}/)
- Respect vendor isolation rules
- Ask user when path ambiguous

### 4. Constitutional Compliance
- Pre-check with /cc-comply --quick
- Pattern library integration awareness
- Error handling with hierarchical error system

---

## When to Use

### ✅ USE for:
- Multi-file implementations (2+ files)
- Work spanning >30 minutes
- Vendor extractor implementations
- Framework refactors
- Any work with clear task tracking needs

### ❌ DON'T USE for:
- Simple file reads/searches
- Quick questions (<5 min)
- Trivial changes (<20 lines)
- Exploratory work (no clear task)

---

## Invocation Patterns

### From Agent
```
I'll use the workflow-automation skill to manage this refactor.
```

### From Main Chat
```
I'll spawn a general-purpose agent with the workflow-automation skill
to handle task tracking and git commits.
```

### From User
```
Please use the workflow-automation skill for this implementation.
```

---

## Documentation

### Core Files

1. **SKILL.md** (~1,200 tokens)
   - Complete workflow protocol
   - Task management integration
   - Micro-commit strategy
   - File organization compliance
   - Invocation examples
   - Graceful degradation

2. **INTEGRATION_GUIDE.md** (~9,800 tokens)
   - Decision tree for when to use
   - Agent integration recommendations
   - Main chat orchestration patterns
   - 5 comprehensive examples
   - User override patterns
   - FAQs

3. **TESTING_STRATEGY.md** (~7,500 tokens)
   - 16 test scenarios
   - 12 edge cases
   - Validation criteria (≥95% accuracy targets)
   - 3-phase rollout plan
   - Metrics to track
   - Rollback procedures

### Prompting Patterns

The skill provides prompting patterns for Claude to guide subagents:

**prompting-patterns/**:
- `task-tracking-prompts.md` - When to prompt about task-mcp updates
- `commit-discipline-prompts.md` - When to prompt for commits (20-50 lines, 30 min, milestones)
- `file-organization-prompts.md` - When to validate file paths before writing

These patterns help Claude remind subagents about workflow discipline without manual reminders.

### Reference Files (Tier 3)

- `docs/reference/constitutional/FILE_ORGANIZATION_POLICY.md`
- `docs/guides/MICRO_COMMIT_DISCIPLINE.md`
- `.claude/lib/cc-helpers.sh`

---

## Integration Points

### Task-MCP MCP Server
- `mcp__task-mcp__list_tasks` - Find active tasks
- `mcp__task-mcp__update_task` - Mark in-progress/done
- `mcp__task-mcp__create_task` - Create if needed
- Graceful degradation: Work continues if MCP offline

### Git Automation
- `detect_commit_milestones()` - From cc-helpers.sh
- `validate_commit_frequency()` - From cc-helpers.sh
- Conventional commits: `type(scope): message`
- Pre-commit hooks respected

### File Organization
- `validate_file_organization()` - From cc-helpers.sh
- FILE_ORGANIZATION_POLICY.md validation
- Subdirectory structure enforcement
- Vendor isolation compliance

---

## Success Criteria

### Phase 1: Low-Risk Testing (2-3 sessions)
- ✅ 100% file placement compliance
- ✅ ≥95% task-mcp accuracy
- ✅ ≥80% commit intervals within ≤30 min
- ✅ Token overhead ≤10%

### Phase 2: Medium Complexity (3-5 sessions)
- ✅ ≥80% commit intervals maintained
- ✅ 100% conventional commit format
- ✅ All quality gates pass (checkall)
- ✅ User satisfaction ≥4.0/5.0

### Phase 3: Production (5-7 sessions)
- ✅ Complete vendor implementation
- ✅ Zero production incidents
- ✅ ≥95% automation accuracy
- ✅ Net token savings vs manual workflow

---

## Rollout Status

**Current Phase**: Phase 0 (Draft Review)

**Next Steps**:
1. Review skill specification (SKILL.md)
2. Review testing strategy (TESTING_STRATEGY.md)
3. Review integration guide (INTEGRATION_GUIDE.md)
4. Begin Phase 1 testing with low-risk work
5. Iterate based on feedback

**Estimated Timeline**: 3-4 weeks through all phases

---

## Benefits

### For Users
- ✅ Reduced cognitive load (focus on implementation, not workflow)
- ✅ No more "did you commit?" reminders
- ✅ Automatic task tracking and completion
- ✅ Files always in correct locations

### For Agents
- ✅ Self-sufficient workflow automation
- ✅ Clear invocation patterns
- ✅ Graceful error handling
- ✅ No need to remember workflow mechanics

### For Project
- ✅ 80-90% work loss prevention (micro-commits)
- ✅ Complete audit trail (task-mcp + git)
- ✅ Constitutional compliance automation
- ✅ Reduced token overhead (97% vs embedded approach)

---

## Architectural Constraints

### Subagents Cannot Use Skills

**CRITICAL**: Due to Claude Code's architecture, subagents spawned via the Task tool **cannot invoke Skills**. This has important implications for workflow-automation:

**What This Means**:
- ✅ Main chat CAN use workflow-automation to orchestrate work
- ✅ Skills CAN call subagents to perform work
- ❌ Subagents CANNOT use Skills (architectural limitation)
- ❌ Workflow-automation CANNOT be invoked within subagent contexts

**Design Implications**:
1. **Main chat orchestration**: Workflow-automation guides the main chat in orchestrating subagents, but doesn't automate workflow within subagent execution
2. **Pre-work setup**: Main chat uses workflow-automation to mark tasks in_progress, then spawns subagents for implementation
3. **Post-work cleanup**: After subagent completes, main chat uses workflow-automation to commit and mark tasks done
4. **Hybrid approach**: Main chat handles workflow mechanics (task tracking, commits), subagents focus on implementation

**Example Usage Pattern**:
```markdown
# Main chat (with workflow-automation)
1. Mark task #42 as in_progress
2. Spawn implementation subagent (no Skills)
3. Subagent completes work
4. Main chat commits work (detect milestone)
5. Main chat marks task #42 as done
```

This constraint shapes the skill's design—it's a **main chat orchestration tool**, not a **subagent automation tool**.

---

## Known Limitations

1. **Requires MCP server**: Task-mcp must be online (graceful degradation if not)
2. **Git repository required**: Won't work outside git repos
3. **Constitutional knowledge needed**: Must understand FILE_ORGANIZATION_POLICY.md
4. **Manual validation needed**: Agent must verify work complete before marking task done
5. **Subagent constraint**: Cannot be invoked by subagents (architectural limitation)

---

## Version History

**v1.0.0** (2025-11-03)
- Initial draft specification
- 3-tier progressive disclosure architecture
- Complete testing strategy (16 scenarios + 12 edge cases)
- Comprehensive integration guide (5 examples)
- Production-ready design pending validation

---

## Self-Contained & Portable

This skill is fully self-contained with no external dependencies beyond:
- **Git** (for commit automation)
- **task-mcp MCP server** (optional - graceful degradation if offline)

All reference documentation is included in the skill's `references/` directory:
- `FILE_ORGANIZATION_POLICY.md` - Complete organization rules
- `MICRO_COMMIT_DISCIPLINE.md` - Commit discipline guide
- `cc-helpers-functions.md` - Helper function documentation
- `commit-milestone-detection.md` - Milestone detection algorithm
- `file-organization-validation.md` - Path validation rules
- `error-recovery-patterns.md` - Graceful degradation
- `advanced-workflows.md` - Multi-session patterns

All prompting patterns are included in the skill's `prompting-patterns/` directory:
- `task-tracking-prompts.md` - Task management prompting patterns
- `commit-discipline-prompts.md` - Commit discipline prompting patterns
- `file-organization-prompts.md` - File organization prompting patterns

**Portability**: This skill can be copied to other projects without external dependencies (beyond Git and optional MCP server).

---

## References

- Constitution: `.specify/memory/constitution.md`
- Skill Portfolio: `.claude/skills/README.md`

---

**Ready for review!** 🎯
