# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Universal Workflow Orchestrator** project that provides Claude Code with automated session management, workflow automation, and development discipline enforcement through a skill-based architecture.

**Core Purpose**: Eliminate repetitive session setup work and enforce best practices (micro-commits, parallel orchestration, quality gates) across any project type through reusable skills and slash commands.

## Architecture

### Skill-Based System

This project uses Claude Code's **skill architecture** where skills are model-invoked knowledge resources (not executable agents):

- **Skills provide procedural knowledge** that guides Claude's actions
- **Skills activate automatically** when task context matches their descriptions
- **Skills don't invoke other skills** - they activate in parallel based on context
- **Skills are loaded progressively** (3-tier token budget: metadata → core → references)

### Directory Structure

```
.claude/
├── commands/           # Slash commands for workflow automation
│   ├── uwo-ready.md       # Session initialization
│   ├── uwo-checkpoint.md  # Quick progress saves
│   └── uwo-handoff.md     # Session end handoffs
└── skills/            # Skill definitions
    ├── workflow-automation/           # Task management + git automation
    │   ├── SKILL.md                   # Core skill protocol
    │   ├── references/                # Lazy-loaded guidance
    │   └── prompting-patterns/        # Subagent guidance patterns
    └── universal-workflow-orchestrator/  # Session initialization
        ├── SKILL.md                   # Core orchestration knowledge
        ├── references/                # Context detection, health validation
        └── assets/                    # Templates (handoff, research)
```

### Key Concepts

**1. Parallel Subagent Orchestration**
- Main chat acts as orchestration layer ONLY
- Complex tasks → spawn multiple subagents in parallel for speed
- ALL subagents MUST write findings to files (not just verbal reports)
- File locations: `docs/subagent-reports/{agent-type}/{component}/YYYY-MM-DD-HHMM-description.md`
- Subagents must micro-commit their files immediately

**2. Micro-Commit Discipline**
- Target: ≤30 minutes between commits
- Frequency: Every 20-50 lines OR logical milestone
- Risk reduction: 80-90% work loss prevention
- Auto-commit via checkpoint commands during research

**3. File Naming Conventions**
- Format: `YYYY-MM-DD-HHMM-description.md`
- Apply to: Session handoffs, research reports, subagent outputs
- Rationale: Chronological tracking when multiple files created per day

**4. Session Handoffs**
- Document session state for continuity between sessions
- Location: `session-handoffs/YYYY-MM-DD-HHMM-description.md`
- Use template: `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_SESSION_HANDOFF.md`
- Automated via `/uwo-handoff` command

## Common Commands

### Session Management

```bash
# Initialize new session (context detection + health checks)
/uwo-ready

# Quick mid-session checkpoint (2-3 min, auto-commits subagent reports)
/uwo-checkpoint

# Full session handoff at end (5 min, comprehensive context transfer)
/uwo-handoff "brief-description-of-work-completed"
```

### Development Workflow

Since this is a documentation/skill project (no build system), common operations are:

```bash
# Check for uncommitted work
git status --short

# View recent commits
git log --oneline -10

# Find specific skill documentation
find .claude/skills -name "*.md" | grep {keyword}

# Read session handoffs chronologically
ls -t session-handoffs/*.md | head -5
```

### Git Branch Masking

This project uses **opaque session identifiers** for all public-facing git operations to prevent exposure of internal task IDs and implementation details.

**Quick Reference:**

```bash
# Create new masked branch
.git-masking/new-branch.sh

# Check current task/session
.git-masking/check-branch.sh

# Commit with masked session ID (always use session-NNN format)
git commit -m "feat: session-004 - implement search functionality"

# NEVER expose task IDs or implementation details in commits
# ❌ WRONG:  git commit -m "feat: task 5.1 - implement RRF algorithm"
# ✅ RIGHT:  git commit -m "feat: session-004 - implement ranking"
```

**Branch Naming Convention:**
- Format: `work/session-{NNN}` (e.g., `work/session-001`, `work/session-042`)
- Session IDs are completely opaque (no task details visible)
- Internal mapping stored in `.git-masking/branch-mapping.json` (never committed)

**Key Concepts:**
- Only you have the mapping between session IDs and actual task IDs
- Helper scripts in `.git-masking/` handle branch creation and lookup
- All documentation in `BRANCH_MASKING.md` and `.git-masking/README.md`
- Template commit messages in `.git-masking/template-commit-messages.txt`

**Security:**
- `branch-mapping.json` is gitignored (never committed to remote)
- Pre-commit hook prevents accidental commits of sensitive mapping file
- Public git history shows only session numbers, no task details
- Code changes are visible, but task context is hidden

**For detailed usage and troubleshooting, see:** `BRANCH_MASKING.md`

### Quality Validation

This project has no linting/testing configured. When adding to projects that do:

```bash
# Node.js projects
npm run lint && npm run typecheck && npm test

# Python projects
ruff check . && mypy . && pytest

# Rust projects
cargo clippy && cargo test

# Go projects
go vet ./... && go test ./...
```

## Workflow Principles

### When Working in This Repository

1. **Session Initialization**: Always start with `/uwo-ready` to:
   - Detect context from git branch/directory
   - Validate system health (git status, tools)
   - Create required directories (session-handoffs/, docs/subagent-reports/)
   - Generate session checklist

2. **During Development**:
   - Use `/uwo-checkpoint` every 30-60 minutes
   - Commit frequently (≤30 min intervals)
   - Document research findings in timestamped markdown files
   - Create subagent reports in appropriate subdirectories

3. **Session End**:
   - Use `/uwo-handoff` for comprehensive context transfer
   - Auto-fills: metadata, git status, session statistics
   - Prompts for: executive summary, completed work, next priorities
   - Auto-commits handoff file

### Subagent File Output Protocol

**CRITICAL**: When spawning subagents for analysis/research:

1. **Main chat orchestrates** (doesn't do analysis itself)
2. **Subagents write findings to files**:
   - Research: `docs/subagent-reports/{type}/{component}/YYYY-MM-DD-HHMM-description.md`
   - Types: api-analysis, architecture-review, security-analysis, performance-analysis, code-review
3. **Subagents micro-commit** their files immediately
4. **Main chat synthesizes** subagent reports into recommendations

**Benefits**:
- Work preserved if session crashes
- Audit trail of all analysis
- Async review capability
- Session handoff includes all findings

## Skill Integration

### workflow-automation

**Purpose**: Automate task management (task-mcp), micro-commits (git), and file organization compliance

**When to use**:
- Multi-file implementations (2+ files)
- Work spanning >30 minutes
- Any implementation with clear task tracking needs

**Token budget**: 200 (metadata) → 1,400 (invoked) → 7,000 (full reference)

**Integration points**:
- task-mcp MCP server (optional - graceful degradation)
- Git automation (detect_commit_milestones)
- File organization validation

### universal-workflow-orchestrator

**Purpose**: Universal session initialization with context detection, health validation, and workflow setup

**When to use**: Start of every development session

**Token budget**: ~1,500 tokens (core skill)

**Capabilities**:
- Context detection (branch name, directory path, project type)
- System health validation (git, tools, documentation)
- Workflow enforcement (parallel orchestration, micro-commits, quality gates)
- Session checklist generation

## Context Detection Patterns

The universal-workflow-orchestrator detects context from:

1. **Git branch name**:
   - `feat/*` → DEVELOPMENT
   - `test/*` → TESTING
   - `docs/*` → DOCUMENTATION
   - `fix/*` → BUGFIX

2. **Directory path**:
   - `/src/`, `/lib/`, `/app/` → DEVELOPMENT
   - `/tests/`, `/__tests__/` → TESTING
   - `/docs/` → DOCUMENTATION

3. **Project type** (from config files):
   - `package.json` → Node.js
   - `pyproject.toml` → Python
   - `Cargo.toml` → Rust
   - `go.mod` → Go

## Design Principles

### Progressive Disclosure (3-Tier Token Budget)

Skills use 3-tier loading to minimize token overhead:

- **Tier 1**: Metadata (~200 tokens, always loaded in portfolio)
- **Tier 2**: Core SKILL.md (~1,200-1,500 tokens, loaded on invocation)
- **Tier 3**: Reference files (~3,500 tokens, lazy-loaded when needed)

**Efficiency**: 97% token reduction vs embedding everything in agents

### Graceful Degradation

All workflows handle missing dependencies gracefully:

- **MCP server offline**: Task tracking continues with manual updates
- **Template missing**: Creates fallback structure
- **Git unavailable**: Uses placeholder values, doesn't block workflow
- **Quality gates missing**: Reports "not configured", continues

### Universal Design

Skills work across any project type:

- No hard-coded project assumptions
- Configurable context types per project
- Adaptable quality gates and health checks
- Domain-agnostic workflow patterns

## Architectural Constraints

### Subagents Cannot Use Skills

**CRITICAL LIMITATION**: Subagents spawned via Task tool **cannot invoke Skills**.

**Implications**:
- ✅ Main chat CAN use skills to orchestrate work
- ✅ Skills CAN call subagents to perform work
- ❌ Subagents CANNOT use Skills (architectural limitation)
- ❌ Workflow-automation CANNOT be invoked within subagent contexts

**Design approach**: Skills are main chat orchestration tools, not subagent automation tools.

## Templates

### Session Handoff Template

Location: `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_SESSION_HANDOFF.md`

**Sections**:
- Metadata (date, time, branch, context)
- Executive Summary
- Completed Work (with evidence)
- Next Priorities
- Blockers & Challenges
- Session Statistics (auto-generated)
- Subagent Results (if any)
- Quality Gates Summary
- Git Status

### Research Report Template

Location: `.claude/skills/universal-workflow-orchestrator/assets/TEMPLATE_RESEARCH_REPORT.md`

For subagent research findings and analysis results.

## ROI Metrics

### Checkpoint Command (`/uwo-checkpoint`)
- **Time savings**: 2-3 min per checkpoint
- **Risk reduction**: 80-90% work loss prevention
- **Frequency**: 2-5x per session

### Handoff Command (`/uwo-handoff`)
- **Time savings**: 75% reduction (20 min → 5 min)
- **ROI**: 30:1 return
- **Annual savings**: 45 hours/year at 3 handoffs/week

### Workflow Automation Skill
- **Token overhead**: ≤10% (vs manual workflow)
- **Commit discipline**: ≥80% intervals within ≤30 min
- **File placement**: 100% compliance with organization policy

## Future Extensions

This workflow system is designed to be portable to other projects:

1. **Customization points**:
   - Context types (adapt to project domains)
   - Quality gates (configure linters/tests)
   - Health checks (project-specific tools)
   - File organization rules (per project policy)

2. **Integration patterns**:
   - MCP servers (semantic search, work tracking)
   - Project-specific slash commands
   - Custom skill implementations
   - Template variations

3. **Rollout phases**:
   - Phase 1: Low-risk testing (2-3 sessions)
   - Phase 2: Medium complexity (3-5 sessions)
   - Phase 3: Production (5-7 sessions)

## Key Files to Read

When extending or modifying this system:

- **Skill specifications**: `.claude/skills/*/SKILL.md`
- **Integration guides**: `.claude/skills/*/INTEGRATION_GUIDE.md`
- **Testing strategies**: `.claude/skills/*/TESTING_STRATEGY.md`
- **Reference documentation**: `.claude/skills/*/references/`
- **Slash command definitions**: `.claude/commands/*.md`

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
