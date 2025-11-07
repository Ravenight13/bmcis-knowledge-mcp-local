---
name: universal-workflow-orchestrator
description: Universal session initialization and workflow automation for any project type. Automates context detection, loads task-specific guidance, enforces workflow principles (parallel orchestration, micro-commits, quality gates), and validates system health. Use at session start for consistent, efficient workflow setup across web apps, data pipelines, DevOps automation, or any development project.
---

# Universal Workflow Orchestrator

---

## Architectural Note: Skill vs Agent Distinction

**Why This Is a Skill (Not an Agent)**:
- Provides **instructional procedural knowledge** for session initialization
- Does NOT execute orchestration independently—guides Claude's actions
- Describes what Claude SHOULD do (knowledge) vs what it WILL do (behavior)
- Model-invoked based on session start context

**What It Instructs**:
1. How to detect task context (knowledge)
2. Which workflows to apply (procedural)
3. Workflow norms to enforce (policy)
4. Health checks to perform (checklist)

Claude interprets these instructions and executes using its tools. The skill itself
does not perform autonomous orchestration—it provides the orchestration playbook.

---

## 📘 How This Skill Works

**This is a KNOWLEDGE resource**, not executable behavior.

**Skills are model-invoked**:
- Claude activates skills when task context matches skill descriptions
- No @ syntax, no load_skill() function, no special invocation
- Skills provide reference knowledge, not executable code

**When this skill activates**:
- User requests session initialization or workflow orchestration
- Keywords detected: "session start", "orchestration", "workflow", "setup"
- Claude model-invokes this skill to provide orchestration guidance

**Relationship to other skills**:
This skill references other skills conceptually, but doesn't invoke them programmatically. Other skills activate independently when their contexts match.

---

**For detailed explanation of skills vs agents architecture**, load `references/model-invoked-architecture.md` (800 tokens)

---

## Overview

Automate session initialization by intelligently detecting task context, loading relevant workflows, setting quality expectations, and validating system health. Eliminates repetitive setup work while ensuring best practices from the start.

**Core Value Proposition**: Universal benefit across all development sessions, regardless of project domain.

**Key Differentiators**:
- Context-aware workflow loading (customizable contexts)
- Parallel orchestration enforcement (multi-agent coordination)
- Quality gate validation (linting, type checking, testing)
- Micro-commit discipline tracking (≤30 min intervals)
- Health validation automation (git, tools, documentation)
- Flexible customization points for project-specific needs

## When to Use This Skill

Apply universal-workflow-orchestrator at session start when:
- Beginning new development session for any project type
- Starting web development, data engineering, DevOps, or infrastructure work
- Need consistent workflow setup with quality discipline
- Want automated context detection and tool validation
- Require best practice enforcement from session start
- Orchestrating parallel subagents with file output requirements

**Frequency**: Every development session

## Core Capabilities

### 1. Session Initialization

**Purpose**: Automate repetitive session start tasks

**Automated Actions**:
1. **Context Detection**: Identify task type from git branch, working directory, or user input
2. **Workflow Loading**: Apply task-specific workflow guidance
3. **Health Validation**: Check git status, quality tools, documentation
4. **Checklist Generation**: Provide structured session start template
5. **Directory Setup**: Early creation of subagent output directories (if not exists)

**Benefits**:
- Consistent session start across all projects
- Reduced manual setup time (typical 40% reduction)
- Early issue detection (git state, tool availability)
- Clear next actions based on context

---

### 2. Context Detection (Generic)

**Purpose**: Identify task type to apply appropriate workflow guidance

**Detection Algorithm**:
1. Parse git branch name (e.g., `feat/auth-service` → BACKEND_DEV)
2. Check current working directory path
   - `src/components/`, `frontend/` → FRONTEND_DEV
   - `src/api/`, `backend/`, `services/` → BACKEND_DEV
   - `tests/`, `__tests__/` → TESTING
   - `infra/`, `terraform/`, `k8s/` → INFRASTRUCTURE
   - `data/`, `pipelines/`, `etl/` → DATA_ENGINEERING
3. Read project status file (customizable location)
4. Fallback to user confirmation if ambiguous

**Context Types (Customizable)**:
- **FRONTEND_DEV**: UI/UX development (React, Vue, Angular, etc.)
- **BACKEND_DEV**: API/service development (Node, Python, Go, etc.)
- **DATA_ENGINEERING**: Data pipelines, ETL, analytics
- **INFRASTRUCTURE**: DevOps, IaC, deployment automation
- **TESTING**: Test implementation, QA automation
- **DOCUMENTATION**: Technical writing, API docs

**Output**: Context type + confidence score (HIGH/MEDIUM/LOW)

**Customization Point**: Configure context types in project-specific config file

**Implementation**: Load `references/context-detection.md` when detection is ambiguous for detailed context definitions (1,000 tokens)

---

### 3. Workflow Principles Enforcement

**Purpose**: Set behavioral norms explicitly to prevent anti-patterns

**Universal Principles**:

**1. Parallel Subagent Orchestration (MANDATORY)**:
- Main chat = orchestration layer ONLY
- Complex tasks → spawn multiple subagents in parallel
- Example: 10 component reviews → 10 parallel agents = 10x speedup
- Benefits: Speed, expertise, token efficiency, clarity

**Subagent File Output Protocols (CRITICAL)**:
- **MANDATORY**: All subagents MUST write their findings to files (not just report back verbally)
- **File locations**:
  - Research findings: `docs/subagent-reports/{agent-type}/{component}/YYYY-MM-DD-HHMM-description.md`
  - Session handoffs: `session-handoffs/YYYY-MM-DD-HHMM-description.md`
  - Analysis results: `docs/analysis/YYYY-MM-DD-HHMM-description.md`
- **Micro-commit requirement**: Subagents must commit their files immediately upon completion
- **Rationale**: Prevents work loss if main session crashes, provides audit trail, enables async review

**Example workflow**:
```
User: "Analyze these 5 API endpoints using parallel subagents"

Claude (Main Chat):
  → Spawns 5 subagents (one per endpoint)

Subagent 1:
  → Analyzes endpoint-auth
  → Writes findings to docs/subagent-reports/api-analysis/auth/2025-10-27-1400-auth-analysis.md
  → Commits file: "docs(api): add auth endpoint analysis by subagent"
  → Returns summary to main chat

Subagent 2-5: (same pattern)

Main Chat:
  → Synthesizes 5 subagent reports
  → Creates consolidated recommendation
```

**Benefits**:
- Work preserved even if session crashes mid-analysis
- Audit trail of all subagent work
- Can be reviewed asynchronously
- Enables handoff between sessions

**2. Micro-Commit Discipline**:
- Target: ≤30 minutes between commits
- Frequency: Every 20-50 lines OR logical milestone
- Risk reduction: 80-90% work loss prevention
- Research phase: Auto-commit via checkpoint commands

**3. Quality Gates Before Commit**:
- Run linting (project-specific: ESLint, Ruff, Prettier, etc.)
- Run type checking (TypeScript, mypy, Flow, etc.)
- Run tests (unit, integration, as configured)
- Verify build succeeds (if applicable)

**4. Session Handoffs**:
- Document session state for continuity
- Format: Timestamped markdown files (YYYY-MM-DD-HHMM-description.md)
- Location: Configurable (e.g., `session-handoffs/`, `docs/sessions/`)
- Content: Completed work, next priorities, blockers, subagent results
- **Template**: Use `assets/TEMPLATE_SESSION_HANDOFF.md` as structure
- **Automation** (optional): Projects may define slash commands (e.g., `/handoff`, `/checkpoint`) for automated creation

**5. File Naming Conventions with Timestamps**:
- Format: `YYYY-MM-DD-HHMM-description.md` (e.g., `2025-10-27-1500-api-design.md`)
- Apply to: Session handoffs, research reports, documentation, analysis files
- Rationale: Track chronological order when multiple files created in one day
- Headers use: `HH:MM:SS` format for timestamped sections

**Customization Point**: Configure quality gates and checkpoint commands per project

**Implementation**: Load `references/workflow-principles.md` when establishing workflow discipline for complete guidance (900 tokens)

---

### 4. System Health Validation

**Purpose**: Identify issues before development starts

**Validation Checks**:

**1. Git Status**:
```bash
git status --short --branch
# Check: branch name, uncommitted files, sync status
```

**2. Quality Gates**:
```bash
# Project-specific quality commands
npm run lint && npm run typecheck  # JavaScript/TypeScript
ruff check . && mypy .              # Python
cargo clippy && cargo test          # Rust
go vet ./... && go test ./...       # Go
```

**3. Tool Availability**:
- Language runtime (Node, Python, Go, etc.)
- Package manager (npm, pip, cargo, etc.)
- Build tools (webpack, vite, make, etc.)
- Test frameworks (jest, pytest, cargo test, etc.)

**4. Documentation Structure**:
- Check: Project-specific docs directory exists
- Verify: README.md, CONTRIBUTING.md presence
- Load: Session-specific workflow guidance (if available)

**5. Project Status File**:
- Primary: Database-driven tracking (if MCP available)
- Fallback: Markdown status file (configurable location)
- Extract: Active work, blockers, priorities

**6. Environment Configuration**:
- Verify: .env files exist (if required)
- Check: Required environment variables set
- Validate: Configuration files present (tsconfig.json, pyproject.toml, etc.)

**Output**: Health report with warnings/errors flagged

**Customization Point**: Configure health checks per project type and toolchain

---

### 5. Session Checklist Generation

**Purpose**: Provide structured session start template

**Checklist Sections**:
1. **Context Confirmation**: Task type, branch, active work
2. **Skills Loaded**: Which Skills loaded, token budget status
3. **Workflow Reminders**: Parallel orchestration, quality gates, micro-commits
4. **Health Status**: Git, tools, documentation, configuration
5. **Next Actions**: Recommended first steps based on context

**Customization Point**: Customize checklist template per project

**Template**: Use `assets/session-checklist.md` as session start template

---

### 6. Integration Patterns

**Purpose**: Work seamlessly with existing project tools and workflows

**Git Integration**:
- Enforce branch naming conventions
- Prevent pushes to protected branches (main, production)
- Auto-generate commit message templates
- Track commit frequency for micro-commit discipline

**Tool Integration**:
- Run quality gates before commits
- Execute tests automatically
- Build verification
- Documentation generation

**Other Skills Integration**:
```
Session Start → universal-workflow-orchestrator (model-invoked, 1,500 tokens)
    ↓
Context Detected → Task-specific skill (model-invoked when context matches)
    ↓
Complex Work → Research/planning skills (model-invoked when complexity detected)
    ↓
Implementation → Micro-commit tracking ACTIVE
    ↓
Pre-Commit → Quality gates enforcement
    ↓
COMMIT ✅
```

**Model-Invoked Composition**: Claude activates skills independently based on
description matching. Skills don't invoke other skills—they activate in parallel
when context matches their descriptions. No special syntax or programmatic loading
is required or available.

**Customization Point**: Configure integration commands and workflows per project

## Usage Workflow

### Step 1: Invoke Skill

User starts session with trigger phrase:
```
"Start new development session"
"Initialize workflow for [feature/task]"
"Set up session for [project work]"
```

Skill will be model-invoked based on description matching.

---

### Step 2: Context Detection

**Detection Logic** (illustrative - not executable code):

1. Parse git branch name: `feat/auth-api` → BACKEND_DEV
2. Check current working directory: `/src/api/` → BACKEND_DEV
3. Read project status file for active work context
4. Return context type with confidence score: (BACKEND_DEV, HIGH)

---

### Step 3: Workflow Loading

Apply task-specific workflow guidance:

**Conceptual Workflow** (Illustrative - NOT executable):
```markdown
When context == BACKEND_DEV:
    → Load API development best practices
    → Reference relevant documentation
    → Suggest appropriate tools and patterns
    → Set quality gates for backend work

When context == FRONTEND_DEV:
    → Load UI/UX development guidance
    → Reference component patterns
    → Suggest styling and accessibility tools
    → Set quality gates for frontend work

**Note**: This is conceptual documentation. Actual workflow loading adapts
to detected context automatically.
```

---

### Step 4: Workflow Enforcement

Set explicit expectations:
```
✅ Parallel orchestration: ENABLED
✅ Quality gates: CONFIGURED ([project-specific tools])
✅ Micro-commit tracking: ACTIVE (≤30 min intervals)
✅ Session handoffs: ENABLED
```

---

### Step 5: Health Validation

Check system state:
```bash
Git status: feat/auth-api (clean working tree)
Quality gates: [linter] ✅ | [type checker] ✅ | [tests] ✅
Tools: [runtime] ✅ | [package manager] ✅ | [build tool] ✅
Documentation: README.md ✅ | CONTRIBUTING.md ✅
Environment: .env ✅ | config files ✅
```

---

### Step 6: Generate Session Checklist

Provide structured template:
```markdown
## Session Checklist

**Context**: BACKEND_DEV (HIGH confidence)
**Branch**: feat/auth-api
**Active Work**: Implement authentication service endpoints

**Skills Loaded**:
- universal-workflow-orchestrator (1,500 tokens)
- [Context-specific skills as activated]

**Workflow Reminders**:
- [ ] Use parallel subagents for complex tasks
- [ ] Run quality gates before each commit
- [ ] Commit every 20-50 lines or ≤30 min
- [ ] Use timestamp format: YYYY-MM-DD-HHMM-description.md
- [ ] Document session state at checkpoints

**Health Status**:
- Git: ✅ Clean working tree
- Quality gates: ✅ Passing
- Tools: ✅ All available
- Documentation: ✅ Present

**Next Actions**:
1. Review existing API documentation
2. Analyze authentication requirements
3. Design endpoint structure
4. Implement with micro-commit discipline
5. Run quality gates before commits
```

## Customization Guide

**Project-Specific Configuration**:

### 1. Define Context Types

Edit `references/context-detection.md` to define your project's context types:
```markdown
# Project: E-commerce Platform
Contexts:
- STOREFRONT_DEV: Customer-facing UI (src/storefront/)
- ADMIN_DEV: Admin dashboard (src/admin/)
- API_DEV: Backend services (src/api/)
- INFRASTRUCTURE: Deployment/infra (infra/, k8s/)
```

### 2. Configure Quality Gates

Edit `references/workflow-principles.md` to set quality commands:
```markdown
# Quality Gates
Linting: npm run lint
Type checking: npm run typecheck
Testing: npm test
Build: npm run build
```

### 3. Set Up Session Handoffs

Configure session handoff location:
```markdown
# Session Handoffs
Location: .session-handoffs/
Format: YYYY-MM-DD-HHMM-description.md
Auto-commit: Yes (via checkpoint command)
```

### 4. Customize Health Checks

Edit `references/health-validation.md` to configure checks:
```markdown
# Health Checks
Git: Standard checks
Tools: [node, npm, docker]
Environment: .env.local required
Documentation: docs/ directory required
```

### 5. Adapt Checklist Template

Edit `assets/session-checklist.md` to customize session start template with project-specific sections.

---

## Integration Points

### Slash Commands (Project-Specific)

Projects can define custom slash commands that integrate with this skill:
- **Session initialization**: `/start`, `/init`, `/ready` - Automated workflow setup
- **Session handoff**: `/handoff`, `/checkpoint`, `/save-session` - Automated handoff generation using TEMPLATE_SESSION_HANDOFF.md
- **Compliance validation**: `/validate`, `/check`, `/comply` - Project standards verification

**Example: Handoff Command**
```
User: "/handoff Feature implementation complete"

What the command does:
1. Copies assets/TEMPLATE_SESSION_HANDOFF.md to session-handoffs/
2. Auto-fills context (branch, date, time, git status)
3. Prompts for completed work and next priorities
4. Commits the handoff file
```

**Note**: These are optional. Projects without slash commands can use templates manually.

### MCP Servers (Optional)

If MCP servers are available:
- **Semantic search**: Find relevant patterns, examples
- **Work item tracking**: Query active tasks, update status
- **Project status**: Read current operational state

### Template Directory (Optional)

If template directory exists:
- **Location**: Configurable (e.g., `docs/templates/`, `.templates/`)
- **Purpose**: Centralized templates for documentation, reports, checklists
- **Usage**: Reference templates for consistent documentation

### Other Skills

Context-specific skills that may be model-invoked based on detected context:
- **Frontend development**: Component patterns, accessibility, styling
- **Backend development**: API design, database patterns, security
- **Data engineering**: Pipeline patterns, data quality, transformations
- **Infrastructure**: Deployment patterns, scaling, monitoring

## Complete Example: Backend API Session Start

**User Query**:
```
"Start new session for authentication API implementation"
```

**Skill Response**:
```
✅ SESSION INITIALIZED - Universal Workflow

**Context**: BACKEND_DEV (HIGH confidence)
**Branch**: feat/auth-api
**Skills Loaded**: Universal Workflow Orchestrator (1,500 tokens)

**Workflow Enforcement**:
- Parallel orchestration: ENABLED
- Quality gates: CONFIGURED (linter + type checker + tests)
- Micro-commit tracking: ACTIVE (≤30 min intervals)
- Session handoffs: ENABLED

**System Health**:
- Git: ✅ feat/auth-api (clean working tree)
- Quality gates: ✅ All tools passing
- Tools: ✅ Runtime + package manager + build tool available
- Documentation: ✅ README.md + CONTRIBUTING.md present
- Environment: ✅ Configuration files valid

**Best Practices Available**:
1. API endpoint design patterns
2. Authentication security guidelines
3. Error handling standards
4. Testing strategies for auth flows
5. Documentation requirements

**Next Steps**:
1. Review authentication requirements documentation
2. Analyze existing API structure and patterns
3. Design authentication endpoint structure
4. Implement with micro-commit discipline (≤30 min intervals)
5. Run quality gates before each commit

Ready to begin work. What would you like to tackle first?
```

---

## References

**Location**: `.claude/skills/universal-workflow-orchestrator/references/`

Load these reference files when detailed guidance needed:

- **`references/model-invoked-architecture.md`** (800 tokens): Skills vs agents distinction, knowledge vs behavior, integration patterns
- **`references/context-detection.md`** (1,000 tokens): Context type definitions, detection algorithms, customization guide
- **`references/workflow-principles.md`** (900 tokens): Parallel orchestration, micro-commits, quality gates, session handoffs
- **`references/health-validation.md`** (800 tokens): Git checks, tool validation, documentation verification, environment configuration

**Total Available**: 3,500 tokens (lazy-loaded)

**Asset**:
- **`assets/session-checklist.md`**: Session start checklist template (customizable)
