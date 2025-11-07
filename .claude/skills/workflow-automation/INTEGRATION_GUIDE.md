# Workflow Automation Skill - Agent Integration Guide

**Last Updated**: 2025-11-03
**Version**: 1.0.0
**Target Audience**: Claude Code agents, main chat orchestrator, skill developers

---

## Overview

The **workflow-automation** skill provides automated task tracking, git commit discipline, and file organization capabilities for multi-file implementations and long-running work sessions. This guide shows agents and the main chat how to effectively integrate and use this skill.

**Core Capabilities**:
- Automated task tracking (work item management)
- Git micro-commit discipline (≤30 min intervals)
- File organization enforcement (constitutional compliance)
- Milestone detection and commit triggering
- Work session continuity (handoff generation)

**Key Philosophy**: The skill is a **process enforcer**, not a code generator. It ensures workflow discipline while agents focus on implementation.

---

## 1. When to Use - Decision Tree

Use this decision tree to determine if the workflow-automation skill should be invoked:

```
┌─────────────────────────────────────┐
│ Is this a multi-file implementation │
│ (3+ files changing)?                │
└──────────┬────────────────┬─────────┘
           │ YES            │ NO
           ▼                ▼
    ┌──────────────┐   ┌────────────────┐
    │ USE SKILL    │   │ Does work span  │
    └──────────────┘   │ >30 minutes?    │
                       └────┬────────┬───┘
                            │ YES    │ NO
                            ▼        ▼
                     ┌──────────┐  ┌─────────────┐
                     │USE SKILL │  │ Is there a  │
                     └──────────┘  │ clear task  │
                                   │ tracking    │
                                   │ need?       │
                                   └──┬─────┬────┘
                                      │YES  │NO
                                      ▼     ▼
                               ┌──────────┐ ┌────────────┐
                               │USE SKILL │ │ SKIP SKILL │
                               └──────────┘ └────────────┘
```

### ✅ YES - Use Workflow Automation When:

1. **Multi-File Implementations**
   - Vendor extractor implementation (5-15 files)
   - Framework refactoring (multiple modules)
   - Documentation overhaul (10+ files)
   - Test suite creation (fixtures + tests + utilities)

2. **Work Spanning >30 Minutes**
   - Complex debugging sessions
   - Research → Implementation cycle
   - Multi-phase feature development
   - Architecture changes

3. **Clear Task Tracking Needs**
   - Work items need to be tracked in database
   - Session continuity required (handoff generation)
   - Multiple milestones to track
   - Dependencies between tasks

4. **Git Discipline Requirements**
   - Micro-commit discipline needed (≤30 min intervals)
   - Work loss prevention critical
   - Audit trail required
   - Multiple logical milestones

### ❌ NO - Skip Workflow Automation When:

1. **Simple File Operations**
   - Single file read
   - Quick search operation (Grep/Glob)
   - Documentation typo fix
   - Simple file move/rename

2. **Quick Questions**
   - Code explanation request
   - Documentation lookup
   - Simple status check
   - Quick analysis (<5 minutes)

3. **Trivial Changes**
   - Single-line fix
   - Comment addition
   - Minor formatting change
   - Quick config update

4. **Exploratory Work**
   - Initial codebase exploration
   - Quick prototype (throwaway code)
   - Proof-of-concept testing
   - Ad-hoc experimentation

---

## 2. Invocation Patterns

### 2.1 From Agent (Self-Invocation)

**Pattern**: Agent recognizes need and invokes skill autonomously

**Agent Prompt Addition** (~50 tokens):
```markdown
**Workflow Automation Integration**:
If implementing multi-file changes or work spanning >30 minutes, I'll use the
workflow-automation skill for task tracking, git discipline, and file organization.

Invocation: "I'll use the workflow-automation skill for this {task-type}"
```

**Example Agent Dialogue**:
```
Agent (implementation-specialist): "I'm implementing the COASTAL_SOURCE vendor
extractor, which will involve 8 files (extractor.py, config.py, brand enum,
tests, fixtures, documentation). I'll use the workflow-automation skill to
ensure proper task tracking and micro-commit discipline throughout this
implementation."

[Skill activates and provides workflow guidance]
```

**When Agent Should Self-Invoke**:
- Implementation task clearly meets decision tree criteria
- Agent has autonomy to manage workflow
- No user override detected
- Work complexity warrants automation

---

### 2.2 From Main Chat (Orchestrated Invocation)

**Pattern**: Main chat detects need and spawns agent with skill

**Main Chat Orchestration Logic**:
```markdown
When delegating complex task to agent:
  1. Assess task complexity (files, duration, tracking needs)
  2. If decision tree indicates YES → Include workflow-automation skill
  3. Spawn agent with explicit skill instruction
  4. Monitor skill usage in agent output

Example delegation:
"I'll spawn an implementation-specialist agent with the workflow-automation
skill to handle this vendor extractor implementation."
```

**Example Main Chat Dialogue**:
```
User: "Implement the COASTAL_SOURCE vendor extractor"

Main Chat: "This is a complex implementation involving multiple files
(extractor, config, tests, docs). I'll spawn an implementation-specialist
agent with the workflow-automation skill to ensure proper task tracking
and git discipline.

[Spawns agent with skill loaded]

Agent (with workflow-automation): "I'll track work items in task-mcp
database and commit at logical milestones. Starting with vendor config..."
```

**Main Chat Coordination**:
- Detects complexity before delegation
- Explicitly mentions skill loading to user
- Monitors agent for skill usage
- Synthesizes workflow artifacts in final report

---

### 2.3 From User (Explicit Request)

**Pattern**: User directly requests skill usage

**User Request Examples**:
```
"Please use the workflow-automation skill for this vendor implementation"
"Enable workflow automation for this refactoring session"
"I want task tracking and git discipline for this work"
"Use workflow-automation to track this multi-file change"
```

**Agent/Main Chat Response**:
```
Agent: "Workflow automation skill activated. I'll:
  1. Track work items in task-mcp database
  2. Enforce micro-commit discipline (≤30 min intervals)
  3. Follow file organization policy
  4. Generate session handoff at completion

Starting implementation with these workflow controls in place."
```

**User Override Acknowledgment**: Always confirm skill activation and explain what it provides

---

## 3. Integration Examples

### Example 1: Vendor Implementation (Multi-File, Task Tracking, Commits)

**Scenario**: Implementing new COASTAL_SOURCE vendor extractor

**Files Involved** (8 files):
- `backend/src/extractors/coastal_source/extractor.py`
- `backend/src/extractors/coastal_source/config.py`
- `backend/src/extractors/coastal_source/__init__.py`
- `backend/src/extractors/coastal_source/brand.py`
- `backend/tests/acceptance/test_coastal_source.py`
- `backend/tests/acceptance/fixtures/vendors/COASTAL_SOURCE/golden/`
- `docs/vendors/COASTAL_SOURCE/README.md`
- `docs/vendors/COASTAL_SOURCE/CHANGELOG.md`

**Workflow Automation Actions**:

1. **Task Tracking**:
```bash
# Create parent task
mcp__task-mcp__create_task \
  title="Implement COASTAL_SOURCE vendor extractor" \
  status="in_progress" \
  priority="high" \
  tags="vendor implementation coastal-source"

# Create subtasks
mcp__task-mcp__create_task \
  title="Create TypedExtractorConfig for COASTAL_SOURCE" \
  parent_task_id=42 \
  status="pending"

mcp__task-mcp__create_task \
  title="Implement brand detection logic" \
  parent_task_id=42 \
  depends_on=[43]
```

2. **Micro-Commit Discipline** (≤30 min intervals):
```bash
# Milestone 1: Config created (15 min elapsed)
git add backend/src/extractors/coastal_source/config.py
git commit -m "feat(coastal-source): Add TypedExtractorConfig"

# Milestone 2: Brand enum (28 min elapsed)
git add backend/src/extractors/coastal_source/brand.py
git commit -m "feat(coastal-source): Add VendorBrand enum with 3 brands"

# Milestone 3: Core extractor (25 min elapsed)
git add backend/src/extractors/coastal_source/extractor.py
git commit -m "feat(coastal-source): Implement core extraction logic"
```

3. **File Organization**:
- Enforce vendor isolation (all vendor code in `coastal_source/`)
- Documentation in `docs/vendors/COASTAL_SOURCE/`
- Tests in `backend/tests/acceptance/`
- Fixtures in standardized golden/ structure

4. **Session Handoff Generation**:
```markdown
## Session Handoff - COASTAL_SOURCE Implementation
**Date**: 2025-11-03 14:30:00

**Completed Work**:
- ✅ TypedExtractorConfig created and validated
- ✅ VendorBrand enum with COASTAL_1, COASTAL_2, COASTAL_3
- ✅ Core extraction logic implemented
- ✅ 3 micro-commits completed

**Next Actions**:
- [ ] Implement reconciliation validation
- [ ] Create golden test fixtures
- [ ] Write acceptance tests
- [ ] Update vendor documentation

**Task Status**: Task #42 (60% complete, 3/5 subtasks done)
```

**Agent Workflow**:
```
Agent: "Starting COASTAL_SOURCE vendor implementation with workflow-automation
skill. I'll track tasks in database, commit at logical milestones, and
generate handoff at completion.

[15 minutes later]
Agent: "Config complete (TypedExtractorConfig). Committing milestone 1/5..."

[28 minutes later]
Agent: "Brand enum complete. Committing milestone 2/5..."

[25 minutes later]
Agent: "Core extractor logic complete. Committing milestone 3/5..."

[Session end]
Agent: "3/5 milestones complete. Generating session handoff with remaining
work items. Task #42 updated in database (60% progress)."
```

---

### Example 2: Framework Refactor (Multiple Components, Git Discipline)

**Scenario**: Refactoring error handling hierarchy across 5 vendor extractors

**Files Involved** (10+ files):
- `backend/src/extractors/base.py` (error hierarchy)
- `backend/src/extractors/{vendor}/extractor.py` (5 vendors)
- `backend/tests/unit/extractors/test_errors.py`
- `docs/guides/ERROR_HANDLING_MIGRATION_GUIDE.md`

**Workflow Automation Actions**:

1. **Task Breakdown**:
```bash
# Parent task
mcp__task-mcp__create_task \
  title="Refactor error handling hierarchy" \
  description="Migrate 5 vendors to new ExtractorError subclasses" \
  status="in_progress" \
  priority="high"

# Component subtasks
mcp__task-mcp__create_task \
  title="Update base.py error hierarchy" \
  parent_task_id=50

mcp__task-mcp__create_task \
  title="Migrate LEGRAND to new errors" \
  parent_task_id=50 \
  depends_on=[51]
```

2. **Micro-Commit Strategy** (component-based):
```bash
# Commit 1: Base hierarchy (20 min)
git add backend/src/extractors/base.py
git commit -m "refactor(framework): Add ExtractorError subclass hierarchy"

# Commit 2: First vendor (25 min)
git add backend/src/extractors/legrand/extractor.py
git commit -m "refactor(legrand): Migrate to ExtractorValidationError"

# Commit 3: Second vendor (22 min)
git add backend/src/extractors/sound_united/extractor.py
git commit -m "refactor(sound-united): Migrate to new error hierarchy"
```

3. **File Organization Enforcement**:
- Base changes in `backend/src/extractors/base.py`
- Vendor changes isolated to vendor modules
- Tests updated in parallel
- Documentation synchronized

**Agent Workflow**:
```
Agent: "Refactoring error hierarchy across 5 vendors. Using workflow-automation
to track component progress and ensure each vendor migration is committed
separately for easy rollback if needed.

[20 minutes - base hierarchy complete]
Agent: "Base error hierarchy complete. Committing before vendor migrations..."

[25 minutes - LEGRAND migrated]
Agent: "LEGRAND migrated successfully. Committing vendor 1/5..."

[Continues through all vendors with separate commits]
```

---

### Example 3: Documentation Overhaul (File Organization, Commits)

**Scenario**: Reorganizing vendor documentation to 7-category structure

**Files Involved** (40+ files across 7 vendors):
- `docs/vendors/{VENDOR}/README.md`
- `docs/vendors/{VENDOR}/CHANGELOG.md`
- `docs/vendors/{VENDOR}/format-analysis/`
- `docs/vendors/{VENDOR}/implementation-notes/`
- (7 vendors × 5-8 files each)

**Workflow Automation Actions**:

1. **Task Tracking** (vendor-by-vendor):
```bash
# Parent task
mcp__task-mcp__create_task \
  title="Migrate all vendors to 7-category structure" \
  status="in_progress"

# Per-vendor subtasks
for vendor in LEGRAND LUTRON EPSON "Sound United" TPD AMINA COASTAL_SOURCE; do
  mcp__task-mcp__create_task \
    title="Migrate $vendor to 7-category structure" \
    parent_task_id=60
done
```

2. **Micro-Commit Strategy** (vendor-by-vendor):
```bash
# Commit after each vendor (prevents massive changes)
git add docs/vendors/LEGRAND/
git commit -m "docs(legrand): Migrate to 7-category structure"

git add docs/vendors/LUTRON/
git commit -m "docs(lutron): Migrate to 7-category structure"
```

3. **File Organization Enforcement**:
- Validate 7-category structure for each vendor
- Ensure YYYY-MM-DD-HHMM timestamp format
- Check README.md and CHANGELOG.md exist
- Verify file placement per policy

**Agent Workflow**:
```
Agent: "Migrating 7 vendors to new documentation structure. Using
workflow-automation to track vendor-by-vendor progress and commit each
vendor separately for atomic changes.

[30 minutes - LEGRAND complete]
Agent: "LEGRAND migration complete (8 files reorganized). Committing
vendor 1/7... Task #61 marked complete."

[Continues through all vendors]

[Session end]
Agent: "7/7 vendors migrated. Generating handoff with validation
checklist for documentation standards compliance."
```

---

### Example 4: Test Suite Creation (Milestone Commits, File Placement)

**Scenario**: Creating comprehensive test suite for COASTAL_SOURCE vendor

**Files Involved** (15+ files):
- Golden fixtures (inputs/ and outputs/)
- Test scenarios (3 scenarios × inputs/outputs)
- Acceptance tests (`test_coastal_source.py`)
- Test utilities (`test_helpers.py`)
- Manifest validation tests

**Workflow Automation Actions**:

1. **Task Breakdown** (by test phase):
```bash
mcp__task-mcp__create_task \
  title="Create COASTAL_SOURCE test suite" \
  status="in_progress"

# Subtasks
mcp__task-mcp__create_task \
  title="Create golden fixtures (inputs + outputs)" \
  parent_task_id=70

mcp__task-mcp__create_task \
  title="Implement happy path tests" \
  parent_task_id=70 \
  depends_on=[71]

mcp__task-mcp__create_task \
  title="Add edge case test scenarios" \
  parent_task_id=70 \
  depends_on=[72]
```

2. **Micro-Commit Strategy** (milestone-based):
```bash
# Milestone 1: Golden fixtures (30 min)
git add backend/tests/acceptance/fixtures/vendors/COASTAL_SOURCE/golden/
git commit -m "test(coastal-source): Add golden fixtures for 2025-07 period"

# Milestone 2: Happy path tests (25 min)
git add backend/tests/acceptance/test_coastal_source.py
git commit -m "test(coastal-source): Add happy path acceptance tests"

# Milestone 3: Edge cases (28 min)
git add backend/tests/acceptance/fixtures/vendors/COASTAL_SOURCE/test-scenarios/
git commit -m "test(coastal-source): Add 3 edge case test scenarios"
```

3. **File Placement Validation**:
- Golden fixtures in `fixtures/vendors/COASTAL_SOURCE/golden/`
- Test scenarios in `fixtures/vendors/COASTAL_SOURCE/test-scenarios/`
- Tests in `backend/tests/acceptance/`
- Follow fixture README structure

**Agent Workflow**:
```
Agent: "Creating test suite for COASTAL_SOURCE. Using workflow-automation
to track test phases and commit at each milestone (fixtures → happy path →
edge cases).

[30 minutes - fixtures complete]
Agent: "Golden fixtures created (inputs + 3 brand outputs). Committing
test milestone 1/3..."

[25 minutes - happy path complete]
Agent: "Happy path tests passing (5/5). Committing test milestone 2/3..."

[28 minutes - edge cases complete]
Agent: "3 edge case scenarios implemented. Committing test milestone 3/3.
Test suite complete - 100% coverage achieved."
```

---

### Example 5: Debug Session (Task Tracking, Git Discipline)

**Scenario**: Debugging reconciliation failure in Sound United extractor

**Files Involved** (variable, could be 1-5 files):
- `backend/src/extractors/sound_united/extractor.py`
- Possibly config, brand logic, or calculation utilities

**Workflow Automation Actions**:

1. **Task Tracking** (debug phases):
```bash
# Parent debug task
mcp__task-mcp__create_task \
  title="Debug Sound United reconciliation failure" \
  description="Input total: \$50,000 | Output total: \$49,856 | Delta: \$144" \
  status="in_progress" \
  priority="high" \
  tags="debug reconciliation sound-united"

# Debug phase subtasks
mcp__task-mcp__create_task \
  title="Identify root cause (5 Whys analysis)" \
  parent_task_id=80

mcp__task-mcp__create_task \
  title="Implement fix" \
  parent_task_id=80 \
  depends_on=[81]

mcp__task-mcp__create_task \
  title="Validate reconciliation passes" \
  parent_task_id=80 \
  depends_on=[82]
```

2. **Micro-Commit Strategy** (debug loop phases):
```bash
# Commit 1: Root cause identified (investigation phase)
git add docs/debug-reports/2025-11-03-1430-sound-united-reconciliation.md
git commit -m "docs(sound-united): Document reconciliation failure root cause"

# Commit 2: Fix implemented (fix phase)
git add backend/src/extractors/sound_united/extractor.py
git commit -m "fix(sound-united): Correct Decimal quantization for commission calc"

# Commit 3: Validation passed (verify phase)
git commit --allow-empty -m "test(sound-united): Reconciliation now passing (100%)"
```

3. **Debug Loop Integration**:
- Workflow automation integrates with debug-loop-enforcer skill
- Enforces TEST → EVALUATE → PLAN → FIX → COMMIT cycle
- Tracks progress through debug phases
- Documents root cause analysis

**Agent Workflow**:
```
Agent: "Debugging Sound United reconciliation failure. Using workflow-automation
with debug-loop-enforcer to track debug phases and ensure each fix is
committed with validation.

[Phase 1: TEST - 10 min]
Agent: "Reconciliation failing: \$144 delta. Running root cause analysis..."

[Phase 2: EVALUATE - 15 min]
Agent: "Root cause: Decimal quantization error in commission calculation.
Committing analysis document..."

[Phase 3: PLAN - 5 min]
Agent: "Fix strategy: Use quantize_to_cents() helper. Impact: 1 file, 3 lines."

[Phase 4: FIX - 20 min]
Agent: "Fix implemented. Committing code change..."

[Phase 5: COMMIT - 5 min]
Agent: "Validation passed - reconciliation 100%. Recording validation commit.
Debug session complete."
```

---

## 4. Agent Recommendations

### 4.1 Which Agents Benefit Most?

**High Benefit** (Recommend for all instances):

1. **implementation-specialist**
   - **Why**: Multi-file implementations are core use case
   - **Frequency**: Every vendor implementation (~15.5/year)
   - **Prompt Addition** (~50 tokens):
   ```markdown
   When implementing multi-file features or vendor extractors, I'll use the
   workflow-automation skill to track work items, enforce micro-commit
   discipline, and ensure file organization compliance.
   ```

2. **debugger / test-automator**
   - **Why**: Debug sessions benefit from structured tracking
   - **Frequency**: ~10-15 debug sessions/year
   - **Prompt Addition** (~50 tokens):
   ```markdown
   For complex debugging spanning >30 minutes, I'll use workflow-automation
   to track debug phases, commit fixes at each milestone, and document
   root cause analysis.
   ```

3. **documentation-automator**
   - **Why**: Documentation overhauls involve many files
   - **Frequency**: ~5-10 doc sessions/year
   - **Prompt Addition** (~50 tokens):
   ```markdown
   When reorganizing documentation across multiple files or vendors, I'll
   use workflow-automation to track file organization compliance and commit
   changes atomically (per vendor or category).
   ```

**Medium Benefit** (Recommend when work is complex):

4. **framework-research / architect-review**
   - **Why**: Framework changes often span multiple files
   - **Frequency**: ~5-8 sessions/year
   - **Usage**: Conditional based on complexity
   - **Prompt Addition** (~50 tokens):
   ```markdown
   When research leads to multi-file framework changes, I'll recommend
   workflow-automation skill for implementation phase to ensure tracking
   and git discipline.
   ```

5. **general-purpose**
   - **Why**: Handles diverse tasks, some requiring automation
   - **Frequency**: Variable
   - **Usage**: Assess per task using decision tree
   - **Prompt Addition** (~50 tokens):
   ```markdown
   For multi-file work or tasks spanning >30 minutes, I'll use
   workflow-automation to provide structure and prevent work loss through
   micro-commit discipline.
   ```

**Low Benefit** (Skip unless explicitly requested):

6. **quick-search / code-navigator**
   - **Why**: Tasks are typically exploratory and short
   - **Frequency**: Many, but brief
   - **Recommendation**: Don't add to default prompt

7. **question-answerer**
   - **Why**: No file changes, just information retrieval
   - **Recommendation**: Don't add to default prompt

---

### 4.2 Lightweight Prompt Additions

**Template** (~50 tokens per agent):
```markdown
**Workflow Automation Integration**:
When {specific-use-case}, I'll use the workflow-automation skill to
{specific-benefits}.

Invocation trigger: {decision-criteria}
```

**Examples by Agent Type**:

**Implementation Specialist** (52 tokens):
```markdown
**Workflow Automation Integration**:
When implementing multi-file features or vendor extractors, I'll use the
workflow-automation skill to track work items in task-mcp database, enforce
micro-commit discipline (≤30 min), and ensure file organization compliance.

Invocation trigger: 3+ files changing OR work duration >30 min
```

**Debugger** (48 tokens):
```markdown
**Workflow Automation Integration**:
For debugging sessions spanning >30 minutes, I'll use workflow-automation
to track debug phases (TEST→EVALUATE→PLAN→FIX→COMMIT), commit fixes at
milestones, and document root cause analysis.

Invocation trigger: Complex debugging OR reconciliation failures
```

**Documentation Automator** (45 tokens):
```markdown
**Workflow Automation Integration**:
When reorganizing documentation across multiple files or vendors, I'll use
workflow-automation to enforce file organization policy and commit changes
atomically (per vendor or category).

Invocation trigger: 5+ documentation files changing
```

**Framework Research** (50 tokens):
```markdown
**Workflow Automation Integration**:
When research leads to multi-file framework changes, I'll recommend
workflow-automation skill for implementation phase to ensure task tracking
and git discipline throughout refactoring.

Invocation trigger: Framework changes affecting 3+ modules
```

**General Purpose** (47 tokens):
```markdown
**Workflow Automation Integration**:
For multi-file implementations or work spanning >30 minutes, I'll use
workflow-automation to track progress, enforce micro-commit discipline,
and prevent work loss.

Invocation trigger: Decision tree indicates YES
```

---

### 4.3 How to Recommend Skill Usage (Without Being Pushy)

**Principle**: Inform, don't insist. Explain value, let user decide.

**Good Recommendation Pattern**:
```
Agent: "This implementation involves {X files/Y duration}. I recommend using
the workflow-automation skill to {specific-benefits}. This will help us
{outcome-1} and {outcome-2}.

Would you like me to proceed with workflow automation enabled, or would
you prefer manual tracking for this work?"
```

**Examples**:

**Multi-File Implementation**:
```
Agent: "This vendor implementation will modify 8 files across extractor,
config, tests, and documentation. I recommend using the workflow-automation
skill to track work items in the database and ensure micro-commits at
logical milestones. This prevents work loss and provides clear audit trail.

Should I enable workflow automation, or would you prefer to track this manually?"
```

**Long Duration Work**:
```
Agent: "This debugging session may take 60+ minutes given the complexity.
I recommend using workflow-automation to track debug phases and commit
fixes incrementally. This ensures we don't lose progress if the session
is interrupted.

Would you like me to activate workflow automation for this debug session?"
```

**User Override Respected**:
```
User: "Skip workflow automation, just fix it quickly"

Agent: "Understood. I'll proceed without workflow automation - quick fix
mode. I'll still commit when the fix is complete, but won't track
intermediate milestones."
```

**Gentle Reminder** (if work becomes complex):
```
Agent: "I've now modified 4 files and we're 35 minutes into this work.
This would benefit from workflow-automation skill (task tracking + micro-commits).

Would you like me to activate it now, or continue with manual workflow?"
```

**Key Principles**:
1. **Explain value** - Say what skill provides, not just "I recommend skill X"
2. **Give choice** - Always offer opt-out
3. **Respect override** - Don't argue if user says no
4. **Gentle reminders** - If work becomes complex, suggest once (not repeatedly)
5. **Benefits-focused** - Explain outcomes (work loss prevention, audit trail)

---

## 5. Main Chat Orchestration

### 5.1 How Should Main Chat Suggest Skill Usage?

**Main chat role**: Orchestrator and strategist, not executor

**Suggestion Pattern**:
1. **Assess task complexity** before delegation
2. **Recommend skill** if decision tree indicates benefit
3. **Explain value** to user in orchestration plan
4. **Spawn agent** with skill loaded
5. **Monitor progress** and synthesize results

**Example Orchestration Dialogue**:

```
User: "Implement the COASTAL_SOURCE vendor extractor"

Main Chat: "I'll orchestrate this vendor implementation by spawning an
implementation-specialist agent. This is a complex task involving:
  - 8 files (extractor, config, tests, docs)
  - Multiple milestones (config → logic → tests → docs)
  - Task tracking needs (work items in database)

I recommend using the workflow-automation skill to ensure:
  ✅ Task tracking in task-mcp database
  ✅ Micro-commit discipline (≤30 min intervals)
  ✅ File organization compliance
  ✅ Session handoff generation

Spawning implementation-specialist agent with workflow-automation skill..."

[Agent executes with workflow automation]

Main Chat: "Implementation complete. The workflow-automation skill tracked
5 work items, created 4 micro-commits, and generated a session handoff
with remaining test tasks. Agent reports COASTAL_SOURCE extractor is 80%
complete (core logic done, tests pending)."
```

**Key Elements**:
- Assess before delegation
- Explain skill value to user
- Show orchestration transparency
- Synthesize workflow artifacts in summary

---

### 5.2 When to Auto-Invoke vs. Ask User?

**Decision Matrix**:

| Task Characteristics | Action | Rationale |
|---------------------|--------|-----------|
| Multi-file (5+ files) + User expects structured work | **Auto-invoke** | Clear benefit, user expectation |
| Multi-file (3-4 files) + Uncertain duration | **Ask user** | Moderate benefit, unclear need |
| Work duration >60 min + Complex task | **Auto-invoke** | Work loss risk high |
| Work duration 30-60 min + Simple task | **Ask user** | Moderate risk, may be overkill |
| Clear task tracking need (user mentions "track", "organize") | **Auto-invoke** | User explicitly wants structure |
| Exploratory work (user says "let's see...") | **Don't invoke** | Premature optimization |
| User has previously used skill | **Auto-invoke** | User preference established |
| First-time user | **Ask user** | Educational opportunity |

**Auto-Invoke Example**:
```
User: "Implement COASTAL_SOURCE vendor extractor with full test suite"

Main Chat: "Spawning implementation-specialist with workflow-automation
skill (8 files, 60+ min estimated, task tracking required)..."

[No user confirmation needed - clear benefit]
```

**Ask User Example**:
```
User: "Can you add error handling to the LEGRAND extractor?"

Main Chat: "This could involve 3-5 files (extractor + config + tests).
Would you like me to use the workflow-automation skill for task tracking
and micro-commits, or keep it simple for a quick update?"

[User confirms preference before proceeding]
```

**Don't Invoke Example**:
```
User: "Let's explore the COASTAL_SOURCE input file structure"

Main Chat: "I'll spawn a data-analysis agent to examine the file structure..."

[No workflow automation - exploratory work, no file changes expected]
```

---

### 5.3 Coordination with Other Skills

**Integration Architecture**:

```
Main Chat Orchestration Layer
    ↓
┌───────────────────────────────────────────────────────┐
│ Skill Composition (Model-Invoked, Context-Aware)      │
├───────────────────────────────────────────────────────┤
│                                                        │
│  workflow-automation ← (Process enforcement)          │
│         ↓                                             │
│  Coordinates with:                                    │
│         ↓                                             │
│  ┌─────────────────┐  ┌──────────────────┐           │
│  │ pattern-library │  │ debug-loop       │           │
│  │ -guide          │  │ -enforcer        │           │
│  └─────────────────┘  └──────────────────┘           │
│         ↓                      ↓                      │
│  (Domain patterns)      (Debug methodology)           │
│                                                        │
│  ┌──────────────────┐  ┌──────────────────┐          │
│  │ research-        │  │ commission-      │          │
│  │ orchestrator     │  │ workflow-orch    │          │
│  └──────────────────┘  └──────────────────┘          │
│         ↓                      ↓                      │
│  (Research process)     (Session init)                │
└───────────────────────────────────────────────────────┘
```

**Coordination Patterns**:

**1. Workflow Automation + Pattern Library Guide**:
```
Use case: Vendor implementation
Main chat: "Spawning agent with workflow-automation + pattern-library-guide"

Benefits:
  - Workflow-automation tracks tasks and commits
  - Pattern-library-guide provides domain patterns
  - No conflicts - complementary capabilities
```

**2. Workflow Automation + Debug Loop Enforcer**:
```
Use case: Complex debugging session
Main chat: "Spawning debugger agent with workflow-automation + debug-loop-enforcer"

Integration:
  - Debug-loop-enforcer provides methodology (TEST→EVALUATE→PLAN→FIX→COMMIT)
  - Workflow-automation tracks debug phases and enforces commits
  - Combined: Structured debugging with automated tracking
```

**3. Workflow Automation + Research Orchestrator**:
```
Use case: Research → Implementation workflow
Main chat: "Phase 1: Research-orchestrator for analysis"
          "Phase 2: Workflow-automation for implementation"

Sequence:
  1. Research-orchestrator runs research phase
  2. Generates implementation plan
  3. Workflow-automation activates for implementation
  4. Tracks implementation tasks from research plan
```

**4. Workflow Automation + Commission Workflow Orchestrator**:
```
Use case: Session initialization
Main chat: "Commission-workflow-orchestrator initializes session"
          "Workflow-automation activates for implementation work"

Relationship:
  - Commission-workflow: Session start, health checks, context detection
  - Workflow-automation: Ongoing work tracking and git discipline
  - Handoff: Commission-workflow enables, workflow-automation executes
```

**Coordination Principles**:
1. **No conflicts** - Skills provide different capabilities
2. **Additive value** - Skills enhance each other
3. **Clear boundaries** - Each skill has distinct purpose
4. **Model-invoked** - Claude activates based on context
5. **User transparency** - Main chat explains which skills active

---

## 6. User Override Patterns

Users can override workflow automation behavior. Agents must respect these overrides.

### 6.1 "Skip task tracking for this work"

**User Intent**: Want git commits but not database tracking

**Agent Response**:
```
Agent: "Understood. I'll skip task-mcp database tracking but maintain
micro-commit discipline. Changes will be committed at logical milestones
without creating work items."

Workflow adjustments:
  ✅ Micro-commits enabled (≤30 min intervals)
  ✅ File organization enforcement
  ❌ Task tracking in database
  ❌ Session handoff generation
```

**Implementation**:
- Continue micro-commits
- Skip `mcp__task-mcp__create_task` calls
- Document work in commit messages instead of database
- No session handoff at end

---

### 6.2 "Don't auto-commit, I'll do it manually"

**User Intent**: Want full control over git workflow

**Agent Response**:
```
Agent: "Understood. I'll implement changes without automatic commits.
I'll track milestones and remind you when logical commit points occur,
but won't execute git commands.

You'll be responsible for:
  - Creating commits at appropriate milestones
  - Following micro-commit discipline (≤30 min intervals)
  - Using conventional commit format

I'll provide commit message suggestions at each milestone."
```

**Implementation**:
- Track work and milestones
- Suggest commit points with draft messages
- Skip `git commit` execution
- User controls all git operations

**Milestone Reminder Example**:
```
Agent: "Milestone reached: TypedExtractorConfig implementation complete
(28 minutes elapsed).

Suggested commit:
  git add backend/src/extractors/coastal_source/config.py
  git commit -m 'feat(coastal-source): Add TypedExtractorConfig'

Ready to proceed when you've committed."
```

---

### 6.3 "Use directory X instead of Y"

**User Intent**: Override file organization policy for specific reason

**Agent Response**:
```
Agent: "Understood. I'll place files in {user-specified-directory}
instead of the standard {policy-directory}.

Note: This deviates from file organization policy
(docs/reference/constitutional/FILE_ORGANIZATION_POLICY.md). I'll document
this decision in commit message for future reference.

Proceeding with custom file placement..."
```

**Implementation**:
- Respect user's directory choice
- Document deviation in commit message
- Flag for constitutional review if needed
- Suggest policy update if pattern emerges

**Commit Message Example**:
```
feat(coastal-source): Add vendor documentation

BREAKING: Files placed in docs/custom/coastal-source/ instead of
docs/vendors/COASTAL_SOURCE/ per user override.

Rationale: [User's explanation or "User preference"]
```

---

### 6.4 Override Acknowledgment Best Practices

**Principles**:
1. **Acknowledge immediately** - Confirm you heard the override
2. **Explain adjustments** - Show what changes in workflow
3. **Document deviations** - Record in commits or tasks
4. **Don't argue** - Respect user autonomy
5. **Educate gently** - Mention policy/risk once, then move on

**Good Override Response Template**:
```
Agent: "Understood. I'll {adjusted-behavior} as requested.

Workflow adjustments:
  ✅ {What still applies}
  ❌ {What's disabled}

{Optional: One-sentence note about policy/risk if relevant}

Proceeding with your preferences..."
```

**Bad Override Response** (Don't do this):
```
Agent: "I strongly recommend against skipping task tracking because
[long explanation]. Are you sure you want to do this? It could cause
[problem 1], [problem 2], [problem 3]..."

[Too pushy, doesn't respect user decision]
```

---

## 7. Token Budget Considerations

**Skill Size**: ~800 tokens (core) + ~2,000 tokens (references) = **~2,800 tokens total**

**When Loaded**:
- Progressive disclosure: Load core (~800 tokens) initially
- References lazy-loaded on demand
- Typical usage: 800-1,200 tokens per session

**Integration Token Cost**:

**Per-Agent Prompt Addition**: ~50 tokens
**Main Chat Orchestration**: ~100 tokens (when explaining to user)
**Active Skill Context**: ~800 tokens (during usage)

**Total Session Cost** (agent with skill):
```
Agent base prompt:        ~500 tokens
Workflow-automation:      ~800 tokens (core)
Pattern-library-guide:  ~1,300 tokens (if VENDOR context)
Debug-loop-enforcer:      ~900 tokens (if debugging)
                        ─────────────────────────
Total:                  ~3,500 tokens (well within budget)
```

**Optimization Tips**:
1. Load skill only when decision tree indicates benefit
2. Use progressive disclosure (core → references)
3. Lazy-load references when detailed guidance needed
4. Don't load multiple workflow skills (commission-workflow vs workflow-automation)

---

## 8. Quick Reference

### Decision Tree (One-Liner)
```
Multi-file (3+) OR duration (>30min) OR tracking-need → USE SKILL
```

### Invocation Patterns (Quick)
```
Agent:     "I'll use workflow-automation for this {task}"
Main chat: "Spawning agent with workflow-automation"
User:      "Use workflow-automation for this work"
```

### Integration Points (Quick)
```
✅ pattern-library-guide (domain patterns)
✅ debug-loop-enforcer (debug methodology)
✅ research-orchestrator (research → implementation)
✅ commission-workflow-orchestrator (session init)
```

### User Overrides (Quick)
```
"Skip task tracking"     → Commits only, no database
"Don't auto-commit"      → Manual git workflow
"Use directory X"        → Custom file placement
```

### Agent Prompt Template (Quick)
```markdown
**Workflow Automation**: When {trigger}, use workflow-automation
for {benefits}. Trigger: {criteria}
```

---

## Appendix A: Real-World Examples

### A.1 LEGRAND Vendor Implementation (Actual Session)

**Context**: Implementing LEGRAND vendor extractor (very complex, 11 brands)

**Files Changed**: 12 files
- Config, extractor, brand enum, tests, fixtures (8 input files), docs

**Workflow Automation Usage**:

**Task Tracking**:
```bash
# Parent task created
Task #25: "Implement LEGRAND vendor extractor"
  Subtasks:
    #26: Create TypedExtractorConfig (DONE - 15 min)
    #27: Implement brand detection logic (DONE - 28 min)
    #28: Add entity name fallback (DONE - 22 min)
    #29: Implement reconciliation (DONE - 30 min)
    #30: Create test fixtures (IN_PROGRESS)
```

**Micro-Commits** (7 commits over 2 hours):
```
1. feat(legrand): Add TypedExtractorConfig (15 min)
2. feat(legrand): Implement VendorBrand enum with 11 brands (28 min)
3. feat(legrand): Add entity name fallback logic (22 min)
4. feat(legrand): Implement core extraction logic (30 min)
5. feat(legrand): Add reconciliation validation (25 min)
6. test(legrand): Create golden fixtures for 11 brands (27 min)
7. docs(legrand): Add vendor documentation (18 min)
```

**Outcome**:
- ✅ 12 files organized correctly
- ✅ 7 commits at logical milestones
- ✅ 100% reconciliation achieved
- ✅ Session handoff generated with test tasks
- ✅ Zero work loss (all progress tracked)

**Agent Reflection**:
```
"Workflow-automation skill prevented work loss during this 2-hour
implementation. Micro-commits ensured each milestone was saved. Task
tracking kept me focused on priorities (core logic before tests).
File organization enforcement prevented vendor isolation violations."
```

---

### A.2 Sound United Reconciliation Debug (Actual Session)

**Context**: Debugging \$144 reconciliation discrepancy

**Duration**: 55 minutes

**Workflow Automation Usage**:

**Task Tracking**:
```bash
Task #40: "Debug Sound United reconciliation failure"
  Description: "Input: \$50,000 | Output: \$49,856 | Delta: \$144"
  Subtasks:
    #41: Root cause analysis (DONE - 15 min)
    #42: Implement Decimal quantization fix (DONE - 20 min)
    #43: Validate reconciliation passes (DONE - 10 min)
```

**Debug Loop Integration**:
```
Phase 1: TEST (10 min)
  - Run production extractor
  - Identify \$144 discrepancy
  - Document failure

Phase 2: EVALUATE (15 min)
  - 5 Whys analysis
  - Root cause: Decimal precision loss in commission calculation
  - Commit: "docs(sound-united): Document reconciliation root cause"

Phase 3: PLAN (5 min)
  - Strategy: Use quantize_to_cents() helper
  - Impact: 1 file, 3 lines changed

Phase 4: FIX (20 min)
  - Implement Decimal quantization
  - Commit: "fix(sound-united): Use quantize_to_cents for commission calc"

Phase 5: COMMIT (5 min)
  - Validate: 100% reconciliation
  - Commit: "test(sound-united): Reconciliation now passing"
```

**Outcome**:
- ✅ 3 micro-commits (analysis → fix → validation)
- ✅ Root cause documented
- ✅ 100% reconciliation achieved
- ✅ Debug audit trail complete

**Agent Reflection**:
```
"Workflow-automation + debug-loop-enforcer combination was critical.
Enforced evaluation phase prevented jumping to wrong fix. Task tracking
maintained focus through 3 debug phases. Micro-commits provided rollback
points if fix failed."
```

---

### A.3 Documentation Reorganization (Actual Session)

**Context**: Migrating 7 vendors to new 7-category structure

**Files Changed**: 42 files across 7 vendors

**Duration**: 3 hours (90 min session 1, 90 min session 2)

**Workflow Automation Usage**:

**Task Tracking**:
```bash
Task #50: "Migrate all vendors to 7-category structure"
  Subtasks:
    #51: Migrate LEGRAND (DONE - Session 1)
    #52: Migrate LUTRON (DONE - Session 1)
    #53: Migrate EPSON (DONE - Session 1)
    #54: Migrate Sound United (DONE - Session 2)
    #55: Migrate TPD (DONE - Session 2)
    #56: Migrate AMINA (DONE - Session 2)
    #57: Migrate COASTAL_SOURCE (DONE - Session 2)
```

**Micro-Commits** (vendor-by-vendor):
```
Session 1:
1. docs(legrand): Migrate to 7-category structure (6 files, 25 min)
2. docs(lutron): Migrate to 7-category structure (5 files, 22 min)
3. docs(epson): Migrate to 7-category structure (7 files, 28 min)

Session 2:
4. docs(sound-united): Migrate to 7-category structure (8 files, 30 min)
5. docs(tpd): Migrate to 7-category structure (5 files, 20 min)
6. docs(amina): Migrate to 7-category structure (4 files, 18 min)
7. docs(coastal-source): Migrate to 7-category structure (6 files, 22 min)
```

**Session Handoff** (End of Session 1):
```markdown
## Session Handoff - Vendor Documentation Migration

**Completed** (3/7 vendors):
- ✅ LEGRAND (6 files migrated)
- ✅ LUTRON (5 files migrated)
- ✅ EPSON (7 files migrated)

**Remaining** (4/7 vendors):
- [ ] Sound United
- [ ] TPD
- [ ] AMINA
- [ ] COASTAL_SOURCE

**Next Session**:
1. Continue vendor-by-vendor migration
2. Commit after each vendor (atomic changes)
3. Validate 7-category structure compliance
4. Update VENDOR_DOCUMENTATION_STANDARDS.md when complete
```

**Outcome**:
- ✅ 42 files organized correctly
- ✅ 7 atomic commits (one per vendor)
- ✅ Session continuity across 2 sessions (handoff used)
- ✅ 100% compliance with documentation standards

**Agent Reflection**:
```
"Multi-session work benefited hugely from workflow-automation. Session
handoff preserved exact state (3/7 done). Task tracking prevented forgetting
which vendors remained. Vendor-by-vendor commits allowed easy rollback if
structure needed adjustment. File organization enforcement caught 2 placement
errors during migration."
```

---

## Appendix B: FAQ

**Q1: Can I use workflow-automation for single-file changes?**
A1: You can, but it's overkill. The skill is designed for multi-file work. For single-file changes, manual workflow is simpler and faster.

**Q2: What if task-mcp database is offline?**
A2: Workflow-automation degrades gracefully:
  - Primary: task-mcp database (live tracking)
  - Fallback: `.task-mcp-backup.json` (30-min cache)
  - Last resort: Manual tracking (skill provides templates)

**Q3: Can I use workflow-automation with other skills?**
A3: Yes! It's designed to integrate with pattern-library-guide, debug-loop-enforcer, research-orchestrator, and commission-workflow-orchestrator. See Section 5.3 for coordination patterns.

**Q4: How do I know when to use workflow-automation vs commission-workflow-orchestrator?**
A4:
  - **commission-workflow-orchestrator**: Session initialization, context detection, health checks
  - **workflow-automation**: Ongoing work tracking, git discipline, file organization
  - **Relationship**: Commission-workflow enables, workflow-automation executes

**Q5: What if I want git commits but not task tracking?**
A5: Use override: "Skip task tracking for this work". Agent will maintain micro-commits but skip database tracking.

**Q6: Can workflow-automation create PRs?**
A6: No. Workflow-automation handles task tracking, commits, and file organization. Use `prup` helper command or manual `gh pr create` for PR creation.

**Q7: Does workflow-automation work for frontend/API tasks?**
A7: Yes! It works for any multi-file implementation or >30 min work session, regardless of context (VENDOR/FRONTEND/API/TESTING).

**Q8: How does workflow-automation detect milestones?**
A8:
  - Time-based: ≤30 minutes elapsed
  - Logical: Component complete (config done, tests passing, docs written)
  - Agent judgment: Significant progress deserving commit
  - User signal: "Commit this milestone"

**Q9: Can I customize commit message format?**
A9: Workflow-automation uses conventional commit format (`type(scope): message`). If you need custom format, use override: "Don't auto-commit, I'll do it manually".

**Q10: What's the token cost of using workflow-automation?**
A10: ~800 tokens (core skill) + ~50 tokens (agent prompt integration) = ~850 tokens total. References are lazy-loaded only when needed (~2,000 tokens available).

---

## Document Metadata

**File**: `.claude/skills/workflow-automation/INTEGRATION_GUIDE.md`
**Token Count**: ~9,800 tokens
**Last Updated**: 2025-11-03
**Version**: 1.0.0
**Author**: Claude Code (Sonnet 4.5)

**Related Documents**:
- `.claude/skills/workflow-automation/SKILL.md` (Core skill definition)
- `.claude/skills/commission-workflow-orchestrator/SKILL.md` (Session initialization)
- `.claude/skills/debug-loop-enforcer/SKILL.md` (Debug methodology)
- `docs/guides/MICRO_COMMIT_DISCIPLINE.md` (Git discipline guide)
- `docs/reference/constitutional/FILE_ORGANIZATION_POLICY.md` (File organization rules)

---

**End of Integration Guide**
