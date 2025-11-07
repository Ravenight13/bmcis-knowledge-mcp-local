# Workflow Automation Skill - Testing & Validation Strategy

**Created**: 2025-11-03
**Status**: Pre-Production Testing Plan
**Target Skill Version**: v1.0.0

---

## Executive Summary

This document outlines the comprehensive testing and validation plan for the workflow-automation skill, which automates task-mcp tracking, git micro-commits, and file organization compliance. The skill aims to reduce cognitive overhead by 60-70% while maintaining constitutional compliance and workflow discipline.

**Testing Phases**:
1. **Phase 1**: Low-risk documentation work (2-3 sessions, ~6 hours)
2. **Phase 2**: Medium complexity refactors (3-5 sessions, ~12 hours)
3. **Phase 3**: Production vendor implementations (5-7 sessions, ~20 hours)

**Success Criteria**: ≥80% automation accuracy, ≤10% token overhead, zero critical failures

---

## Test Scenarios

### Scenario 1: Low-Risk Documentation Updates

**Complexity**: Low  
**Duration**: 30-60 minutes  
**Risk**: Minimal (no production code changes)

**Test Case 1.1: Update Vendor README**
```markdown
Context: Update docs/vendors/LEGRAND/README.md
Expected Behavior:
- Detects context: VENDOR_PROCESSING (confidence: MEDIUM)
- No task-mcp task required (documentation-only)
- No micro-commit automation (single file edit)
- File organization: ✅ Correct location
- Commit: Manual (documentation updates don't require micro-commits)

Validation:
✅ Skill correctly identifies low-complexity work
✅ Does NOT create unnecessary task-mcp entries
✅ Does NOT trigger micro-commit automation
✅ Validates file is in correct location
```

**Test Case 1.2: Create New Guide in docs/guides/**
```markdown
Context: Create docs/guides/NEW_FEATURE_GUIDE.md
Expected Behavior:
- Detects context: DOCUMENTATION (confidence: HIGH)
- Creates task-mcp task: "Document new feature guide"
- No micro-commit automation (documentation treated as single unit)
- File organization: ✅ Validates docs/guides/ location
- Commit: Manual at completion

Validation:
✅ Task-mcp entry created with correct metadata
✅ File placed in correct directory
✅ No premature commits during draft phase
✅ Skill suggests commit at logical completion
```

**Test Case 1.3: Archive Old Session Handoff**
```markdown
Context: Move session-handoffs/2025-10-15-vendor-work.md to docs/archive/
Expected Behavior:
- Detects context: MAINTENANCE (confidence: HIGH)
- No task-mcp task (housekeeping operation)
- File organization: ✅ Validates archival structure
- Validates: docs/archive/YYYY-MM-DD-[topic]/ pattern
- Commit: Manual after archival verification

Validation:
✅ Archival structure validated
✅ Timestamp convention enforced
✅ No task-mcp overhead for maintenance
✅ File organization compliance check passes
```

---

### Scenario 2: Medium Complexity (Helper Functions & Refactors)

**Complexity**: Medium  
**Duration**: 1-3 hours  
**Risk**: Low-Medium (isolated code changes)

**Test Case 2.1: Add Helper Function to base.py**
```markdown
Context: Add calculate_percentage_commission() to backend/src/extractors/base.py
Expected Behavior:
- Detects context: FRAMEWORK_DEV (confidence: HIGH)
- Creates task-mcp task: "Add percentage commission helper to base.py"
- Micro-commit milestone: After function implementation (~30-50 lines)
- File organization: ✅ backend/src/extractors/base.py
- Commit message: "feat(framework): Add percentage commission calculation helper"

Validation:
✅ Task created with file_references: ["backend/src/extractors/base.py"]
✅ Milestone detected at ~40 lines
✅ Auto-commit triggered with conventional commit format
✅ Task status updated: "in_progress" → "done"
✅ checkall passes before commit
```

**Test Case 2.2: Refactor Vendor Extractor (Single File)**
```markdown
Context: Refactor backend/src/extractors/lutron/extractor.py
Expected Behavior:
- Detects context: VENDOR_PROCESSING (confidence: HIGH, vendor=LUTRON)
- Creates task-mcp task: "Refactor LUTRON extractor for type safety"
- Links to vendor entity: mcp__task-mcp__link_entity_to_task
- Micro-commit milestones: Every 30-50 lines OR logical grouping
- Expected commits: 2-3 during 150-line refactor
- File organization: ✅ backend/src/extractors/lutron/
- Commit messages: "refactor(lutron): <milestone description>"

Validation:
✅ Vendor entity linked to task
✅ Milestone detection at 45, 90, 145 lines
✅ Each commit passes checkall
✅ Commit frequency: ≤30 minutes between commits
✅ Task metadata updated with commit SHAs
```

**Test Case 2.3: Multi-File Refactor (Coordinated)**
```markdown
Context: Refactor base.py + 2 vendor extractors
Expected Behavior:
- Detects context: FRAMEWORK_DEV + VENDOR_PROCESSING (confidence: MEDIUM)
- Creates parent task: "Type system migration for extractors"
- Creates subtasks: ["Migrate base.py", "Migrate LUTRON", "Migrate LEGRAND"]
- Micro-commit per file: 3 commits minimum
- File organization: ✅ Multiple file validation
- Dependency tracking: base.py → vendor extractors

Validation:
✅ Hierarchical task structure created
✅ Dependencies enforced (base.py first)
✅ Micro-commits per file
✅ Cross-file validation passes
✅ Parent task completion tracks subtask status
```

---

### Scenario 3: High Complexity (Multi-File Vendor Implementation)

**Complexity**: High  
**Duration**: 3-6 hours  
**Risk**: Medium-High (production code, financial logic)

**Test Case 3.1: New Vendor Implementation (Full Workflow)**
```markdown
Context: Implement backend/src/extractors/coastal_source/ (5-7 files)
Expected Behavior:
- Detects context: VENDOR_PROCESSING (confidence: HIGH, vendor=COASTAL_SOURCE)
- Creates vendor entity: metadata with phase="implementation"
- Creates parent task: "Implement COASTAL SOURCE vendor extractor"
- Creates subtasks (7):
  1. "Create TypedExtractorConfig"
  2. "Implement format detection"
  3. "Add entity name fallback pattern"
  4. "Implement CSV output path"
  5. "Implement JSON output path"
  6. "Add reconciliation validation"
  7. "Create golden test fixtures"
- Micro-commit frequency: 8-12 commits over 3-6 hours
- Expected milestone commits:
  - Config creation (~30 lines)
  - Format detection (~50 lines)
  - CSV path complete (~60 lines)
  - JSON path complete (~60 lines)
  - Reconciliation added (~40 lines)
  - Test fixtures complete (~50 lines)
- File organization: ✅ All files in backend/src/extractors/coastal_source/
- Vendor documentation: ✅ Creates docs/vendors/COASTAL_SOURCE/ structure

Validation:
✅ Vendor entity created with correct metadata
✅ Task hierarchy matches implementation plan
✅ 8-12 commits at ≤30 min intervals
✅ Each commit passes checkall
✅ Subtask status updates automatically
✅ Parent task completes when all subtasks done
✅ Vendor documentation structure validated
✅ File references tracked per subtask
```

**Test Case 3.2: Vendor Enhancement (Multi-Path Changes)**
```markdown
Context: Add new output format to Sound United extractor (3 files)
Expected Behavior:
- Detects context: VENDOR_PROCESSING (confidence: HIGH, vendor=SOUND_UNITED)
- Links to existing vendor entity (already in database)
- Creates task: "Add XML output format to Sound United"
- Micro-commit milestones: Config update, format implementation, tests
- Expected commits: 3-4
- File organization: ✅ backend/src/extractors/sound_united/
- Pattern usage: Validates reconciliation_validation still passes

Validation:
✅ Existing vendor entity detected and linked
✅ Vendor metadata updated (formats: ["excel", "xml"])
✅ 3-4 commits at logical milestones
✅ Reconciliation validation enforced
✅ Golden files updated in test-scenarios/
✅ CHANGELOG.md updated automatically
```

---

### Scenario 4: Error Conditions

**Test Case 4.1: Task-MCP Offline**
```markdown
Condition: MCP server unavailable during session
Expected Behavior:
- Graceful degradation to manual workflow
- Display warning: "⚠️ task-mcp offline - manual tracking required"
- Fallback to .task-mcp-backup.json (if exists, age < 30 min)
- Micro-commit automation: DISABLED (can't track milestones)
- File organization: Still enforced (no MCP dependency)
- User prompt: "Please re-enable task-mcp for full automation"

Validation:
✅ No crashes or exceptions
✅ Clear warning message displayed
✅ Backup file used if available
✅ File organization checks still pass
✅ Manual workflow guidance provided
✅ Session continues without blocking
```

**Test Case 4.2: Git Merge Conflict During Auto-Commit**
```markdown
Condition: Auto-commit triggered but git has merge conflicts
Expected Behavior:
- Detect conflict: git add fails with merge error
- Halt automation: Do NOT retry commit
- Display error: "⚠️ Git merge conflict detected - resolve manually"
- Provide guidance: "Run: git status && git mergetool"
- Micro-commit automation: PAUSED until conflict resolved
- Task status: Mark as "blocked" with blocker_reason

Validation:
✅ Auto-commit safely aborted
✅ No data loss or corruption
✅ Clear resolution instructions provided
✅ Task status updated to "blocked"
✅ Automation resumes after manual resolution
```

**Test Case 4.3: Ambiguous File Placement**
```markdown
Condition: New file doesn't clearly match FILE_ORGANIZATION_POLICY.md
Example: "vendor_commission_report.md" - docs/vendors/ or docs/reports/?
Expected Behavior:
- File organization check: ❌ AMBIGUOUS
- Halt file creation
- Prompt user: "Where should this file go?"
  - Option 1: docs/vendors/{VENDOR}/status-reports/
  - Option 2: docs/reports/ (if new category needed)
  - Option 3: Custom path (user-specified)
- Wait for user decision
- Validate chosen path against policy
- Create file if validation passes

Validation:
✅ Ambiguity detected correctly
✅ User prompted for decision
✅ No files created in wrong location
✅ Validation enforced on chosen path
✅ Constitutional compliance maintained
```

**Test Case 4.4: Multiple Tasks Matching Work Context**
```markdown
Condition: Branch feat/vendor-legrand but 3 in-progress LEGRAND tasks
Expected Behavior:
- Detect ambiguity: Multiple active tasks for same vendor
- Display task list:
  1. [HIGH] "Fix reconciliation tolerance issue" (ID: 42)
  2. [MEDIUM] "Add XML format support" (ID: 43)
  3. [LOW] "Update documentation" (ID: 44)
- Prompt user: "Which task are you working on?"
- Link commits to selected task
- Update task status automatically

Validation:
✅ All matching tasks displayed
✅ User prompted to choose
✅ Commits linked to correct task
✅ Other tasks remain in "in_progress"
✅ No cross-contamination between tasks
```

**Test Case 4.5: Premature Task Completion**
```markdown
Condition: Skill attempts to mark task "done" but work incomplete
Example: Only 2/5 subtasks completed, parent task marked "done"
Expected Behavior:
- Validate subtask status before parent completion
- Block parent completion: "❌ Cannot complete - 3 subtasks still pending"
- Display pending subtasks:
  - [TODO] "Implement JSON output path"
  - [TODO] "Add reconciliation validation"
  - [TODO] "Create golden test fixtures"
- Require manual confirmation: "Force complete anyway? (not recommended)"
- If forced: Add metadata flag "forced_completion: true"

Validation:
✅ Premature completion blocked
✅ Pending subtasks clearly listed
✅ Manual override available (with warning)
✅ Forced completions flagged in metadata
✅ Constitutional compliance maintained
```

---

### Scenario 5: User Override Scenarios

**Test Case 5.1: Manual Task-MCP Override**
```markdown
Context: User wants to manually manage task-mcp (no automation)
Expected Behavior:
- User command: "/workflow-automation disable task-tracking"
- Disable task-mcp automation
- Display confirmation: "✅ Task tracking automation disabled"
- Micro-commit automation: Still active (independent feature)
- File organization: Still enforced (constitutional requirement)
- Re-enable: "/workflow-automation enable task-tracking"

Validation:
✅ Automation correctly disabled
✅ No task-mcp calls during session
✅ Other features still functional
✅ Re-enable works correctly
✅ User preference persisted for session
```

**Test Case 5.2: Manual Micro-Commit Control**
```markdown
Context: User wants full control over commits (disable automation)
Expected Behavior:
- User command: "/workflow-automation disable micro-commits"
- Disable auto-commit automation
- Still display milestone suggestions: "💡 Consider commit at line 45"
- User commits manually via cc command
- Task-mcp tracking: Still active (independent feature)
- Re-enable: "/workflow-automation enable micro-commits"

Validation:
✅ Auto-commits disabled
✅ Milestone suggestions still shown
✅ Manual commits tracked correctly
✅ Task status updates still work
✅ Re-enable restores automation
```

**Test Case 5.3: Custom File Organization Rule**
```markdown
Context: User needs to place file outside standard policy
Example: "test_coastal_source_debug.py" → docs/vendors/COASTAL_SOURCE/tests/debugging/
Expected Behavior:
- File organization check: ❌ NON-STANDARD LOCATION
- Display warning: "⚠️ File outside standard policy"
- Prompt user: "Confirm custom location?"
  - Option 1: Continue with custom path
  - Option 2: Use standard location (suggest: backend/tests/scratch/)
- If confirmed: Create file + add exception to session metadata
- Document exception: "Custom path approved by user"

Validation:
✅ Non-standard location detected
✅ User confirmation required
✅ Exception documented in metadata
✅ File created at approved location
✅ No constitutional violation logged
```

---

## Edge Cases (10+ Scenarios)

### Edge Case 1: Task-MCP Database Offline
**Condition**: PostgreSQL database unreachable  
**Expected**: Fallback to .task-mcp-backup.json → Manual tracking → Clear warning  
**Validation**: No crashes, graceful degradation, user guidance provided

### Edge Case 2: Git Detached HEAD State
**Condition**: Repository in detached HEAD (not on branch)  
**Expected**: Block auto-commits, display error: "⚠️ Detached HEAD - checkout branch first"  
**Validation**: No commits in detached state, clear resolution steps

### Edge Case 3: Commit Milestone False Positive
**Condition**: 50 lines of comments trigger milestone detection  
**Expected**: Milestone suggestion displayed but NOT auto-committed (requires code changes)  
**Validation**: Comments don't trigger auto-commits, only substantive code changes

### Edge Case 4: File Placement Policy Update
**Condition**: FILE_ORGANIZATION_POLICY.md updated mid-session  
**Expected**: Reload policy, re-validate all pending file operations, notify user of changes  
**Validation**: Policy changes detected, new rules applied immediately

### Edge Case 5: Premature Task Completion Attempt
**Condition**: User says "task done" but subtasks still in "todo" state  
**Expected**: Block completion, list pending subtasks, require manual override  
**Validation**: Premature completion prevented, clear feedback provided

### Edge Case 6: Multiple Concurrent Agent Operations
**Condition**: 3 subagents + main chat all creating files simultaneously  
**Expected**: Serialize file operations, queue automation tasks, no race conditions  
**Validation**: All files created correctly, no conflicts, automation completes for each

### Edge Case 7: Git Commit Hook Failure
**Condition**: Pre-commit hook fails (ruff/mypy errors)  
**Expected**: Abort auto-commit, display hook errors, require manual fix  
**Validation**: No commits with failing tests, clear error messages

### Edge Case 8: Ambiguous Vendor Context
**Condition**: Branch feat/vendor-multi-brand but working on LEGRAND + LUTRON  
**Expected**: Prompt for vendor selection, create separate tasks per vendor  
**Validation**: No cross-vendor contamination, clear task separation

### Edge Case 9: Session Interruption Mid-Commit
**Condition**: Network disconnect during git push  
**Expected**: Local commit preserved, retry push next session, no data loss  
**Validation**: Work not lost, git state recoverable, automation resumes

### Edge Case 10: Repository in Rebase State
**Condition**: User in middle of git rebase operation  
**Expected**: Block all automation until rebase complete, display warning  
**Validation**: No automation during rebase, clear resolution guidance

### Edge Case 11: Circular Task Dependencies
**Condition**: Task A depends on Task B, Task B depends on Task A  
**Expected**: Detect circular dependency, reject task creation, suggest fix  
**Validation**: No circular dependencies created, clear error message

### Edge Case 12: Stale Backup File
**Condition**: .task-mcp-backup.json age > 30 minutes, database offline  
**Expected**: Use backup but display "⚠️ STALE DATA" warning, suggest MCP re-enable  
**Validation**: Stale data warning shown, user aware of limitations

---

## Validation Criteria

### 1. Task-MCP Integration Correctness

**Metrics**:
- Task creation accuracy: ≥95% (correct context, metadata, vendor linking)
- Task status updates: ≥98% (in_progress → done transitions)
- Vendor entity linking: 100% (critical for vendor work)
- Task hierarchy correctness: ≥95% (parent/subtask relationships)

**Test Method**:
```bash
# Query all tasks created during test session
mcp__task-mcp__list_tasks status="done" created_by="test-session-id"

# Validate:
✅ All tasks have correct metadata
✅ File references match actual files
✅ Vendor entities linked correctly
✅ Task completion timestamps accurate
```

**Pass Criteria**: ≥95% accuracy across 20+ task operations

---

### 2. Micro-Commit Frequency & Quality

**Metrics**:
- Commit interval: ≤30 minutes (80%+ of commits)
- Milestone detection accuracy: ≥90% (true positives)
- False positive rate: ≤5% (inappropriate commits)
- Commit message quality: 100% conventional commit format
- checkall pass rate: 100% (no commits with failing tests)

**Test Method**:
```bash
# Analyze commit history
git log --since="1 hour ago" --pretty=format:"%h %s %ar"

# Validate:
✅ Commits spaced ≤30 min apart
✅ Commit messages follow "type(scope): message" format
✅ Each commit passes checkall (ruff + mypy)
✅ No premature commits (incomplete work)
```

**Pass Criteria**: ≥80% commits at ≤30 min intervals, 100% conventional format

---

### 3. File Placement Compliance

**Metrics**:
- Correct placement rate: 100% (constitutional requirement)
- Ambiguity detection: ≥95% (catches unclear cases)
- Policy violation prevention: 100% (blocks invalid locations)

**Test Method**:
```bash
# Validate all files created during session
find . -type f -newer /tmp/session-start-marker

# Check against FILE_ORGANIZATION_POLICY.md:
✅ No files in root (except CLAUDE.md, README.md, .project_status.md)
✅ Documentation in docs/ subdirectories
✅ Vendor files in docs/vendors/{VENDOR}/
✅ Test files in backend/tests/ or docs/vendors/{VENDOR}/tests/
✅ No debug/scratch files committed
```

**Pass Criteria**: 100% compliance (zero violations)

---

### 4. Graceful Degradation Behavior

**Metrics**:
- Error recovery rate: ≥95% (automation continues after errors)
- Fallback activation: 100% (when MCP offline)
- User notification clarity: ≥90% (clear error messages)
- Session continuity: 100% (no blocking errors)

**Test Method**:
```bash
# Simulate failures:
1. Disable task-mcp → Verify fallback to backup file
2. Create merge conflict → Verify auto-commit aborted safely
3. Trigger checkall failure → Verify commit blocked
4. Detached HEAD state → Verify automation paused

# Validate:
✅ No crashes or exceptions
✅ Clear error messages displayed
✅ Fallback mechanisms activate
✅ User guidance provided
✅ Session continues productively
```

**Pass Criteria**: 100% error handling, ≥95% recovery rate

---

### 5. Token Budget Validation

**Metrics**:
- Overhead vs manual workflow: ≤10% additional tokens
- Automation efficiency: ≥60% cognitive overhead reduction
- Context loading: ≤2,000 tokens for skill activation
- Per-operation cost: ≤200 tokens/automation event

**Test Method**:
```bash
# Manual workflow baseline (no automation):
- Session start: 0 tokens (no skill loaded)
- Task tracking: ~150 tokens/reminder (manual)
- Commit reminders: ~100 tokens/reminder (manual)
- File validation: ~75 tokens/check (manual)
- Total manual session: ~3,000 tokens (10 reminders)

# Automated workflow (skill active):
- Session start: 1,500 tokens (skill loading)
- Automation overhead: ~50 tokens/operation (automated)
- Total automated session: ~2,000 tokens (30 operations)

# Calculate efficiency:
Efficiency = (Manual - Automated) / Manual
           = (3,000 - 2,000) / 3,000
           = 33% token reduction ✅
```

**Pass Criteria**: ≤10% overhead, ≥30% token reduction vs manual

---

## Rollout Plan

### Phase 1: Low-Risk Testing (Documentation Work)

**Duration**: 2-3 sessions (~6 hours)  
**Risk Level**: Minimal  
**Scope**: Documentation updates, guide creation, archival tasks

**Success Criteria**:
- ✅ 10+ file operations with 100% correct placement
- ✅ 5+ task-mcp operations with ≥95% accuracy
- ✅ Zero constitutional violations
- ✅ Graceful degradation tested (MCP offline scenario)
- ✅ Token overhead ≤10%

**Test Cases**: Scenarios 1.1, 1.2, 1.3 + Edge Cases 1, 4  
**Go/No-Go**: All success criteria met → Proceed to Phase 2

**Rollback Procedure**:
1. Disable skill via `/workflow-automation disable`
2. Revert to manual workflow (existing slash commands)
3. Document issues in TESTING_RESULTS.md
4. Fix issues before retrying Phase 1

---

### Phase 2: Medium Complexity (Helper Functions & Refactors)

**Duration**: 3-5 sessions (~12 hours)  
**Risk Level**: Low-Medium  
**Scope**: Helper function additions, single-file refactors, multi-file coordinated changes

**Success Criteria**:
- ✅ 8-12 micro-commits with ≤30 min intervals (80%+)
- ✅ 100% conventional commit format
- ✅ 15+ task-mcp operations with ≥95% accuracy
- ✅ Vendor entity linking works correctly
- ✅ Hierarchical task structure validated
- ✅ Zero checkall failures (100% quality gate pass rate)

**Test Cases**: Scenarios 2.1, 2.2, 2.3 + Edge Cases 2, 3, 7  
**Go/No-Go**: All success criteria met → Proceed to Phase 3

**Rollback Procedure**:
1. Review commit history for quality issues
2. If ≥2 false positive commits: Disable micro-commit automation
3. If task-mcp accuracy <90%: Disable task automation
4. Document issues, fix, retry Phase 2

---

### Phase 3: Production Work (Vendor Implementations)

**Duration**: 5-7 sessions (~20 hours)  
**Risk Level**: Medium-High  
**Scope**: Full vendor implementations, multi-file production changes

**Success Criteria**:
- ✅ Complete vendor implementation with 15-20 micro-commits
- ✅ Vendor entity created + metadata accurate
- ✅ Task hierarchy (parent + 7 subtasks) tracked correctly
- ✅ 100% file organization compliance
- ✅ CHANGELOG.md updates tracked
- ✅ Reconciliation validation enforced
- ✅ Golden test fixtures created and validated
- ✅ Zero production incidents

**Test Cases**: Scenarios 3.1, 3.2 + All Edge Cases  
**Production Readiness**: All success criteria met → Full deployment

**Rollback Procedure**:
1. If critical failure: Immediately disable skill
2. Review all commits from session
3. Revert any problematic commits
4. Manual completion of work if needed
5. Post-mortem analysis → Fix → Retry Phase 3

---

## Metrics to Track

### 1. Token Overhead vs Manual Reminders

**Baseline (Manual Workflow)**:
- Task tracking reminders: ~150 tokens × 5 per session = 750 tokens
- Micro-commit reminders: ~100 tokens × 8 per session = 800 tokens
- File organization checks: ~75 tokens × 4 per session = 300 tokens
- **Total manual overhead**: ~1,850 tokens/session

**Automated Workflow**:
- Skill loading: 1,500 tokens (one-time per session)
- Automation operations: ~50 tokens × 20 operations = 1,000 tokens
- **Total automated overhead**: ~2,500 tokens/session

**Comparison**:
- Token increase: +650 tokens (35% increase)
- Cognitive overhead reduction: -60% (fewer manual reminders)
- **Trade-off**: Acceptable (more tokens, less mental overhead)

**Target**: ≤10% net overhead after accounting for reduced manual reminders

---

### 2. Commit Quality & Frequency

**Track Per Session**:
- Total commits: Count
- Commits at ≤30 min intervals: Count (target ≥80%)
- Conventional commit format adherence: % (target 100%)
- checkall pass rate: % (target 100%)
- False positive commits: Count (target ≤1 per session)

**Analysis**:
```bash
# Generate commit report
git log --since="session-start" --pretty=format:"%h|%s|%ar" > commit-report.txt

# Calculate metrics:
- Total commits: wc -l commit-report.txt
- Interval analysis: time between commits
- Format validation: grep -E "^[a-f0-9]+ (feat|fix|docs|test|refactor|chore)"
```

**Target**: ≥80% commits at ≤30 min, 100% format compliance

---

### 3. File Organization Compliance Rate

**Track Per Session**:
- Files created: Count
- Correct placement: Count (target 100%)
- Ambiguity detections: Count (measure accuracy)
- Policy violations: Count (target 0)

**Analysis**:
```bash
# List all files created during session
git diff --name-status $(git merge-base main HEAD) HEAD

# Validate each file against FILE_ORGANIZATION_POLICY.md:
- Root files: Should be 0 (except allowed)
- docs/ subdirectories: Correct categorization
- Vendor files: Correct vendor folder
- Test files: Correct test location
```

**Target**: 100% compliance (zero violations)

---

### 4. Error Recovery Success Rate

**Track Per Session**:
- Errors encountered: Count
- Successful recoveries: Count (target ≥95%)
- Blocking errors: Count (target 0)
- Graceful degradations: Count

**Analysis**:
```bash
# Review automation logs
grep "ERROR\|WARNING\|FALLBACK" workflow-automation.log

# Categorize:
- Recoverable errors: Automation continued
- Blocking errors: Manual intervention required
- Fallback activations: MCP offline, etc.
```

**Target**: ≥95% error recovery, 0 blocking errors

---

### 5. User Satisfaction

**Survey After Each Phase**:
1. "Automation accuracy felt accurate": 1-5 (target ≥4)
2. "Cognitive overhead reduced": 1-5 (target ≥4)
3. "Would use in production": Yes/No (target Yes)
4. "Automation felt intrusive": 1-5 (target ≤2)
5. "Error messages were clear": 1-5 (target ≥4)

**Open Feedback**:
- What worked well?
- What felt wrong or unexpected?
- What features would you add/remove?

**Target**: ≥4.0 average satisfaction, 100% production willingness

---

## Success Criteria Summary

**Phase 1 (Documentation)**:
- ✅ 100% file placement accuracy
- ✅ ≥95% task-mcp accuracy
- ✅ Zero constitutional violations
- ✅ Graceful degradation validated

**Phase 2 (Refactors)**:
- ✅ ≥80% commits at ≤30 min intervals
- ✅ 100% conventional commit format
- ✅ ≥95% task-mcp accuracy
- ✅ 100% checkall pass rate

**Phase 3 (Production)**:
- ✅ Complete vendor implementation successful
- ✅ 100% file organization compliance
- ✅ Zero production incidents
- ✅ ≥4.0 user satisfaction

**Overall Production Readiness**:
- ✅ All phase criteria met
- ✅ ≥95% automation accuracy across 30+ operations
- ✅ ≤10% token overhead (net)
- ✅ Zero critical failures
- ✅ User satisfaction ≥4.0

---

## Testing Results Log

**Location**: `.claude/skills/workflow-automation/TESTING_RESULTS.md`

**Format** (per session):
```markdown
## Session: YYYY-MM-DD-HHMM - [Phase] [Description]

**Phase**: 1 / 2 / 3
**Duration**: X hours
**Test Cases**: [List]

### Metrics:
- Task-mcp operations: X (Y% accurate)
- Micro-commits: X (Y% at ≤30 min)
- File operations: X (Y% correct)
- Errors encountered: X (Y% recovered)
- Token overhead: X tokens (Y% vs manual)

### Pass/Fail:
- ✅ / ❌ Task-mcp integration
- ✅ / ❌ Micro-commit frequency
- ✅ / ❌ File organization compliance
- ✅ / ❌ Error recovery
- ✅ / ❌ Token budget

### Issues Found:
1. [Description] → [Resolution]
2. [Description] → [Resolution]

### User Feedback:
[Open-ended feedback]

### Go/No-Go Decision:
✅ Proceed to Phase X / ❌ Fix issues before retry
```

---

## Implementation Notes

**Skill Location**: `.claude/skills/workflow-automation/SKILL.md`

**Core Components**:
1. **Task-MCP Integration**: Automatic task creation, status updates, vendor linking
2. **Micro-Commit Automation**: Milestone detection, auto-commit triggers
3. **File Organization Validation**: Policy compliance checks, placement guidance
4. **Graceful Degradation**: Fallback mechanisms, error recovery
5. **User Overrides**: Disable automation, manual control options

**Dependencies**:
- task-mcp MCP server (primary tracking)
- Git repository (commit automation)
- FILE_ORGANIZATION_POLICY.md (validation rules)
- checkall helper (quality gates)
- .task-mcp-backup.json (fallback data)

**Integration with Existing Skills**:
- commission-workflow-orchestrator: Complementary session initialization
- debug-loop-enforcer: Coordinates with testing workflow
- pattern-library-guide: No conflicts, independent operation

---

## Next Steps

1. **Create SKILL.md**: Implement core automation logic (~3,000 tokens)
2. **Create TESTING_RESULTS.md**: Log template for session tracking
3. **Phase 1 Testing**: Run 2-3 documentation sessions
4. **Phase 1 Review**: Analyze metrics, fix issues
5. **Phase 2 Testing**: Run 3-5 refactor sessions
6. **Phase 2 Review**: Validate micro-commit automation
7. **Phase 3 Testing**: Run 5-7 production sessions
8. **Production Readiness Review**: Final go/no-go decision
9. **Documentation**: Update skill README, usage guide
10. **Deployment**: Enable for all sessions

**Estimated Timeline**: 3-4 weeks (including testing phases)

---

**Document Status**: ✅ READY FOR REVIEW
**Next Action**: Begin Phase 1 testing with low-risk documentation work
