# Micro-Commit Workflow

**Category**: Workflow Enforcement
**Token Budget**: ~700 tokens

---

## Purpose

Prevent work loss through frequent, atomic commits during development.

**Problem**: Large commit gaps (45-60 minutes) can result in significant work loss during system crashes, session disconnects, or unexpected failures.

**Solution**: Commit every 20-50 lines or at logical milestones, targeting ≤30 minute average intervals.

---

## Commit Frequency Target

**Goal**: ≤30 minutes average between commits

**Risk Reduction**: 80-90% work loss prevention

**Time to Recovery**: 15-30 minutes (vs 1-2 hours without micro-commits)

---

## Commit Frequency Strategy

### Research/Planning Phase (Auto-commit)

Research artifacts should be automatically committed using project-specific tools:

**Examples**:
- Checkpoint commands (if available)
- Milestone markers
- Research artifact tools

**Benefits**:
- Prevents loss of research work
- Creates audit trail
- Enables session continuity

### Implementation Phase (Manual)

Commit every **20-50 lines** OR at **logical milestones**:

**Code Examples**:
```bash
# Helper functions (20-50 lines)
git add {relevant-files}
git commit -m "type(scope): Add helper function for {purpose}"

# Feature implementation (one commit per logical unit)
git add {relevant-files}
git commit -m "feat(scope): Add {feature-name} implementation"

# Test development (commit per suite)
git add {test-files}
git commit -m "test(scope): Add {component} tests"
```

**Conventional Commits**: Use `type(scope): message` format
- Types: feat, fix, docs, chore, refactor, test, ci, build, perf, revert
- Scope: Component/module/area being changed
- Message: Clear, concise description

---

## What to Commit

### ✅ Commit Immediately:

- Every 20-50 lines of implementation code
- After completing helper functions
- After each feature component implementation
- After each test suite creation
- After golden file/reference data verification
- Research artifacts (via auto-commit tools)
- Configuration changes
- Documentation updates

### ❌ Wait for Logical Grouping:

- Single-line comment changes
- Formatting-only changes (batch with next feature)
- Work-in-progress debugging code
- Temporary experimental changes

---

## Milestone Detection Examples

### Backend/Core Development

**Helper Functions** (commit after each):
```python
def process_input_data(data: dict) -> ProcessedData:
    """Transform raw input to structured format."""
    # 15-30 lines of transformation logic
    return processed_data

# COMMIT HERE: "feat(core): Add input data processing helper"
```

**Feature Implementation** (commit per component):
```python
# Component A complete (40-60 lines)
# COMMIT HERE: "feat(feature): Add Component A implementation"

# Component B complete (40-60 lines)
# COMMIT HERE: "feat(feature): Add Component B implementation"
```

### Frontend Development

**Component Creation** (commit per component):
```javascript
// UserProfile component complete (50-80 lines)
// COMMIT HERE: "feat(ui): Add UserProfile component"

// ProfileSettings component complete (50-80 lines)
// COMMIT HERE: "feat(ui): Add ProfileSettings component"
```

### Test Development

**Test Suite** (commit after each module):
```python
def test_input_validation():
    """Test input validation logic."""
    # 20-40 lines of test cases
    assert result.is_valid

# COMMIT HERE: "test(core): Add input validation tests"
```

---

## Risk Reduction Analysis

### Without Micro-Commits:
- **Average loss per incident**: 500-700 lines (45-60 min of work)
- **Recovery time**: 1-2 hours to recreate lost work
- **Frustration**: High (recreating completed work)

### With Micro-Commits (≤30 min):
- **Average loss per incident**: 100-200 lines (10-15 min of work)
- **Recovery time**: 15-30 minutes
- **Frustration**: Low (minimal recreation needed)

### Risk Reduction: 80-90%

---

## Practical Workflow

### Development Loop

```bash
# 1. Write 20-50 lines or complete milestone

# 2. Run quality gates (if configured)
{quality-check-command}  # e.g., npm test, make check, etc.

# 3. Commit if successful
git add {files}
git commit -m "{type}({scope}): {message}"

# 4. Repeat
```

### Quality Gate Integration

```bash
# ALWAYS run quality gates before commit (if available)
{quality-check-command}  # Must pass before ANY commit

# If checks fail:
# 1. Fix issues
# 2. Run checks again
# 3. Then commit
```

**Note**: If your project doesn't have quality gates, commit at logical milestones and rely on code review.

---

## Language-Agnostic Examples

### Python Projects
```bash
# Quality gates
python -m pytest tests/
python -m mypy .
python -m ruff check .

# Commit
git commit -m "feat(parser): Add JSON parsing logic"
```

### JavaScript/TypeScript Projects
```bash
# Quality gates
npm run lint
npm run type-check
npm test

# Commit
git commit -m "feat(api): Add REST client wrapper"
```

### Go Projects
```bash
# Quality gates
go test ./...
go vet ./...
golangci-lint run

# Commit
git commit -m "feat(handler): Add HTTP request handler"
```

### Rust Projects
```bash
# Quality gates
cargo test
cargo clippy

# Commit
git commit -m "feat(parser): Add config file parser"
```

---

## Benefits

**Work Loss Prevention**:
- 80-90% reduction in work loss risk
- Faster recovery from failures (15-30 min vs 1-2 hours)

**Code Quality**:
- Cleaner git history with atomic commits
- Easier debugging (git bisect-friendly)
- Better code review experience (focused diffs)

**Collaboration**:
- Frequent pushes reduce merge conflicts
- Clear progression of work
- Easier to track team progress

---

## Summary

**Key Principles**:
1. Commit every 20-50 lines OR at logical milestones
2. Target ≤30 minutes between commits
3. ALWAYS run quality gates before committing (if available)
4. Use conventional commit format
5. Preserve 80-90% of work during system failures

**Universal Applicability**: This workflow applies to any programming language, framework, or project type.
