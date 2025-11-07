# Micro-Commit Discipline Guide

**Purpose**: Prevent work loss through frequent, atomic commits during development.

**Problem**: Large commit gaps (45-60 minutes) can result in significant work loss during system crashes or session disconnects.

**Solution**: Commit every 20-50 lines or at logical milestones, targeting ≤30 minute average intervals.

---

## Commit Frequency Strategy

### Research Phase (Auto-commit)

Research artifacts are automatically committed via `/cc-checkpoint`:

```bash
/cc-checkpoint {milestone-name}  # Auto-commits research outputs
```

**Examples**:
- `/cc-checkpoint vendor-analysis-complete`
- `/cc-checkpoint integration-research-done`

### Implementation Phase (Manual)

Commit every 20-50 lines OR at logical milestones:

```bash
# Helper functions (20-50 lines)
git add backend/src/extractors/{vendor}/*.py
cc feat {vendor} "Add DataFrame conversion helpers"

# Format path implementation (one commit per path)
git add backend/src/extractors/{vendor}/*.py
cc feat {vendor} "Add file writing to {format} path"

# Test development (commit per suite)
git add backend/tests/acceptance/test_{vendor}*.py
cc test {vendor} "Add {format} acceptance tests"

# Golden file generation (commit per pattern)
git add backend/tests/acceptance/fixtures/vendors/{VENDOR}/golden/
cc test {vendor} "Add golden files for {pattern}"
```

---

## Commit Frequency Target

**Goal**: ≤30 minutes average between commits

**Monitoring**:
```bash
detect_commit_milestones    # Check for logical commit points
validate_commit_frequency   # Warns if >30 min since last commit
```

---

## What to Commit

✅ **Commit immediately**:
- Every 20-50 lines of implementation code
- After completing helper functions
- After each format path implementation
- After each test suite creation
- After golden file verification
- Research artifacts (auto-committed via `/cc-checkpoint`)

❌ **Wait for logical grouping**:
- Single-line comment changes
- Formatting-only changes (batch with next feature)
- Work-in-progress debugging code

---

## Milestone Detection Examples

### Backend Development

**Helper Functions** (commit after each):
```python
def parse_header_row(row: pd.Series) -> dict:
    """Extract header metadata."""
    # 15-30 lines
    return metadata

# COMMIT HERE: "feat(vendor): Add header parsing helper"

def parse_line_items(df: pd.DataFrame) -> list[dict]:
    """Extract line items from DataFrame."""
    # 30-50 lines
    return items

# COMMIT HERE: "feat(vendor): Add line item parsing helper"
```

**Format Path Implementation** (commit per path):
```python
# CSV path complete (40-60 lines)
# COMMIT HERE: "feat(vendor): Add CSV output path"

# JSON path complete (40-60 lines)
# COMMIT HERE: "feat(vendor): Add JSON output path"
```

### Test Development

**Test Suite** (commit after each format):
```python
def test_csv_output_basic(vendor_fixture):
    """Test basic CSV output."""
    # 20-40 lines
    assert result.success

# COMMIT HERE: "test(vendor): Add CSV acceptance tests"

def test_json_output_basic(vendor_fixture):
    """Test basic JSON output."""
    # 20-40 lines
    assert result.success

# COMMIT HERE: "test(vendor): Add JSON acceptance tests"
```

---

## Work Loss Prevention

### Real-World Example: EPSON Incident

**Without Micro-Commits**:
- Lost 692 lines (~1 hour of work)
- 45-60 minute commit gaps
- Complete loss of all in-progress work

**With Micro-Commits** (≤30 min intervals):
- Maximum loss: ≤200 lines (~15 min work)
- 80-90% risk reduction
- Most recent work preserved

### Risk Reduction Calculation

```
Without micro-commits:
- Average loss per incident: 500-700 lines (45-60 min)
- Recovery time: 1-2 hours

With micro-commits (≤30 min):
- Average loss per incident: 100-200 lines (10-15 min)
- Recovery time: 15-30 minutes

Risk reduction: 80-90%
```

---

## Practical Workflow

### Start Session
```bash
git checkout main && git pull --ff-only
newfeat_here
checkall
```

### Development Loop
```bash
# 1. Write 20-50 lines or complete milestone
# 2. Run quality gates
checkall

# 3. Commit if successful
git add {files}
cc {type} {scope} "{message}"

# 4. Repeat
```

### Quality Gate Integration
```bash
# NEVER commit without passing quality gates
checkall  # Must pass before ANY commit

# If tests fail:
# 1. Fix issues
# 2. Run checkall again
# 3. Then commit
```

---

## Integration with Git Discipline

### Conventional Commits
```bash
cc feat vendor "message"       # New feature
cc fix vendor "message"        # Bug fix
cc test vendor "message"       # Test addition
cc refactor vendor "message"   # Code refactor
cc docs vendor "message"       # Documentation
```

### Clean History (Optional)
```bash
# Make fixup commits during development
fx  # Create fixup commit

# Squash before PR
rbi  # Interactive rebase with autosquash
git push --force-with-lease
```

---

## Constitutional Alignment

**Principle I (TDD)**: Micro-commits enforce test-driven development by requiring frequent quality gate passes.

**Principle III (Git Discipline)**: Small, atomic commits create clean history and reduce merge conflicts.

**Principle XI (Agentic Decomposition)**: `/cc-checkpoint` integration enables research-first workflow with automatic work preservation.

---

## Summary

**Key Principles**:
1. Commit every 20-50 lines OR at logical milestones
2. Target ≤30 minutes between commits
3. ALWAYS run `checkall` before committing
4. Use `/cc-checkpoint` for research phases
5. Preserve 80-90% of work during system failures

**Benefits**:
- 80-90% reduction in work loss risk
- Cleaner git history with atomic commits
- Easier debugging (bisect-friendly)
- Faster recovery from failures
- Better code review experience

**Reference**: EPSON incident analysis (lost 692 lines, 45-60 min gaps)
