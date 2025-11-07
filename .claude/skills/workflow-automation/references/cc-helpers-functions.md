# cc-helpers.sh Function Documentation

This file documents the key helper functions from `.claude/lib/cc-helpers.sh` used by the workflow-automation skill.

---

## validate_file_organization()

**Purpose**: Validate root directory file count against constitutional limits.

**Usage**:
```bash
validate_file_organization
```

**Parameters**: None

**Returns**:
- `0` - Compliant (≤10 files in root)
- `1` - Needs cleanup (>10 files in root)

**Behavior**:
- Counts files in root directory (excluding `.`, `backend`, `frontend`, `docs`, `scripts`, `outputs`)
- Compares against max_files=10 threshold
- Provides graceful degradation (returns 0 if ls fails or no write permissions)

**Output**:
```
✅ File Organization: COMPLIANT (8/10 files)
```
or
```
⚠️ File Organization: NEEDS CLEANUP (13/10 files, 3 over limit)
```

**Constitutional Reference**: Principle III (File Organization)

---

## validate_commit_frequency()

**Purpose**: Check time since last commit and warn if >30 minutes.

**Usage**:
```bash
validate_commit_frequency
```

**Parameters**: None

**Returns**:
- `0` - Recent commit (<30 minutes)
- `1` - Warning (>30 minutes since last commit)

**Behavior**:
- Gets timestamp of last commit via `git log -1 --format=%ct`
- Compares against current time
- Warns if >30 minutes elapsed (micro-commit discipline)
- Gracefully skips if not a git repo or no commits exist

**Output**:
```
✅ Last commit: 15 minutes ago (within 30 min threshold)
```
or
```
⚠️ Last commit was 45 minutes ago (>30 min threshold)
   Consider committing recent work to prevent loss
```

**Constitutional Reference**: Principle I (Git Discipline), Micro-Commit Discipline Guide

---

## detect_commit_milestones()

**Purpose**: Detect logical commit milestones based on file patterns.

**Usage**:
```bash
detect_commit_milestones
```

**Parameters**: None

**Returns**:
- `0` - No milestone detected (or no uncommitted changes)
- `1` - Milestone detected (commit recommended)

**Behavior**:
Detects milestones based on uncommitted file patterns:
1. **Research artifacts** - `docs/subagent-reports/*.{md,json}`
2. **Helper functions** - `.claude/lib/*.sh`, `backend/src/extractors/base.py`
3. **Test files** - `backend/tests/*.py`, `.claude/tests/*.bats`
4. **Extractor implementations** - `backend/src/extractors/*/extractor.py`
5. **Documentation** - `docs/session-prompts/*.md`, `CLAUDE.md`

**Output**:
```
📍 Commit milestone detected: research artifact
   Uncommitted files: 3
   Suggestion: Commit now to prevent work loss
```

**Use Case**: Automated reminder system in `/cc-comply` and `/cc-checkpoint`

---

## auto_commit_research_artifact()

**Purpose**: Auto-commit research artifacts with comprehensive validation.

**Usage**:
```bash
auto_commit_research_artifact "path/to/artifact.md" "research|planning-review|architecture-review|revised-plan"
```

**Parameters**:
- `$1` - Artifact path (required, must be in `docs/subagent-reports/`)
- `$2` - Artifact type (optional, default: "research")
  - `research` or `initial-research`
  - `planning-review`
  - `architecture-review`
  - `revised-plan`

**Returns**:
- `0` - Success (artifact committed)
- `1` - Failure (validation failed or commit failed)

**Behavior**:

### 11-Point Validation Checklist:
1. **Git repository check** - Verify `.git` directory exists
2. **File existence** - Verify file exists at path
3. **Path validation** - Must be in `docs/subagent-reports/`
4. **Not .gitignore'd** - Verify file is trackable
5. **Clean working directory** - Warn if other changes exist (still commits artifact)
6. **Not detached HEAD** - Must be on a named branch
7. **No merge conflicts** - Git index must be clean
8. **Disk space check** - At least 100MB free
9. **File size check** - File must be <10MB
10. **Valid branch** - Cannot be main/master
11. **Git operations not locked** - No `.git/index.lock`

### Commit Message Template:
```
docs(scope): add {artifact_type}

Auto-committed research artifact from three-step research process.

File: artifact-name.md
Size: 45.2 KiB
Phase: research

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Scope Extraction**: Automatically extracted from path
- Path: `docs/subagent-reports/framework-research/git-discipline/research.md`
- Scope: `git-discipline`

**Output (Success)**:
```
✅ Auto-committed research artifact: docs/subagent-reports/.../research.md
   Commit message: docs(git-discipline): add initial research
   Branch: feat/vendor-coastal-source
```

**Output (Failure)**:
```
❌ Auto-commit failed: Not a research artifact (must be in docs/subagent-reports/)
```

**Error Handling**:
- All validations fail gracefully with clear error messages
- Rollback staging if commit fails (`git reset HEAD`)
- Returns non-zero exit code for script integration

**Use Case**: Called by `/cc-checkpoint` to auto-commit research outputs

---

## Integration Examples

### Example 1: Validate Before Commit
```bash
# Check file organization and commit frequency
validate_file_organization
validate_commit_frequency

# Detect if commit is recommended
if detect_commit_milestones; then
    echo "No milestone detected, continue working"
else
    echo "Milestone detected, committing..."
fi
```

### Example 2: Auto-Commit Research Artifact
```bash
# After generating research report
research_file="docs/subagent-reports/framework-research/type-system/research.md"

if auto_commit_research_artifact "$research_file" "research"; then
    echo "Research committed successfully"
else
    echo "Auto-commit failed, manual commit required"
fi
```

### Example 3: Full Validation Pipeline (cc-comply)
```bash
# Run all validation checks
validate_file_organization
validate_commit_frequency
detect_commit_milestones

# If milestone detected, auto-commit research artifacts
if [ $? -eq 1 ]; then
    for artifact in $(git status --porcelain | grep "docs/subagent-reports" | awk '{print $2}'); do
        auto_commit_research_artifact "$artifact" "research"
    done
fi
```

---

## Error Recovery Patterns

All functions implement graceful degradation:

1. **Missing dependencies** - Skip validation if git/ls/df not available
2. **Permission errors** - Return success (0) if cannot validate
3. **Invalid state** - Return success (0) to avoid blocking workflow
4. **Explicit failures** - Return error (1) only for actionable violations

**Philosophy**: Never block workflow due to validation failures. Warn user but allow work to continue.

---

## Token Budget

- **Total**: ~800 tokens
- **Function signatures**: ~150 tokens
- **Behavior documentation**: ~400 tokens
- **Examples**: ~200 tokens
- **Error recovery**: ~50 tokens

---

## References

- Source: `.claude/lib/cc-helpers.sh`
- Lines: 75-98 (validate_file_organization), 556-583 (validate_commit_frequency), 585-642 (detect_commit_milestones), 646-782 (auto_commit_research_artifact)
- Constitutional Principles: I (Git Discipline), III (File Organization)
- Related: `MICRO_COMMIT_DISCIPLINE.md`, `FILE_ORGANIZATION_POLICY.md`
