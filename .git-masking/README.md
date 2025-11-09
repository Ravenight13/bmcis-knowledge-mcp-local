# Git Branch Masking System - Internal Documentation

## Overview

This directory contains the implementation of a **branch masking system** that prevents exposure of internal task identifiers and descriptions in public git history. All public-facing git operations use opaque session identifiers instead of revealing internal project structure.

## Security Model

### Problem

Traditional branch naming exposes internal information:
- `feat/task-5.1-implement-rrf-algorithm` reveals task IDs and implementation details
- Commit messages like `feat: task 5.1 - implement RRF` expose project structure
- PR titles show internal task organization

### Solution

Use opaque session identifiers for all public git operations:
- Branch names: `work/session-001`, `work/session-002`, etc.
- Commit messages: `feat: session-001 - implement ranking algorithm`
- PR titles: `feat: session-001 - ranking system implementation`

The mapping between session IDs and real task information is maintained in `branch-mapping.json`, which is **never committed** to version control.

## Directory Structure

```
.git-masking/
├── .gitignore                      # Prevents branch-mapping.json from being committed
├── branch-mapping.json             # SECURITY: Never committed! Local mapping only
├── branch-mapping.template.json    # Template for new projects (committed)
├── new-branch.sh                   # Helper: Create new masked branches
├── check-branch.sh                 # Helper: Look up current branch info
├── README.md                       # This file (internal documentation)
└── template-commit-messages.txt    # Examples of masked vs exposed messages
```

## Branch Naming Convention

### Format

```
work/session-{NNN}
```

Where `{NNN}` is a zero-padded 3-digit session identifier:
- `work/session-001`
- `work/session-002`
- `work/session-042`
- `work/session-137`

### Why "session" instead of "task"?

The term "session" is deliberately generic and reveals nothing about:
- Whether this is a feature, bugfix, or refactor
- What component is being worked on
- What the implementation involves
- The project's internal task structure

## Mapping File Structure

### branch-mapping.json

This file is the **security-critical component** and must never be committed.

```json
{
  "version": "1.0.0",
  "description": "Maps opaque session IDs to internal task information for security",
  "nextSessionId": 3,
  "branches": {
    "work/session-001": {
      "sessionId": "001",
      "taskId": "task-5.1",
      "description": "Implement RRF ranking algorithm",
      "created": "2025-11-08",
      "status": "active"
    },
    "work/session-002": {
      "sessionId": "002",
      "taskId": "task-5.2",
      "description": "Add performance benchmarks",
      "created": "2025-11-08",
      "status": "completed"
    }
  },
  "metadata": {
    "created": "2025-11-08",
    "lastModified": "2025-11-08",
    "purpose": "Prevent exposure of internal task details in public git history"
  }
}
```

### Fields

- `version`: Mapping file format version (semver)
- `nextSessionId`: Next available session number (auto-incremented)
- `branches`: Object mapping branch names to task information
- `branches[].sessionId`: Zero-padded session identifier
- `branches[].taskId`: Internal task identifier (NEVER exposed publicly)
- `branches[].description`: Full task description (NEVER exposed publicly)
- `branches[].created`: Branch creation date
- `branches[].status`: Branch status (active, completed, abandoned)

## Helper Scripts

### new-branch.sh

Creates a new masked branch with automatic session ID assignment.

**Usage:**

```bash
# Interactive mode (prompts for task ID and description)
./new-branch.sh

# Non-interactive mode
./new-branch.sh --task-id task-5.3 --description "Implement caching layer"

# Show help
./new-branch.sh --help
```

**What it does:**

1. Reads `nextSessionId` from `branch-mapping.json`
2. Formats as `work/session-{NNN}` (3 digits with leading zeros)
3. Prompts for Task ID and Description (if not provided)
4. Creates the git branch
5. Updates `branch-mapping.json` with new entry
6. Increments `nextSessionId`
7. Displays next steps and usage examples

**Requirements:**

- `jq` command-line JSON processor
- Must be run from within a git repository

### check-branch.sh

Looks up the current branch in the mapping and displays task information.

**Usage:**

```bash
# Check current branch
./check-branch.sh

# Check specific branch
./check-branch.sh work/session-042

# Use in scripts
TASK_ID=$(./check-branch.sh | grep "Task ID:" | awk '{print $3}')
```

**What it displays:**

- Session ID
- Task ID (internal, never expose publicly!)
- Task description (internal, never expose publicly!)
- Branch creation date
- Branch status
- Example commit messages
- Example PR title
- Security reminders

### template-commit-messages.txt

Examples showing the difference between exposed and masked commit messages.

**Usage:**

```bash
# View examples
cat template-commit-messages.txt

# Use as reference when committing
git commit -m "$(grep 'feat:' template-commit-messages.txt | head -1)"
```

## Workflow Examples

### Creating a New Masked Branch

```bash
# Navigate to .git-masking directory
cd .git-masking

# Create new branch interactively
./new-branch.sh

# Or provide details directly
./new-branch.sh --task-id task-6.1 --description "Add authentication system"

# The script will:
# 1. Assign session ID (e.g., 003)
# 2. Create branch: work/session-003
# 3. Update branch-mapping.json
# 4. Check out the new branch
```

### Working on a Masked Branch

```bash
# Check what you're working on
cd .git-masking
./check-branch.sh

# Output shows:
# Session ID:    003
# Task ID:       task-6.1
# Description:   Add authentication system
# Created:       2025-11-08
# Status:        active

# Make changes
cd ..
# ... implement feature ...

# Commit with MASKED message
git commit -m "feat: session-003 - implement JWT authentication"
```

### Creating a Pull Request

```bash
# WRONG (exposes internal task ID):
gh pr create --title "feat: task-6.1 - add authentication system"

# CORRECT (uses masked session ID):
gh pr create --title "feat: session-003 - add authentication system"
```

### Checking Multiple Branches

```bash
# List all masked branches
git branch | grep "work/session"

# Check each one
for branch in $(git branch | grep "work/session" | tr -d ' *'); do
    echo "=== $branch ==="
    ./check-branch.sh "$branch"
    echo
done
```

## Security Best Practices

### DO

- ✅ Always use session IDs in commit messages
- ✅ Always use session IDs in PR titles
- ✅ Keep `branch-mapping.json` local only
- ✅ Use `check-branch.sh` to look up task details
- ✅ Create branches with `new-branch.sh`
- ✅ Review commits before pushing to ensure no task IDs leaked

### DON'T

- ❌ Never reference task IDs in commit messages
- ❌ Never reference task IDs in PR titles
- ❌ Never commit `branch-mapping.json`
- ❌ Never paste task descriptions in PR bodies
- ❌ Never create branches manually with task IDs
- ❌ Never share screenshots showing internal task IDs

### Verification Checklist

Before pushing commits or creating PRs:

```bash
# Check recent commits for exposed task IDs
git log --oneline -10 | grep -E "task-[0-9]+"

# If any matches found, rewrite commit messages
git rebase -i HEAD~10  # Edit the problematic commits

# Check branch name
git rev-parse --abbrev-ref HEAD | grep "work/session"

# Verify you're on a masked branch
./check-branch.sh
```

## Installation & Setup

### Prerequisites

```bash
# Install jq (required for JSON manipulation)
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# CentOS/RHEL
sudo yum install jq
```

### Initialize for New Project

```bash
# Create directory structure
mkdir -p .git-masking

# Copy template
cp .git-masking/branch-mapping.template.json .git-masking/branch-mapping.json

# Make scripts executable
chmod +x .git-masking/*.sh

# Verify .gitignore is configured
cat .git-masking/.gitignore
# Should show: branch-mapping.json

# Create first masked branch
cd .git-masking
./new-branch.sh
```

## Troubleshooting

### Problem: "jq: command not found"

**Solution:** Install jq using your package manager (see Prerequisites).

### Problem: Branch mapping file not found

**Solution:**

```bash
# Check if file exists
ls -la .git-masking/branch-mapping.json

# If missing, copy from template
cp .git-masking/branch-mapping.template.json .git-masking/branch-mapping.json
```

### Problem: Script shows "Permission denied"

**Solution:**

```bash
# Make scripts executable
chmod +x .git-masking/*.sh

# Verify permissions
ls -l .git-masking/*.sh
```

### Problem: Invalid JSON after update

**Solution:**

```bash
# Validate JSON
jq empty .git-masking/branch-mapping.json

# If invalid, restore from backup
cp .git-masking/branch-mapping.json.bak .git-masking/branch-mapping.json

# Or manually fix with text editor
```

### Problem: Session IDs out of sync

**Solution:**

```bash
# Check current nextSessionId
jq '.nextSessionId' .git-masking/branch-mapping.json

# Check highest existing session
jq -r '.branches | keys[]' .git-masking/branch-mapping.json | \
  grep -oE '[0-9]+' | sort -n | tail -1

# Manually set nextSessionId to highest + 1
jq '.nextSessionId = 10' .git-masking/branch-mapping.json > temp.json
mv temp.json .git-masking/branch-mapping.json
```

## Maintenance

### Cleaning Up Old Branches

```bash
# List all branches in mapping
jq -r '.branches | keys[]' .git-masking/branch-mapping.json

# Remove merged branches from mapping
# (Manual edit or create cleanup script)

# Update branch status to "completed"
jq '.branches["work/session-001"].status = "completed"' \
  .git-masking/branch-mapping.json > temp.json
mv temp.json .git-masking/branch-mapping.json
```

### Backup Mapping File

```bash
# Create backup before major changes
cp .git-masking/branch-mapping.json \
   .git-masking/branch-mapping.json.bak.$(date +%Y%m%d)

# Restore from backup
cp .git-masking/branch-mapping.json.bak.20251108 \
   .git-masking/branch-mapping.json
```

## Architecture Decisions

### Why JSON instead of a database?

- ✅ Simple to use and edit
- ✅ Works without additional dependencies
- ✅ Easy to backup and restore
- ✅ Human-readable for debugging
- ✅ Works well with jq command-line tool

### Why shell scripts instead of a CLI tool?

- ✅ No compilation or installation required
- ✅ Works on any Unix-like system
- ✅ Easy to customize per project
- ✅ Transparent and auditable
- ✅ Minimal dependencies (just jq and bash)

### Why .git-masking directory?

- ✅ Clear separation from application code
- ✅ Easy to find and maintain
- ✅ Conventional dot-prefix for tooling
- ✅ Can be .gitignore'd easily for the mapping file

## Future Enhancements

Potential improvements for future versions:

1. **Automated cleanup**: Script to remove completed branches from mapping
2. **Validation**: Pre-commit hook to verify no task IDs in commit messages
3. **Search**: Script to search mapping by task ID or description
4. **Export**: Generate report of all sessions and their status
5. **Integration**: Claude Code slash commands for branch operations
6. **Backup**: Automated backup of mapping file before changes

## References

- [Git Branch Naming Conventions](https://www.conventionalcommits.org/)
- [jq Manual](https://stedolan.github.io/jq/manual/)
- [Bash Scripting Guide](https://www.gnu.org/software/bash/manual/)

## Support

For questions or issues with the branch masking system:

1. Check this README for troubleshooting steps
2. Review the template files for examples
3. Validate JSON with `jq empty branch-mapping.json`
4. Check script permissions with `ls -l *.sh`
5. Verify git repository status with `git status`
