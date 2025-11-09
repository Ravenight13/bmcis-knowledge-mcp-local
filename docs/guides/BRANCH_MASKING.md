# Git Branch Masking System - Quick Start Guide

## What is Branch Masking?

This project uses **opaque session identifiers** instead of task IDs in all public git operations (branches, commits, PRs) to prevent exposure of internal project structure and task details.

### Before (Exposed)

```bash
# Branch name reveals task ID
git checkout feat/task-5.1-implement-rrf-algorithm

# Commit exposes internal details
git commit -m "feat: task 5.1 - implement RRF ranking algorithm"

# PR title shows project structure
gh pr create --title "feat: task 5.1 - RRF implementation"
```

### After (Masked)

```bash
# Branch name is opaque
git checkout work/session-001

# Commit reveals nothing
git commit -m "feat: session-001 - implement ranking algorithm"

# PR title is generic
gh pr create --title "feat: session-001 - ranking implementation"
```

## Quick Start

### Prerequisites

```bash
# Install jq (required for JSON manipulation)
brew install jq   # macOS
# OR
sudo apt-get install jq   # Ubuntu/Debian
```

### Create Your First Masked Branch

```bash
# Navigate to the masking directory
cd .git-masking

# Run the helper script
./new-branch.sh

# Follow the prompts:
# Enter Task ID: task-5.1
# Enter Task Description: Implement RRF ranking algorithm

# The script will:
# ✓ Create branch: work/session-001
# ✓ Update mapping file
# ✓ Check out the new branch
# ✓ Show you next steps
```

### Check What You're Working On

```bash
# View current branch info
cd .git-masking
./check-branch.sh

# Output:
# Session ID:    001
# Task ID:       task-5.1
# Description:   Implement RRF ranking algorithm
# Created:       2025-11-08
# Status:        active
```

## Daily Workflow

### 1. Create a New Branch

```bash
cd .git-masking
./new-branch.sh --task-id task-6.2 --description "Add caching layer"
```

### 2. Make Changes and Commit

```bash
# Make your changes
vim src/ranking.py

# Stage changes
git add src/ranking.py

# Commit with MASKED message
git commit -m "feat: session-001 - implement core ranking logic"
```

### 3. Create Pull Request

```bash
# CORRECT: Use session ID
gh pr create --title "feat: session-001 - ranking implementation" \
  --body "Implements advanced ranking algorithm with performance optimizations"

# WRONG: Never use task ID
# gh pr create --title "feat: task-5.1 - RRF implementation"
```

### 4. Check Progress

```bash
# View current task
cd .git-masking
./check-branch.sh

# List all masked branches
git branch | grep "work/session"
```

## Commit Message Examples

### Feature Implementation

```bash
✅ git commit -m "feat: session-001 - add ranking algorithm"
✅ git commit -m "feat(search): session-001 - implement rank fusion"
❌ git commit -m "feat: task 5.1 - add RRF algorithm"  # NEVER!
```

### Bug Fixes

```bash
✅ git commit -m "fix: session-002 - resolve query parsing bug"
✅ git commit -m "fix(parser): session-002 - handle edge cases"
❌ git commit -m "fix: task 5.2 - fix parser bug"  # NEVER!
```

### Documentation

```bash
✅ git commit -m "docs: session-003 - update API documentation"
✅ git commit -m "docs(api): session-003 - add usage examples"
❌ git commit -m "docs: task 5.3 - document RRF API"  # NEVER!
```

### Tests

```bash
✅ git commit -m "test: session-004 - add unit tests"
✅ git commit -m "test(ranking): session-004 - add coverage"
❌ git commit -m "test: task 5.4 - test RRF algorithm"  # NEVER!
```

## Security Checklist

Before pushing commits or creating PRs:

- [ ] Branch name uses `work/session-NNN` format
- [ ] No task IDs in commit messages
- [ ] No task IDs in PR title
- [ ] No internal details in PR description
- [ ] Verified with: `git log --oneline -10 | grep -E "task-[0-9]+"` (should be empty)

## Helper Scripts

### new-branch.sh

Create a new masked branch with automatic session ID assignment.

```bash
# Interactive mode
./new-branch.sh

# Non-interactive mode
./new-branch.sh --task-id task-6.1 --description "Add authentication"

# Show help
./new-branch.sh --help
```

### check-branch.sh

Look up task information for current or specified branch.

```bash
# Check current branch
./check-branch.sh

# Check specific branch
./check-branch.sh work/session-042
```

## Branch Naming Convention

### Format

```
work/session-{NNN}
```

Where `{NNN}` is a zero-padded 3-digit session identifier:
- ✅ `work/session-001` - First session
- ✅ `work/session-042` - 42nd session
- ✅ `work/session-137` - 137th session
- ❌ `feat/task-5.1-rrf` - NEVER use task IDs!

### Why "session"?

The term **session** is deliberately generic and reveals:
- ❌ Nothing about the feature being implemented
- ❌ Nothing about the component being modified
- ❌ Nothing about the project structure
- ✅ Only that this is a work session

## Common Mistakes

### ❌ Mistake 1: Task ID in Commit Message

```bash
# WRONG
git commit -m "feat: session-001 - implement task 5.1"
                                    ^^^^^^^^ Exposed!

# CORRECT
git commit -m "feat: session-001 - implement ranking algorithm"
```

### ❌ Mistake 2: Task ID in PR Body

```bash
# WRONG
gh pr create --title "feat: session-001" --body "Implements task-5.1"

# CORRECT
gh pr create --title "feat: session-001 - ranking implementation" \
  --body "Implements advanced ranking algorithm"
```

### ❌ Mistake 3: Too Specific Description

```bash
# WRONG (reveals too much)
git commit -m "feat: session-001 - implement RRF with pgvector HNSW"

# CORRECT (appropriately generic)
git commit -m "feat: session-001 - implement ranking algorithm"
```

### ❌ Mistake 4: Referencing Other Tasks

```bash
# WRONG
git commit -m "feat: session-001 - depends on task-4.4"

# CORRECT
git commit -m "feat: session-001 - integrate with search pipeline"
```

## Troubleshooting

### "jq: command not found"

Install jq:
```bash
brew install jq   # macOS
sudo apt-get install jq   # Ubuntu/Debian
```

### "Permission denied" when running scripts

Make scripts executable:
```bash
chmod +x .git-masking/*.sh
```

### "Branch mapping file not found"

Initialize from template:
```bash
cp .git-masking/branch-mapping.template.json .git-masking/branch-mapping.json
```

### "Not in a git repository"

Make sure you're in the project root:
```bash
cd /path/to/project
git status
```

## Verification Commands

```bash
# Check for exposed task IDs in recent commits
git log --oneline -20 | grep -E "task-?[0-9]+"
# (Should return nothing)

# Verify current branch is masked
git rev-parse --abbrev-ref HEAD | grep "work/session"
# (Should show: work/session-NNN)

# Check all commits for task IDs
git log --all --grep="task-[0-9]" --oneline
# (Should return nothing)
```

## Resources

- **Internal Documentation**: `.git-masking/README.md` - Detailed implementation guide
- **Commit Examples**: `.git-masking/template-commit-messages.txt` - Comprehensive examples
- **Helper Scripts**: `.git-masking/new-branch.sh`, `.git-masking/check-branch.sh`

## FAQ

### Q: Can I create branches manually?

**A:** Yes, but use the helper script to ensure proper mapping:
```bash
cd .git-masking
./new-branch.sh
```

### Q: What if I accidentally expose a task ID?

**A:** Rewrite the commit history immediately:
```bash
# For the last commit
git commit --amend -m "feat: session-001 - implement feature"

# For older commits
git rebase -i HEAD~10  # Edit the problematic commits
```

### Q: Can collaborators see the mapping file?

**A:** No! The `branch-mapping.json` file is in `.gitignore` and never committed. Each developer maintains their own local mapping.

### Q: What happens when session IDs run out?

**A:** Session IDs are 3 digits (001-999), supporting 999 sessions. If needed, the format can be extended to 4 digits (0001-9999).

### Q: Can I use session IDs in PR descriptions?

**A:** Yes! Session IDs are safe to use anywhere. Just never use task IDs or reveal internal project structure.

---

**Remember:** The goal is **opacity**. Public git history should reveal nothing about internal task organization.

For detailed documentation, see: `.git-masking/README.md`
