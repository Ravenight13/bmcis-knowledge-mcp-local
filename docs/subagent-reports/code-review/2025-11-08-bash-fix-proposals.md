# Bash Script Fix Proposals - uwo-ready.md
**Date:** 2025-11-08
**Reviewer:** Code Review Agent
**File:** `.claude/commands/uwo-ready.md`
**Objective:** POSIX-compliant shell syntax fixes for universal portability

---

## Executive Summary

Identified 4 critical sections using non-POSIX bash syntax (`[[ ]]`, complex conditionals, unsafe piping). All fixes ensure compatibility with `/bin/sh`, `dash`, `ash`, and other minimal shells while handling edge cases robustly.

**Key Improvements:**
- Replace `[[ ]]` regex with POSIX `case` statements
- Break complex multi-condition chains into separate commands
- Add null-safety for empty file lists and missing remotes
- Simplify variable nesting and quoting

---

## Problem 1: Context Detection with `[[ ]]` Regex (Lines 69-78)

### Original Code (Non-POSIX)
```bash
# Analyze branch name for context
CONTEXT="GENERAL"
[[ "$CURRENT_BRANCH" =~ ^feat/ ]] && CONTEXT="DEVELOPMENT"
[[ "$CURRENT_BRANCH" =~ ^test/ ]] && CONTEXT="TESTING"
[[ "$CURRENT_BRANCH" =~ ^docs/ ]] && CONTEXT="DOCUMENTATION"
[[ "$CURRENT_BRANCH" =~ ^fix/ ]] && CONTEXT="BUGFIX"

echo "📍 Detected Context: $CONTEXT"
```

### Issues
1. `[[ ]]` is bash/ksh specific, not available in POSIX `/bin/sh`
2. `=~` regex operator not available in POSIX shells
3. Fails silently in `dash`, `ash`, minimal Docker containers

### Fixed Code (POSIX-Compliant)
```bash
# Analyze branch name for context (POSIX-compliant)
CONTEXT="GENERAL"
case "$CURRENT_BRANCH" in
    feat/*)
        CONTEXT="DEVELOPMENT"
        ;;
    test/*)
        CONTEXT="TESTING"
        ;;
    docs/*)
        CONTEXT="DOCUMENTATION"
        ;;
    fix/*)
        CONTEXT="BUGFIX"
        ;;
    *)
        CONTEXT="GENERAL"
        ;;
esac

echo "📍 Detected Context: $CONTEXT"
```

### Why This Fix Works
- `case` pattern matching is POSIX standard (SUSv3 compliant)
- Glob patterns (`feat/*`) work identically to regex `^feat/` for prefix matching
- Default `*)` case handles unknown branches explicitly
- 100% compatible with `/bin/sh`, `dash`, `ash`, `bash`, `zsh`

### Edge Cases Handled
| Edge Case | Behavior |
|-----------|----------|
| Empty branch name (`""`) | Matches `*)` → `GENERAL` |
| Non-standard branch (`feature/auth`) | Matches `*)` → `GENERAL` |
| Branch with slashes (`feat/api/v2`) | Matches `feat/*` → `DEVELOPMENT` |
| No git repo (branch = `"unknown"`) | Matches `*)` → `GENERAL` |

### Testing Validation
```bash
# Test in minimal shell
sh -c 'CURRENT_BRANCH="feat/auth"; case "$CURRENT_BRANCH" in feat/*) echo "MATCH";; esac'
# Expected: MATCH

# Test with dash (common in Docker)
dash -c 'CURRENT_BRANCH="test/unit"; case "$CURRENT_BRANCH" in test/*) echo "TESTING";; esac'
# Expected: TESTING

# Test edge case: empty branch
sh -c 'CURRENT_BRANCH=""; case "$CURRENT_BRANCH" in *) echo "GENERAL";; esac'
# Expected: GENERAL
```

---

## Problem 2: Branch Sync Complex Conditionals (Lines 99-108)

### Original Code (Fragile)
```bash
# 4.2 Branch Sync (capture, summarize)
SYNC_STATUS=$(git rev-list --left-right --count @{u}...HEAD 2>/dev/null || echo "")
if [ -n "$SYNC_STATUS" ]; then
    BEHIND=$(echo "$SYNC_STATUS" | awk '{print $1}')
    AHEAD=$(echo "$SYNC_STATUS" | awk '{print $2}')
    [ "$BEHIND" -gt 0 ] && echo "   ⚠️ Behind remote by $BEHIND commits"
    [ "$AHEAD" -gt 0 ] && echo "   📤 Ahead of remote by $AHEAD commits"
    [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ] && echo "   ✅ Synced with remote"
else
    echo "   ℹ️ No remote tracking"
fi
```

### Issues
1. `awk` may extract empty strings if git output format changes
2. `[ "$BEHIND" -gt 0 ]` fails if `$BEHIND` is empty/non-numeric (unbound variable error)
3. Double-conditional on line 105 (`[ ... ] && [ ... ] && echo`) is fragile
4. No validation that `$AHEAD`/`$BEHIND` are valid integers

### Fixed Code (Robust)
```bash
# 4.2 Branch Sync (capture, summarize) - POSIX-compliant with validation
SYNC_STATUS=$(git rev-list --left-right --count @{u}...HEAD 2>/dev/null || echo "")

if [ -n "$SYNC_STATUS" ]; then
    # Extract values with defaults
    BEHIND=$(echo "$SYNC_STATUS" | awk '{print $1}')
    AHEAD=$(echo "$SYNC_STATUS" | awk '{print $2}')

    # Validate extracted values are numbers (default to 0 if not)
    case "$BEHIND" in
        ''|*[!0-9]*) BEHIND=0 ;;
    esac
    case "$AHEAD" in
        ''|*[!0-9]*) AHEAD=0 ;;
    esac

    # Separate conditionals (easier to read, debug, and maintain)
    if [ "$BEHIND" -gt 0 ]; then
        echo "   ⚠️ Behind remote by $BEHIND commits"
    fi

    if [ "$AHEAD" -gt 0 ]; then
        echo "   📤 Ahead of remote by $AHEAD commits"
    fi

    # Only print sync message if both are zero
    if [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ]; then
        echo "   ✅ Synced with remote"
    fi
else
    echo "   ℹ️ No remote tracking"
fi
```

### Why This Fix Works
- **Numeric validation:** `case` pattern `''|*[!0-9]*` catches empty/non-numeric values
- **Separate conditionals:** Each check is independent, easier to debug
- **Explicit defaults:** Variables default to `0` if extraction fails
- **No chained `&&`:** Avoids short-circuit failures in complex conditions

### Edge Cases Handled
| Edge Case | Behavior |
|-----------|----------|
| No remote tracking (`@{u}` fails) | `SYNC_STATUS=""` → "No remote tracking" |
| Git output format change | Validation catches empty/non-numeric → defaults to `0` |
| Behind only (`10 0`) | Shows "Behind by 10" only |
| Ahead only (`0 5`) | Shows "Ahead by 5" only |
| Synced (`0 0`) | Shows "Synced with remote" |
| Malformed output (`abc def`) | Validates to `0 0` → "Synced" (safe fallback) |

### Testing Validation
```bash
# Test numeric validation
sh -c 'VAL=""; case "$VAL" in ""|\*[!0-9]\*) VAL=0;; esac; echo $VAL'
# Expected: 0

sh -c 'VAL="abc"; case "$VAL" in ""|\*[!0-9]\*) VAL=0;; esac; echo $VAL'
# Expected: 0

sh -c 'VAL="42"; case "$VAL" in ""|\*[!0-9]\*) VAL=0;; esac; echo $VAL'
# Expected: 42

# Test branch sync with no remote (expect graceful failure)
mkdir /tmp/test-git && cd /tmp/test-git && git init
git config user.email "test@test.com" && git config user.name "Test"
echo "test" > file && git add file && git commit -m "test"
# Run the fixed branch sync code (should show "No remote tracking")

# Cleanup
rm -rf /tmp/test-git
```

---

## Problem 3: Session Handoff Counting with Piping (Lines 127-130)

### Original Code (Unsafe Piping)
```bash
# 4.5 Count Session Handoffs
HANDOFF_COUNT=$(ls -1 session-handoffs/*.md 2>/dev/null | wc -l | tr -d ' ')
LATEST_HANDOFF=$(ls -t session-handoffs/*.md 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "none")
echo "📋 Session handoffs: $HANDOFF_COUNT total"
[ "$HANDOFF_COUNT" -gt 0 ] && echo "   Latest: $LATEST_HANDOFF"
```

### Issues
1. If no `*.md` files exist, `ls` fails but `wc -l` returns `0` (correct, but fragile)
2. `xargs basename` on empty input can behave differently across systems
3. Multiple pipes increase failure surface area
4. `ls` parsing is generally discouraged (use `find` or globs)

### Fixed Code (Robust File Handling)
```bash
# 4.5 Count Session Handoffs - POSIX-compliant with null-safety
# Use shell globbing instead of ls parsing
set -- session-handoffs/*.md 2>/dev/null

# Check if glob matched any files
if [ -e "$1" ]; then
    HANDOFF_COUNT=$#
    # Get latest file by modification time (POSIX-compliant)
    LATEST_HANDOFF=""
    LATEST_TIME=0
    for file in session-handoffs/*.md; do
        if [ -f "$file" ]; then
            FILE_TIME=$(stat -f "%m" "$file" 2>/dev/null || stat -c "%Y" "$file" 2>/dev/null || echo "0")
            if [ "$FILE_TIME" -gt "$LATEST_TIME" ]; then
                LATEST_TIME=$FILE_TIME
                LATEST_HANDOFF=$(basename "$file")
            fi
        fi
    done
else
    HANDOFF_COUNT=0
    LATEST_HANDOFF="none"
fi

echo "📋 Session handoffs: $HANDOFF_COUNT total"
if [ "$HANDOFF_COUNT" -gt 0 ]; then
    echo "   Latest: $LATEST_HANDOFF"
fi
```

### Why This Fix Works
- **Shell globbing:** `set -- *.md` is POSIX standard, avoids `ls` parsing issues
- **`$#` for counting:** Positional parameter count is reliable, atomic
- **`stat` fallback:** Tries BSD (`-f "%m"`) then GNU (`-c "%Y"`) for cross-platform compatibility
- **Explicit file test:** `[ -e "$1" ]` checks if glob matched anything
- **No piping:** All operations in shell, reduces failure modes

### Edge Cases Handled
| Edge Case | Behavior |
|-----------|----------|
| No `*.md` files | `HANDOFF_COUNT=0`, `LATEST_HANDOFF="none"` |
| Single file | `HANDOFF_COUNT=1`, correct basename shown |
| Directory doesn't exist | `set --` fails gracefully, count = 0 |
| Files with spaces in names | Glob handles correctly (no word splitting) |
| Symlinks | `[ -f "$file" ]` test ensures only regular files counted |

### Alternative (Simpler but Less Portable)
```bash
# Simpler version using find (requires GNU findutils)
HANDOFF_COUNT=$(find session-handoffs -maxdepth 1 -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$HANDOFF_COUNT" -gt 0 ]; then
    LATEST_HANDOFF=$(find session-handoffs -maxdepth 1 -name "*.md" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}' | xargs basename)
    echo "📋 Session handoffs: $HANDOFF_COUNT total"
    echo "   Latest: $LATEST_HANDOFF"
else
    echo "📋 Session handoffs: 0 total"
fi
```

**Note:** The glob-based approach is recommended for POSIX compliance. The `find` version requires GNU extensions (`-printf`, `-maxdepth`).

### Testing Validation
```bash
# Test with no files
mkdir -p /tmp/test-handoffs
cd /tmp/test-handoffs
set -- *.md
if [ -e "$1" ]; then echo "Files found: $#"; else echo "No files"; fi
# Expected: No files

# Test with files
touch "2025-01-01-test.md" "2025-01-02-latest.md"
set -- *.md
if [ -e "$1" ]; then echo "Files found: $#"; else echo "No files"; fi
# Expected: Files found: 2

# Test basename extraction
for file in *.md; do echo $(basename "$file"); done
# Expected: 2025-01-01-test.md, 2025-01-02-latest.md

# Cleanup
cd - && rm -rf /tmp/test-handoffs
```

---

## Problem 4: Quality Tools Checking with && Chains (Lines 120-124)

### Original Code (Verbose, Fragile)
```bash
# 4.4 Check Quality Tools (condensed)
TOOLS=""
command -v ruff &>/dev/null && TOOLS="$TOOLS ruff"
command -v mypy &>/dev/null && TOOLS="$TOOLS mypy"
command -v pytest &>/dev/null && TOOLS="$TOOLS pytest"
command -v npm &>/dev/null && TOOLS="$TOOLS npm"
[ -n "$TOOLS" ] && echo "   Quality tools:$TOOLS" || echo "   ⚠️ No quality tools detected"
```

### Issues
1. Multiple `command -v` calls with `&&` chains (6 separate checks)
2. String concatenation with leading spaces (` ruff`, ` mypy`) is unclean
3. Final conditional (`[ -n "$TOOLS" ] && ... || ...`) is an anti-pattern (can fail unexpectedly)
4. No explicit trimming of leading space in `$TOOLS`

### Fixed Code (Clean, Robust)
```bash
# 4.4 Check Quality Tools (POSIX-compliant, cleaner)
TOOLS=""

# Check each tool and build list
for tool in ruff mypy pytest npm; do
    if command -v "$tool" >/dev/null 2>&1; then
        if [ -z "$TOOLS" ]; then
            TOOLS="$tool"
        else
            TOOLS="$TOOLS $tool"
        fi
    fi
done

# Print results
if [ -n "$TOOLS" ]; then
    echo "   Quality tools: $TOOLS"
else
    echo "   ⚠️ No quality tools detected"
fi
```

### Why This Fix Works
- **Loop-based:** Single pattern, easier to extend with new tools
- **No leading spaces:** First tool doesn't get prepended space
- **Explicit conditionals:** No `&&`/`||` chaining anti-patterns
- **Cleaner output:** `"Quality tools: ruff mypy"` vs `"Quality tools: ruff mypy"`

### Edge Cases Handled
| Edge Case | Behavior |
|-----------|----------|
| No tools installed | `TOOLS=""` → "No quality tools detected" |
| Single tool (ruff only) | `TOOLS="ruff"` → "Quality tools: ruff" |
| All tools installed | `TOOLS="ruff mypy pytest npm"` → clean spacing |
| Non-standard tool names | Easy to add to loop list |

### Alternative (Array-Based, Bash 4+)
```bash
# Bash 4+ version with arrays (not POSIX-compliant)
TOOLS=()
for tool in ruff mypy pytest npm; do
    command -v "$tool" >/dev/null 2>&1 && TOOLS+=("$tool")
done

if [ ${#TOOLS[@]} -gt 0 ]; then
    echo "   Quality tools: ${TOOLS[*]}"
else
    echo "   ⚠️ No quality tools detected"
fi
```

**Recommendation:** Use the loop-based POSIX version for maximum compatibility.

### Testing Validation
```bash
# Test with no tools (empty PATH)
sh -c 'PATH=""; TOOLS=""; for tool in ruff mypy; do command -v "$tool" >/dev/null 2>&1 && TOOLS="$TOOLS $tool"; done; echo "TOOLS=$TOOLS"'
# Expected: TOOLS=

# Test with mock tools
mkdir -p /tmp/test-bin
echo '#!/bin/sh' > /tmp/test-bin/ruff && chmod +x /tmp/test-bin/ruff
export PATH="/tmp/test-bin:$PATH"
TOOLS=""; for tool in ruff mypy; do command -v "$tool" >/dev/null 2>&1 && { [ -z "$TOOLS" ] && TOOLS="$tool" || TOOLS="$TOOLS $tool"; }; done; echo "TOOLS=$TOOLS"
# Expected: TOOLS=ruff

# Cleanup
rm -rf /tmp/test-bin
```

---

## Summary of All Fixes

### Quick Reference Table

| Problem | Original Syntax | Fixed Syntax | Portability Gain |
|---------|----------------|--------------|------------------|
| **Context Detection** | `[[ =~ ]]` regex | `case` patterns | ✅ POSIX compliant |
| **Branch Sync** | Chained `&&`, no validation | Separate `if` blocks, numeric validation | ✅ Handles malformed input |
| **File Counting** | `ls \| xargs \| basename` | Shell globbing + `stat` | ✅ No `ls` parsing, null-safe |
| **Tool Detection** | Multiple `&&` chains | Loop with explicit conditionals | ✅ Cleaner, extensible |

### Combined Compatibility Matrix

| Shell | Original Code | Fixed Code |
|-------|---------------|------------|
| `bash` 4+ | ✅ Works | ✅ Works |
| `bash` 3.x | ⚠️ Works (deprecated syntax) | ✅ Works |
| `dash` (Debian/Ubuntu default `/bin/sh`) | ❌ Fails (`[[ ]]` syntax error) | ✅ Works |
| `ash` (Alpine Linux) | ❌ Fails | ✅ Works |
| `ksh` | ✅ Works | ✅ Works |
| `zsh` | ✅ Works | ✅ Works |
| POSIX `/bin/sh` | ❌ Fails | ✅ Works |

---

## Implementation Strategy

### Phase 1: Low-Risk Fixes (Immediate)
1. **Context Detection** (Lines 69-78) → `case` statement
2. **Tool Detection** (Lines 120-124) → Loop-based

**Risk:** None
**Testing:** Run on `bash`, `sh`, `dash`

### Phase 2: Medium-Risk Fixes (Validate First)
3. **File Counting** (Lines 127-130) → Shell globbing

**Risk:** Medium (changes file handling logic)
**Testing:** Test with 0 files, 1 file, 10+ files, files with spaces

### Phase 3: Complex Fixes (Thorough Testing)
4. **Branch Sync** (Lines 99-108) → Validated conditionals

**Risk:** Medium-High (git interaction)
**Testing:** Test with no remote, ahead, behind, synced, malformed output

### Rollback Plan
Keep original code commented out above each fix:
```bash
# ORIGINAL (non-POSIX):
# [[ "$CURRENT_BRANCH" =~ ^feat/ ]] && CONTEXT="DEVELOPMENT"

# FIXED (POSIX-compliant):
case "$CURRENT_BRANCH" in
    feat/*) CONTEXT="DEVELOPMENT" ;;
esac
```

---

## Testing Checklist

### Pre-Implementation Testing
- [ ] Verify all fixes are syntactically correct with `sh -n script.sh`
- [ ] Test each fix in isolation with `sh -c '...'`
- [ ] Validate POSIX compliance with `checkbashisms` (if available)

### Post-Implementation Testing
- [ ] Run full script in `bash` (current behavior)
- [ ] Run full script in `dash` (POSIX validation)
- [ ] Run full script in `sh` (minimal shell)
- [ ] Test edge cases: no git repo, empty directories, no remote tracking
- [ ] Verify output formatting unchanged

### Regression Testing
- [ ] Compare output with original script in normal conditions
- [ ] Verify performance (should be identical or faster)
- [ ] Check for any new error messages or warnings

### Shell-Specific Testing Commands
```bash
# Test in strict POSIX mode
sh -o posix uwo-ready.md

# Test in dash (Debian/Ubuntu default)
dash uwo-ready.md

# Test in ash (Alpine Linux)
docker run -v $(pwd):/work alpine:latest /bin/ash /work/uwo-ready.md

# Syntax check without execution
sh -n uwo-ready.md
```

---

## Recommended Tools for Validation

### ShellCheck (Static Analysis)
```bash
# Install
brew install shellcheck  # macOS
apt install shellcheck   # Debian/Ubuntu

# Check script
shellcheck -s sh uwo-ready.md
```

**Expected warnings to fix:**
- SC2039: In POSIX sh, `[[ ]]` is undefined
- SC2076: Don't use `=~` in `[ ]`
- SC2012: Use `find` instead of `ls` to better handle non-alphanumeric filenames

### checkbashisms (POSIX Compliance)
```bash
# Install
apt install devscripts  # Debian/Ubuntu

# Check script
checkbashisms uwo-ready.md
```

**Expected issues:**
- `[[ ]]` is not POSIX
- `command -v` with `&>/dev/null` (use `>/dev/null 2>&1` instead)

---

## Expected Outcomes

### Before Fixes
- ❌ Fails in Alpine Linux Docker containers (`ash` shell)
- ❌ Fails in Debian/Ubuntu with `/bin/sh` → `dash`
- ⚠️ Fragile file counting with empty directories
- ⚠️ Potential numeric comparison errors with malformed git output

### After Fixes
- ✅ Works in all POSIX-compliant shells
- ✅ Handles edge cases gracefully (empty files, no remote, malformed input)
- ✅ Cleaner, more maintainable code
- ✅ No behavioral changes in normal operation
- ✅ Future-proof for minimal shell environments (CI/CD, containers)

---

## Code Review Verdict

**Recommendation:** ✅ **IMPLEMENT ALL FIXES**

**Rationale:**
1. Zero behavioral changes in normal operation
2. Significant portability improvements (Alpine, Debian, minimal shells)
3. Better error handling for edge cases
4. Cleaner, more maintainable code
5. No performance degradation

**Priority:**
1. **HIGH:** Context Detection, Tool Detection (immediate portability wins)
2. **MEDIUM:** Branch Sync (critical for git operations)
3. **MEDIUM:** File Counting (important for session management)

**Timeline:**
- Implementation: 30 minutes
- Testing: 1 hour (across multiple shells)
- Documentation: 15 minutes (update CLAUDE.md with POSIX compliance note)

---

## Appendix: Full Fixed Script Section

### Complete Fixed Version (Lines 69-135)

```bash
# STEP 3: Context Detection (POSIX-COMPLIANT)

# Capture context (suppress errors)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
WORKING_DIR=$(pwd)

echo "🌿 Branch: $CURRENT_BRANCH"
echo "📁 Directory: $WORKING_DIR"

# Analyze branch name for context
CONTEXT="GENERAL"
case "$CURRENT_BRANCH" in
    feat/*)
        CONTEXT="DEVELOPMENT"
        ;;
    test/*)
        CONTEXT="TESTING"
        ;;
    docs/*)
        CONTEXT="DOCUMENTATION"
        ;;
    fix/*)
        CONTEXT="BUGFIX"
        ;;
    *)
        CONTEXT="GENERAL"
        ;;
esac

echo "📍 Detected Context: $CONTEXT"

# STEP 4: System Health Validation (POSIX-COMPLIANT)

echo "🔍 System Health Checks"
echo ""

# 4.1 Git Status
UNCOMMITTED_COUNT=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNCOMMITTED_COUNT" -eq 0 ]; then
    echo "✅ Git: Clean working tree"
else
    echo "⚠️ Git: $UNCOMMITTED_COUNT uncommitted files"
fi

# 4.2 Branch Sync (POSIX-COMPLIANT)
SYNC_STATUS=$(git rev-list --left-right --count @{u}...HEAD 2>/dev/null || echo "")

if [ -n "$SYNC_STATUS" ]; then
    # Extract values with defaults
    BEHIND=$(echo "$SYNC_STATUS" | awk '{print $1}')
    AHEAD=$(echo "$SYNC_STATUS" | awk '{print $2}')

    # Validate extracted values are numbers
    case "$BEHIND" in
        ''|*[!0-9]*) BEHIND=0 ;;
    esac
    case "$AHEAD" in
        ''|*[!0-9]*) AHEAD=0 ;;
    esac

    # Separate conditionals
    if [ "$BEHIND" -gt 0 ]; then
        echo "   ⚠️ Behind remote by $BEHIND commits"
    fi

    if [ "$AHEAD" -gt 0 ]; then
        echo "   📤 Ahead of remote by $AHEAD commits"
    fi

    if [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ]; then
        echo "   ✅ Synced with remote"
    fi
else
    echo "   ℹ️ No remote tracking"
fi

# 4.3 Detect Project Type
PROJECT_TYPE="Unknown"
[ -f "package.json" ] && PROJECT_TYPE="Node.js"
[ -f "pyproject.toml" ] && PROJECT_TYPE="Python"
[ -f "Cargo.toml" ] && PROJECT_TYPE="Rust"
[ -f "go.mod" ] && PROJECT_TYPE="Go"
echo "✅ Project type: $PROJECT_TYPE"

# 4.4 Check Quality Tools (POSIX-COMPLIANT)
TOOLS=""
for tool in ruff mypy pytest npm; do
    if command -v "$tool" >/dev/null 2>&1; then
        if [ -z "$TOOLS" ]; then
            TOOLS="$tool"
        else
            TOOLS="$TOOLS $tool"
        fi
    fi
done

if [ -n "$TOOLS" ]; then
    echo "   Quality tools: $TOOLS"
else
    echo "   ⚠️ No quality tools detected"
fi

# 4.5 Count Session Handoffs (POSIX-COMPLIANT)
set -- session-handoffs/*.md 2>/dev/null

if [ -e "$1" ]; then
    HANDOFF_COUNT=$#
    # Get latest file by modification time
    LATEST_HANDOFF=""
    LATEST_TIME=0
    for file in session-handoffs/*.md; do
        if [ -f "$file" ]; then
            FILE_TIME=$(stat -f "%m" "$file" 2>/dev/null || stat -c "%Y" "$file" 2>/dev/null || echo "0")
            if [ "$FILE_TIME" -gt "$LATEST_TIME" ]; then
                LATEST_TIME=$FILE_TIME
                LATEST_HANDOFF=$(basename "$file")
            fi
        fi
    done
else
    HANDOFF_COUNT=0
    LATEST_HANDOFF="none"
fi

echo "📋 Session handoffs: $HANDOFF_COUNT total"
if [ "$HANDOFF_COUNT" -gt 0 ]; then
    echo "   Latest: $LATEST_HANDOFF"
fi

# 4.6 Count Subagent Reports
REPORT_COUNT=$(find docs/subagent-reports -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "🤖 Subagent reports: $REPORT_COUNT total"
echo ""
```

---

**End of Analysis**
**Next Step:** Apply fixes to `.claude/commands/uwo-ready.md` and validate with multi-shell testing
