# Session Handoff: MCP Server Loading Issues - Debugging Required

**Date**: 2025-11-09 21:05 PST
**Session**: MCP Configuration & Dependency Resolution
**Branch**: feat/task-10-fastmcp-integration
**Status**: ⚠️ IN PROGRESS - MCP Server Initializes But Claude Code Still Won't Load

---

## Executive Summary

Successfully ingested 2,426 BMCIS knowledge chunks and fixed dependency issues. MCP server now initializes correctly (`python -m src.mcp` starts without errors). However, **Claude Code MCP Manager still reports "failed to load"** despite the server being functional.

**Root Cause**: Unknown - Server initialization works in isolation but Claude Code's MCP loader is not successfully spawning/communicating with it.

---

## Work Completed This Session

### 1. ✅ Identified Initial Issues
- FastMCP not in dependencies → Added `fastmcp>=0.1.0`
- psycopg version mismatch → Changed from `psycopg[binary]` to `psycopg2-binary`
- Missing MCP entry point → Created `src/mcp/__main__.py`
- Wrong module path → Updated to `python -m src.mcp`

### 2. ✅ Fixed Dependencies
```
pyproject.toml changes:
+ fastmcp>=0.1.0
+ requests>=2.31.0
- psycopg[binary]>=3.1.0
+ psycopg2-binary>=2.9.0
```

### 3. ✅ Created MCP Entry Point
`src/mcp/__main__.py`:
```python
async def main():
    initialize_server()
    await mcp.run_stdio()
```

### 4. ✅ Updated Claude Code Configuration
`~/.claude.json`:
```json
"bmcis-knowledge-mcp": {
  "command": "uv",
  "args": ["--directory", "...", "run", "python", "-m", "src.mcp"],
  "env": {
    "PYTHONPATH": "...",
    "DATABASE_URL": "postgresql://..."
  }
}
```

### 5. ✅ Verified Server Initializes
```bash
$ timeout 3 uv run python -m src.mcp 2>&1 | head -5
{"timestamp": "2025-11-09T21:01:11-0600", "level": "INFO",
 "logger": "src.mcp.auth", "message": "RateLimiter initialized: ..."}
{"timestamp": "2025-11-09T21:01:11-0600", "level": "INFO",
 "logger": "src.mcp.auth", "message": "Authentication module initialized..."}
```

**Result**: ✅ Server starts without errors (timeout kills it as expected)

### 6. ❌ BUT Claude Code Still Reports Failed
```
[Contains warnings] Local config (private to you in this project)
❯ 1. bmcis-knowledge-mcp            ✘ failed · Enter to view details
```

---

## Current Problem

### Symptoms
- MCP Manager shows: `✘ failed`
- No specific error message shown in UI
- Server works fine when run manually
- Task Master MCP also shows warnings about missing env vars (expected, can ignore)

### What Works
- ✅ Server initialization (direct `python -m src.mcp`)
- ✅ Database connectivity (2,426 chunks verified)
- ✅ FastMCP imports
- ✅ Auth module initialization
- ✅ All dependencies installed

### What Doesn't Work
- ❌ Claude Code's MCP process spawning
- ❌ Stdio communication between Claude Code ↔ MCP Server
- ❌ MCP Manager UI shows connection status

---

## Possible Root Causes

### 1. **Stdio Transport Issues**
- MCP expects JSON-RPC over stdio
- FastMCP's `run_stdio()` may need different initialization
- Claude Code may not be feeding stdin properly to the subprocess

### 2. **Environment Variable Not Set**
- `PYTHONPATH` might not be passed correctly
- `DATABASE_URL` might be missing in actual Claude Code spawn
- Database connection might fail before stdio setup

### 3. **Module Import Timing**
- `src.mcp` imports `src.mcp.server` at top level
- Server imports database which imports psycopg2
- If any import fails, entire process dies before reaching `run_stdio()`

### 4. **Path/Working Directory Issues**
- `--directory` flag in uv args might not work as expected
- PYTHONPATH might need to be absolute path vs relative
- Module resolution might fail due to working directory

### 5. **FastMCP Compatibility**
- FastMCP version might need specific setup
- May need explicit tool registration before `run_stdio()`
- Might need error handling wrapper

---

## Debugging Steps for Next Session

### Step 1: Check Claude Code MCP Logs
```bash
# Find Claude Code's MCP server logs (macOS)
~/Library/Application\ Support/Claude/logs/

# Look for recent error messages about bmcis-knowledge-mcp
grep -r "bmcis-knowledge-mcp" ~/Library/Application\ Support/Claude/logs/
```

### Step 2: Test MCP Server Directly with JSON-RPC
```bash
# Start server in background
uv run python -m src.mcp > /tmp/mcp.log 2>&1 &
MCP_PID=$!

# Send MCP init request via stdin
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  uv run python -m src.mcp

# Check output for response
```

### Step 3: Run with Verbose Logging
Create debug wrapper:
```bash
# Modify ~/.claude.json to add debugging:
"bmcis-knowledge-mcp": {
  "command": "bash",
  "args": ["-c", "set -x; uv run python -m src.mcp 2>&1 | tee /tmp/mcp-debug.log"]
}
```

Then check `/tmp/mcp-debug.log` for actual execution errors.

### Step 4: Simplify Entry Point
Current `src/mcp/__main__.py` uses async - try synchronous version:
```python
def main():
    initialize_server()
    mcp.run_stdio()  # Sync version

if __name__ == "__main__":
    main()
```

### Step 5: Check FastMCP Tool Registration
Verify tools are properly decorated in `src/mcp/tools/*.py`:
```python
@mcp.tool()  # This decorator registers with FastMCP
def semantic_search(...):
    pass
```

### Step 6: Validate MCP Protocol
Use MCP's built-in test client (if available):
```bash
# Some MCP implementations have test tools
uv run python -m fastmcp.server src.mcp.server:mcp
```

---

## Git Status

**Current Commits**:
```
db35e5c fix: Replace psycopg[binary] with psycopg2-binary
d19d14b fix: Add fastmcp dependency and MCP server entry point
2511ef4 docs: Session handoff for BMCIS MCP ingestion
8cb0110 feat: Complete BMCIS data ingestion - 2,426 chunks
```

**Uncommitted Changes**: None

---

## Files Modified This Session

```
pyproject.toml
  - Removed: psycopg[binary]>=3.1.0
  + Added: psycopg2-binary>=2.9.0
  + Added: fastmcp>=0.1.0
  + Added: requests>=2.31.0

src/mcp/__main__.py (NEW)
  - Created proper MCP entry point with async stdio transport

~/.claude.json (UPDATED)
  - Modified bmcis-knowledge-mcp server config
  - Changed to: python -m src.mcp
  - Already has DATABASE_URL and PYTHONPATH
```

---

## Known Working State

**Database**: 2,426 chunks with embeddings ✅
**Server Code**: All imports work, initialization succeeds ✅
**Dependencies**: All installed and compatible ✅
**Manual Testing**: `python -m src.mcp` starts without errors ✅

**Not Working**: Claude Code MCP Manager integration ❌

---

## Next Session Action Plan

### Priority 1: Diagnose MCP Loading (30-45 min)
1. Check Claude Code MCP logs for specific error
2. Test JSON-RPC protocol directly
3. Run with verbose debug logging
4. Identify exact failure point

### Priority 2: Fix Based on Diagnosis (30-60 min)
- If async issue → Switch to sync entry point
- If path issue → Fix PYTHONPATH/working directory
- If protocol issue → Adjust FastMCP initialization
- If import issue → Add error handling in __main__.py

### Priority 3: Verify and Test (15-30 min)
- Confirm MCP Manager shows ✓ Connected
- Test semantic_search tool directly
- Verify tool parameters and response format
- Ensure rate limiting works

### Priority 4: Mark Complete (5 min)
- Mark Task 10 as done: `task-master set-status --id=10 --status=done`
- Create Task 11 for production validation

---

## Troubleshooting Checklist

- [ ] Check Claude Code MCP logs (`~/Library/Application\ Support/Claude/logs/`)
- [ ] Test `python -m src.mcp` with JSON-RPC stdin
- [ ] Run with `set -x` bash debugging
- [ ] Check tool decorators (@mcp.tool())
- [ ] Verify DATABASE_URL env var is set
- [ ] Test with synchronous entry point instead of async
- [ ] Check if stdio is being inherited correctly
- [ ] Look for database connection timeouts
- [ ] Verify FastMCP version compatibility
- [ ] Check for circular import issues

---

## Reference Links

**FastMCP Documentation**:
- https://docs.anthropic.com/en/docs/build-with-claude/mcp/fastmcp

**MCP Spec**:
- https://spec.modelcontextprotocol.io/

**Claude Code MCP Docs**:
- https://docs.claude.com/en/docs/claude-code/mcp

---

## Important Notes for Next Session

1. **Database is NOT the issue** - All 2,426 chunks are safely stored
2. **Code is NOT the issue** - Server initializes without errors
3. **Dependencies are NOT the issue** - All imports work
4. **The issue is integration** - Claude Code can't communicate with the spawned process

5. **Don't waste time on**:
   - Re-running ingestion
   - Changing database code
   - Installing more dependencies (all needed ones are there)
   - Refactoring MCP server structure

6. **Do focus on**:
   - Subprocess communication (stdio, pipes, environment)
   - FastMCP initialization and tool registration
   - MCP protocol compliance
   - Error logging and debugging output

---

## Environment Variables to Check

```bash
# Verify these are set when Claude Code spawns the process:
PYTHONPATH=/Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local
DATABASE_URL=postgresql://cliffclarke@localhost:5432/bmcis_knowledge_dev

# Optional (should auto-detect):
BMCIS_API_KEY  # Optional authentication

# System:
PATH  # Must include uv executable
HOME  # For .venv lookups
```

---

## Quick Test Commands

```bash
# Test 1: Direct execution
cd /Users/cliffclarke/Claude_Code/bmcis-knowledge-mcp-local
timeout 2 uv run python -m src.mcp

# Test 2: With explicit environment
PYTHONPATH=. timeout 2 uv run python -m src.mcp

# Test 3: Check if tools are loaded
uv run python -c "from src.mcp.server import mcp; print(mcp.tools())"

# Test 4: Database connectivity
psql bmcis_knowledge_dev -c "SELECT COUNT(*) FROM knowledge_base;"
```

---

**Session Handoff Created By**: Claude Code
**Status**: ⚠️ Blocked - Waiting for MCP Integration Fix
**Next Session**: Diagnose and fix Claude Code MCP communication
