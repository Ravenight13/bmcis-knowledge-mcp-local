# Parallel Orchestration Guide

**Category**: Workflow Orchestration
**Token Budget**: ~2,400 tokens

---

## Orchestration Philosophy

The main chat (user ↔ Claude) is the **orchestration layer**. All complex tasks should be delegated to parallel subagents for maximum efficiency.

**Core Principle**: Main chat orchestrates, subagents execute specialized work.

---

## When to Use Parallel Subagents

### ALWAYS consider parallel subagents for:

- **Multiple independent tasks** (research, analysis, implementation)
- **Simultaneous evaluations** (analyze multiple components/files/commands)
- **Documentation generation** (create multiple reports in parallel)
- **Code reviews** across multiple files or modules
- **Research requiring diverse expertise** (architecture, security, testing, performance)
- **File output required**: When research must be preserved beyond session

### Example Triggers:

**Complexity Indicators**:
- Task requires analyzing 10+ files
- Multiple independent research questions
- Need domain expertise across different areas
- Work can be parallelized (no sequential dependencies)
- Estimated effort >30 minutes if done sequentially

---

## How to Orchestrate

### Model-Invoked Architecture

**Important**: Users request subagent delegation via natural language, not by calling functions directly.

**User Request Pattern**:
```
"Please use parallel subagents to analyze the vendor format,
review base patterns, and check security requirements."
```

**What Claude Does**:
1. Evaluates request complexity
2. Decides delegation is beneficial
3. Spawns appropriate subagents with isolated contexts
4. Each subagent executes independently
5. Synthesizes results when complete

### Orchestration Pattern:

1. **Decide** which subagents to launch and what tasks to assign
2. **Launch** all agents (Claude handles parallel execution internally)
3. **Wait** for completion (non-blocking)
4. **Synthesize** outputs into actionable next steps
5. **Communicate** results to user in concise summary

---

## Subagent File Output Protocol (MANDATORY)

### The Golden Rule

**Every subagent MUST write their findings to a file and micro-commit before reporting back.**

**Why this is critical**:
- **Work preservation**: If main session crashes, subagent work is not lost
- **Audit trail**: Complete history of all research and analysis
- **Async review**: Team members can review subagent outputs independently
- **Handoff continuity**: Next session can pick up from subagent files

### Directory Structure for Subagent Outputs

**Research & Analysis**:
```
docs/subagent-reports/{agent-type}/{component}/
└── YYYY-MM-DD-HHMM-{description}.md
```

**Examples**:
```
docs/subagent-reports/api-analysis/auth/2025-10-27-1400-authentication-security.md
docs/subagent-reports/architecture-review/database/2025-10-27-1430-schema-analysis.md
docs/subagent-reports/performance-analysis/queries/2025-10-27-1500-slow-query-investigation.md
```

**Session Handoffs** (if subagent creates intermediate handoff):
```
session-handoffs/YYYY-MM-DD-HHMM-{description}.md
```

**Analysis Results**:
```
docs/analysis/YYYY-MM-DD-HHMM-{description}.md
```

### Subagent Workflow (Required Steps)

1. **Create output directory** (if not exists):
   ```bash
   mkdir -p docs/subagent-reports/{agent-type}/{component}
   ```

2. **Conduct research/analysis** (execute assigned task)

3. **Write findings to file**:
   ```bash
   # Use TEMPLATE_RESEARCH_REPORT.md as structure
   # Write to: docs/subagent-reports/{agent-type}/{component}/YYYY-MM-DD-HHMM-{description}.md
   ```

4. **Micro-commit immediately**:
   ```bash
   git add docs/subagent-reports/{agent-type}/{component}/YYYY-MM-DD-HHMM-{description}.md
   git commit -m "docs({component}): add {description} research by subagent"
   ```

5. **Report summary to main chat**:
   ```
   "Analysis complete. Full findings in docs/subagent-reports/{path}.

   Key takeaways:
   - {Takeaway 1}
   - {Takeaway 2}
   - {Takeaway 3}

   Recommendation: {High-level recommendation}"
   ```

### Example: Complete Subagent Workflow

**User request:**
"Analyze these 5 API endpoints for security vulnerabilities using parallel subagents"

**Main Chat (Orchestrator):**
```markdown
Spawning 5 security analysis agents:
1. Auth endpoint agent
2. User data endpoint agent
3. Payment endpoint agent
4. Admin endpoint agent
5. Public API endpoint agent
```

**Subagent 1 (Auth Endpoint Security Analysis):**

Step 1: Create directory
```bash
mkdir -p docs/subagent-reports/security-analysis/auth
```

Step 2: Conduct security analysis
- Review auth endpoint code
- Check for common vulnerabilities (SQL injection, XSS, auth bypass)
- Verify input validation
- Check rate limiting
- Review token handling

Step 3: Write findings to file
```bash
# Writes to: docs/subagent-reports/security-analysis/auth/2025-10-27-1400-auth-security-audit.md
# Uses TEMPLATE_RESEARCH_REPORT.md structure
# Includes: Executive Summary, Findings, Vulnerabilities, Recommendations
```

Step 4: Micro-commit
```bash
git add docs/subagent-reports/security-analysis/auth/2025-10-27-1400-auth-security-audit.md
git commit -m "docs(auth): add security audit findings by subagent"
```

Step 5: Report back to main chat
```markdown
Auth endpoint security analysis complete.

Full report: docs/subagent-reports/security-analysis/auth/2025-10-27-1400-auth-security-audit.md

Key findings:
- ✅ SQL injection: Protected (parameterized queries)
- ⚠️ Rate limiting: Insufficient (10 req/sec, recommend 5 req/sec)
- ❌ Token validation: Missing expiry check on refresh tokens (CRITICAL)

Recommendation: Fix token validation immediately (High priority)
```

**Subagents 2-5:** Follow same pattern for their respective endpoints

**Main Chat (Synthesis):**
```markdown
All 5 subagents complete. Security audit findings:

Critical Issues:
- Auth endpoint: Missing token expiry validation (CRITICAL)
- Payment endpoint: No CSRF protection (CRITICAL)

Medium Issues:
- User data: Rate limiting too permissive (3 endpoints)

Low Issues:
- Admin endpoint: Missing request logging

Recommendation: Fix critical issues before deployment.

All detailed reports available in docs/subagent-reports/security-analysis/{endpoint}/
```

### Anti-Pattern: What NOT to Do

❌ **WRONG: Subagent reports back verbally only**
```
Subagent: "I analyzed the auth endpoint. Found SQL injection protection is good,
rate limiting needs improvement, and token validation is missing expiry check."
```

**Problem**: If session crashes before main chat synthesizes, all subagent work is lost.

✅ **CORRECT: Subagent writes file + commits + reports back**
```
Subagent:
1. Writes docs/subagent-reports/security-analysis/auth/2025-10-27-1400-auth-security-audit.md
2. Commits file
3. Reports: "Auth security audit complete. Full report in {file-path}. Key finding: Missing token expiry check (CRITICAL)"
```

**Benefit**: Work is preserved even if session crashes. Can resume from file.

---

## Main Chat Responsibilities

### 1. Orchestrate
Decide which subagents to launch and what tasks to assign. Consider:
- Task independence (can run in parallel)
- Required expertise (specialized vs general)
- Estimated effort (overhead vs benefit)

### 2. Coordinate
Ensure subagents have:
- Clear, independent objectives
- Isolated contexts (no overlap)
- Well-defined deliverables
- **File output paths specified** (MANDATORY)

### 3. Synthesize
Combine subagent outputs:
- Identify common themes
- Resolve conflicts
- Create unified action plan
- **Reference subagent file paths** in synthesis

### 4. Communicate
Present results to user:
- High-level summary
- Key findings from each subagent
- Recommended next steps
- **Links to detailed subagent reports**

---

## Main Chat Should NOT

❌ **Perform complex research directly** (delegate to research agents)
❌ **Read 10+ files sequentially** (use exploration/analysis agents)
❌ **Analyze multiple independent components** (spawn parallel analysis agents)
❌ **Generate comprehensive reports** (use documentation agents)

---

## Benefits of Parallel Orchestration

### 1. Speed
- **Parallelism**: N tasks in parallel vs N sequential = up to N× faster
- **Concurrency**: Claude Code supports up to 10 concurrent subagents
- **Queueing**: Additional tasks automatically queued and executed as slots free

### 2. Expertise
- Specialized agents bring domain knowledge
- Each agent focuses on specific expertise area
- Better quality outputs from focused contexts

### 3. Token Efficiency
- Subagents have isolated contexts (no cross-contamination)
- No token bloat in main chat
- Progressive disclosure enforced per agent

### 4. Clarity
- Main chat focuses on orchestration, not execution
- Clean separation of concerns
- Easier to track progress and debug

### 5. Work Preservation
- **All subagent findings committed to files** (survives crashes)
- **Audit trail**: Complete history of all parallel work in git
- **Handoff continuity**: Next session can resume from subagent outputs

---

## Real-World Examples

### Example 1: Multi-Language Codebase Analysis

**Task**: Analyze codebase with Python backend + JavaScript frontend

**Orchestration**:
```
User: "Please use parallel subagents to analyze the Python
backend patterns and JavaScript frontend architecture."

Claude spawns:
- backend-analysis agent (Python expertise)
- frontend-analysis agent (JavaScript expertise)
```

**Result**:
- 2 agents complete in ~40 minutes (parallel)
- Sequential estimate: ~80 minutes
- Speedup: 2×

### Example 2: Documentation Review

**Task**: Review 15 documentation files for consistency

**Orchestration**:
```
User: "Please use parallel subagents to review all 15
documentation files for consistency and completeness."

Claude spawns:
- 10 concurrent doc-review agents (first batch)
- 5 queued agents (second batch)
```

**Result**:
- 10 concurrent + 5 queued = ~30 minutes total
- Sequential estimate: ~150 minutes
- Speedup: 5×

### Example 3: Security Audit

**Task**: Audit application for security vulnerabilities

**Orchestration**:
```
User: "Please use parallel subagents to audit authentication,
API security, data validation, and deployment security."

Claude spawns:
- auth-security agent
- api-security agent
- validation-security agent
- deployment-security agent
```

**Result**:
- 4 agents complete in ~45 minutes (parallel)
- Sequential estimate: ~180 minutes
- Speedup: 4×

---

## Parallelism Reality

**Concurrent Execution**: Claude Code supports up to **10 concurrent task executions**.

**Queueing**: If more than 10 subagents are requested, additional tasks are automatically queued and executed as slots become available.

**Transparency**: Queueing is handled transparently—just request the work you need.

**Example**:
```
Request 15 parallel subagents:
- First batch: 10 agents execute concurrently
- Second batch: 5 agents queued, execute when slots free
```

---

## Anti-Patterns (NEVER)

❌ **Sequential subagent execution** (defeats parallelism benefits)
❌ **Main chat doing specialized work** (lose expertise benefits)
❌ **Launching subagents for trivial tasks** (overhead not justified)
❌ **Too many subagents without justification** (>10 = coordination overhead)
❌ **Overlapping contexts** (duplicate work across agents)
❌ **Subagents reporting verbally without file output** (work loss risk)
❌ **Skipping micro-commits after subagent file creation** (no audit trail)

---

## Summary

**Philosophy**: Main chat orchestrates, subagents execute

**When**: Complex tasks, multiple independent objectives, diverse expertise needed

**How**: Request delegation via natural language, Claude handles parallel execution

**Benefits**: Up to N× speedup, expertise, token efficiency, clarity, work preservation

**File Output Protocol**: MANDATORY for all subagents (write + commit + report)

**Key Insight**: Parallel orchestration maximizes efficiency for complex multi-faceted work while preserving all research outputs for continuity.
