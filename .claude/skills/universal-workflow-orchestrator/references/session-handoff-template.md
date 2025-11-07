# Session Handoff Template

**Category**: Session Continuity
**Token Budget**: ~600 tokens

---

## Purpose

Preserve session state and context across Claude Code sessions through timestamped, structured narratives.

**Problem**: Context loss between sessions wastes time on re-discovery and re-planning.

**Solution**: Create handoff files at session end with completed work, next priorities, blockers, and critical context.

---

## File Structure

### Naming Convention

**Format**: `YYYY-MM-DD-HHMM-{description}.md`

**Examples**:
- `2025-10-27-0930-feature-implementation.md`
- `2025-10-27-1445-debugging-session.md`
- `2025-10-28-0815-research-phase.md`

**Time Format**:
- **Filename**: HHMM (24-hour, no colons)
- **Headers**: HH:MM:SS (24-hour, with colons)

### Directory Organization

**Location**: `session-handoffs/` (project root)

**Structure**:
```
session-handoffs/
├── 2025-10-27-0930-feature-implementation.md
├── 2025-10-27-1445-debugging-session.md
├── 2025-10-28-0815-research-phase.md
└── README.md  (optional: index of handoffs)
```

---

## Template Structure

### 1. Header Section

```markdown
# Session Handoff: {Brief Description}

**Date**: YYYY-MM-DD
**Time**: HH:MM:SS
**Branch**: {git-branch-name}
**Author**: Claude Code + {User-Name}
```

### 2. Session Summary (2-3 sentences)

**What was accomplished**:
- High-level overview of session work
- Key decisions made
- Major milestones reached

**Example**:
```markdown
## Session Summary

Completed initial implementation of user authentication module
including JWT token generation and validation. Established testing
framework with 15 unit tests covering core authentication flows.
Identified performance bottleneck in token refresh logic requiring
optimization in next session.
```

### 3. Completed Work (Bulleted List)

**Deliverables from this session**:

```markdown
## Completed Work

- ✅ Implemented JWT token generation (auth/tokens.py)
- ✅ Added token validation middleware (auth/middleware.py)
- ✅ Created 15 unit tests for authentication (tests/test_auth.py)
- ✅ Updated API documentation with auth endpoints (docs/api.md)
- ✅ Committed all changes (5 commits, conventional format)
```

### 4. Next Priorities (Ordered List)

**What needs to happen next**:

```markdown
## Next Priorities

1. **Optimize token refresh** (performance bottleneck identified)
   - Current: 500ms average
   - Target: <50ms
   - File: auth/tokens.py lines 45-67

2. **Add integration tests** (unit tests complete)
   - Test full auth flow (login → refresh → logout)
   - File: tests/integration/test_auth_flow.py

3. **Security review** (before production deployment)
   - Review token expiration logic
   - Validate CSRF protection
   - Check rate limiting
```

### 5. Blockers/Questions (If Any)

**Open issues requiring resolution**:

```markdown
## Blockers/Questions

- **Performance**: Token refresh taking 500ms (investigate database query)
- **Security**: Need clarification on acceptable token TTL (30 min vs 1 hour?)
- **Testing**: Integration test environment not configured (requires Docker setup)
```

### 6. Context for Next Session

**Critical background information**:

```markdown
## Context for Next Session

**Decisions Made**:
- Using JWT (not session cookies) for stateless auth
- Token expiration: 1 hour (access), 7 days (refresh)
- CORS enabled for frontend domain only

**Key Files**:
- `auth/tokens.py` - Token generation/validation logic
- `auth/middleware.py` - Request authentication middleware
- `tests/test_auth.py` - Unit test suite

**References**:
- JWT spec: RFC 7519
- Security best practices: OWASP Auth Cheat Sheet
- Performance benchmark: auth/BENCHMARKS.md
```

---

## When to Create Handoffs

### Always Create:

- ✅ **End of work session** (before closing Claude Code)
- ✅ **Major milestone completion** (feature done, phase complete)
- ✅ **Context switch** (switching to different project/task)
- ✅ **Before significant breaks** (lunch, overnight, weekend)
- ✅ **Complex work** (>2 hours, multiple files, research phase)

### Optional:

- Short sessions (<30 min, simple changes)
- Emergency hotfixes (if clearly documented in git commit)
- Pure documentation updates (if self-explanatory)

---

## Examples from Different Domains

### Example 1: Web Application Development

```markdown
# Session Handoff: User Dashboard Implementation

**Date**: 2025-10-27
**Time**: 14:30:00
**Branch**: feat/user-dashboard

## Completed Work
- ✅ Created Dashboard component (src/components/Dashboard.vue)
- ✅ Integrated API calls for user statistics
- ✅ Added responsive layout (mobile + desktop)

## Next Priorities
1. Add data visualization charts (Chart.js integration)
2. Implement real-time updates (WebSocket connection)
```

### Example 2: Data Pipeline Development

```markdown
# Session Handoff: ETL Pipeline Optimization

**Date**: 2025-10-27
**Time**: 09:15:00
**Branch**: perf/etl-optimization

## Completed Work
- ✅ Profiled pipeline bottlenecks (processing: 45 min → 12 min)
- ✅ Implemented batch processing (chunk size: 10,000 records)
- ✅ Added progress tracking with logging

## Next Priorities
1. Optimize database writes (bulk insert instead of individual)
2. Add error recovery mechanism (checkpoint/resume)
```

### Example 3: Machine Learning Model

```markdown
# Session Handoff: Model Training Infrastructure

**Date**: 2025-10-27
**Time**: 16:45:00
**Branch**: feat/training-pipeline

## Completed Work
- ✅ Set up training pipeline (PyTorch Lightning)
- ✅ Configured hyperparameter sweep (Optuna integration)
- ✅ Added model checkpointing (every 10 epochs)

## Next Priorities
1. Run full hyperparameter sweep (overnight job)
2. Evaluate best model on test set
3. Export final model for deployment
```

---

## Benefits

**Time Savings**:
- 5-15 minutes to create handoff
- 20-45 minutes saved on next session (no re-discovery)
- Net gain: 15-30 minutes per session

**Context Preservation**:
- Decisions documented (why choices were made)
- Blockers tracked (no forgotten issues)
- Files catalogued (quick navigation)

**Collaboration**:
- Team members can pick up work
- Code reviews have context
- Project continuity maintained

---

## Summary

**Structure**: Timestamped markdown files with 6 sections

**Naming**: YYYY-MM-DD-HHMM-description.md

**Location**: session-handoffs/ directory

**When**: End of session, major milestones, context switches

**Benefits**: Time savings, context preservation, collaboration

**Universal**: Applies to any programming language, framework, or project type.
