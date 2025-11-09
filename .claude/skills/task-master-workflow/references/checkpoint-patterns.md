# Session Checkpoint Patterns

Create a session checkpoint every 5 subtasks completed. Checkpoints preserve session state for multi-session work.

## When to Create Checkpoint

**Trigger**: Every 5 subtasks marked `done`

Examples:
- After completing subtasks 1.1, 1.2, 1.3, 1.4, 1.5 → Checkpoint
- After completing subtasks 1.5, 1.6, 2.1, 2.2, 2.3 → Checkpoint
- After completing subtasks 2.4, 2.5, 3.1, 3.2, 3.3 → Checkpoint

**Count**: Track completed subtasks this session
```bash
# After marking subtasks done
task-master list | grep "Completed:"
# Example: "Completed: 5/50"

# If count = 5, 10, 15, 20, etc. → Create checkpoint
```

## Checkpoint Command

```bash
/uwo-checkpoint
```

This interactive command:
1. Displays session metadata (branch, time, context)
2. Summarizes completed work (subtasks done, commits, blockers)
3. Prompts for checkpoint notes
4. Auto-commits checkpoint file to git
5. Saves session state for next session

**Location**: Checkpoint file created in `session-handoffs/YYYY-MM-DD-HHMM-checkpoint.md`

## Manual Checkpoint Creation (If /uwo-checkpoint unavailable)

If slash command not available, manually create checkpoint:

```bash
# Create checkpoint file
cat > session-handoffs/2025-11-08-1430-checkpoint.md << 'EOF'
# Session Checkpoint - Task Master Development

**Date:** 2025-11-08
**Time:** 14:30
**Branch:** master

## Completed This Session

### Subtasks Done (5 total)
- ✅ 1.1 - PostgreSQL 16 + pgvector installation
- ✅ 1.2 - Database schema creation
- ✅ 1.3 - Pydantic configuration system
- ✅ 1.4 - Connection pooling setup
- ✅ 1.5 - Structured logging configuration

### Commits (6 total)
1. feat(db): Install PostgreSQL 16 with pgvector
2. feat(db): Execute schema from sql/schema_768.sql
3. feat(config): Pydantic settings system
4. feat(pooling): Connection pooling with psycopg2
5. feat(logging): Structured JSON logging
6. docs(task-master): Session checkpoint at 5/50 subtasks

## Next Priorities

1. **Immediate (Next Subtask)**
   - Task 1.6: Virtual environment setup with pytest, pre-commit hooks

2. **Short-term (This Session if Time)**
   - Task 2.1: Markdown file reader with metadata extraction
   - Task 2.2: Tiktoken tokenization system

3. **This Week**
   - Complete Task 2 (Document Parsing) fully
   - Begin Task 3 (Embedding Pipeline) setup

## Blockers & Challenges

None identified. Foundation layer progressing smoothly.

## Quality Gates Status

- ✅ All commits with clear conventional format
- ✅ All tests passing (PostgreSQL, schema, config)
- ✅ No uncommitted changes
- ✅ Progress notes updated per subtask

## Statistics

- **Subtasks Complete:** 5/50 (10%)
- **Task 1 Progress:** 5/6 subtasks (83%)
- **Commits Today:** 6
- **Session Duration:** ~120 minutes
- **Average Subtask Time:** ~20 minutes

---

Ready for next session. Continue with Task 1.6.
EOF

# Commit checkpoint
git add session-handoffs/2025-11-08-1430-checkpoint.md
git commit -m "checkpoint: 5 subtasks complete - foundation layer ready (1.1-1.5)"
```

## Checkpoint Format Reference

Checkpoints should include:

```markdown
# Session Checkpoint - {Project Name}

**Date:** YYYY-MM-DD
**Time:** HH:MM
**Branch:** {branch}
**Duration:** ~X minutes

## Completed This Session

### Subtasks Done (X total)
- ✅ Task ID - Description
- ✅ Task ID - Description
...

### Commits (Y total)
1. conventional(scope): message
2. conventional(scope): message
...

## Next Priorities

1. **Immediate (Next Subtask)**
   - Task X.Y: Description

2. **Short-term (This Session if Time)**
   - Task X.Y: Description
   - Task X.Y: Description

3. **Longer-term (Next Session)**
   - Task X: Description

## Blockers & Challenges

- [Description of any blockers]
- None if no blockers

## Quality Gates Status

- ✅ Code committed
- ✅ Tests passing
- ✅ Progress notes updated
- [Any other quality checks]

## Statistics

- **Subtasks Complete:** X/Y
- **Task Progress:** X% of current task
- **Commits Today:** N
- **Session Duration:** ~ X hours/minutes

---

Ready for next session. Continue with Task X.Y.
```

## Examples by Session Type

### Example 1: Foundation Phase Checkpoint (After Task 1)

```markdown
# Session Checkpoint - bmcis-knowledge-mcp-local

**Date:** 2025-11-08
**Time:** 16:45
**Branch:** master

## Completed This Session

### Subtasks Done (6 total)
- ✅ 1.1 - PostgreSQL 16 + pgvector installation
- ✅ 1.2 - Database schema creation
- ✅ 1.3 - Pydantic configuration system
- ✅ 1.4 - Connection pooling setup
- ✅ 1.5 - Structured logging configuration
- ✅ 1.6 - Virtual environment + pytest + pre-commit

### Commits (7 total)
All commits merged, 6 logical milestones plus final task completion

## Next Priorities

1. **Immediate**
   - Task 2.1: Markdown file reader

2. **This Week**
   - Complete Task 2 (Document Parsing System)
   - Begin Task 3 (Embedding Pipeline)

## Blockers & Challenges

None - Foundation layer complete and validated

## Quality Gates Status

- ✅ All 6/6 subtasks of Task 1 done
- ✅ All tests passing
- ✅ PostgreSQL + pgvector verified
- ✅ Connection pooling tested
- ✅ All commits have clear messages

## Statistics

- **Task 1 Complete:** 6/6 (100%)
- **Overall Progress:** 6/50 subtasks (12%)
- **Commits This Session:** 7
- **Session Duration:** ~180 minutes

---

Foundation phase complete. Ready to begin Phase 1 (Document Parsing & Embedding).
```

### Example 2: Mid-Project Checkpoint (During Task 3-4)

```markdown
# Session Checkpoint - bmcis-knowledge-mcp-local

**Date:** 2025-11-09
**Time:** 17:30
**Branch:** master

## Completed This Session

### Subtasks Done (5 total)
- ✅ 2.1 - Markdown file reader
- ✅ 2.2 - Tiktoken tokenization
- ✅ 2.3 - 512-token chunking with overlap
- ✅ 2.4 - Context header generation
- ✅ 3.1 - Sentence-transformers model loading

### Commits (5 total)
1. feat(parsing): Markdown file reader with metadata
2. feat(parsing): Tiktoken-based tokenization
3. feat(parsing): Token chunking (512-token, 20% overlap)
4. feat(parsing): Context header generation from document structure
5. feat(embeddings): Sentence-transformers model loading

## Next Priorities

1. **Immediate (Next Subtask)**
   - Task 3.2: Parallel embedding generation with batch processing

2. **Short-term (This Session if Time)**
   - Task 3.3: Database insertion with HNSW index
   - Task 3.4: Embedding validation and quality checks

## Blockers & Challenges

- Tiktoken model loading slow (~30s first run). Implemented caching - now <100ms subsequent runs.

## Quality Gates Status

- ✅ Document parsing tests passing (10 sample files)
- ✅ Tokenization accuracy: 99.8% (vs expected 100%)
- ✅ Chunking algorithm verified with edge cases
- ✅ All progress notes updated

## Statistics

- **Task 2 Complete:** 4/5 subtasks (80%)
- **Task 3 Progress:** 1/4 subtasks (25%)
- **Overall Progress:** 11/50 subtasks (22%)
- **Commits This Session:** 5
- **Session Duration:** ~150 minutes
- **Average Subtask Time:** ~30 minutes

---

Document parsing nearly complete. Task 3 (Embeddings) in progress. No blockers identified.
```

## Checkpoint Frequency Guidelines

| Metric | Threshold | Action |
|--------|-----------|--------|
| Subtasks Completed | 5 | Create checkpoint |
| Time Elapsed | 150+ minutes | Consider checkpoint |
| Blocker Found | Any | Create checkpoint before hand-off |
| Session End | Always | Create checkpoint if any work done |
| Branch Change | Any | Create checkpoint before switching |

## Multi-Session Recovery

To resume after checkpoint:

1. **Read latest checkpoint:**
   ```bash
   cat session-handoffs/2025-11-08-1430-checkpoint.md
   ```

2. **Review next task:**
   ```bash
   task-master show 1.6  # Or whatever task checkpoint recommends
   ```

3. **Check git status:**
   ```bash
   git log --oneline | head -5
   ```

4. **Resume work:**
   ```bash
   task-master set-status --id=1.6 --status=in-progress
   ```

Checkpoints should make session resumption seamless - you should be able to pick up exactly where you left off.
