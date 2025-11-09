# Logical Milestone Detection

Guide for recognizing when a logical milestone has been reached (and progress should be committed/logged).

## What is a Logical Milestone?

A **logical milestone** is a meaningful unit of work completion that should trigger:
1. A git commit (micro-commit discipline)
2. A progress note update (task-master update-subtask)
3. Potentially a checkpoint (every 5 subtasks)

Milestones are NOT based on arbitrary line counts (not "50 lines changed"). They're based on **feature completion**.

## Milestone Examples by Task Type

### Infrastructure Setup (Task 1.1 - PostgreSQL)

**Milestones**:
1. ✅ PostgreSQL 16 installed and running
   - System-level: binary installed, service started
   - Git commit: "feat(db): Install PostgreSQL 16"
   - Progress: "PostgreSQL installed and verified"

2. ✅ pgvector extension loaded
   - PostgreSQL extension compiled and loaded
   - Git commit: "feat(db): Install and load pgvector extension"
   - Progress: "pgvector ready, tested CREATE EXTENSION"

3. ✅ Test connection working
   - Connection pooling verified
   - Git commit: "feat(db): Verify connection pooling"
   - Progress: "Connection pooling tested, latency < 100ms"

### Code Implementation (Task 2.1 - Document Parser)

**Milestones**:
1. ✅ File reader implemented
   - Markdown file reading functional
   - Handles common cases (simple .md files)
   - Git commit: "feat(parsing): Implement markdown file reader"
   - Progress: "Markdown reader working, tested with 5 sample files"

2. ✅ Metadata extraction working
   - Frontmatter parsing functional
   - Path-based metadata extraction complete
   - Git commit: "feat(parsing): Add metadata extraction from frontmatter"
   - Progress: "Metadata extraction complete, 100% parse rate"

3. ✅ Tokenization system complete
   - Tiktoken integration working
   - Token counting accurate
   - Git commit: "feat(parsing): Add tiktoken-based tokenization"
   - Progress: "Tokenization system ready, 99.8% accuracy"

4. ✅ Chunking algorithm implemented
   - 512-token chunks with 20% overlap
   - Edge cases handled
   - Git commit: "feat(parsing): Implement 512-token chunking"
   - Progress: "Chunking algorithm complete, tested with edge cases"

### ML Model Integration (Task 3.1 - Embeddings)

**Milestones**:
1. ✅ Model loading complete
   - sentence-transformers model loads successfully
   - Caching strategy implemented
   - Git commit: "feat(embeddings): Set up model loading with caching"
   - Progress: "Model loads in < 100ms (cached), first load ~30s"

2. ✅ Batch processing working
   - Parallel embedding generation functional
   - Performance verified (e.g., "1000 embeddings in 15s")
   - Git commit: "feat(embeddings): Implement batch embedding generation"
   - Progress: "Batch processing complete, throughput: 1000 vectors/15s"

3. ✅ Database insertion complete
   - Embeddings inserted into database
   - HNSW index created
   - Git commit: "feat(embeddings): Implement database insertion with HNSW"
   - Progress: "Database integration complete, index created"

### Testing (Task 8.3 - Golden Query Set)

**Milestones**:
1. ✅ Query extraction done
   - 50 representative queries extracted
   - Documented selection criteria
   - Git commit: "test(golden): Extract 50 queries from production"
   - Progress: "Query extraction complete, 50 queries validated"

2. ✅ Annotation complete
   - Expected results manually annotated for all 50
   - Ground truth documented
   - Git commit: "test(golden): Add manual annotations for golden queries"
   - Progress: "All 50 queries annotated, ready for A/B testing"

3. ✅ Framework ready
   - Golden query set JSON created
   - A/B testing framework implemented
   - Git commit: "test(golden): Create golden query test framework"
   - Progress: "Framework ready, can run A/B comparison"

## Milestone Duration

**Expected timing between milestones**: 15-45 minutes

| Duration | Action |
|----------|--------|
| < 10 min | Might be too fine-grained (combine with next step?) |
| 15-45 min | ✅ Perfect milestone size |
| 45-90 min | Still acceptable (but could split into 2 milestones) |
| > 90 min | Too long without checkpoint - split into smaller milestones |

## Recognizing Milestone Completion

Ask yourself these questions:

**1. Is a feature/component complete?**
- ✅ PostgreSQL installed → YES (component done)
- ❌ "Added 30 lines to extractor" → NO (partial feature)
- ✅ Schema created and indexed → YES (component done)

**2. Can the next person pick up from here?**
- ✅ "PostgreSQL running, connection pooling verified" → YES
- ❌ "Partially implemented parser" → NO
- ✅ "Tokenization system complete, tested" → YES

**3. Is it a testable checkpoint?**
- ✅ "Model loads and generates embeddings for 1000 vectors" → YES (can verify)
- ❌ "Started implementing embeddings" → NO (can't verify)
- ✅ "Tests passing for file parser" → YES (can verify)

**4. Is this a logical stopping point?**
- ✅ "Config system working with env variable support" → YES
- ❌ "Half-wrote error handling" → NO
- ✅ "Dependency validation implemented and tested" → YES

If answers are mostly YES → **Milestone reached → Commit & log progress**

## Anti-Patterns: What's NOT a Milestone

❌ **Line-based**: "50 lines changed" - Arbitrary and meaningless

❌ **Time-based**: "30 minutes elapsed" - Time is a reminder to commit, not the milestone

❌ **Partial feature**: "Function signature written" - Incomplete feature

❌ **Unverified work**: "Code written but not tested" - Needs validation

❌ **Multiple features**: "Parser, chunking, and tokenization" - Too broad (should split)

## Examples of Bad vs Good Milestones

### ❌ Bad: "Document parser progress"
- Vague
- Unverifiable
- Could mean anything

**Better**: "Markdown file reader complete - handles frontmatter, tested with 10 files"

---

### ❌ Bad: "50 lines of code added to embedding system"
- Arbitrary line count
- Not feature-focused
- Could be incomplete

**Better**: "Sentence-transformers model loading complete - 100ms latency with caching"

---

### ❌ Bad: "30 minutes of work on search optimization"
- Time-based, not feature-based
- No measure of completion

**Better**: "HNSW parameter tuning complete - latency reduced from 120ms to 45ms"

---

### ✅ Good: "PostgreSQL 16 installed and pgvector extension loaded - verified with CREATE EXTENSION test"
- Specific and verifiable
- Feature complete
- Clear next step (schema creation)

---

### ✅ Good: "Tokenization system complete - tested on 100 documents, 99.8% accuracy"
- Measurable
- Feature complete
- Ready to build on

---

### ✅ Good: "Golden query set created and annotated - 50 queries with expected results, ready for A/B testing"
- Specific deliverable
- Verifiable
- Next step is clear

## Milestone Template

When you think you've reached a milestone, use this template to verify:

```
## Milestone: [Feature/Component Name]

**Completion Criteria**:
- [Specific, measurable completion indicator]
- [Verification method]
- [Next step ready?]

**Time Elapsed**: ~X minutes

**Testing/Verification**:
- [What was tested?]
- [Test results?]

**Commit Message**:
```
type(scope): [feature complete]

Milestone achievement: [what works now]
Verified: [how it was verified]
```

**Progress Note**:
[Summary for task-master update-subtask]
```

## Milestone Recognition in Real-Time

**During work, watch for these signals**:

🎯 **Signals a milestone is reached**:
- ✅ Feature works end-to-end
- ✅ Tests pass for this component
- ✅ Documentation updated
- ✅ ~30 minutes elapsed since last commit
- ✅ Clear logical stopping point
- ✅ "Next task is ready to start"

⚠️ **Signals you should keep working**:
- ❌ Feature is partially working
- ❌ Tests failing
- ❌ Missing error handling
- ❌ Only 5 minutes elapsed
- ❌ "Need to finish X before this is useful"

## Checkpoint Triggers Within Milestones

If a logical milestone takes > 90 minutes:

**Example**: "Database schema creation"
- Starts: 10:00 AM
- 30 min: Schema file executed → Milestone 1 (commit)
- 60 min: Indexes created → Milestone 2 (commit)
- 90 min: All constraints added and verified → Milestone 3 (commit)

Split into **multiple smaller milestones** rather than one huge 90-minute chunk.

## Decision Tree: Is This a Milestone?

```
┌─ Does this feature/component work end-to-end?
│  ├─ NO  → Keep working
│  └─ YES ↓
├─ Have you tested it?
│  ├─ NO  → Test it, then commit
│  └─ YES ↓
├─ Has ~30 min+ elapsed since last commit?
│  ├─ YES ↓
│  └─ NO  → Could wait (but OK to commit if feature done)
├─ Is there a clear next step?
│  ├─ YES ↓
│  └─ NO  → Finish this feature completely first
└─ Commit this milestone? YES ✅
```

## Summary

- **Milestone** = Verifiable feature/component completion
- **Duration** = 15-45 minutes (max 90 before checkpoint)
- **Trigger** = Feature works end-to-end + tests pass
- **Action** = git commit + task-master update-subtask
- **Not** = Arbitrary line counts or time thresholds

When in doubt: **Commit at logical stopping points, not arbitrary times.**
