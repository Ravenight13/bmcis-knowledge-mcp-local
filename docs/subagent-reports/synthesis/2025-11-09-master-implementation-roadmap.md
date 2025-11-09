# Master Implementation Roadmap - Task 7 Phase 1 Fixes

**Date**: 2025-11-09
**Status**: All 8 issues planned and ready for execution
**Planning Complete**: 4 subagents delivered detailed task breakdowns

---

## Executive Summary

**4 Planning Documents Created**:
1. Blocker 1: Schema/Query Mismatch (1,169 lines) ✅
2. Blockers 2-3: Constraint Tests + Repository Integration (1,671 lines) ✅
3. High Priority 4-5, 7: Indexes, Pooling, Enums (1,400+ lines) ✅
4. High Priority 6, 8: Dependency Injection + Stress Tests (1,200+ lines) ✅

**Total Planning Effort Delivered**: 16+ hours of detailed task breakdowns
**Implementation Effort Required**: 27-35 hours
**Parallelization Opportunity**: 40-50% time reduction (to ~15-20 hours with 2-3 parallel developers)

---

## 🎯 Execution Order (Dependency Graph)

### **CRITICAL PATH** (Must complete sequentially)

```
┌─────────────────────────────────────────────────────────────────┐
│ BLOCKER 1: Schema/Query Mismatch (3.5-4.5 hours)               │
│ - Task 1.1: Audit mismatches (30-45 min)                       │
│ - Task 1.2: Fix queries (1-1.5 hours)                          │
│ - Task 1.3: Validate ORM (30-45 min)                           │
│ - Task 1.4: Create tests (1-1.5 hours)                         │
│ Status: FOUNDATION FOR ALL OTHER FIXES                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        (All other fixes depend on Blocker 1)
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │   BLOCKER 2: Constraint Tests (5-6 hours)               │
    │   - Task 2.1: ORM model tests (2-3 hours)              │
    │   - Task 2.2: PostgreSQL schema tests (3-4 hours)      │
    │   Status: TESTS VALIDATE DATA INTEGRITY                │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │   BLOCKER 3: Repository Integration (3-4 hours)        │
    │   - Task 2.3: Wire repository (3-4 hours)              │
    │   - Task 2.4: Integration tests (3-4 hours)            │
    │   Status: SERVICE LAYER FUNCTIONAL                     │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │   HIGH PRIORITY 6: Dependency Injection (3.75 hours)   │
    │   (Builds on Blocker 3)                                 │
    │   - Define cache protocol (45 min)                      │
    │   - Update service (30 min)                             │
    │   - Create mocks (1 hour)                              │
    │   - Write DI tests (1.5 hours)                         │
    └─────────────────────────────────────────────────────────┘
```

### **PARALLEL TRACKS** (Can run simultaneously after Blocker 1)

```
Track A: Performance & Security (6-7 hours, after Blocker 1)
├─ HIGH PRIORITY 4: Add Indexes (1-2 hours)
├─ HIGH PRIORITY 5: Connection Pooling (0 hours - ALREADY DONE!)
└─ HIGH PRIORITY 7: Enum Validation (2-3 hours)

Track B: Testing & Validation (5 hours, after Blocker 1)
├─ BLOCKER 2: Constraint Tests (5-6 hours)
└─ BLOCKER 3: Repository Integration (3-4 hours, starts after Blocker 2)

Track C: Architecture & Resilience (5-6 hours, after Blocker 3)
├─ HIGH PRIORITY 6: Dependency Injection (3.75 hours)
└─ HIGH PRIORITY 8: Stress Tests (5 hours, can start immediately)
```

---

## 📊 Timeline & Effort Analysis

### **Sequential Execution** (1 developer)
| Phase | Blocker/HP | Hours | Duration |
|-------|-----------|-------|----------|
| **1** | Blocker 1 | 3.5-4.5 | 1 day |
| **2** | Blocker 2 | 5-6 | 1.5 days |
| **3** | Blocker 3 | 6-8 | 2 days |
| **4** | HP 4,7 | 3-5 | 1 day |
| **5** | HP 6,8 | 8-10 | 2 days |
| **Total** | **All** | **27-35** | **7.5 days** |

### **Parallel Execution** (2 developers, optimal strategy)
| Timeline | Blocker 1 | Track A/B | Track C | Cumulative |
|----------|-----------|-----------|---------|-----------|
| Day 1 | Blocker 1 (4h) | — | — | 4h |
| Day 2-3 | — | Blockers 2-3 (9h) | HP 8 (2h) | 15h |
| Day 4 | — | HP 4,7 (4h) | HP 6 (2h) | 21h |
| **Total** | **4h** | **13h** | **4h** | **~3-4 days** |

### **Aggressive Parallel Execution** (3 developers)
| Timeline | Dev 1 | Dev 2 | Dev 3 | Cumulative |
|----------|-------|-------|-------|-----------|
| Day 1 | Blocker 1 (4h) | — | — | 4h |
| Day 2-3 | Blocker 2 (5.5h) | HP 4,7 (3h) | HP 8 (2h) | 14.5h |
| Day 4 | Blocker 3 (4h) | — | HP 6 (2h) | 20.5h |
| **Total** | **13.5h** | **3h** | **4h** | **~2-3 days** |

---

## 📋 Prioritized Task Queue

### **Immediate (MUST DO FIRST)**

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 1: Blocker 1 - Schema/Query Mismatch                   │
│ Status: Foundation - blocks everything                      │
│ Effort: 3.5-4.5 hours (1 developer, 1 day)                 │
│ Subtasks:                                                    │
│  • Task 1.1: Audit mismatches (30-45 min)                  │
│  • Task 1.2: Fix 5 query methods (1-1.5h)                  │
│  • Task 1.3: Validate ORM models (30-45 min)               │
│  • Task 1.4: Create 15 alignment tests (1-1.5h)            │
│ Success Criteria:                                            │
│  • All column references match schema                        │
│  • 15 alignment tests pass (prevent regressions)            │
│  • mypy validation passes                                    │
│ Blocker Removal: CRITICAL ✓                                │
└─────────────────────────────────────────────────────────────┘
```

### **High Priority (Unlock Other Fixes)**

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 2: Blocker 2 - Constraint Tests (0% → 100%)            │
│ Depends On: Blocker 1 (schema correct)                      │
│ Status: Data integrity validator                            │
│ Effort: 5-6 hours (1 developer, 1.5 days)                  │
│ Subtasks:                                                    │
│  • Task 2.1: ORM constraint tests (2-3h, 12-15 tests)      │
│  • Task 2.2: PostgreSQL tests (3-4h, 13-15 tests)          │
│ Success Criteria:                                            │
│  • 100% constraint coverage (25-30 tests)                   │
│  • Invalid data rejected at both layers                      │
│  • All tests pass against real PostgreSQL                   │
│ Blocker Removal: CRITICAL ✓                                │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 3: Blocker 3 - Repository Integration                  │
│ Depends On: Blocker 2 (tests pass)                          │
│ Status: Service layer functional                            │
│ Effort: 6-8 hours (1 developer, 2 days)                    │
│ Subtasks:                                                    │
│  • Task 2.3: Wire repository (3-4h)                         │
│  • Task 2.4: Integration tests (3-4h, 16-20 tests)         │
│ Success Criteria:                                            │
│  • Service methods fully implemented (not stubs)            │
│  • Cache hit/miss logic working                            │
│  • Cascade invalidation functional                          │
│  • 40-50 integration tests passing                         │
│ Blocker Removal: CRITICAL ✓                                │
└─────────────────────────────────────────────────────────────┘
```

### **Performance Optimization (Parallel to Blockers 2-3)**

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 4: High Priority 4 - Add Indexes (60-73% faster)      │
│ Depends On: Blocker 1 (schema correct)                      │
│ Status: Performance optimization                            │
│ Effort: 1-2 hours (1 developer, <1 day)                    │
│ What's Included:                                             │
│  • 4 composite indexes (source_confidence, entity_type, etc)|
│  • 1 new Alembic migration                                  │
│  • EXPLAIN ANALYZE before/after                             │
│ Performance Gains:                                           │
│  • 1-hop: 8-12ms → 3-5ms (60-70% faster)                   │
│  • 2-hop: 30-50ms → 15-25ms (50% faster)                   │
│  • Type filter: 18.5ms → 2.5ms (86% faster)                │
│ Success Criteria:                                            │
│  • All 4 indexes created                                    │
│  • Query plans use indexes                                  │
│  • P95 latency targets achieved                            │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 5: High Priority 5 - Connection Pooling               │
│ Status: ALREADY DONE ✅ (in src/core/database.py)          │
│ Effort: 0 hours (no work needed!)                          │
│ Current State:                                               │
│  • SimpleConnectionPool implemented                         │
│  • Pool size: 10, overflow: 20                             │
│  • Health checks & retry logic in place                    │
│ No Action Required - Proceed to other tasks!               │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 6: High Priority 7 - Enum Validation                   │
│ Depends On: Blocker 1 (schema correct)                      │
│ Status: Security & data integrity                           │
│ Effort: 2-3 hours (1 developer, 1 day)                     │
│ What's Included:                                             │
│  • PostgreSQL enum types (entity_type, relationship_type)  │
│  • ORM Pydantic validators                                 │
│  • 1 new Alembic migration                                  │
│  • Validation tests (15+ tests)                            │
│ Security Benefits:                                           │
│  • Prevents invalid entity types                           │
│  • Database-enforced constraints                           │
│  • ORM-level backup validation                             │
│ Success Criteria:                                            │
│  • Enum types created in PostgreSQL                        │
│  • ORM validators reject invalid types                     │
│  • 15+ validation tests passing                            │
└─────────────────────────────────────────────────────────────┘
```

### **Architecture & Resilience (Parallel to Blockers, after Blocker 3)**

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 7: High Priority 6 - Dependency Injection              │
│ Depends On: Blocker 3 (repository integrated)               │
│ Status: Architecture refactor (enables Redis migration)     │
│ Effort: 3.75 hours (1 developer, 1 day)                    │
│ What's Included:                                             │
│  • CacheProtocol definition (45 min)                       │
│  • Service constructor refactor (30 min)                    │
│  • Mock cache implementation (1 hour)                       │
│  • DI tests (1.5 hours, 8+ tests)                         │
│ Architecture Benefits:                                       │
│  • Future Redis migration (0 service code changes)          │
│  • Improved testability (inject mocks)                      │
│  • Dependency Inversion compliance                         │
│ Success Criteria:                                            │
│  • CacheProtocol defined with 8 methods                    │
│  • Service accepts cache as parameter                      │
│  • Backward compatibility maintained                        │
│  • 8+ DI tests passing                                     │
│  • mypy --strict compliant                                 │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│ TASK 8: High Priority 8 - Concurrent Stress Tests           │
│ Depends On: None (independent, can start immediately!)      │
│ Status: Thread-safety & resilience                          │
│ Effort: 5 hours (1 developer, 1.5 days)                    │
│ What's Included:                                             │
│  • 5 concurrent read scenarios (3 tests, 1 hour)           │
│  • 5 concurrent write scenarios (3 tests, 45 min)          │
│  • Mixed read/write contention (2 tests, 45 min)           │
│  • Bidirectional invalidation (2 tests, 30 min)            │
│  • LRU eviction under load (2 tests, 30 min)               │
│  • Load testing framework (3 tests, 1.5 hours)             │
│ Resilience Benefits:                                         │
│  • Validated thread-safety (100+ threads)                  │
│  • Race condition detection                                │
│  • Performance validation under load (>10k ops/sec)        │
│  • Cache hit rate verification (>80%)                      │
│ Success Criteria:                                            │
│  • All 15+ stress tests passing                            │
│  • P95 latency <2µs maintained under concurrency           │
│  • No deadlocks or cache corruption                        │
│  • Throughput >10k operations/second                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Recommended Execution Plan

### **Option A: Sequential (1 Developer, 7.5 Days)**
Best for: Single developer, low risk tolerance
```
Day 1:   Task 1 (Blocker 1) - 3.5-4.5h
Day 2-3: Task 2 (Blocker 2) - 5-6h
Day 4-5: Task 3 (Blocker 3) - 6-8h
Day 6:   Tasks 4,6 (HP 4,7) - 3-5h
Day 7:   Tasks 7,8 (HP 6, Stress) - 8-10h
```

### **Option B: Parallel (2 Developers, 3-4 Days) ⭐ RECOMMENDED**
Best for: Two developers, balanced parallelization
```
Dev 1 - Main Path:
  Day 1:   Task 1 (Blocker 1) - 4h
  Day 2-3: Task 2 (Blocker 2) - 5.5h
  Day 4:   Task 3 (Blocker 3) - 4h

Dev 2 - Parallel Path:
  Day 1:   Task 8 (HP 8 Stress) - 2h
  Day 2-3: Tasks 4,6 (HP 4,7) - 3h + Task 8 continued
  Day 4:   Task 7 (HP 6 DI) - 2h
```

### **Option C: Aggressive Parallel (3 Developers, 2-3 Days)**
Best for: Urgent delivery, multiple developers
```
Dev 1:   Blocker 1 (4h) → Blocker 2 (5.5h) → Blocker 3 (4h) = 13.5h
Dev 2:   HP 4 & 7 (3h) + buffer
Dev 3:   HP 8 (2h) → HP 6 (2h) + buffer
Total: ~2-3 days wall-clock
```

---

## ✅ Success Criteria by Phase

### **Blocker 1 Complete** ✓
- [ ] All 5 column name mismatches fixed
- [ ] 15 alignment tests created and passing
- [ ] All query patterns use correct schema columns
- [ ] ORM models validated against schema

### **Blocker 2 Complete** ✓
- [ ] 25-30 constraint tests created
- [ ] 100% ORM model constraint coverage
- [ ] 100% PostgreSQL schema constraint coverage
- [ ] No invalid data can enter database

### **Blocker 3 Complete** ✓
- [ ] KnowledgeGraphQueryRepository wired into service
- [ ] 5 core query methods fully implemented (not stubs)
- [ ] Cache hit/miss logic working
- [ ] Cascade invalidation on entity updates
- [ ] 40-50 integration tests passing

### **All High-Priority Issues Complete** ✓
- [ ] 4 composite indexes created (60-73% latency reduction)
- [ ] Enum validation enforced (PostgreSQL + ORM)
- [ ] Dependency injection protocol defined
- [ ] 15+ concurrent stress tests passing
- [ ] Cache performance validated (>80% hit rate)

---

## 📂 All Planning Documents

Located in: `/docs/subagent-reports/task-planning/`

1. **2025-11-09-blocker1-schema-mismatch-plan.md** (1,169 lines)
   - Complete audit of column mismatches
   - Task breakdown with exact file/line numbers
   - Test strategy for 15 alignment tests

2. **2025-11-09-blockers2-3-tests-integration-plan.md** (1,671 lines)
   - 40-50 test cases specified
   - Repository integration architecture
   - Cache invalidation strategy

3. **2025-11-09-highpriority4-5-7-optimizations-plan.md** (1,400+ lines)
   - 4 composite index specifications with EXPLAIN ANALYZE
   - **Discovery: Connection pooling already implemented!**
   - Enum validation two-layer strategy

4. **2025-11-09-highpriority6-8-di-stress-plan.md** (1,200+ lines)
   - CacheProtocol specification
   - 15+ concurrent stress test scenarios
   - Load testing framework

---

## 🎯 Next Actions

### **To Proceed with Implementation:**

1. **Choose execution option** (A, B, or C above)
2. **Assign tasks** to developer(s)
3. **Start with Task 1** (Blocker 1 - schema/query fix)
4. **Parallelize Tasks 4, 6, 8** while Task 1 completes
5. **Proceed to Blockers 2 & 3** after Task 1

### **To Request Implementation Help:**

Just ask to:
- "Spawn subagents to implement Task 1" (schema/query fix)
- "Implement all 8 fixes in parallel" (comprehensive)
- "Focus on Blocker 1 first" (critical path)

All planning is complete and implementation-ready. The 4 planning documents contain enough detail for engineering teams to execute without additional context.

---

**Status**: ✅ **READY FOR IMPLEMENTATION**
**Planning Complete**: 2025-11-09
**Estimated Delivery**: 2-4 days (depending on parallelization)
