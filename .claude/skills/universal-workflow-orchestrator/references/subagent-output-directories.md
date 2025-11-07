# Subagent Output Directory Structure Guide

**Category**: Workflow Organization
**Token Budget**: ~600 tokens

---

## Overview

This guide specifies WHERE subagents should write their output files to ensure consistent organization and discoverability across all projects.

**Golden Rule**: Every subagent output MUST be written to a file and committed before the subagent reports back to main chat.

---

## Standard Directory Structure

```
project-root/
├── docs/
│   ├── subagent-reports/           ← PRIMARY location for subagent outputs
│   │   ├── {agent-type}/           ← Group by agent type
│   │   │   └── {component}/        ← Group by component/feature
│   │   │       └── YYYY-MM-DD-HHMM-{description}.md
│   │   │
│   │   ├── api-analysis/
│   │   │   ├── auth/
│   │   │   ├── payments/
│   │   │   └── user-data/
│   │   │
│   │   ├── architecture-review/
│   │   │   ├── database/
│   │   │   ├── frontend/
│   │   │   └── backend/
│   │   │
│   │   ├── security-analysis/
│   │   ├── performance-analysis/
│   │   ├── code-review/
│   │   ├── testing-strategy/
│   │   └── documentation-review/
│   │
│   └── analysis/                    ← Alternative for standalone analysis
│       └── YYYY-MM-DD-HHMM-{description}.md
│
└── session-handoffs/                ← For session continuity documents
    └── YYYY-MM-DD-HHMM-{description}.md
```

---

## Agent Type Directories

### Research & Analysis Agents

**Location**: `docs/subagent-reports/{agent-type}/{component}/`

**Agent Types**:
- `api-analysis/` - API endpoint analysis, REST/GraphQL reviews
- `architecture-review/` - System architecture, design patterns, scalability
- `security-analysis/` - Security audits, vulnerability assessments
- `performance-analysis/` - Performance profiling, bottleneck identification
- `code-review/` - Code quality, maintainability, best practices
- `testing-strategy/` - Test coverage, testing approaches, QA analysis
- `documentation-review/` - Documentation quality, completeness, accuracy
- `data-analysis/` - Data structure, schema, ETL pipeline analysis
- `framework-research/` - Technology evaluation, library comparison
- `vendor-research/` - Third-party tool evaluation (domain-specific example)

**Example**:
```
docs/subagent-reports/api-analysis/auth/2025-10-27-1400-authentication-security.md
```

### Component Grouping

**Purpose**: Further organize by component/feature/module within agent type

**Examples**:
- `api-analysis/auth/` - Authentication endpoint analysis
- `api-analysis/payments/` - Payment endpoint analysis
- `architecture-review/database/` - Database architecture
- `architecture-review/frontend/` - Frontend architecture
- `security-analysis/endpoints/` - Endpoint security
- `security-analysis/dependencies/` - Dependency vulnerabilities

---

## File Naming Convention

**Format**: `YYYY-MM-DD-HHMM-{description}.md`

**Examples**:
- `2025-10-27-1400-authentication-security-audit.md`
- `2025-10-27-1430-database-schema-analysis.md`
- `2025-10-27-1500-payment-endpoint-review.md`

**Why timestamps**:
- Chronological ordering
- Multiple analyses per day (hours differentiate)
- Easy discovery of most recent analysis

---

## Decision Matrix: Where to Write Output

| Output Type | Location | Example |
|-------------|----------|---------|
| **API Analysis** | `docs/subagent-reports/api-analysis/{endpoint}/` | `.../auth/2025-10-27-1400-auth-analysis.md` |
| **Architecture Review** | `docs/subagent-reports/architecture-review/{component}/` | `.../database/2025-10-27-1430-schema-review.md` |
| **Security Audit** | `docs/subagent-reports/security-analysis/{scope}/` | `.../endpoints/2025-10-27-1500-security-audit.md` |
| **Performance Analysis** | `docs/subagent-reports/performance-analysis/{component}/` | `.../queries/2025-10-27-1600-slow-query-analysis.md` |
| **Code Review** | `docs/subagent-reports/code-review/{module}/` | `.../utils/2025-10-27-1700-code-quality-review.md` |
| **Documentation Review** | `docs/subagent-reports/documentation-review/{area}/` | `.../api-docs/2025-10-27-1800-doc-completeness.md` |
| **Session Handoff** | `session-handoffs/` | `2025-10-27-1900-feature-implementation-handoff.md` |
| **Standalone Analysis** | `docs/analysis/` | `2025-10-27-2000-system-wide-analysis.md` |

---

## Multi-Domain Examples

### Web Development Project

**API Security Analysis**:
```
docs/subagent-reports/security-analysis/
├── auth/2025-10-27-1400-auth-security.md
├── user-data/2025-10-27-1430-data-privacy.md
└── payments/2025-10-27-1500-payment-security.md
```

**Frontend Performance Analysis**:
```
docs/subagent-reports/performance-analysis/
├── components/2025-10-27-1600-component-render-perf.md
├── state-management/2025-10-27-1630-state-optimization.md
└── api-calls/2025-10-27-1700-api-request-optimization.md
```

### Data Science Project

**Data Pipeline Analysis**:
```
docs/subagent-reports/data-analysis/
├── ingestion/2025-10-27-1400-data-ingestion-review.md
├── transformation/2025-10-27-1430-etl-logic-analysis.md
└── validation/2025-10-27-1500-data-quality-checks.md
```

**Model Architecture Review**:
```
docs/subagent-reports/architecture-review/
├── feature-engineering/2025-10-27-1600-feature-analysis.md
├── model-selection/2025-10-27-1630-model-comparison.md
└── hyperparameters/2025-10-27-1700-tuning-strategy.md
```

### DevOps/Infrastructure Project

**Infrastructure Security Analysis**:
```
docs/subagent-reports/security-analysis/
├── network/2025-10-27-1400-network-security.md
├── access-control/2025-10-27-1430-iam-review.md
└── secrets/2025-10-27-1500-secrets-management.md
```

**Deployment Architecture Review**:
```
docs/subagent-reports/architecture-review/
├── kubernetes/2025-10-27-1600-k8s-config-review.md
├── terraform/2025-10-27-1630-iac-analysis.md
└── cicd/2025-10-27-1700-pipeline-optimization.md
```

---

## Creating Directories

**Before first subagent run** (in session initialization):

```bash
# Create base directory structure
mkdir -p docs/subagent-reports
mkdir -p docs/analysis
mkdir -p session-handoffs

# Create common agent-type directories
mkdir -p docs/subagent-reports/api-analysis
mkdir -p docs/subagent-reports/architecture-review
mkdir -p docs/subagent-reports/security-analysis
mkdir -p docs/subagent-reports/performance-analysis
mkdir -p docs/subagent-reports/code-review
```

**During subagent execution** (subagent creates specific component directory):

```bash
# Subagent creates component directory if needed
mkdir -p docs/subagent-reports/api-analysis/auth
```

---

## Integration with Session Handoffs

**Session handoff should reference subagent outputs**:

```markdown
## Subagent Results

### Subagent 1: API Security Analysis (Auth Endpoint)

**Output File:** `docs/subagent-reports/security-analysis/auth/2025-10-27-1400-auth-security-audit.md`

**Key Findings:**
- Critical: Missing token expiry validation
- Medium: Rate limiting too permissive
- Low: Request logging missing

**Recommendation:** See full report for implementation details
```

---

## Discoverability

**Finding recent subagent work**:

```bash
# Most recent subagent reports (any type)
ls -lt docs/subagent-reports/**/*.md | head -10

# Most recent security analysis
ls -lt docs/subagent-reports/security-analysis/**/*.md | head -5

# All reports for specific component
ls -l docs/subagent-reports/*/auth/*.md
```

**Git log of subagent work**:

```bash
# All subagent commits
git log --grep="by subagent" --oneline

# Recent subagent work on security
git log --grep="security.*subagent" --oneline
```

---

## Customization for Project-Specific Needs

**Add custom agent types**:

```bash
# Data science project might add:
mkdir -p docs/subagent-reports/model-analysis
mkdir -p docs/subagent-reports/experiment-review

# Mobile development project might add:
mkdir -p docs/subagent-reports/ui-review
mkdir -p docs/subagent-reports/performance-profiling

# Game development project might add:
mkdir -p docs/subagent-reports/gameplay-analysis
mkdir -p docs/subagent-reports/asset-review
```

---

## Summary

**Three-Level Organization**:
1. **Agent Type** - What kind of analysis (api-analysis, security-analysis, etc.)
2. **Component** - What part of system (auth, database, payments, etc.)
3. **Timestamped File** - Specific analysis (YYYY-MM-DD-HHMM-description.md)

**Benefits**:
- ✅ Consistent organization across all subagent work
- ✅ Easy discovery of relevant analysis
- ✅ Chronological tracking via timestamps
- ✅ Survives session crashes (committed files)
- ✅ Enables async review and handoffs

**Remember**: Every subagent MUST write output file + commit before reporting back!
