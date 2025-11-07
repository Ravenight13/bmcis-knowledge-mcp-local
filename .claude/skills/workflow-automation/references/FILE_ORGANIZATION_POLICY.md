# File Organization Policy

**Created**: 2025-10-01
**Status**: Constitutional (MANDATORY)
**Last Updated**: 2025-10-25 (added vendor documentation structure reference)

---

## Root Directory Rules (CONSTITUTIONAL)

**ONLY these files belong in root:**

1. `CLAUDE.md` - Operating rules (NEVER move)
2. `README.md` - Project overview (standard location)
3. `.project_status.md` - Current operational status
4. `.project_archive.md` - Historical archive
5. `.project_status_backup.md` - Status backup
6. `pyproject.toml`, `uv.lock` - Project configuration

**Allowed directories in root:**
- `backend/` - Backend code and tests
- `frontend/` - Frontend code
- `docs/` - All documentation
- `scripts/` - Utility scripts
- `sdk/` - SDK and client libraries (TypeScript, etc.)
- `outputs/` - Production outputs (gitignored)

**Rule**: If it's not on this list, it belongs somewhere else.

---

## docs/ Subdirectory Organization (CONSTITUTIONAL)

**Purpose-Based Categorization:**

### guides/
Process knowledge and operational procedures.
- HOW_TO guides (e.g., `HOW_TO_USE_PATTERNS.md`)
- Operational procedures (e.g., `operational-procedures.md`)
- Developer workflow documentation
- System architecture (ARCHITECTURE.md)
- Developer onboarding (ONBOARDING.md)

### status/
System state and operational status documentation.
- API development status
- Validation system operational guides
- Production health and monitoring documentation

### reference/
Static reference documentation organized by domain.
- **vendor-formats/** - Vendor-specific file format documentation
- **agent-debugging/** - Agent troubleshooting and debugging reference
- **constitutional/** - Governance and compliance documentation (includes FILE_ORGANIZATION_POLICY.md, token-budget-governance.md)
- **framework/** - Framework integration and architecture reference (cc-commands, research-subagent-system)
- **strategic/** - Strategic analysis and architectural deep-dives
- **testing/** - Testing methodologies and validation frameworks
- **diagrams/** - Architecture diagrams and visual documentation
- **manifest-pattern/** - Extraction manifest pattern specifications
- **operational/** - Operational procedures and deployment guides
- **session-grounding/** - Session context grounding documentation
- **speckit/** - SpecKit integration and usage documentation
- **subagent-policies/** - Subagent usage policies and file path validation
- **database-migration/** - Database schema and migration documentation

### checklists/
Quality assurance and maintenance workflows.
- Vendor implementation checklists
- Slash command maintenance checklists
- Testing and validation checklists

### archive/
Historical documentation organized by date and topic.
- Format: `YYYY-MM-DD-[topic]/`
- Example: `2025-09-28-test-artifacts/`
- See "Archival Triggers" section for archival policy

### implementation-plans/
Feature implementation specifications organized by type.
- **vendor/** - Vendor-specific implementation plans (LUTRON, TPD, etc.)
- **framework/** - Framework enhancement specifications (cc-ready, task-mcp, etc.)
- **api/** - API design and implementation plans

**Note**: Replaced deprecated `feature-plans/` folder - all planning documentation now unified here.

### session-prompts/
Session starter prompts for structured workflows.
- `vendor-implementation/` - Vendor-specific session prompts
- `framework-development/` - Framework and tooling session prompts
- `maintenance/` - Quality and performance session prompts

### session-summaries/
Session completion summaries and post-session documentation.
- Format: `YYYY-MM-DD-[topic]-summary.md`
- Contains session outcomes, decisions, and follow-up items
- Complements `context-sessions/` (active context) vs `session-summaries/` (completed sessions)

### vendors/
**⭐ NEW**: Vendor-specific documentation following standardized 7-category structure.

**See**: `docs/reference/VENDOR_DOCUMENTATION_STANDARDS.md` for complete vendor folder structure specification.

**Standard Structure** (per vendor):
```
docs/vendors/{VENDOR}/
├── README.md                           # ⭐ MANDATORY - Folder overview
├── CHANGELOG.md                        # ⭐ MANDATORY - Milestone tracking
├── research/                           # ⭐ MANDATORY - Phase 1 research
│   ├── README.md
│   └── YYYY-MM-DD-HHMM-vendor-format-research.md (1,000+ lines)
├── design/                             # Feature specifications
│   └── README.md
├── analysis/                           # Strategic analysis
│   └── README.md
├── debugging/                          # Issue tracking
│   ├── README.md
│   ├── archived/YYYY-MM/              # Time-based archival
│   └── QUICK_FIX_GUIDE.md             # Named reference docs
├── status-reports/                     # Milestone summaries
│   └── README.md
└── validation-reports/                 # Production validation
    └── README.md
```

**Key Requirements**:
- ✅ All new vendors MUST follow 7-category structure
- ✅ Root README.md + CHANGELOG.md are MANDATORY
- ✅ research/vendor-format-research.md MANDATORY (minimum 1,000 lines)
- ✅ README.md in each category folder
- ✅ Timestamp convention: `YYYY-MM-DD-HHMM-{description}.md`
- ✅ Archival strategy: debugging files >30 days → `archived/YYYY-MM/`
- ❌ Subagent reports NOT in vendor folder (see "Subagent Research Reports" section)

**Migration**: See `docs/reference/VENDOR_DOCUMENTATION_STANDARDS.md` for existing vendor migration guidelines (30-45 min per vendor).

**Rationale**: 60+ vendors require standardized structure for navigation, onboarding, and maintainability. Template proven through Sound United implementation (3+ weeks, 100+ files).

### Supporting Directories
- **api/** - API specifications (OpenAPI, WebSocket patterns)
- **context-sessions/** - Context management for research-first workflow
- **patterns/** - Pattern library documentation
- **subagent-reports/** - Research agent outputs (MUST use subdirectory structure)

---

## File Type Destinations

### Documentation (.md files)
- **Session summaries**: `docs/session-summaries/YYYY-MM-DD-*.md`
- **Implementation plans**: `docs/implementation-plans/`
- **Technical guides**: `docs/guides/`
- **Archived work**: `docs/archive/YYYY-MM-DD-[topic]/`
- **Vendor documentation**: `docs/vendors/{VENDOR}/` (see VENDOR_DOCUMENTATION_STANDARDS.md)
- **General docs**: `docs/`

**Examples**:
```
docs/session-summaries/2025-10-13-docs-restructure-summary.md
docs/implementation-plans/vendor/LUTRON_IMPLEMENTATION_PLAN.md
docs/implementation-plans/framework/task-mcp-migration-plan.md
docs/guides/ARCHITECTURE.md
docs/guides/HOW_TO_USE_PATTERNS.md
docs/reference/constitutional/FILE_ORGANIZATION_POLICY.md
docs/reference/VENDOR_DOCUMENTATION_STANDARDS.md
docs/reference/framework/research-first-pattern-guide.md
docs/reference/strategic/Commission Processing Vendor Extractors- Strategic Architecture Analysis.md
docs/archive/2025-10-13-reference-library-migration/
docs/vendors/SOUND_UNITED/research/2025-10-21-vendor-format-research.md
docs/vendors/LEGRAND/README.md
```

### Test Data (vendor files)
- **Master files**: `backend/tests/acceptance/fixtures/vendors/[VENDOR]/golden/inputs/`
- **Golden outputs**: `backend/tests/acceptance/fixtures/vendors/[VENDOR]/golden/outputs/`
- **Test scenarios**: `backend/tests/acceptance/fixtures/vendors/[VENDOR]/test-scenarios/`
- **Vendor config**: `backend/tests/acceptance/fixtures/vendors/[VENDOR]/config.yaml`

### Production Output (processed CSV files)
- **Location**: `outputs/` (gitignored)
- **Purpose**: Final processed commission data
- **NEVER commit production output files - they contain real vendor data**

### Debug/Temporary Data
- **Location**: `backend/tests/scratch/` (gitignored)
- **Purpose**: Temporary test output during development
- **Lifecycle**: Generated → Verified → Deleted

### Vendor Test Files (Development & Debugging)
**⚠️ CRITICAL**: Test and debug scripts MUST be organized under vendor-specific directories

**Structure**:
```
docs/vendors/{VENDOR}/tests/
├── debugging/              # Debug scripts and validation tests
│   ├── debug_*.py         # Flow tracing and debugging scripts
│   ├── test_*.py          # Unit and integration test scripts
│   ├── check_*.py         # Validation and verification scripts
│   └── verify_*.py        # Data verification scripts
├── analysis/              # Data analysis and investigation
│   ├── analyze_*.py       # Discrepancy analysis scripts
│   ├── find_*.py          # Data discovery scripts
│   ├── trace_*.py         # Data flow tracing scripts
│   └── *_results.csv      # Analysis output files
└── production-validation/ # Production test runners
    ├── test_production_*.py   # Production scenario tests
    └── test_*_output.py       # Output validation tests
```

**Rules**:
- ✅ ALWAYS organize under `docs/vendors/{VENDOR}/tests/` (NOT root or backend/)
- ✅ Use subdirectories: debugging/, analysis/, production-validation/
- ✅ Keep test data paired with test scripts
- ❌ NEVER place test files in project root
- ❌ NEVER place test files in backend/ root
- ❌ NEVER create test_data/, test_outputs/, test_reports/ in root

**Rationale**: 45+ vendors × 10-15 test files each = 450-675 files. Vendor-specific organization prevents navigation chaos and maintains separation of concerns.

**Note**: This `tests/` structure is for vendor-specific debugging and analysis scripts. It complements (but does not replace) the standard vendor documentation categories defined in VENDOR_DOCUMENTATION_STANDARDS.md.

### Subagent Research Reports (MANDATORY SUBDIRECTORIES)
**⚠️ CRITICAL**: Research reports MUST use subdirectory structure for scalability

**Structure**:
```
docs/subagent-reports/
├── vendor-format-research/
│   ├── EPSON/
│   ├── TPD/
│   ├── LUTRON/
│   ├── AMINA/
│   ├── SOUND_UNITED/
│   └── LEGRAND/
├── framework-research/
│   ├── cc-commands/
│   │   ├── cc-ready/
│   │   ├── cc-checkpoint/
│   │   └── cc-organize/
│   ├── research-subagent-system/
│   ├── context-management/
│   └── validation-framework/
├── api-design-research/
├── frontend-architecture-research/
└── testing-strategy-research/
```

**Rules**:
- ✅ ALWAYS create subdirectory per research topic
- ✅ Use category → component → date structure for framework/infrastructure research
- ✅ **For vendor-specific research**: Place in `docs/vendors/{VENDOR}/subagent-reports/`
- ✅ Keep 3-step research files together (initial → planning → architecture → revised)
- ❌ NEVER dump research files in root of `docs/subagent-reports/`
- ❌ **NEVER place vendor-specific subagent reports in `docs/subagent-reports/{agent-type}/{VENDOR}/`** (all vendor files belong in vendor folders)

**Rationale**: All vendor-related artifacts (including subagent reports) should be co-located in the vendor's folder for easier navigation and context.

**Filename Pattern**: `YYYY-MM-DD-{phase}-{topic}.md`
- Example: `2025-10-02-initial-research.md`
- Example: `2025-10-02-planning-review.md`
- Example: `2025-10-02-architecture-review.md`
- Example: `2025-10-02-revised-plan.md`

**See**: `docs/reference/VENDOR_DOCUMENTATION_STANDARDS.md` for complete subagent report location requirements.

### NEVER
- ❌ Test files in `docs/` (except under `docs/vendors/{VENDOR}/tests/`)
- ❌ Test files in project root
- ❌ Test files in backend/ root
- ❌ Test directories in root (test_data/, test_outputs/, test_reports/)
- ❌ Documentation in root (except CLAUDE.md, README.md, .project_status.md)
- ❌ Debug output committed anywhere
- ❌ Session files older than 7 days in root
- ❌ Production CSV files in root or committed to git
- ❌ Duplicate src/ or tests/ directories
- ❌ **Research reports in root of docs/subagent-reports/** (MUST use subdirectories)
- ❌ **Vendor-specific subagent reports outside vendor folder** (MUST use docs/vendors/{VENDOR}/subagent-reports/)
- ❌ **Framework/infrastructure subagent reports in vendor folders** (MUST use docs/subagent-reports/{agent-type}/)
- ❌ **Vendor folders without README.md + CHANGELOG.md** (MANDATORY per VENDOR_DOCUMENTATION_STANDARDS.md)

---

## Archival Triggers

**Archive when**:
1. Session handoff completed (vendor implemented)
2. Implementation plan executed
3. File >7 days old and not actively referenced
4. Root directory >10 files (emergency cleanup)
5. Monthly review (first Monday of month)
6. **Vendor debugging files >30 days old** → `docs/vendors/{VENDOR}/debugging/archived/YYYY-MM/`

**HOW to archive**:
```bash
# Create dated directory
mkdir -p docs/archive/YYYY-MM-DD-[topic]/

# Move related files
mv [FILES] docs/archive/YYYY-MM-DD-[topic]/

# Vendor debugging archival (per VENDOR_DOCUMENTATION_STANDARDS.md)
mkdir -p docs/vendors/{VENDOR}/debugging/archived/YYYY-MM/
mv docs/vendors/{VENDOR}/debugging/YYYY-MM-DD-*.md docs/vendors/{VENDOR}/debugging/archived/YYYY-MM/
```

---

## Cross-References

**Related Documentation**:
- **Vendor Documentation Standards**: `docs/reference/VENDOR_DOCUMENTATION_STANDARDS.md` (v2.0.0)
  - 7-category vendor folder structure
  - Timestamp conventions (YYYY-MM-DD-HHMM)
  - Mandatory files (README.md, CHANGELOG.md)
  - Migration guidelines
- **Vendor Registry**: `docs/vendors/VENDOR_REGISTRY.md`
- **Pattern Library Guide**: `docs/guides/HOW_TO_USE_PATTERNS.md`
- **Three-Step Research Process**: `docs/guides/THREE_STEP_RESEARCH_PROCESS.md`

---

## Constitutional Enforcement

This policy is **CONSTITUTIONAL** and must be enforced by:
- Claude Code at session start
- Manual review before commits
- Pre-commit hooks (future)
- Monthly compliance audits
- `/cc-comply` and `/cc-comply --quick` commands

**Version History**:
- 2025-10-25: Added vendor documentation structure reference (VENDOR_DOCUMENTATION_STANDARDS.md v2.0.0)
- 2025-10-13: Restructured per architecture review
- 2025-10-01: Initial policy creation
