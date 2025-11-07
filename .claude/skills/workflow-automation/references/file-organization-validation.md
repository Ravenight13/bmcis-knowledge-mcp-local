# File Organization Guidance

**Purpose**: Decision trees for when Claude should validate file paths and organize files.

---

## When Claude Should Validate File Paths

### Before Every File Write

Claude should check file paths before using Write tool:

**Decision Tree**:
```
About to write file
├─ Is path vendor code?
│   └─ Must be: backend/src/extractors/{VENDOR}/
├─ Is path tests?
│   └─ Must be: backend/tests/acceptance/test_{vendor}.py
├─ Is path documentation?
│   ├─ Research → docs/subagent-reports/{type}/{component}/
│   ├─ Vendor docs → docs/vendors/{VENDOR}/
│   └─ Guides → docs/guides/
├─ Is path in root?
│   └─ Only allowed: CLAUDE.md, README.md, pyproject.toml, etc.
└─ Ask user if unsure
```

---

## Common Path Violations and Prompts

### Violation 1: Test Files in Root Directory

**Detection**: File named `test_*.py` not in `backend/tests/`

**Prompt**:
```
This test file should be in backend/tests/, not root directory.
Moving to backend/tests/acceptance/test_{vendor}.py
```

---

### Violation 2: Vendor Code Mixing

**Detection**: Code references multiple vendor names, or path contains wrong vendor

**Prompt**:
```
This LEGRAND code is in the SOUND_UNITED/ directory.
Vendors must be isolated per Principle II.

Correct path: backend/src/extractors/legrand/
Current path: backend/src/extractors/sound_united/
```

---

### Violation 3: Session Files Not Archived

**Detection**: Session handoff >7 days old in `session-handoffs/`

**Prompt**:
```
This session handoff from {date} is >7 days old.
Archiving to docs/archived-context/session-handoffs/{YYYY}/{MM}/
```

---

### Violation 4: Documentation in Wrong Category

**Detection**: Vendor doc not in 7-category structure

**Prompt**:
```
This vendor documentation should be in a category folder.

Available categories:
- 01-format-analysis/
- 02-implementation/
- 03-testing/
- 04-debugging/
- 05-optimization/
- 06-troubleshooting/
- 07-archive/

Suggested: docs/vendors/{VENDOR}/02-implementation/
```

---

## Vendor Isolation Check

### Before Writing Vendor Code

**Check**:
```
Is path: backend/src/extractors/{correct_vendor}/?
Does code import from other vendors?
Is vendor name consistent (snake_case in code, UPPER_SNAKE_CASE in docs)?
```

**Prompt if violated**:
```
Vendor isolation violated. EPSON code cannot import from LEGRAND.

Allowed imports:
✅ from extractors.base import ...
✅ from extractors.config import ...
❌ from extractors.legrand import ...

Fix: Move shared code to base.py or create pattern in Pattern Library.
```

---

## Naming Convention Prompts

### Wrong Case for Vendor Directory

**Detection**: Vendor code path uses wrong case

**Prompt**:
```
Vendor directories use different casing depending on context:

- Code modules: snake_case (backend/src/extractors/sound_united/)
- Documentation: UPPER_SNAKE_CASE (docs/vendors/SOUND_UNITED/)
- Test fixtures: UPPER_SNAKE_CASE (backend/tests/acceptance/fixtures/vendors/SOUND_UNITED/)

Current path: {current_path}
Correct path: {suggested_path}
```

---

### Session Handoff Wrong Format

**Detection**: Session handoff filename doesn't match `YYYY-MM-DD-HHMM-description.md`

**Prompt**:
```
Session handoffs must follow format: YYYY-MM-DD-HHMM-description.md

Examples:
✅ 2025-11-03-1430-vendor-implementation.md
✅ 2025-11-03-0900-framework-research.md
❌ session-handoff.md (missing timestamp)
❌ 2025-11-03-vendor.md (missing time component)

Suggested: {corrected_filename}
```

---

## Subagent Report Organization

### Before Writing Research Artifacts

**Check structure**:
```
docs/subagent-reports/
  {agent-type}/           # framework-research, vendor-format-research, etc.
    {component}/          # type-system, coastal-source, etc.
      {timestamp}-{description}.md
```

**Prompt if wrong**:
```
Research artifacts use subdirectory structure:

Pattern: docs/subagent-reports/{agent-type}/{component}/{timestamp}-{description}.md

Current: {current_path}
Correct: {suggested_path}

This prevents docs/ root bloat per Article IV (File Organization Policy).
```

---

## Root Directory Protection

### Before Writing to Root

**Check**: Is this file allowed in root?

**Allowed files**:
- CLAUDE.md
- README.md
- pyproject.toml
- .gitignore
- .pre-commit-config.yaml
- LICENSE

**Prompt if not allowed**:
```
Files should not be added to root directory except for:
- Project configuration (pyproject.toml, .gitignore, etc.)
- Documentation index (CLAUDE.md, README.md)

This file should be in a subdirectory:
- Code: backend/src/
- Tests: backend/tests/
- Docs: docs/
- Scripts: scripts/

Suggested location: {suggested_path}
```

---

## Decision Workflow

**When About to Write File**:

1. **Extract path components** (vendor name, category, filename)
2. **Check validation rules** (case, structure, allowed locations)
3. **If valid** → Proceed with write
4. **If invalid** → Prompt with correction, wait for confirmation
5. **If unsure** → Ask user for clarification

**Never**:
- Silently write files to wrong locations
- Mix vendor code across directories
- Create files in root without checking allowed list
- Skip validation "just this once"

**Always**:
- Validate before writing
- Prompt user with corrected path
- Explain why path is wrong (reference constitutional principle)
- Offer to fix automatically (with user consent)
