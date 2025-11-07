# File Organization Prompting Patterns

## When Claude Should Validate File Paths

### Before Writing Files

**Prompt Pattern**:
"Before writing `{filepath}`, let me validate the location:
- Vendor files → `backend/src/extractors/{vendor}/`
- Tests → `backend/tests/acceptance/test_{vendor}.py`
- Research → `docs/subagent-reports/{type}/{component}/`

Is this path correct? Checking FILE_ORGANIZATION_POLICY.md..."

### Path Validation Decision Tree

```
Is it vendor code?
├─ YES → backend/src/extractors/{vendor}/
├─ NO → Is it tests?
    ├─ YES → backend/tests/
    ├─ NO → Is it documentation?
        ├─ YES → docs/{category}/
        └─ NO → Ask user for guidance
```

### Common Mistakes to Catch

1. **Root directory clutter**: Test files, debug output in root
   **Prompt**: "This looks like it belongs in tests/, not root. Moving to backend/tests/"

2. **Vendor isolation violated**: Code mixing vendors
   **Prompt**: "This LEGRAND code is in SOUND_UNITED directory. Each vendor must be isolated."

3. **Session files not archived**: Old handoffs in root
   **Prompt**: "This session file is >7 days old. Should we archive to docs/archived-context/?"
