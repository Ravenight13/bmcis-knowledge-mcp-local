# Context Detection Patterns

**Category**: Workflow Intelligence
**Token Budget**: ~800 tokens

---

## Purpose

Automatically detect the type of work being performed to load appropriate context, patterns, and tools.

**Problem**: Different work types (development, testing, documentation) require different workflows, but manually specifying context is tedious.

**Solution**: Use heuristics to detect work type from git branches, directory structure, and file patterns.

---

## Generic Context Types

### 1. Development Context

**Definition**: Active feature development or bug fixing in application code.

**Characteristics**:
- Writing production code
- Implementing new features
- Fixing bugs
- Refactoring existing code

### 2. Testing Context

**Definition**: Test implementation, test infrastructure, and quality assurance work.

**Characteristics**:
- Writing test cases
- Setting up test infrastructure
- Debugging test failures
- Creating test fixtures/data

### 3. Documentation Context

**Definition**: Documentation writing, API documentation, and knowledge base updates.

**Characteristics**:
- Writing markdown files
- Updating README files
- API documentation
- Architecture diagrams

### 4. Maintenance Context

**Definition**: Dependency updates, configuration changes, and infrastructure work.

**Characteristics**:
- Updating dependencies
- Configuration file changes
- CI/CD pipeline updates
- Build system modifications

---

## Detection Heuristics

### Git Branch Patterns

**Development Context**:
```
feat/*          # Feature development
feat/{feature}
feature/*
fix/*           # Bug fixes
fix/{bug}
bugfix/*
refactor/*      # Code refactoring
```

**Testing Context**:
```
test/*          # Test implementation
test/{area}
tests/*
qa/*            # Quality assurance
```

**Documentation Context**:
```
docs/*          # Documentation work
doc/*
documentation/*
```

**Maintenance Context**:
```
chore/*         # Maintenance work
deps/*          # Dependency updates
ci/*            # CI/CD changes
build/*         # Build system changes
```

### Directory Structure Patterns

**Development Context**:
```
src/                    # Source code
lib/                    # Library code
app/                    # Application code
{lang}/src/             # Language-specific (e.g., backend/src/)
```

**Testing Context**:
```
tests/                  # Test directory
test/
spec/                   # RSpec-style tests
__tests__/              # Jest-style tests
{lang}/tests/           # Language-specific test directory
```

**Documentation Context**:
```
docs/                   # Documentation
doc/
documentation/
*.md                    # Markdown files (README, etc.)
```

**Maintenance Context**:
```
.github/                # GitHub workflows
.gitlab-ci.yml          # GitLab CI
Dockerfile              # Docker configuration
package.json            # Node dependencies
requirements.txt        # Python dependencies
Cargo.toml              # Rust dependencies
go.mod                  # Go dependencies
```

### File Type Analysis

**Development Context** (code files):
```
*.py                    # Python
*.js, *.ts              # JavaScript/TypeScript
*.jsx, *.tsx            # React
*.go                    # Go
*.rs                    # Rust
*.java                  # Java
*.rb                    # Ruby
*.c, *.cpp, *.h         # C/C++
*.cs                    # C#
*.php                   # PHP
*.swift                 # Swift
*.kt                    # Kotlin
```

**Testing Context** (test files):
```
test_*.py               # Python tests
*_test.py
*_test.go               # Go tests
*.test.js               # JavaScript tests
*.spec.js               # Jest/Jasmine specs
*.test.ts               # TypeScript tests
*.spec.ts
*_spec.rb               # RSpec tests
*Test.java              # JUnit tests
```

**Documentation Context** (documentation files):
```
*.md                    # Markdown
*.rst                   # reStructuredText
*.adoc                  # AsciiDoc
*.txt                   # Plain text
openapi.yaml            # API specs
swagger.json
```

**Maintenance Context** (configuration files):
```
*.yaml, *.yml           # YAML configs
*.toml                  # TOML configs
*.json                  # JSON configs
Dockerfile              # Docker
Makefile                # Make build
*.sh                    # Shell scripts
.env                    # Environment variables
```

---

## Confidence Scoring

### HIGH Confidence (Multiple Indicators)

**Criteria**: 2+ indicators from different categories match

**Example 1** (Development):
- Branch: `feat/user-auth`
- Directory: `src/auth/`
- Files: `auth.py`, `middleware.py`
- **Confidence**: HIGH

**Example 2** (Testing):
- Branch: `test/integration`
- Directory: `tests/integration/`
- Files: `test_api.py`, `test_auth.py`
- **Confidence**: HIGH

### MEDIUM Confidence (Single Strong Indicator)

**Criteria**: 1 indicator matches strongly

**Example 1** (Development):
- Branch: `feat/dashboard`
- Directory: (root)
- Files: (mixed)
- **Confidence**: MEDIUM (branch indicates development)

**Example 2** (Documentation):
- Branch: (main)
- Directory: `docs/`
- Files: `architecture.md`
- **Confidence**: MEDIUM (docs directory indicates documentation)

### LOW Confidence (Ambiguous or No Indicators)

**Criteria**: No clear indicators or conflicting signals

**Example 1** (Ambiguous):
- Branch: `main`
- Directory: (root)
- Files: (mixed types)
- **Confidence**: LOW → **Request user confirmation**

**Example 2** (Conflicting):
- Branch: `feat/tests`
- Directory: `src/`
- Files: `test_helpers.py`
- **Confidence**: LOW → **Request user confirmation**

---

## Customization (Project-Specific Contexts)

### Defining Custom Contexts

**Example**: Web Application with Frontend/Backend Split

```yaml
# .workflow-context.yaml (project-specific)

contexts:
  - name: Backend Development
    patterns:
      branches: [feat/backend-*, fix/api-*]
      directories: [backend/src/, backend/api/]
      files: [*.py]

  - name: Frontend Development
    patterns:
      branches: [feat/frontend-*, fix/ui-*]
      directories: [frontend/src/, frontend/components/]
      files: [*.vue, *.jsx, *.tsx]

  - name: Database Migrations
    patterns:
      branches: [db/*, migration/*]
      directories: [migrations/]
      files: [*.sql, alembic.ini]
```

### Domain-Specific Contexts

**Example**: Data Science Project

```yaml
contexts:
  - name: Model Development
    patterns:
      branches: [model/*, feat/model-*]
      directories: [models/, notebooks/]
      files: [*.ipynb, train.py]

  - name: Data Processing
    patterns:
      branches: [data/*, feat/pipeline-*]
      directories: [pipelines/, etl/]
      files: [pipeline.py, transform.py]

  - name: Experimentation
    patterns:
      branches: [experiment/*]
      directories: [experiments/]
      files: [*.ipynb, experiment.py]
```

---

## Fallback: User Confirmation

### When to Request Confirmation

**Scenarios**:
- LOW confidence (no clear indicators)
- Ambiguous signals (conflicting patterns)
- First session in new project
- Context switch (branch change, directory change)

### Confirmation Prompt Format

```
Context detection confidence: LOW

Detected indicators:
- Branch: main
- Directory: /project-root
- Recent files: mixed types

Please confirm work type:
1. Development (writing production code)
2. Testing (writing tests)
3. Documentation (writing docs)
4. Maintenance (config/dependencies)
```

### User Response Handling

```python
# User selects option 1 (Development)
context = "Development"

# Store preference (optional)
# If user frequently works in main branch on development,
# learn this preference for future sessions
```

---

## Context Switching

### When to Re-Detect Context

**Triggers**:
- Branch change (git checkout)
- Directory change (cd command)
- File type change (editing different file types)
- Explicit user request ("switch to testing context")

### Switching Workflow

```bash
# 1. Detect trigger (e.g., branch change)
git checkout feat/backend-auth

# 2. Re-run context detection
# New indicators:
#   - Branch: feat/backend-auth
#   - Directory: backend/src/auth/
#   - Files: auth.py
# Confidence: HIGH → Backend Development

# 3. Unload previous context (if different)
# 4. Load new context (patterns, tools, workflows)
```

---

## Integration with Workflow Orchestrator

### Context-Aware Skill Loading

**Pattern**:
```
Detect context → Load appropriate skills/patterns → Configure workflows
```

**Example**:
```python
# Context: Development (Backend)
Load:
  - Code architecture patterns
  - API design guidelines
  - Database schema patterns

# Context: Testing
Load:
  - Test frameworks (pytest, jest, etc.)
  - Fixture patterns
  - Mocking/stubbing patterns

# Context: Documentation
Load:
  - Documentation templates
  - Markdown style guide
  - API documentation patterns
```

---

## Benefits

**Automatic Context Loading**:
- No manual specification needed
- Appropriate patterns/tools loaded
- Workflow optimized for work type

**Reduced Token Overhead**:
- Only load relevant context
- No unnecessary skill loading
- Progressive disclosure enforced

**Improved Accuracy**:
- Context-appropriate suggestions
- Relevant pattern recommendations
- Better code generation

---

## Summary

**Generic Contexts**: Development, Testing, Documentation, Maintenance

**Detection Methods**: Git branches, directory structure, file types

**Confidence Levels**: HIGH (2+ indicators), MEDIUM (1 indicator), LOW (ambiguous → confirm)

**Customization**: Project-specific contexts via configuration files

**Fallback**: User confirmation when confidence is LOW

**Benefits**: Automatic context loading, reduced token overhead, improved accuracy

**Universal Applicability**: Works across any programming language, framework, or project type.
