# Quality Gates Setup

**Category**: Code Quality Enforcement
**Token Budget**: ~700 tokens

---

## Purpose

Establish automated quality checks that run before every commit to maintain code quality and prevent defects.

**Problem**: Manual quality checks are inconsistent and easy to forget, leading to bugs and technical debt.

**Solution**: Configure automated quality gates that must pass before code can be committed or deployed.

---

## Universal Quality Gate Concept

**Definition**: A quality gate is an automated check that validates code meets predefined standards before integration.

**Common Quality Gates**:
1. **Linting** - Code style and formatting checks
2. **Type Checking** - Static type validation (if applicable)
3. **Unit Tests** - Automated test execution
4. **Security Scans** - Vulnerability detection
5. **Code Coverage** - Test coverage thresholds
6. **Build Verification** - Compilation/bundling success

---

## Language-Specific Examples

### Python Projects

**Quality Gates**:
```bash
# Linting (style + best practices)
ruff check .                    # Fast Python linter
# OR
pylint src/                     # Traditional linter
# OR
flake8 .                        # Alternative linter

# Type checking
mypy .                          # Static type checker
# OR
pyright .                       # Alternative type checker

# Testing
pytest tests/                   # Unit/integration tests
pytest --cov=src --cov-report=term-missing  # With coverage

# Security
bandit -r src/                  # Security vulnerability scanner
safety check                    # Dependency vulnerability check

# Formatting (optional auto-fix)
black .                         # Code formatter
isort .                         # Import sorter
```

**Combined Command** (checkall):
```bash
#!/bin/bash
ruff check . && mypy . && pytest tests/
```

### JavaScript/TypeScript Projects

**Quality Gates**:
```bash
# Linting
npm run lint                    # ESLint
# OR
eslint src/                     # Direct ESLint

# Type checking
npm run type-check              # TypeScript compiler
# OR
tsc --noEmit                    # Direct TypeScript check

# Testing
npm test                        # Jest/Vitest/Mocha
npm run test:coverage           # With coverage

# Security
npm audit                       # Dependency vulnerabilities
# OR
snyk test                       # Snyk security scan

# Formatting (optional auto-fix)
npm run format                  # Prettier
# OR
prettier --write .              # Direct Prettier
```

**Combined Command** (checkall):
```bash
npm run lint && npm run type-check && npm test
```

### Go Projects

**Quality Gates**:
```bash
# Testing
go test ./...                   # Run all tests
go test -cover ./...            # With coverage

# Linting
golangci-lint run               # Comprehensive linter
# OR
go vet ./...                    # Go's built-in vet tool

# Formatting
go fmt ./...                    # Go formatter
goimports -w .                  # Import management

# Security
gosec ./...                     # Security scanner

# Build verification
go build ./...                  # Ensure code compiles
```

**Combined Command** (checkall):
```bash
go test ./... && golangci-lint run && go build ./...
```

### Rust Projects

**Quality Gates**:
```bash
# Testing
cargo test                      # Run all tests
cargo test --all-features       # Test all feature combinations

# Linting
cargo clippy                    # Rust linter
cargo clippy -- -D warnings     # Strict mode (warnings = errors)

# Formatting
cargo fmt --check               # Check formatting
cargo fmt                       # Auto-format

# Security
cargo audit                     # Dependency vulnerabilities

# Build verification
cargo build                     # Debug build
cargo build --release           # Release build
```

**Combined Command** (checkall):
```bash
cargo test && cargo clippy -- -D warnings && cargo build
```

### Java Projects

**Quality Gates**:
```bash
# Testing (Maven)
mvn test                        # Run tests
mvn verify                      # Run integration tests

# Testing (Gradle)
./gradlew test                  # Run tests
./gradlew check                 # Run all checks

# Linting
mvn checkstyle:check            # Maven Checkstyle
./gradlew checkstyleMain        # Gradle Checkstyle

# Security
mvn dependency-check:check      # OWASP dependency check

# Build verification
mvn clean install               # Maven build
./gradlew build                 # Gradle build
```

**Combined Command** (checkall):
```bash
mvn test && mvn checkstyle:check && mvn clean install
```

### Ruby Projects

**Quality Gates**:
```bash
# Testing
rspec                           # RSpec tests
rails test                      # Rails tests

# Linting
rubocop                         # Ruby linter
rubocop -a                      # Auto-fix

# Security
brakeman                        # Rails security scanner
bundle audit                    # Dependency vulnerabilities

# Type checking (optional)
sorbet tc                       # Sorbet type checker
```

**Combined Command** (checkall):
```bash
rspec && rubocop && brakeman
```

---

## Pre-Commit Integration

### Setup Git Pre-Commit Hook

**Manual Setup** (.git/hooks/pre-commit):
```bash
#!/bin/bash

echo "Running quality gates..."

# Run your checkall command
{checkall-command}

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Quality gates failed. Commit aborted."
    exit 1
fi

echo "✅ Quality gates passed. Proceeding with commit."
exit 0
```

**Pre-Commit Framework** (recommended):
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: quality-gates
        name: Run Quality Gates
        entry: {checkall-command}
        language: system
        pass_filenames: false
```

---

## Continuous Validation Workflow

### Development Loop with Quality Gates

```bash
# 1. Make code changes

# 2. Run quality gates BEFORE commit
{checkall-command}

# 3. If gates pass, commit
git add {files}
git commit -m "{type}({scope}): {message}"

# 4. If gates fail, fix issues and repeat
```

### Benefits:
- **Catch issues early** (before code review)
- **Consistent quality** (automated enforcement)
- **Faster feedback** (immediate vs waiting for CI)
- **Cleaner history** (no "fix linting" commits)

---

## Custom Quality Gates (Project-Specific)

### Example: Documentation Checks

```bash
# Ensure all public functions have docstrings (Python)
pydocstyle src/

# Check Markdown files for broken links
markdown-link-check docs/**/*.md

# Validate OpenAPI spec
openapi-generator validate -i openapi.yaml
```

### Example: Performance Checks

```bash
# Ensure bundle size under threshold (JavaScript)
bundlesize                      # Checks bundle size limits

# Profile performance (Python)
py-spy record --output profile.svg -- python script.py
```

### Example: Architecture Checks

```bash
# Validate dependency constraints
dependency-cruiser --validate .dependency-cruiser.js src/

# Check for circular dependencies
madge --circular src/
```

---

## Graceful Degradation

### When Quality Gates Are Unavailable

**Fallback Strategy**:
1. **Rely on code review** (manual quality checks)
2. **Use CI/CD checks** (gates run on push)
3. **Document expectations** (CONTRIBUTING.md)
4. **Gradual adoption** (add gates incrementally)

**Example**:
```bash
# If mypy not configured, skip type checking
if command -v mypy &> /dev/null; then
    mypy .
else
    echo "⚠️  mypy not found, skipping type checking"
fi
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Quality Gates

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Quality Gates
        run: {checkall-command}
```

### GitLab CI Example

```yaml
quality_gates:
  script:
    - {checkall-command}
  only:
    - merge_requests
    - main
```

---

## Summary

**Universal Concept**: Automated checks before code integration

**Common Gates**: Linting, type checking, testing, security scans, build verification

**Integration**: Pre-commit hooks + CI/CD pipelines

**Workflow**: Make changes → Run gates → Commit if passing → Fix if failing

**Benefits**: Early issue detection, consistent quality, faster feedback, cleaner history

**Graceful Degradation**: Fallback to manual review if gates unavailable

**Language-Agnostic**: Apply to any programming language or framework with appropriate tools.
