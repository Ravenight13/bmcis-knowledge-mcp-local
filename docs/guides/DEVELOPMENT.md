# Development Environment Setup Guide

This guide provides instructions for setting up and maintaining the development environment for the BMCIS Knowledge MCP project.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 18+ (for local testing)
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/bmcis-knowledge-mcp.git
cd bmcis-knowledge-mcp

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest tests/ -v
mypy src/
ruff check src/
black --check src/
```

## Project Structure

```
bmcis-knowledge-mcp/
├── src/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── database.py        # Database connection pooling
│   │   └── logging.py         # Logging configuration
│   └── __init__.py
├── tests/
│   ├── test_config.py         # Configuration tests
│   ├── test_database.py       # Database tests
│   ├── test_database_pool.py  # Pool management tests
│   ├── test_logging.py        # Logging tests
│   └── __init__.py
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── pyproject.toml             # Project metadata and tool configuration
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
└── DEVELOPMENT.md             # This file
```

## Development Workflow

### 1. Code Formatting

Format all code using Black and Ruff:

```bash
# Format code with Black
black src/ tests/

# Sort imports with Ruff
ruff check --fix src/ tests/

# Format with Ruff formatter
ruff format src/ tests/
```

### 2. Linting and Code Quality

Check code quality with Ruff:

```bash
# Check for linting issues
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Check docstring coverage
interrogate --vv src/ --ignore-init-method --ignore-init-module
```

### 3. Type Checking

Perform static type checking with MyPy:

```bash
# Run strict type checking
mypy src/

# Run with all strict rules enabled
mypy src/ --strict

# Check specific module
mypy src/core/config.py
```

### 4. Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run specific test class
pytest tests/test_config.py::TestDatabaseConfig -v

# Run with verbose output
pytest tests/ -vv --tb=long
```

### 5. Pre-commit Hooks

Pre-commit hooks automatically check code before commits:

```bash
# Install hooks (run once after setup)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Skip hooks for a specific commit
git commit --no-verify
```

## Quality Gates

All code must pass these quality gates before being committed:

### Required Quality Checks

1. **Black Formatting**: Code must be formatted according to Black's style
2. **Ruff Linting**: Code must pass Ruff linting rules
3. **MyPy Type Checking**: Code must pass strict type checking
4. **Pytest Testing**: All tests must pass with >95% coverage

### Running All Checks

```bash
# Run all quality checks
./scripts/quality-check.sh

# Or manually:
black --check src/ tests/
ruff check src/ tests/
mypy src/ --strict
pytest tests/ --cov=src --cov-report=term-missing
```

## Configuration Files

### pyproject.toml

Main project configuration containing:

- **Project metadata**: Name, version, description, dependencies
- **Tool configuration**: pytest, mypy, ruff, black settings
- **Build system**: setuptools configuration

Key settings:

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
python_version = "3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--verbose --cov=src --cov-report=term-missing"
```

### .pre-commit-config.yaml

Defines pre-commit hooks that run automatically before commits:

- **Whitespace cleanup**: Trailing whitespace, end-of-file newlines
- **Syntax validation**: YAML, JSON, TOML file validation
- **Code formatting**: Black, Ruff
- **Type checking**: MyPy
- **Docstring coverage**: Interrogate

## Dependency Management

### Production Dependencies

```bash
# View production dependencies
pip list | grep -E "pydantic|psycopg|pgvector"

# Update production dependencies
pip install --upgrade -r requirements.txt
```

### Development Dependencies

```bash
# View development dependencies
pip list | grep -E "pytest|black|mypy|ruff"

# Update all dependencies
pip install --upgrade -r requirements-dev.txt
```

### Adding New Dependencies

1. **Production dependency**: Add to `pyproject.toml` [project] > dependencies
2. **Development dependency**: Add to `pyproject.toml` [project.optional-dependencies] > dev
3. **Update files**: Run `pip install -r requirements-dev.txt`

## Common Tasks

### Creating a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Write tests first (TDD)
# Add test cases in tests/test_*.py

# 3. Implement feature
# Add code in src/core/*.py

# 4. Check code quality
black src/
ruff check --fix src/
mypy src/
pytest tests/

# 5. Pre-commit hooks will run automatically
git add .
git commit -m "feat: my-feature description"
```

### Fixing Type Errors

```bash
# 1. Identify type errors
mypy src/

# 2. Add type annotations as needed
# Edit files to add proper type hints

# 3. Verify fixes
mypy src/ --strict

# 4. Run all tests
pytest tests/ --cov=src
```

### Running Tests in Isolation

```bash
# Run specific test
pytest tests/test_config.py::TestDatabaseConfig::test_default_values -v

# Run tests matching pattern
pytest tests/ -k "test_pool" -v

# Run with debugging output
pytest tests/ -vv --tb=long --capture=no
```

## Troubleshooting

### Issue: Pre-commit hooks not running

```bash
# Re-install hooks
pre-commit uninstall
pre-commit install

# Verify hooks are installed
pre-commit install --install-hooks
```

### Issue: MyPy errors on imports

```bash
# Update mypy cache
mypy src/ --clear-cache

# Verify all dependencies are installed
pip install -r requirements-dev.txt

# Check specific module
mypy src/core/config.py --show-error-codes
```

### Issue: Tests failing with import errors

```bash
# Reinstall package in development mode
pip install -e .

# Clear pytest cache
pytest --cache-clear tests/

# Run specific test with verbose output
pytest tests/test_config.py -vv
```

### Issue: Virtual environment issues

```bash
# Deactivate current environment
deactivate

# Remove and recreate virtual environment
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

# Reinstall all dependencies
pip install -r requirements-dev.txt
```

## CI/CD Integration

### GitHub Actions

Pre-commit configuration is compatible with GitHub Actions:

```yaml
- name: Run quality checks
  run: |
    pip install -r requirements-dev.txt
    black --check src/ tests/
    ruff check src/ tests/
    mypy src/ --strict
    pytest tests/ --cov=src
```

### Local CI Simulation

To test your changes as they would run in CI:

```bash
# Run all checks as CI would
pre-commit run --all-files
pytest tests/ --cov=src --cov-report=xml
```

## Type Safety and Pydantic

This project uses **Pydantic v2** for configuration management with strict type checking:

### Key Principles

1. **All models are frozen**: Prevents accidental modifications
2. **All fields are typed**: Complete type annotations required
3. **Validation occurs**: Pydantic validates on instantiation
4. **MyPy strict mode**: All type hints verified statically

### Example Configuration Model

```python
from pydantic import BaseModel, Field

class DatabaseConfig(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(5432, gt=0, lt=65536)
    name: str = Field("bmcis", description="Database name")

    model_config = ConfigDict(frozen=True)
```

## Performance Considerations

### Test Execution

- **Baseline**: ~13 seconds for 178 tests
- **Coverage reporting**: Adds ~2-3 seconds
- **MyPy checking**: ~5-10 seconds
- **Ruff/Black formatting**: <1 second each

### Optimization Tips

1. Run only affected tests: `pytest tests/test_config.py`
2. Run mypy on specific files: `mypy src/core/config.py`
3. Use pytest markers for test groups
4. Enable pytest-xdist for parallel testing

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Pre-commit Documentation](https://pre-commit.com/)

## Getting Help

- **Type errors**: Run `mypy src/ --show-error-codes`
- **Test failures**: Run `pytest tests/ -vv --tb=long`
- **Code style**: Run `black --diff src/`
- **Documentation**: See `DEVELOPMENT.md` sections above

## Next Steps

1. Follow the "Quick Start" section above
2. Run pre-commit hooks with `pre-commit run --all-files`
3. Run the full test suite: `pytest tests/ -v`
4. Start developing with proper type annotations!

---

**Last Updated**: November 2024
**Python Version**: 3.11+
**Status**: Production Ready
