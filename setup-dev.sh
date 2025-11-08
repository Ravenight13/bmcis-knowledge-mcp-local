#!/bin/bash

# Development Environment Setup Script for BMCIS Knowledge MCP
# This script automates the setup of a development environment
# Usage: bash setup-dev.sh

set -e  # Exit on first error

echo "==================================="
echo "BMCIS Knowledge MCP - Dev Setup"
echo "==================================="

# Check Python version
echo ""
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python 3.11 or higher
python_check=$(python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)')
if [ $? -ne 0 ]; then
    echo "   ERROR: Python 3.11 or higher required"
    exit 1
fi

# Create virtual environment if not exists
echo ""
echo "2. Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "   Created .venv/"
else
    echo "   Using existing .venv/"
fi

# Activate virtual environment
source .venv/bin/activate
echo "   Activated virtual environment"

# Upgrade pip, setuptools, wheel
echo ""
echo "3. Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel
echo "   Upgraded pip, setuptools, wheel"

# Install development dependencies
echo ""
echo "4. Installing development dependencies..."
pip install -r requirements-dev.txt
echo "   Installed all development dependencies"

# Install pre-commit hooks
echo ""
echo "5. Installing pre-commit hooks..."
pre-commit install
echo "   Pre-commit hooks installed"

# Run pre-commit on all files to cache
echo ""
echo "6. Running pre-commit on all files..."
echo "   (This may take a minute on first run...)"
pre-commit run --all-files || true
echo "   Pre-commit cache created"

# Verify installation
echo ""
echo "7. Verifying installation..."
echo ""
echo "   Python version:"
python --version
echo ""
echo "   Key tools:"
echo "   - pytest: $(pytest --version)"
echo "   - black: $(black --version)"
echo "   - ruff: $(ruff --version)"
echo "   - mypy: $(mypy --version)"
echo "   - pre-commit: $(pre-commit --version)"

# Run a quick test
echo ""
echo "8. Running quick verification..."
echo "   Running pytest (sample)..."
pytest tests/ -q --tb=no
echo "   Running mypy..."
mypy src/ --ignore-missing-imports > /dev/null 2>&1 && echo "   ✓ Type checking passed" || echo "   ⚠ Type checking has issues (review with: mypy src/)"
echo "   Running ruff..."
ruff check src/ > /dev/null 2>&1 && echo "   ✓ Linting passed" || echo "   ⚠ Linting has issues (fix with: ruff check --fix src/)"

# Summary
echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Review any warnings above"
echo "2. Read DEVELOPMENT.md for workflow"
echo "3. Run: pre-commit run --all-files"
echo "4. Start coding!"
echo ""
echo "Useful commands:"
echo "  pytest tests/                    # Run tests"
echo "  black src/ tests/               # Format code"
echo "  ruff check --fix src/ tests/    # Fix linting issues"
echo "  mypy src/                       # Type check"
echo "  pre-commit run --all-files      # Run all hooks"
echo ""
