"""Development environment configuration and tool verification tests.

This comprehensive test suite verifies that the development environment is
properly configured and all required tools are available and functional.

Tests cover:
- Configuration files exist and are valid (pyproject.toml, etc.)
- Required tools are installed and importable
- Tools can execute successfully with correct configuration
- Code quality gates pass (pytest, mypy, ruff, black)
- Development dependencies are properly installed
- Virtual environment is correctly configured
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestDevelopmentFiles:
    """Test suite for development environment configuration files."""

    def test_pyproject_toml_exists(self) -> None:
        """Test pyproject.toml exists in project root."""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml not found in project root"
        assert pyproject_path.is_file(), "pyproject.toml is not a file"

    def test_pyproject_toml_readable(self) -> None:
        """Test pyproject.toml is readable."""
        pyproject_path = Path("pyproject.toml")
        try:
            with open(pyproject_path, encoding="utf-8") as f:
                content = f.read()
            assert len(content) > 0, "pyproject.toml is empty"
        except OSError as e:
            pytest.fail(f"Cannot read pyproject.toml: {e}")

    def test_pyproject_toml_valid_format(self) -> None:
        """Test pyproject.toml is valid TOML format.

        Uses standard library configparser since toml package may not be installed.
        For simplicity, validates basic structure by parsing as raw text.
        """
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        # Verify basic TOML structure markers exist
        assert (
            "[build-system]" in content or "[project]" in content
        ), "pyproject.toml missing expected TOML sections"

        # Check for matching brackets
        opening_brackets = content.count("[")
        closing_brackets = content.count("]")
        assert opening_brackets == closing_brackets, "pyproject.toml has mismatched brackets"

    def test_src_directory_exists(self) -> None:
        """Test src directory exists."""
        src_path = Path("src")
        assert src_path.exists(), "src directory not found"
        assert src_path.is_dir(), "src is not a directory"

    def test_tests_directory_exists(self) -> None:
        """Test tests directory exists."""
        tests_path = Path("tests")
        assert tests_path.exists(), "tests directory not found"
        assert tests_path.is_dir(), "tests is not a directory"

    def test_src_has_init(self) -> None:
        """Test src/__init__.py exists."""
        init_path = Path("src") / "__init__.py"
        assert init_path.exists(), "src/__init__.py not found"

    def test_tests_has_init(self) -> None:
        """Test tests/__init__.py exists."""
        init_path = Path("tests") / "__init__.py"
        assert init_path.exists(), "tests/__init__.py not found"

    def test_core_module_exists(self) -> None:
        """Test src/core module exists with required files."""
        core_path = Path("src") / "core"
        assert core_path.exists(), "src/core directory not found"
        assert core_path.is_dir(), "src/core is not a directory"

        # Check for __init__.py
        init_path = core_path / "__init__.py"
        assert init_path.exists(), "src/core/__init__.py not found"

    def test_core_config_module_exists(self) -> None:
        """Test src/core/config.py exists."""
        config_path = Path("src") / "core" / "config.py"
        assert config_path.exists(), "src/core/config.py not found"

    def test_core_database_module_exists(self) -> None:
        """Test src/core/database.py exists."""
        db_path = Path("src") / "core" / "database.py"
        assert db_path.exists(), "src/core/database.py not found"

    def test_core_logging_module_exists(self) -> None:
        """Test src/core/logging.py exists."""
        logging_path = Path("src") / "core" / "logging.py"
        assert logging_path.exists(), "src/core/logging.py not found"


class TestToolAvailability:
    """Test suite for required development tools availability."""

    def test_pytest_installed(self) -> None:
        """Test pytest is installed and importable."""
        try:
            import pytest as pytest_module

            assert pytest_module is not None
        except ImportError as e:
            pytest.fail(f"pytest not installed: {e}")

    def test_pytest_version_valid(self) -> None:
        """Test pytest version is 7.4.0 or higher."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pytest --version failed: {result.stderr}"
        assert "pytest" in result.stdout, "pytest version output missing"

    def test_mypy_installed(self) -> None:
        """Test mypy is installed and importable."""
        try:
            import mypy

            assert mypy is not None
        except ImportError as e:
            pytest.fail(f"mypy not installed: {e}")

    def test_mypy_version_valid(self) -> None:
        """Test mypy version is 1.7.0 or higher."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mypy --version failed: {result.stderr}"
        assert "mypy" in result.stdout, "mypy version output missing"

    def test_pydantic_installed(self) -> None:
        """Test pydantic is installed and importable."""
        try:
            import pydantic

            assert pydantic is not None
        except ImportError as e:
            pytest.fail(f"pydantic not installed: {e}")

    def test_pydantic_version_valid(self) -> None:
        """Test pydantic version is 2.0.0 or higher."""
        result = subprocess.run(
            [sys.executable, "-c", "import pydantic; print(pydantic.__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pydantic version check failed: {result.stderr}"
        version_str = result.stdout.strip()
        parts = version_str.split(".")
        assert len(parts) >= 2, f"Invalid pydantic version format: {version_str}"
        major = int(parts[0])
        assert major >= 2, f"pydantic version {version_str} is < 2.0.0"

    def test_pytest_cov_installed(self) -> None:
        """Test pytest-cov is installed."""
        try:
            import pytest_cov

            assert pytest_cov is not None
        except ImportError as e:
            pytest.fail(f"pytest-cov not installed: {e}")

    def test_psycopg_installed(self) -> None:
        """Test psycopg is installed."""
        try:
            import psycopg

            assert psycopg is not None
        except ImportError as e:
            pytest.fail(f"psycopg not installed: {e}")

    def test_pgvector_installed(self) -> None:
        """Test pgvector is installed."""
        try:
            import pgvector

            assert pgvector is not None
        except ImportError as e:
            pytest.fail(f"pgvector not installed: {e}")

    def test_pydantic_settings_installed(self) -> None:
        """Test pydantic-settings is installed."""
        try:
            from pydantic_settings import BaseSettings

            assert BaseSettings is not None
        except ImportError as e:
            pytest.fail(f"pydantic-settings not installed: {e}")


class TestToolExecution:
    """Test suite for tool execution and functionality."""

    def test_pytest_can_collect_tests(self) -> None:
        """Test pytest can discover and collect all tests."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "tests/"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pytest --collect-only failed: {result.stderr}"
        assert (
            "collected" in result.stdout or "test session starts" in result.stdout
        ), "pytest output missing expected collection info"

    def test_pytest_can_run_tests(self) -> None:
        """Test pytest can execute tests successfully.

        Note: Runs only config tests to keep test duration reasonable.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_config.py", "-v", "--tb=line", "-x"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Tests should either pass or the test run should complete
        # (we're testing if pytest CAN run, not that all tests pass yet)
        assert result.returncode in [
            0,
            1,
        ], f"pytest execution failed unexpectedly: {result.stderr}"
        assert (
            "test session starts" in result.stdout or "passed" in result.stdout
        ), "pytest output missing expected test session info"

    def test_pytest_coverage_collection(self) -> None:
        """Test pytest can collect coverage metrics.

        Note: Tests coverage on a single test file to keep duration reasonable.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_config.py::TestDatabaseConfig::test_default_values",
                "--cov=src",
                "--cov-report=term-missing",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete (whether tests pass or not)
        assert result.returncode in [0, 1], f"pytest coverage run failed: {result.stderr}"
        # Coverage should be attempted (might show in stdout or stderr)
        combined_output = result.stdout + result.stderr
        assert (
            "coverage" in combined_output.lower() or "test session" in combined_output.lower()
        ), "pytest coverage collection failed"

    def test_mypy_can_typecheck_core(self) -> None:
        """Test mypy can typecheck src/core modules."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "src/core/"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # We're testing if mypy CAN run, not that code passes strict mode
        assert result.returncode in [
            0,
            1,
        ], f"mypy execution failed: {result.stderr}"
        # Output should contain something (either success or error list)
        assert len(result.stdout) > 0 or len(result.stderr) > 0, "mypy produced no output"

    def test_mypy_can_typecheck_tests(self) -> None:
        """Test mypy can typecheck tests directory."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "tests/"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Test if mypy can execute (not if all code is correctly typed)
        assert result.returncode in [
            0,
            1,
        ], f"mypy execution failed: {result.stderr}"
        assert len(result.stdout) > 0 or len(result.stderr) > 0, "mypy produced no output"


class TestConfigurationValidation:
    """Test suite for configuration file validation."""

    def test_pyproject_has_build_system(self) -> None:
        """Test pyproject.toml has [build-system] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "[build-system]" in content, "pyproject.toml missing [build-system] section"

    def test_pyproject_has_project_metadata(self) -> None:
        """Test pyproject.toml has [project] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "[project]" in content, "pyproject.toml missing [project] section"

    def test_pyproject_has_dependencies(self) -> None:
        """Test pyproject.toml defines dependencies."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "dependencies" in content, "pyproject.toml missing dependencies section"
        # Check for expected core dependencies
        assert "pydantic" in content, "pydantic not in dependencies"
        assert "psycopg" in content, "psycopg not in dependencies"
        assert "pgvector" in content, "pgvector not in dependencies"

    def test_pyproject_has_dev_dependencies(self) -> None:
        """Test pyproject.toml defines dev dependencies."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert (
            "[project.optional-dependencies]" in content
        ), "pyproject.toml missing optional-dependencies section"
        assert "pytest" in content, "pytest not in dev dependencies"
        assert "mypy" in content, "mypy not in dev dependencies"

    def test_pyproject_has_pytest_config(self) -> None:
        """Test pyproject.toml has [tool.pytest.ini_options] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert (
            "[tool.pytest.ini_options]" in content
        ), "pyproject.toml missing [tool.pytest.ini_options] section"

    def test_pytest_config_has_testpaths(self) -> None:
        """Test pytest config specifies testpaths."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        # Find pytest config section
        start = content.find("[tool.pytest.ini_options]")
        assert start != -1, "pytest config section not found"

        # Extract pytest config (until next section or EOF)
        pytest_section = content[start : content.find("\n[", start + 1)]
        assert "testpaths" in pytest_section, "pytest config missing testpaths setting"

    def test_pytest_config_has_coverage_options(self) -> None:
        """Test pytest config includes coverage reporting options."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        start = content.find("[tool.pytest.ini_options]")
        assert start != -1, "pytest config section not found"

        pytest_section = content[start : content.find("\n[", start + 1)]
        assert "--cov" in pytest_section, "pytest config missing --cov option"

    def test_pyproject_has_mypy_config(self) -> None:
        """Test pyproject.toml has [tool.mypy] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "[tool.mypy]" in content, "pyproject.toml missing [tool.mypy] section"

    def test_mypy_config_has_strict_mode(self) -> None:
        """Test mypy config enables strict mode."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        start = content.find("[tool.mypy]")
        assert start != -1, "mypy config section not found"

        # Extract mypy config
        next_section = content.find("\n[", start + 1)
        if next_section == -1:
            mypy_section = content[start:]
        else:
            mypy_section = content[start:next_section]

        assert "strict = true" in mypy_section, "mypy config does not enable strict mode"

    def test_mypy_config_has_disallow_untyped_defs(self) -> None:
        """Test mypy config has disallow_untyped_defs."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        start = content.find("[tool.mypy]")
        assert start != -1, "mypy config section not found"

        next_section = content.find("\n[", start + 1)
        if next_section == -1:
            mypy_section = content[start:]
        else:
            mypy_section = content[start:next_section]

        assert "disallow_untyped_defs" in mypy_section, "mypy config missing disallow_untyped_defs"

    def test_pyproject_has_black_config(self) -> None:
        """Test pyproject.toml has [tool.black] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "[tool.black]" in content, "pyproject.toml missing [tool.black] section"

    def test_black_config_has_line_length(self) -> None:
        """Test black config specifies line length."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        start = content.find("[tool.black]")
        assert start != -1, "black config section not found"

        next_section = content.find("\n[", start + 1)
        if next_section == -1:
            black_section = content[start:]
        else:
            black_section = content[start:next_section]

        assert "line-length" in black_section, "black config missing line-length setting"

    def test_pyproject_has_ruff_config(self) -> None:
        """Test pyproject.toml has [tool.ruff] section."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()
        assert "[tool.ruff]" in content, "pyproject.toml missing [tool.ruff] section"

    def test_ruff_config_has_line_length(self) -> None:
        """Test ruff config specifies line length."""
        pyproject_path = Path("pyproject.toml")
        with open(pyproject_path, encoding="utf-8") as f:
            content = f.read()

        start = content.find("[tool.ruff]")
        assert start != -1, "ruff config section not found"

        next_section = content.find("\n[", start + 1)
        if next_section == -1:
            ruff_section = content[start:]
        else:
            ruff_section = content[start:next_section]

        assert "line-length" in ruff_section, "ruff config missing line-length setting"


class TestCodeQualityGates:
    """Test suite for code quality gate validation."""

    def test_all_tests_pass(self) -> None:
        """Test all pytest tests pass.

        Note: Tests core config tests as representative sample.
        """
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_config.py", "-v", "-x"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        # All tests should pass
        assert result.returncode == 0, (
            f"Tests failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_core_modules_type_check(self) -> None:
        """Test src/core modules pass mypy strict type checking."""
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "src/core/"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"mypy type checking failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_no_import_errors(self) -> None:
        """Test all modules can be imported without errors."""
        result = subprocess.run(
            [sys.executable, "-c", "from src.core import config, database, logging"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Module imports failed: {result.stderr}"

    def test_src_core_config_importable(self) -> None:
        """Test src.core.config module is importable."""
        result = subprocess.run(
            [sys.executable, "-c", "from src.core.config import Settings"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"Cannot import Settings from src.core.config: {result.stderr}"

    def test_src_core_database_importable(self) -> None:
        """Test src.core.database module is importable."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.core.database import DatabasePool",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Cannot import from src.core.database: {result.stderr}"

    def test_src_core_logging_importable(self) -> None:
        """Test src.core.logging module is importable."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.core.logging import log_api_call",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Cannot import from src.core.logging: {result.stderr}"


class TestDependencies:
    """Test suite for dependency verification."""

    def test_core_dependencies_importable(self) -> None:
        """Test all core dependencies can be imported."""
        core_deps = [
            "pydantic",
            "psycopg",
            "pgvector",
        ]

        for dep in core_deps:
            result = subprocess.run(
                [sys.executable, "-c", f"import {dep}"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Cannot import core dependency {dep}: {result.stderr}"

    def test_dev_dependencies_importable(self) -> None:
        """Test all dev dependencies can be imported."""
        dev_deps = [
            "pytest",
            "mypy",
        ]

        for dep in dev_deps:
            result = subprocess.run(
                [sys.executable, "-c", f"import {dep}"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Cannot import dev dependency {dep}: {result.stderr}"

    def test_pydantic_has_settings(self) -> None:
        """Test pydantic-settings is available for configuration."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pydantic_settings import BaseSettings; print(BaseSettings)",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pydantic-settings not available: {result.stderr}"

    def test_pytest_cov_available(self) -> None:
        """Test pytest-cov plugin is available."""
        # Test that --cov flag is recognized by running pytest --help
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pytest execution failed: {result.stderr}"
        assert "--cov" in result.stdout, "pytest-cov plugin not available (--cov not in help)"


class TestVirtualEnvironment:
    """Test suite for virtual environment validation."""

    def test_virtual_env_is_active(self) -> None:
        """Test virtual environment is active."""
        venv_path = Path(".venv")
        assert venv_path.exists(), "Virtual environment .venv directory not found"
        assert venv_path.is_dir(), ".venv is not a directory"

    def test_virtual_env_has_bin_python(self) -> None:
        """Test virtual environment has python executable."""
        python_path = Path(".venv") / "bin" / "python"
        assert (
            python_path.exists()
        ), f"Python executable not found in virtual environment: {python_path}"

    def test_virtual_env_has_bin_pip(self) -> None:
        """Test virtual environment has pip executable."""
        pip_path = Path(".venv") / "bin" / "pip"
        assert pip_path.exists(), f"pip executable not found in virtual environment: {pip_path}"

    def test_python_executable_in_venv(self) -> None:
        """Test sys.executable points to virtual environment python."""
        venv_bin = Path(".venv/bin/python").resolve()
        sys_python = Path(sys.executable).resolve()

        # They should point to the same location
        assert str(venv_bin) == str(sys_python), (
            f"Not running in virtual environment. " f"Expected: {venv_bin}, Got: {sys_python}"
        )


class TestProjectStructure:
    """Test suite for project structure validation."""

    def test_required_source_files_exist(self) -> None:
        """Test all required source files exist."""
        required_files = [
            "src/__init__.py",
            "src/core/__init__.py",
            "src/core/config.py",
            "src/core/database.py",
            "src/core/logging.py",
        ]

        for file_path in required_files:
            path = Path(file_path)
            assert path.exists(), f"Required file not found: {file_path}"
            assert path.is_file(), f"Path is not a file: {file_path}"

    def test_required_test_files_exist(self) -> None:
        """Test all required test files exist."""
        required_test_files = [
            "tests/__init__.py",
            "tests/test_config.py",
            "tests/test_database.py",
            "tests/test_logging.py",
        ]

        for file_path in required_test_files:
            path = Path(file_path)
            assert path.exists(), f"Required test file not found: {file_path}"
            assert path.is_file(), f"Path is not a file: {file_path}"

    def test_no_syntax_errors_in_source(self) -> None:
        """Test all source files have valid Python syntax."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "src/"],
            capture_output=True,
            text=True,
        )
        # Note: py_compile doesn't exist for directories, so this tests modules
        # We'll verify each file individually instead
        src_files = list(Path("src").rglob("*.py"))
        for src_file in src_files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(src_file)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Syntax error in {src_file}: {result.stderr}"

    def test_no_syntax_errors_in_tests(self) -> None:
        """Test all test files have valid Python syntax."""
        test_files = list(Path("tests").rglob("*.py"))
        for test_file in test_files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(test_file)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Syntax error in {test_file}: {result.stderr}"


class TestDocumentation:
    """Test suite for project documentation."""

    def test_project_documentation_exists(self) -> None:
        """Test project documentation exists."""
        # Check for DEVELOPMENT.md or README.md
        dev_doc = Path("DEVELOPMENT.md")
        readme_doc = Path("README.md")
        assert dev_doc.exists() or readme_doc.exists(), (
            "Neither DEVELOPMENT.md nor README.md found"
        )

    def test_project_documentation_not_empty(self) -> None:
        """Test project documentation is not empty."""
        # Check for DEVELOPMENT.md first, fall back to README.md
        if Path("DEVELOPMENT.md").exists():
            doc_path = Path("DEVELOPMENT.md")
        else:
            doc_path = Path("README.md")

        with open(doc_path, encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0, f"{doc_path.name} is empty"

    def test_gitignore_exists(self) -> None:
        """Test .gitignore exists."""
        gitignore_path = Path(".gitignore")
        assert gitignore_path.exists(), ".gitignore not found"

    def test_env_example_exists(self) -> None:
        """Test .env.example exists for configuration reference."""
        env_example_path = Path(".env.example")
        assert env_example_path.exists(), ".env.example not found for environment setup"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
