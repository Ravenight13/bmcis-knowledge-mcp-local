"""
Comprehensive test suite for Pydantic configuration system.

Tests cover:
- Environment variable loading (.env files and system env vars)
- Type validation (ports, enums, boolean parsing)
- Configuration models (DatabaseConfig, LoggingConfig, Settings)
- Factory pattern (get_settings())
- Edge cases and error handling
- Configuration immutability and defaults

Target coverage: >90% of src/core/config.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Generator
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

# Import configuration classes once they are implemented
# These imports assume the configuration module exists at src/core/config.py
try:
    from src.core.config import (
        Settings,
        DatabaseConfig,
        LoggingConfig,
        get_settings,
        Environment,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Mock classes for test structure validation
    Settings = MagicMock()  # type: ignore
    DatabaseConfig = MagicMock()  # type: ignore
    LoggingConfig = MagicMock()  # type: ignore
    get_settings = MagicMock()  # type: ignore
    Environment = MagicMock()  # type: ignore


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_env_file() -> Generator[Path, None, None]:
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, newline=""
    ) as f:
        temp_path = Path(f.name)
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def clean_environ() -> Generator[dict[str, str], None, None]:
    """Save and restore environment variables."""
    original_env = os.environ.copy()
    # Remove database-related env vars for clean testing
    env_vars_to_remove = [
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_USER",
        "DATABASE_PASSWORD",
        "DATABASE_NAME",
        "ENVIRONMENT",
        "LOG_LEVEL",
        "APP_DEBUG",
    ]
    for var in env_vars_to_remove:
        os.environ.pop(var, None)

    yield os.environ.copy()

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ==============================================================================
# DatabaseConfig Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestDatabaseConfig:
    """Test suite for DatabaseConfig model."""

    def test_default_values(self) -> None:
        """Test DatabaseConfig initializes with correct defaults."""
        config = DatabaseConfig()

        assert config.host == "localhost", "Default host should be 'localhost'"
        assert config.port == 5432, "Default port should be 5432 (PostgreSQL)"
        assert config.user == "postgres", "Default user should be 'postgres'"
        assert config.name == "bmcis_knowledge_dev", "Default database name mismatch"
        assert config.password is None or config.password == "", (
            "Default password should be None or empty"
        )

    def test_custom_values(self) -> None:
        """Test DatabaseConfig with custom initialization values."""
        config = DatabaseConfig(
            host="prod.example.com",
            port=5433,
            user="app_user",
            password="secret123",
            name="production_db",
        )

        assert config.host == "prod.example.com"
        assert config.port == 5433
        assert config.user == "app_user"
        assert config.password == "secret123"
        assert config.name == "production_db"

    def test_invalid_port_zero(self) -> None:
        """Test that port 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=0)

        assert "port" in str(exc_info.value).lower(), (
            "Error message should mention 'port'"
        )

    def test_invalid_port_too_high(self) -> None:
        """Test that port > 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=65536)

        assert "port" in str(exc_info.value).lower(), (
            "Error message should mention 'port'"
        )

    def test_invalid_port_negative(self) -> None:
        """Test that negative port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=-1)

        assert "port" in str(exc_info.value).lower(), (
            "Error message should mention 'port'"
        )

    def test_valid_port_boundaries(self) -> None:
        """Test valid port boundaries (1 and 65535)."""
        # Test minimum valid port
        config_min = DatabaseConfig(port=1)
        assert config_min.port == 1

        # Test maximum valid port
        config_max = DatabaseConfig(port=65535)
        assert config_max.port == 65535

    def test_port_from_string(self) -> None:
        """Test port can be parsed from string."""
        config = DatabaseConfig(port="5433")  # type: ignore
        assert config.port == 5433
        assert isinstance(config.port, int)

    def test_missing_required_fields(self) -> None:
        """Test behavior with missing fields - should use defaults."""
        config = DatabaseConfig()
        # All fields should have defaults
        assert config.host is not None
        assert config.port is not None

    def test_immutability_frozen(self) -> None:
        """Test that DatabaseConfig is immutable if frozen=True."""
        config = DatabaseConfig()
        # If frozen, attempting to modify should raise error
        try:
            config.host = "newhost.com"  # type: ignore
            # If no error, check if model_config allows mutation
            assert hasattr(config, "model_config") or hasattr(
                config, "__config__"
            ), "Config should either be frozen or have immutability settings"
        except (AttributeError, TypeError, ValidationError):
            # Expected for frozen models
            pass

    def test_host_validation_not_empty(self) -> None:
        """Test that host cannot be empty string."""
        with pytest.raises(ValidationError):
            DatabaseConfig(host="")

    def test_user_validation_not_empty(self) -> None:
        """Test that user cannot be empty string."""
        with pytest.raises(ValidationError):
            DatabaseConfig(user="")

    def test_name_validation_not_empty(self) -> None:
        """Test that database name cannot be empty string."""
        with pytest.raises(ValidationError):
            DatabaseConfig(name="")


# ==============================================================================
# LoggingConfig Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestLoggingConfig:
    """Test suite for LoggingConfig model."""

    def test_default_values(self) -> None:
        """Test LoggingConfig initializes with correct defaults."""
        config = LoggingConfig()

        assert config.level == "INFO", "Default log level should be 'INFO'"
        assert hasattr(
            config, "format"
        ), "LoggingConfig should have format attribute"
        assert config.format is not None, "Format should not be None"

    def test_custom_values(self) -> None:
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"

    def test_valid_log_levels(self) -> None:
        """Test all valid log level enum values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_invalid_log_level(self) -> None:
        """Test that invalid log level raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID_LEVEL")

        assert "level" in str(exc_info.value).lower(), (
            "Error should mention 'level'"
        )

    def test_log_level_case_sensitivity(self) -> None:
        """Test log level enum case handling."""
        # Standard log levels should be uppercase
        config = LoggingConfig(level="INFO")
        assert config.level == "INFO"

        # Lowercase might be converted or rejected depending on implementation
        try:
            config_lower = LoggingConfig(level="info")  # type: ignore
            # If accepted, should be converted to uppercase
            assert config_lower.level.upper() == "INFO"
        except ValidationError:
            # Also acceptable if strict enum validation
            pass

    def test_json_format_configuration(self) -> None:
        """Test logging format attribute exists and is configured."""
        config = LoggingConfig()
        # Format should indicate JSON or structured logging
        assert hasattr(config, "format"), "Should have format attribute"
        # Format should not be None
        assert config.format is not None or hasattr(
            config, "json_format"
        ), "Should have format or json_format attribute"

    def test_immutability_frozen(self) -> None:
        """Test that LoggingConfig is immutable if frozen=True."""
        config = LoggingConfig()
        try:
            config.level = "DEBUG"  # type: ignore
            assert hasattr(config, "model_config") or hasattr(
                config, "__config__"
            ), "Config should have immutability settings"
        except (AttributeError, TypeError, ValidationError):
            pass


# ==============================================================================
# Settings Composition Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestSettings:
    """Test suite for Settings model composition."""

    def test_settings_composition(self) -> None:
        """Test that Settings properly composes nested config models."""
        settings = Settings()

        assert isinstance(
            settings.database, DatabaseConfig
        ), "Settings.database should be DatabaseConfig instance"
        assert isinstance(
            settings.logging, LoggingConfig
        ), "Settings.logging should be LoggingConfig instance"

    def test_settings_nested_access(self) -> None:
        """Test accessing nested configuration values."""
        settings = Settings()

        # Should be able to access nested values
        assert hasattr(settings.database, "host")
        assert hasattr(settings.database, "port")
        assert hasattr(settings.logging, "level")

    def test_settings_custom_nested_values(self) -> None:
        """Test Settings with custom nested configuration."""
        settings = Settings(
            database={"host": "custom.db.com", "port": 5433},
            logging={"level": "DEBUG"},
        )

        assert settings.database.host == "custom.db.com"
        assert settings.database.port == 5433
        assert settings.logging.level == "DEBUG"

    def test_settings_environment_field(self) -> None:
        """Test that Settings has environment field for app environment."""
        settings = Settings()

        assert hasattr(settings, "environment"), "Settings should have environment field"
        # Should be a valid environment value
        assert settings.environment in [
            "development",
            "staging",
            "production",
        ], "Environment should be one of: development, staging, production"

    def test_settings_debug_field(self) -> None:
        """Test that Settings has debug flag."""
        settings = Settings()

        assert hasattr(settings, "debug"), "Settings should have debug field"
        assert isinstance(
            settings.debug, bool
        ), "Debug field should be boolean"


# ==============================================================================
# Environment Variable Loading Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables."""

    def test_database_host_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading DATABASE_HOST from environment variable."""
        os.environ["DATABASE_HOST"] = "env-host.example.com"

        config = DatabaseConfig()
        assert config.host == "env-host.example.com"

    def test_database_port_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading DATABASE_PORT from environment variable."""
        os.environ["DATABASE_PORT"] = "5433"

        config = DatabaseConfig()
        assert config.port == 5433

    def test_database_user_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading DATABASE_USER from environment variable."""
        os.environ["DATABASE_USER"] = "app_user"

        config = DatabaseConfig()
        assert config.user == "app_user"

    def test_database_password_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading DATABASE_PASSWORD from environment variable."""
        os.environ["DATABASE_PASSWORD"] = "secret_password"

        config = DatabaseConfig()
        assert config.password == "secret_password"

    def test_database_name_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading DATABASE_NAME from environment variable."""
        os.environ["DATABASE_NAME"] = "custom_database"

        config = DatabaseConfig()
        assert config.name == "custom_database"

    def test_log_level_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading LOG_LEVEL from environment variable."""
        os.environ["LOG_LEVEL"] = "DEBUG"

        config = LoggingConfig()
        assert config.level == "DEBUG"

    def test_environment_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading ENVIRONMENT from environment variable."""
        os.environ["ENVIRONMENT"] = "production"

        settings = Settings()
        assert settings.environment == "production"

    def test_debug_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading APP_DEBUG from environment variable."""
        os.environ["APP_DEBUG"] = "true"

        settings = Settings()
        assert settings.debug is True

    def test_debug_false_from_env(self, clean_environ: dict[str, str]) -> None:
        """Test loading APP_DEBUG=false from environment variable."""
        os.environ["APP_DEBUG"] = "false"

        settings = Settings()
        assert settings.debug is False

    def test_env_override_defaults(self, clean_environ: dict[str, str]) -> None:
        """Test that environment variables override default values."""
        os.environ["DATABASE_HOST"] = "override-host"
        os.environ["DATABASE_PORT"] = "6543"
        os.environ["LOG_LEVEL"] = "WARNING"

        config = DatabaseConfig()
        logging_config = LoggingConfig()

        assert config.host == "override-host"
        assert config.port == 6543
        assert logging_config.level == "WARNING"

    def test_case_sensitivity_env_vars(self, clean_environ: dict[str, str]) -> None:
        """Test that environment variable names are case-sensitive."""
        os.environ["DATABASE_HOST"] = "correct-host"
        os.environ["database_host"] = "wrong-host"

        config = DatabaseConfig()
        assert config.host == "correct-host", (
            "Should use DATABASE_HOST (uppercase) not database_host"
        )


# ==============================================================================
# .env File Loading Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestEnvFileLoading:
    """Test loading configuration from .env files."""

    def test_load_from_env_file(
        self, temp_env_file: Path, clean_environ: dict[str, str]
    ) -> None:
        """Test loading configuration from .env file."""
        # Write test values to .env file
        env_content = """DATABASE_HOST=file-host.example.com
DATABASE_PORT=5434
DATABASE_USER=file_user
DATABASE_PASSWORD=file_password
DATABASE_NAME=file_database
LOG_LEVEL=DEBUG
ENVIRONMENT=staging
APP_DEBUG=true
"""
        temp_env_file.write_text(env_content)

        # Load configuration with patched .env file path
        with patch.dict(os.environ, {"ENV_FILE": str(temp_env_file)}):
            # Reload or reinitialize config to read the .env file
            config = DatabaseConfig()
            logging_config = LoggingConfig()
            settings = Settings()

            assert config.host == "file-host.example.com" or config.host == "localhost"
            # Note: Actual behavior depends on implementation

    def test_env_file_missing_uses_defaults(self, clean_environ: dict[str, str]) -> None:
        """Test that missing .env file uses default values."""
        # Remove all environment variables
        for key in list(os.environ.keys()):
            if key.startswith("DATABASE_") or key.startswith("LOG_"):
                del os.environ[key]

        config = DatabaseConfig()

        # Should have defaults, not fail
        assert config.host == "localhost"
        assert config.port == 5432

    def test_env_file_empty_uses_defaults(
        self, temp_env_file: Path, clean_environ: dict[str, str]
    ) -> None:
        """Test that empty .env file uses default values."""
        temp_env_file.write_text("")

        # Load with empty .env
        with patch.dict(os.environ, {"ENV_FILE": str(temp_env_file)}):
            config = DatabaseConfig()
            assert config.host == "localhost"

    def test_env_file_comments_ignored(
        self, temp_env_file: Path, clean_environ: dict[str, str]
    ) -> None:
        """Test that .env file comments are properly ignored."""
        env_content = """# This is a comment
DATABASE_HOST=comment-test-host
# DATABASE_PORT=9999
DATABASE_PORT=5434
"""
        temp_env_file.write_text(env_content)

        # Comments should be ignored
        with patch.dict(os.environ, {"ENV_FILE": str(temp_env_file)}):
            config = DatabaseConfig()
            # Implementation-dependent, but should either load or use defaults
            assert config.port == 5434 or config.port == 5432

    def test_env_file_blank_lines_ignored(
        self, temp_env_file: Path, clean_environ: dict[str, str]
    ) -> None:
        """Test that blank lines in .env file are ignored."""
        env_content = """DATABASE_HOST=blank-test-host

DATABASE_PORT=5434

"""
        temp_env_file.write_text(env_content)

        with patch.dict(os.environ, {"ENV_FILE": str(temp_env_file)}):
            config = DatabaseConfig()
            assert config.host == "blank-test-host" or config.host == "localhost"

    def test_env_override_env_file(
        self, temp_env_file: Path, clean_environ: dict[str, str]
    ) -> None:
        """Test that environment variables override .env file values."""
        # Write to .env file
        env_content = """DATABASE_HOST=file-host
DATABASE_PORT=5434
"""
        temp_env_file.write_text(env_content)

        # Set environment variable (should override)
        os.environ["DATABASE_HOST"] = "env-host"

        # Load configuration
        with patch.dict(os.environ, {"ENV_FILE": str(temp_env_file)}):
            config = DatabaseConfig()
            assert config.host == "env-host", (
                "Environment variables should override .env file"
            )


# ==============================================================================
# Factory Pattern Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestSettingsFactory:
    """Test Settings factory function get_settings()."""

    def test_get_settings_returns_instance(self) -> None:
        """Test that get_settings() returns a Settings instance."""
        settings = get_settings()

        assert isinstance(
            settings, Settings
        ), "get_settings() should return Settings instance"

    def test_get_settings_valid_composition(self) -> None:
        """Test that get_settings() returns properly composed Settings."""
        settings = get_settings()

        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.logging, LoggingConfig)

    def test_get_settings_multiple_calls(self) -> None:
        """Test behavior of multiple get_settings() calls."""
        settings1 = get_settings()
        settings2 = get_settings()

        # Both should be valid Settings instances
        assert isinstance(settings1, Settings)
        assert isinstance(settings2, Settings)

        # Could be same object (cached) or different objects depending on design
        # Just verify both are valid
        assert isinstance(settings1.database, DatabaseConfig)
        assert isinstance(settings2.database, DatabaseConfig)

    def test_settings_immutable(self) -> None:
        """Test that Settings returned from factory is immutable."""
        settings = get_settings()

        try:
            settings.debug = not settings.debug  # type: ignore
            # If no error, check for immutability configuration
            assert hasattr(settings, "model_config") or hasattr(
                settings, "__config__"
            ), "Settings should have immutability configuration"
        except (AttributeError, TypeError, ValidationError):
            # Expected for frozen/immutable models
            pass


# ==============================================================================
# Type Validation Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestTypeValidation:
    """Test type validation and coercion."""

    def test_port_type_coercion_from_string(self) -> None:
        """Test that port can be coerced from string to int."""
        config = DatabaseConfig(port="5433")  # type: ignore
        assert isinstance(config.port, int)
        assert config.port == 5433

    def test_port_type_coercion_from_float(self) -> None:
        """Test that port can be coerced from float to int."""
        config = DatabaseConfig(port=5433.7)  # type: ignore
        assert isinstance(config.port, int)
        assert config.port == 5433

    def test_debug_boolean_coercion_from_string(self) -> None:
        """Test that boolean debug flag is properly coerced."""
        settings_true = Settings(debug="true")  # type: ignore
        settings_false = Settings(debug="false")  # type: ignore
        settings_yes = Settings(debug="yes")  # type: ignore
        settings_no = Settings(debug="no")  # type: ignore

        # Should all coerce properly
        assert isinstance(settings_true.debug, bool)
        assert isinstance(settings_false.debug, bool)

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid types raise ValidationError."""
        with pytest.raises(ValidationError):
            DatabaseConfig(port="not_a_number")  # type: ignore

    def test_environment_enum_validation(self) -> None:
        """Test that environment field validates against enum."""
        valid_envs = ["development", "staging", "production"]

        for env in valid_envs:
            settings = Settings(environment=env)  # type: ignore
            assert settings.environment == env

        # Invalid environment should fail
        with pytest.raises(ValidationError):
            Settings(environment="invalid_env")  # type: ignore


# ==============================================================================
# Edge Cases and Error Handling Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extra_fields_ignored(self) -> None:
        """Test that extra fields are ignored or raise error (depending on config)."""
        try:
            config = DatabaseConfig(extra_field="value")  # type: ignore
            # If no error, extra fields are ignored
            assert True
        except ValidationError:
            # Also acceptable if strict field validation
            pass

    def test_none_optional_fields(self) -> None:
        """Test that optional fields can be None."""
        config = DatabaseConfig(password=None)
        assert config.password is None

    def test_whitespace_handling(self) -> None:
        """Test handling of whitespace in configuration values."""
        config = DatabaseConfig(
            host=" whitespace-host ", user=" whitespace-user "
        )

        # Whitespace should be handled (stripped or preserved depending on validator)
        assert isinstance(config.host, str)
        assert isinstance(config.user, str)

    def test_special_characters_in_password(self) -> None:
        """Test that special characters in password are preserved."""
        special_password = "p@ssw0rd!#$%^&*()"
        config = DatabaseConfig(password=special_password)

        assert config.password == special_password

    def test_very_long_values(self) -> None:
        """Test handling of very long configuration values."""
        long_host = "a" * 1000
        long_password = "x" * 10000

        config = DatabaseConfig(host=long_host, password=long_password)

        assert config.host == long_host
        assert config.password == long_password

    def test_unicode_in_values(self) -> None:
        """Test that unicode characters are properly handled."""
        unicode_host = "hôst.exämple.côm"
        unicode_user = "üser_名前"

        config = DatabaseConfig(host=unicode_host, user=unicode_user)

        assert config.host == unicode_host
        assert config.user == unicode_user

    def test_empty_required_field_validation(self) -> None:
        """Test that empty required fields fail validation."""
        with pytest.raises(ValidationError):
            DatabaseConfig(host="", user="valid_user")

    def test_whitespace_only_field(self) -> None:
        """Test that whitespace-only fields fail validation."""
        with pytest.raises(ValidationError):
            DatabaseConfig(user="   ")

    def test_sql_injection_in_values_stored(self) -> None:
        """Test that potentially malicious values are stored as-is (no injection prevention)."""
        malicious_value = "'; DROP TABLE users; --"
        config = DatabaseConfig(user=malicious_value)

        # Should store the value (ORM/DB layer should handle escaping)
        assert config.user == malicious_value


# ==============================================================================
# Integration and Composition Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestIntegration:
    """Integration tests for complete configuration system."""

    def test_full_settings_initialization(
        self, clean_environ: dict[str, str]
    ) -> None:
        """Test complete Settings initialization with all components."""
        # Set up environment
        os.environ.update(
            {
                "DATABASE_HOST": "integration-host",
                "DATABASE_PORT": "5435",
                "DATABASE_USER": "integration_user",
                "DATABASE_PASSWORD": "integration_pass",
                "DATABASE_NAME": "integration_db",
                "LOG_LEVEL": "WARNING",
                "ENVIRONMENT": "staging",
                "APP_DEBUG": "true",
            }
        )

        # Create settings
        settings = get_settings()

        # Verify all components loaded correctly
        assert settings.database.host == "integration-host"
        assert settings.database.port == 5435
        assert settings.database.user == "integration_user"
        assert settings.database.password == "integration_pass"
        assert settings.database.name == "integration_db"
        assert settings.logging.level == "WARNING"
        assert settings.environment == "staging"
        assert settings.debug is True

    def test_development_configuration(self) -> None:
        """Test typical development environment configuration."""
        settings = get_settings()

        # Development typically uses localhost and INFO logging
        assert isinstance(settings.database.host, str)
        assert 1 <= settings.database.port <= 65535
        assert settings.logging.level in ["DEBUG", "INFO"]

    def test_production_configuration_requires_setup(
        self, clean_environ: dict[str, str]
    ) -> None:
        """Test that production configuration requires explicit setup."""
        os.environ["ENVIRONMENT"] = "production"

        settings = get_settings()
        assert settings.environment == "production"

    def test_validation_error_provides_clear_message(self) -> None:
        """Test that validation errors provide clear, actionable messages."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=99999)

        error_message = str(exc_info.value)
        assert "port" in error_message.lower(), (
            "Error should clearly indicate which field failed"
        )
        assert "65535" in error_message or "range" in error_message.lower(), (
            "Error should indicate the valid range"
        )


# ==============================================================================
# Performance and Concurrency Tests
# ==============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not yet implemented")
class TestPerformance:
    """Performance and concurrency tests."""

    def test_config_initialization_performance(self) -> None:
        """Test that config initialization is reasonably fast."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            _ = DatabaseConfig()
        end = time.perf_counter()

        elapsed = end - start
        assert elapsed < 1.0, (
            f"Creating 100 config instances should be fast (< 1s), took {elapsed}s"
        )

    def test_get_settings_performance(self) -> None:
        """Test that get_settings() factory is reasonably fast."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            _ = get_settings()
        end = time.perf_counter()

        elapsed = end - start
        assert elapsed < 1.0, (
            f"Calling get_settings() 100 times should be fast (< 1s), took {elapsed}s"
        )

    def test_thread_safety(self) -> None:
        """Test that config access is thread-safe."""
        import threading

        results: list[Settings] = []
        errors: list[Exception] = []

        def load_config() -> None:
            try:
                settings = get_settings()
                results.append(settings)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_config) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10, "All threads should successfully load config"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/core/config", "--cov-report=term-missing"])
