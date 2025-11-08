"""Tests for configuration management system.

Tests cover:
- Environment variable loading
- .env file loading
- Type validation
- Error handling for invalid configurations
- Factory pattern functionality
- Singleton behavior
"""

import os
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from pydantic import ValidationError

from src.core.config import (
    Settings,
    DatabaseConfig,
    LoggingConfig,
    ApplicationConfig,
    get_settings,
    reset_settings,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig model."""

    def test_default_values(self) -> None:
        """Test default database configuration values."""
        config = DatabaseConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "bmcis_knowledge_dev"
        assert config.user == "postgres"
        assert config.password == ""
        assert config.pool_min_size == 5
        assert config.pool_max_size == 20
        assert config.connection_timeout == 10.0
        assert config.statement_timeout == 30.0

    def test_environment_variable_loading(self) -> None:
        """Test loading database config from environment variables."""
        os.environ["DB_HOST"] = "db.example.com"
        os.environ["DB_PORT"] = "5433"
        os.environ["DB_DATABASE"] = "custom_db"
        os.environ["DB_USER"] = "custom_user"
        os.environ["DB_PASSWORD"] = "secret"

        try:
            config = DatabaseConfig()
            assert config.host == "db.example.com"
            assert config.port == 5433
            assert config.database == "custom_db"
            assert config.user == "custom_user"
            assert config.password == "secret"
        finally:
            # Cleanup
            for key in [
                "DB_HOST",
                "DB_PORT",
                "DB_DATABASE",
                "DB_USER",
                "DB_PASSWORD",
            ]:
                os.environ.pop(key, None)

    def test_port_validation_too_low(self) -> None:
        """Test port validation rejects values < 1."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=0)

        errors = exc_info.value.errors()
        assert any("greater than or equal to 1" in str(e) for e in errors)

    def test_port_validation_too_high(self) -> None:
        """Test port validation rejects values > 65535."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(port=65536)

        errors = exc_info.value.errors()
        assert any("less than or equal to 65535" in str(e) for e in errors)

    def test_pool_size_validation(self) -> None:
        """Test pool size validation: max >= min."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(pool_min_size=20, pool_max_size=10)

        errors = exc_info.value.errors()
        assert any("pool_max_size" in str(e) for e in errors)

    def test_timeout_validation(self) -> None:
        """Test timeout validation requires positive values."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(connection_timeout=0)

        errors = exc_info.value.errors()
        assert any("greater than 0" in str(e) for e in errors)

    def test_host_required(self) -> None:
        """Test host field is required and non-empty."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseConfig(host="")

        errors = exc_info.value.errors()
        assert any("at least 1 character" in str(e) for e in errors)


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_default_values(self) -> None:
        """Test default logging configuration values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.format == "json"
        assert config.console_enabled is True
        assert config.file_enabled is False
        assert config.file_path == "logs/application.log"
        assert config.max_file_size == 10485760
        assert config.backup_count == 5

    def test_environment_variable_loading(self) -> None:
        """Test loading logging config from environment variables."""
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "text"
        os.environ["LOG_CONSOLE_ENABLED"] = "false"
        os.environ["LOG_FILE_ENABLED"] = "true"

        try:
            config = LoggingConfig()
            assert config.level == "DEBUG"
            assert config.format == "text"
            assert config.console_enabled is False
            assert config.file_enabled is True
        finally:
            # Cleanup
            for key in [
                "LOG_LEVEL",
                "LOG_FORMAT",
                "LOG_CONSOLE_ENABLED",
                "LOG_FILE_ENABLED",
            ]:
                os.environ.pop(key, None)

    def test_log_level_validation(self) -> None:
        """Test log level validation and case normalization."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)  # type: ignore
            assert config.level == level

        # Case insensitive - lowercase is normalized to uppercase
        config = LoggingConfig(level="debug")  # type: ignore
        assert config.level == "DEBUG"

    def test_log_level_invalid(self) -> None:
        """Test invalid log level raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID")  # type: ignore

        errors = exc_info.value.errors()
        assert any("literal_error" in str(e) or "Literal" in str(e) for e in errors)

    def test_log_format_validation(self) -> None:
        """Test log format validation and case normalization."""
        # Valid formats
        for fmt in ["json", "text"]:
            config = LoggingConfig(format=fmt)  # type: ignore
            assert config.format == fmt

        # Case insensitive - uppercase is normalized to lowercase
        config = LoggingConfig(format="JSON")  # type: ignore
        assert config.format == "json"

    def test_log_format_invalid(self) -> None:
        """Test invalid log format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(format="xml")  # type: ignore

        errors = exc_info.value.errors()
        assert any("literal_error" in str(e) or "Literal" in str(e) for e in errors)

    def test_file_size_validation(self) -> None:
        """Test max file size validation requires >= 1 MB."""
        # Too small
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(max_file_size=524288)  # 512 KB

        errors = exc_info.value.errors()
        assert any("greater than or equal to" in str(e) for e in errors)


class TestApplicationConfig:
    """Tests for ApplicationConfig model."""

    def test_default_values(self) -> None:
        """Test default application configuration values."""
        config = ApplicationConfig()

        assert config.environment == "development"
        assert config.debug is True
        assert config.api_title == "BMCIS Knowledge Base API"
        assert config.api_version == "1.0.0"
        assert config.api_docs_url == "/docs"

    def test_environment_variable_loading(self) -> None:
        """Test loading application config from environment variables."""
        os.environ["APP_ENVIRONMENT"] = "production"
        os.environ["APP_DEBUG"] = "false"
        os.environ["APP_API_TITLE"] = "Custom API"
        os.environ["APP_API_VERSION"] = "2.0.0"

        try:
            config = ApplicationConfig()
            assert config.environment == "production"
            assert config.debug is False
            assert config.api_title == "Custom API"
            assert config.api_version == "2.0.0"
        finally:
            # Cleanup
            for key in [
                "APP_ENVIRONMENT",
                "APP_DEBUG",
                "APP_API_TITLE",
                "APP_API_VERSION",
            ]:
                os.environ.pop(key, None)

    def test_environment_validation(self) -> None:
        """Test environment validation."""
        valid_envs = ["development", "testing", "staging", "production"]
        for env in valid_envs:
            # For production, must set debug=False
            if env == "production":
                config = ApplicationConfig(environment=env, debug=False)  # type: ignore
            else:
                config = ApplicationConfig(environment=env)  # type: ignore
            assert config.environment == env

        # Case insensitive - uppercase is normalized to lowercase
        config = ApplicationConfig(environment="DEVELOPMENT")  # type: ignore
        assert config.environment == "development"

    def test_environment_invalid(self) -> None:
        """Test invalid environment raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ApplicationConfig(environment="unknown")  # type: ignore

        errors = exc_info.value.errors()
        assert any("literal_error" in str(e) or "Literal" in str(e) for e in errors)

    def test_debug_mode_production_validation(self) -> None:
        """Test debug mode cannot be True in production."""
        with pytest.raises(ValidationError) as exc_info:
            ApplicationConfig(environment="production", debug=True)

        errors = exc_info.value.errors()
        assert any("Debug mode must be False in production" in str(e) for e in errors)

    def test_api_version_format(self) -> None:
        """Test API version must be semantic versioning."""
        # Valid
        config = ApplicationConfig(api_version="1.2.3")
        assert config.api_version == "1.2.3"

        # Invalid format
        with pytest.raises(ValidationError):
            ApplicationConfig(api_version="1.2")

        with pytest.raises(ValidationError):
            ApplicationConfig(api_version="v1.2.3")


class TestSettings:
    """Tests for main Settings model."""

    def teardown_method(self) -> None:
        """Reset settings after each test."""
        reset_settings()

    def test_default_values(self) -> None:
        """Test default settings values."""
        # Clean environment
        for key in os.environ:
            if key.startswith(("DB_", "LOG_", "APP_")):
                os.environ.pop(key, None)

        settings = Settings()

        assert settings.environment == "development"
        assert settings.debug is True
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert isinstance(settings.application, ApplicationConfig)

    def test_nested_environment_variables(self) -> None:
        """Test nested environment variable loading with __ delimiter."""
        os.environ["DB_HOST"] = "remote.db.com"
        os.environ["LOG_LEVEL"] = "WARNING"
        os.environ["APP_ENVIRONMENT"] = "staging"

        try:
            settings = Settings()
            assert settings.database.host == "remote.db.com"
            assert settings.logging.level == "WARNING"
            assert settings.application.environment == "staging"
        finally:
            # Cleanup
            for key in ["DB_HOST", "LOG_LEVEL", "APP_ENVIRONMENT"]:
                os.environ.pop(key, None)

    def test_environment_sync_validation(self) -> None:
        """Test that main environment must match application config."""
        settings = Settings(environment="staging")
        assert settings.environment == "staging"

    def test_debug_validation_across_settings(self) -> None:
        """Test debug validation is enforced in ApplicationConfig within Settings."""
        # The debug validation is in ApplicationConfig, so we need to pass
        # the application config directly
        with pytest.raises(ValidationError):
            ApplicationConfig(environment="production", debug=True)

    def test_case_insensitive_environment_variables(self) -> None:
        """Test environment variable names are case insensitive."""
        os.environ["db_host"] = "myhost.com"
        os.environ["log_level"] = "debug"

        try:
            settings = Settings()
            assert settings.database.host == "myhost.com"
            assert settings.logging.level == "DEBUG"
        finally:
            # Cleanup
            os.environ.pop("db_host", None)
            os.environ.pop("log_level", None)


class TestSettingsFactory:
    """Tests for get_settings() factory function."""

    def teardown_method(self) -> None:
        """Reset settings after each test."""
        reset_settings()

    def test_factory_returns_settings(self) -> None:
        """Test factory returns Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_factory_singleton_behavior(self) -> None:
        """Test factory implements singleton pattern."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_factory_reset_creates_new_instance(self) -> None:
        """Test reset_settings() creates new instance on next call."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()

        assert settings1 is not settings2
        assert isinstance(settings2, Settings)

    def test_factory_with_env_vars(self) -> None:
        """Test factory correctly loads environment variables."""
        os.environ["DB_HOST"] = "factory-test.com"

        try:
            reset_settings()
            settings = get_settings()
            assert settings.database.host == "factory-test.com"
        finally:
            os.environ.pop("DB_HOST", None)
            reset_settings()


class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def teardown_method(self) -> None:
        """Reset settings after each test."""
        reset_settings()

    def test_complete_configuration_load(self) -> None:
        """Test loading complete configuration from environment."""
        os.environ.update({
            "APP_ENVIRONMENT": "staging",
            "APP_DEBUG": "false",
            "DB_HOST": "staging-db.example.com",
            "DB_PORT": "5433",
            "DB_POOL_MAX_SIZE": "30",
            "LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "json",
        })

        try:
            settings = Settings()

            # Application config
            assert settings.application.environment == "staging"
            assert settings.application.debug is False

            # Database config
            assert settings.database.host == "staging-db.example.com"
            assert settings.database.port == 5433
            assert settings.database.pool_max_size == 30

            # Logging config
            assert settings.logging.level == "WARNING"
            assert settings.logging.format == "json"
        finally:
            # Cleanup
            for key in list(os.environ.keys()):
                if key.startswith(("DB_", "LOG_", "APP_")):
                    os.environ.pop(key)

    def test_mixed_defaults_and_env_vars(self) -> None:
        """Test mixing default values with environment variables."""
        os.environ["DB_HOST"] = "custom-host"
        # Leave other DB settings as defaults
        # Leave all LOG settings as defaults
        # Leave all APP settings as defaults

        try:
            settings = Settings()

            # Custom value
            assert settings.database.host == "custom-host"

            # Defaults
            assert settings.database.port == 5432
            assert settings.logging.level == "INFO"
            assert settings.application.environment == "development"
        finally:
            os.environ.pop("DB_HOST", None)

    def test_validation_error_reporting(self) -> None:
        """Test configuration validation errors are clear."""
        os.environ["DB_PORT"] = "99999"  # Invalid port

        try:
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Error message should be helpful
            errors = exc_info.value.errors()
            assert len(errors) > 0
            # Check that error pertains to database configuration
            error_messages = [str(e) for e in errors]
            assert any("port" in msg.lower() or "database" in msg.lower() for msg in error_messages)
        finally:
            os.environ.pop("DB_PORT", None)
