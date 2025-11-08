"""Configuration management using Pydantic v2.

Provides type-safe configuration models for database, logging, and application
settings with environment variable loading and .env file support.

Implements factory pattern for global configuration access with validation
against Pydantic v2 strict mode for type safety.
"""

from typing import Literal, Any
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    ValidationInfo,
    SecretStr,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Type aliases for configuration values
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "text"]
Environment = Literal["development", "testing", "staging", "production"]


class DatabaseConfig(BaseSettings):
    """Database connection and pool configuration.

    Configures PostgreSQL connection parameters and connection pooling settings.
    Environment variables use DB_ prefix.

    Attributes:
        host: PostgreSQL server hostname or IP address.
        port: PostgreSQL server port (1-65535).
        database: Target database name.
        user: Database user for authentication.
        password: Password for database user (can be empty for trusted connections).
        pool_min_size: Minimum size of connection pool.
        pool_max_size: Maximum size of connection pool.
        connection_timeout: Connection timeout in seconds.
        statement_timeout: SQL statement timeout in seconds.
    """

    host: str = Field(
        default="localhost",
        description="PostgreSQL host address",
        min_length=1,
    )
    port: int = Field(
        default=5432,
        description="PostgreSQL port number",
        ge=1,
        le=65535,
    )
    database: str = Field(
        default="bmcis_knowledge_dev",
        description="Target database name",
        min_length=1,
    )
    user: str = Field(
        default="postgres",
        description="Database user for authentication",
        min_length=1,
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Password for database user (never logged or exposed)",
    )
    pool_min_size: int = Field(
        default=5,
        description="Minimum connection pool size",
        ge=1,
    )
    pool_max_size: int = Field(
        default=20,
        description="Maximum connection pool size",
        ge=1,
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds",
        gt=0,
    )
    statement_timeout: float = Field(
        default=30.0,
        description="Statement timeout in seconds",
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
    )

    @field_validator("pool_max_size")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that pool_max_size >= pool_min_size.

        Args:
            v: Maximum pool size value.
            info: Validation context with other field values.

        Returns:
            Validated pool_max_size.

        Raises:
            ValueError: If pool_max_size < pool_min_size.
        """
        min_size = info.data.get("pool_min_size", 5)
        if v < min_size:
            msg = f"pool_max_size ({v}) must be >= pool_min_size ({min_size})"
            raise ValueError(msg)
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration for application.

    Configures log levels, formats, and output handlers. Environment
    variables use LOG_ prefix.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format (json or text).
        console_enabled: Enable console output.
        file_enabled: Enable file output.
        file_path: Path to log file.
        max_file_size: Maximum log file size in bytes before rotation.
        backup_count: Number of backup log files to retain.
    """

    level: LogLevel = Field(
        default="INFO",
        description="Logging level",
    )
    format: LogFormat = Field(
        default="json",
        description="Log output format",
    )
    console_enabled: bool = Field(
        default=True,
        description="Enable console logging",
    )
    file_enabled: bool = Field(
        default=False,
        description="Enable file logging",
    )
    file_path: str = Field(
        default="logs/application.log",
        description="Path to log file",
    )
    max_file_size: int = Field(
        default=10485760,  # 10 MB
        description="Maximum log file size in bytes",
        ge=1048576,  # Minimum 1 MB
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep",
        ge=1,
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False,
    )

    @field_validator("level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        """Normalize log level to uppercase for consistency.

        Args:
            v: Log level value.

        Returns:
            Normalized log level string.
        """
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("format", mode="before")
    @classmethod
    def normalize_log_format(cls, v: Any) -> str:
        """Normalize log format to lowercase for consistency.

        Args:
            v: Log format value.

        Returns:
            Normalized log format string.
        """
        if isinstance(v, str):
            return v.lower()
        return v


class ApplicationConfig(BaseSettings):
    """Application-level configuration.

    Configures general application settings. Environment variables
    use APP_ prefix.

    Attributes:
        environment: Deployment environment (development, testing, staging, production).
        debug: Enable debug mode (typically development only).
        api_title: Title for API documentation.
        api_version: Version string for API.
        api_docs_url: URL path for API documentation (None to disable).
    """

    environment: Environment = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode",
    )
    api_title: str = Field(
        default="BMCIS Knowledge Base API",
        description="API documentation title",
        min_length=1,
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version",
        pattern=r"^\d+\.\d+\.\d+$",
    )
    api_docs_url: str | None = Field(
        default="/docs",
        description="API documentation URL path",
    )

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        case_sensitive=False,
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: Any) -> str:
        """Normalize environment to lowercase for consistency.

        Args:
            v: Environment value.

        Returns:
            Normalized environment string.
        """
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("debug")
    @classmethod
    def validate_debug_mode(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate debug mode is appropriate for environment.

        Args:
            v: Debug flag value.
            info: Validation context with other field values.

        Returns:
            Validated debug flag.

        Raises:
            ValueError: If debug=True in production.
        """
        environment = info.data.get("environment", "development")
        if v and environment == "production":
            msg = "Debug mode must be False in production environment"
            raise ValueError(msg)
        return v


class Settings(BaseSettings):
    """Main application settings combining all configuration modules.

    Loads configuration from environment variables and .env files with
    support for nested settings via double underscore delimiter.
    Validates all configuration values using Pydantic v2 strict mode.

    Attributes:
        environment: Deployment environment (inherited from ApplicationConfig).
        debug: Debug mode enabled (inherited from ApplicationConfig).
        database: Database connection and pool configuration.
        logging: Logging configuration.
        application: Application-level configuration.
    """

    environment: Environment = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode",
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration",
    )
    application: ApplicationConfig = Field(
        default_factory=ApplicationConfig,
        description="Application configuration",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: Any) -> str:
        """Normalize environment to lowercase for consistency.

        Args:
            v: Environment value being set.

        Returns:
            Normalized environment.
        """
        if isinstance(v, str):
            return v.lower()
        return v


# Global settings instance
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Factory function to get or create the global settings instance.

    Implements singleton pattern for configuration access. Configuration
    is loaded once from environment variables and .env files on first call,
    then cached for subsequent calls.

    Returns:
        Settings: Validated Settings instance with all configuration loaded.

    Example:
        >>> settings = get_settings()
        >>> db_url = f"postgresql://{settings.database.user}@{settings.database.host}"
        >>> log_level = settings.logging.level
    """
    global _settings_instance

    if _settings_instance is None:
        _settings_instance = Settings()

    return _settings_instance


def reset_settings() -> None:
    """Reset the global settings instance.

    Used primarily for testing to force reloading configuration.
    After calling this function, the next call to get_settings() will
    create a new Settings instance from environment variables.

    Example:
        >>> reset_settings()
        >>> settings = get_settings()  # Creates new instance
    """
    global _settings_instance
    _settings_instance = None
