"""Type stubs for configuration management module.

Provides Pydantic v2-based configuration models for database, logging,
and application settings with environment variable loading and validation.
"""

from typing import Literal
from pydantic import BaseModel

# Type aliases
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["json", "text"]
Environment = Literal["development", "testing", "staging", "production"]

class DatabaseConfig(BaseModel):
    """Database connection and pool configuration."""

    host: str
    port: int
    database: str
    user: str
    password: str
    pool_min_size: int
    pool_max_size: int
    connection_timeout: float
    statement_timeout: float

    class Config:
        env_prefix: str

class LoggingConfig(BaseModel):
    """Logging configuration for application."""

    level: LogLevel
    format: LogFormat
    console_enabled: bool
    file_enabled: bool
    file_path: str
    max_file_size: int
    backup_count: int

    class Config:
        env_prefix: str

class ApplicationConfig(BaseModel):
    """Application-level configuration."""

    environment: Environment
    debug: bool
    api_title: str
    api_version: str
    api_docs_url: str | None

    class Config:
        env_prefix: str

class Settings(BaseModel):
    """Main application settings combining all configuration modules."""

    environment: Environment
    debug: bool
    database: DatabaseConfig
    logging: LoggingConfig
    application: ApplicationConfig

    class Config:
        env_file: str
        env_nested_delimiter: str
        case_sensitive: bool

def get_settings() -> Settings:
    """Factory function to get or create the global settings instance.

    Returns:
        Settings: Configured Settings instance with validated values.
    """
    ...

def reset_settings() -> None:
    """Reset the global settings instance.

    Used primarily for testing to force reloading configuration.
    """
    ...

_settings_instance: Settings | None
