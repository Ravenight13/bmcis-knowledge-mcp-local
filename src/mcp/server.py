"""FastMCP server initialization for BMCIS Knowledge MCP.

Type-safe MCP server exposing semantic_search tool to Claude Desktop.
Supports progressive disclosure for token efficiency.

Architecture:
- Thin wrapper around existing HybridSearch
- No caching duplication (uses existing query cache)
- Type-safe with Pydantic models
- Async-compatible design

Example:
    # Initialize and run server (production)
    >>> from src.mcp.server import mcp, initialize_server
    >>> initialize_server()
    >>> # Server runs via FastMCP CLI or programmatically

    # Get search instance (for testing)
    >>> from src.mcp.server import get_hybrid_search
    >>> search = get_hybrid_search()
    >>> results = search.search("JWT authentication", top_k=10)
"""

from __future__ import annotations

import logging
from typing import Any

# FastMCP will be available at runtime
try:
    from fastmcp import FastMCP  # type: ignore[import-not-found]
except ImportError:
    # Stub for type checking when FastMCP not installed
    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, name: str) -> None:
            self.name = name
        def tool(self):  # type: ignore[no-untyped-def]
            def decorator(func):  # type: ignore[no-untyped-def]
                return func
            return decorator

from src.core.config import get_settings
from src.core.database import DatabasePool
from src.core.logging import StructuredLogger
from src.mcp.auth import rate_limiter, validate_api_key
from src.search.hybrid_search import HybridSearch

# Initialize logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Initialize FastMCP app
mcp = FastMCP("bmcis-knowledge-mcp")

# Global state (initialized on startup)
_db_pool: DatabasePool | None = None
_hybrid_search: HybridSearch | None = None
_cache_layer: Any | None = None  # CacheLayer from src.mcp.cache (imported below)


def initialize_server() -> None:
    """Initialize server dependencies (database, search, authentication, cache).

    Called on server startup. Initializes:
    - DatabasePool (PostgreSQL connection pool)
    - HybridSearch (vector + BM25 search with RRF merging)
    - Authentication (API key validation and rate limiting)
    - CacheLayer (in-memory cache with TTL and LRU eviction)

    Raises:
        RuntimeError: If initialization fails

    Example:
        >>> initialize_server()
        >>> # Server components ready for use
    """
    global _db_pool, _hybrid_search, _cache_layer

    logger.info("Initializing BMCIS Knowledge MCP server...")

    try:
        # Load settings
        settings = get_settings()

        # Verify API key is configured (optional - log warning if missing)
        try:
            # Test if BMCIS_API_KEY is set (will raise ValueError if not)
            validate_api_key("test_key_for_config_check")
        except ValueError as e:
            if "environment variable not set" in str(e):
                logger.warning(
                    "BMCIS_API_KEY not configured. Authentication disabled. "
                    "Set BMCIS_API_KEY environment variable to enable authentication."
                )
            # Other errors are fine (we're just testing config)

        # Log rate limiter configuration
        logger.info(
            f"Rate limiter configured: {rate_limiter.rpm_limit}/min, "
            f"{rate_limiter.rph_limit}/hr, {rate_limiter.rpd_limit}/day"
        )

        # Initialize database pool
        _db_pool = DatabasePool()
        logger.info("Database pool initialized")

        # Initialize hybrid search
        structured_logger = StructuredLogger.get_logger(__name__)
        _hybrid_search = HybridSearch(
            db_pool=_db_pool,
            settings=settings,
            logger=structured_logger,  # type: ignore[arg-type]
        )
        logger.info("HybridSearch initialized")

        # Initialize cache layer
        from src.mcp.cache import CacheLayer

        _cache_layer = CacheLayer(
            max_entries=1000,  # Max 1000 cached queries
            default_ttl=300,  # 5 minute default TTL
            enable_metrics=True,  # Track hit/miss rates
        )
        logger.info("CacheLayer initialized (max_entries=1000, default_ttl=300s)")

        logger.info("BMCIS Knowledge MCP server ready")

    except Exception as e:
        logger.error(f"Server initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize MCP server: {e}") from e


def get_hybrid_search() -> HybridSearch:
    """Get initialized HybridSearch instance.

    Returns:
        HybridSearch: Initialized search instance

    Raises:
        RuntimeError: If server not initialized

    Example:
        >>> search = get_hybrid_search()
        >>> results = search.search("JWT authentication", top_k=10)
    """
    if _hybrid_search is None:
        raise RuntimeError("Server not initialized. Call initialize_server() first.")
    return _hybrid_search


def get_database_pool() -> DatabasePool:
    """Get initialized DatabasePool instance.

    Returns:
        DatabasePool: Initialized database connection pool

    Raises:
        RuntimeError: If server not initialized

    Example:
        >>> pool = get_database_pool()
        >>> with pool.get_connection() as conn:
        ...     # Use connection
    """
    if _db_pool is None:
        raise RuntimeError("Server not initialized. Call initialize_server() first.")
    return _db_pool


def get_cache_layer() -> Any:
    """Get initialized CacheLayer instance.

    Returns:
        CacheLayer: Initialized cache layer for MCP tools

    Raises:
        RuntimeError: If server not initialized

    Example:
        >>> cache = get_cache_layer()
        >>> cache.set("key", "value", ttl_seconds=60)
        >>> value = cache.get("key")
    """
    if _cache_layer is None:
        raise RuntimeError("Server not initialized. Call initialize_server() first.")
    return _cache_layer


# Tool imports (register tools with FastMCP)
# Import after function definitions to avoid circular imports
from src.mcp.tools.find_vendor_info import find_vendor_info  # noqa: E402, F401
from src.mcp.tools.semantic_search import semantic_search  # noqa: E402, F401

# Initialize on module load (automatic startup)
# Comment out for testing to allow manual initialization
try:
    initialize_server()
except (ImportError, RuntimeError) as e:
    logger.warning(f"Auto-initialization skipped (likely testing): {e}")
