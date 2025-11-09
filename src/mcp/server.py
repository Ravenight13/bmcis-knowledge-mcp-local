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
from typing import Optional

# FastMCP will be available at runtime
try:
    from fastmcp import FastMCP
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
from src.search.hybrid_search import HybridSearch

# Initialize logger
logger: logging.Logger = StructuredLogger.get_logger(__name__)

# Initialize FastMCP app
mcp = FastMCP("bmcis-knowledge-mcp")

# Global state (initialized on startup)
_db_pool: Optional[DatabasePool] = None
_hybrid_search: Optional[HybridSearch] = None


def initialize_server() -> None:
    """Initialize server dependencies (database, search).

    Called on server startup. Initializes:
    - DatabasePool (PostgreSQL connection pool)
    - HybridSearch (vector + BM25 search with RRF merging)

    Raises:
        RuntimeError: If initialization fails

    Example:
        >>> initialize_server()
        >>> # Server components ready for use
    """
    global _db_pool, _hybrid_search

    logger.info("Initializing BMCIS Knowledge MCP server...")

    try:
        # Load settings
        settings = get_settings()

        # Initialize database pool
        _db_pool = DatabasePool()
        logger.info("Database pool initialized")

        # Initialize hybrid search
        _hybrid_search = HybridSearch(
            db_pool=_db_pool,
            settings=settings,
            logger=logger,
        )
        logger.info("HybridSearch initialized")

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


# Tool imports (register tools with FastMCP)
# Import after function definitions to avoid circular imports
from src.mcp.tools.semantic_search import semantic_search  # noqa: E402, F401

# Initialize on module load (automatic startup)
# Comment out for testing to allow manual initialization
try:
    initialize_server()
except Exception as e:
    logger.warning(f"Auto-initialization skipped (likely testing): {e}")
