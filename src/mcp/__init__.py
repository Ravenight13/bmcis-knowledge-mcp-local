"""FastMCP server for BMCIS Knowledge MCP.

Type-safe MCP server exposing semantic search to Claude Desktop via progressive disclosure.

Main Components:
- models: Pydantic schemas for request/response validation
- server: FastMCP initialization and dependency management
- tools: MCP tool implementations (semantic_search, etc.)

Example:
    # Start MCP server (production)
    >>> from src.mcp.server import mcp, initialize_server
    >>> initialize_server()
    >>> # Server runs via FastMCP CLI

    # Use search directly (testing)
    >>> from src.mcp.server import get_hybrid_search
    >>> search = get_hybrid_search()
    >>> results = search.search("JWT authentication", top_k=10)
"""

from src.mcp.models import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SearchResultIDs,
    SearchResultMetadata,
    SearchResultPreview,
    SearchResultFull,
)
from src.mcp.server import (
    mcp,
    initialize_server,
    get_hybrid_search,
    get_database_pool,
)

__all__ = [
    # Models
    "SemanticSearchRequest",
    "SemanticSearchResponse",
    "SearchResultIDs",
    "SearchResultMetadata",
    "SearchResultPreview",
    "SearchResultFull",
    # Server
    "mcp",
    "initialize_server",
    "get_hybrid_search",
    "get_database_pool",
]
