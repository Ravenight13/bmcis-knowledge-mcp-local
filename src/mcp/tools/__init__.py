"""MCP tools for BMCIS Knowledge MCP server.

This module exports all MCP tool implementations.

Available Tools:
- semantic_search: Hybrid semantic search with progressive disclosure

Example:
    >>> from src.mcp.tools import semantic_search
    >>> response = semantic_search("JWT authentication", top_k=10)
"""

from src.mcp.tools.semantic_search import semantic_search

__all__ = [
    "semantic_search",
]
