#!/usr/bin/env python3
"""MCP Server entry point for FastMCP.

Runs the BMCIS Knowledge MCP server via stdio transport for Claude Code/Desktop integration.
"""

import asyncio
import sys

from src.mcp.server import mcp, initialize_server


async def main() -> None:
    """Initialize and run MCP server."""
    # Initialize server components (database, search, auth, cache)
    initialize_server()

    # Run FastMCP server via stdio
    # This is the standard way FastMCP servers communicate with MCP clients
    try:
        await mcp.run_stdio()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
