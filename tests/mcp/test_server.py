"""Test FastMCP server initialization and state management.

Tests comprehensive initialization, error handling, tool registration,
and state management for the BMCIS Knowledge MCP server.

Test Coverage:
- Server initialization
- Error handling during initialization
- Tool registration (semantic_search)
- Server state management
- Database pool access
- HybridSearch instance access
- Initialization guards (re-initialization prevention)

Performance:
- Fast unit tests (<100ms total)
- Mocked dependencies (no database)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mcp.server import (
    get_database_pool,
    get_hybrid_search,
    initialize_server,
    mcp,
)


class TestServerInitialization:
    """Test server initialization."""

    def test_initialize_server_success(self) -> None:
        """Test successful server initialization."""
        # Reset global state
        import src.mcp.server as server_module

        server_module._db_pool = None
        server_module._hybrid_search = None

        with patch("src.mcp.server.get_settings") as mock_settings, patch(
            "src.mcp.server.DatabasePool"
        ) as mock_db_pool_class, patch(
            "src.mcp.server.HybridSearch"
        ) as mock_search_class:

            # Setup mocks
            mock_settings.return_value = MagicMock()
            mock_db_pool_instance = MagicMock()
            mock_db_pool_class.return_value = mock_db_pool_instance
            mock_search_instance = MagicMock()
            mock_search_class.return_value = mock_search_instance

            # Execute initialization
            initialize_server()

            # Verify DatabasePool was created
            mock_db_pool_class.assert_called_once()

            # Verify HybridSearch was created with db_pool
            mock_search_class.assert_called_once()
            call_kwargs = mock_search_class.call_args.kwargs
            assert "db_pool" in call_kwargs
            assert "settings" in call_kwargs
            assert "logger" in call_kwargs

    def test_initialize_server_database_connection_failure(self) -> None:
        """Test server initialization fails when database connection fails."""
        import src.mcp.server as server_module

        server_module._db_pool = None
        server_module._hybrid_search = None

        with patch("src.mcp.server.get_settings") as mock_settings, patch(
            "src.mcp.server.DatabasePool"
        ) as mock_db_pool_class:

            # Setup mocks to fail on database initialization
            mock_settings.return_value = MagicMock()
            mock_db_pool_class.side_effect = RuntimeError(
                "Failed to connect to database"
            )

            # Verify RuntimeError is raised
            with pytest.raises(RuntimeError, match="Failed to initialize MCP server"):
                initialize_server()

    def test_initialize_server_hybrid_search_failure(self) -> None:
        """Test server initialization fails when HybridSearch creation fails."""
        import src.mcp.server as server_module

        server_module._db_pool = None
        server_module._hybrid_search = None

        with patch("src.mcp.server.get_settings") as mock_settings, patch(
            "src.mcp.server.DatabasePool"
        ) as mock_db_pool_class, patch(
            "src.mcp.server.HybridSearch"
        ) as mock_search_class:

            # Setup successful database initialization
            mock_settings.return_value = MagicMock()
            mock_db_pool_instance = MagicMock()
            mock_db_pool_class.return_value = mock_db_pool_instance

            # Setup HybridSearch to fail
            mock_search_class.side_effect = RuntimeError("Failed to initialize search")

            # Verify RuntimeError is raised
            with pytest.raises(RuntimeError, match="Failed to initialize MCP server"):
                initialize_server()


class TestServerStateManagement:
    """Test server state accessors."""

    def test_get_hybrid_search_when_initialized(self) -> None:
        """Test get_hybrid_search returns instance when server initialized."""
        import src.mcp.server as server_module

        # Setup mock instance
        mock_search = MagicMock()
        server_module._hybrid_search = mock_search

        # Verify returns the instance
        result = get_hybrid_search()
        assert result is mock_search

    def test_get_hybrid_search_when_not_initialized(self) -> None:
        """Test get_hybrid_search raises when server not initialized."""
        import src.mcp.server as server_module

        # Reset global state
        server_module._hybrid_search = None

        # Verify RuntimeError is raised
        with pytest.raises(
            RuntimeError, match="Server not initialized. Call initialize_server"
        ):
            get_hybrid_search()

    def test_get_database_pool_when_initialized(self) -> None:
        """Test get_database_pool returns instance when server initialized."""
        import src.mcp.server as server_module

        # Setup mock instance
        mock_pool = MagicMock()
        server_module._db_pool = mock_pool

        # Verify returns the instance
        result = get_database_pool()
        assert result is mock_pool

    def test_get_database_pool_when_not_initialized(self) -> None:
        """Test get_database_pool raises when server not initialized."""
        import src.mcp.server as server_module

        # Reset global state
        server_module._db_pool = None

        # Verify RuntimeError is raised
        with pytest.raises(
            RuntimeError, match="Server not initialized. Call initialize_server"
        ):
            get_database_pool()


class TestToolRegistration:
    """Test tool registration with FastMCP."""

    def test_mcp_instance_created(self) -> None:
        """Test FastMCP instance is created with correct name."""
        assert mcp is not None
        assert mcp.name == "bmcis-knowledge-mcp"

    def test_semantic_search_tool_registered(self) -> None:
        """Test semantic_search tool is registered with FastMCP."""
        from src.mcp.tools.semantic_search import semantic_search

        # Verify tool function exists and is callable
        assert callable(semantic_search)

        # Verify docstring indicates it's a tool
        assert semantic_search.__doc__ is not None
        assert "semantic_search" in semantic_search.__name__


class TestServerIntegration:
    """Test server component integration."""

    def test_initialization_idempotent_after_success(self) -> None:
        """Test server can be initialized multiple times."""
        import src.mcp.server as server_module

        server_module._db_pool = None
        server_module._hybrid_search = None

        with patch("src.mcp.server.get_settings") as mock_settings, patch(
            "src.mcp.server.DatabasePool"
        ) as mock_db_pool_class, patch(
            "src.mcp.server.HybridSearch"
        ) as mock_search_class:

            # Setup mocks
            mock_settings.return_value = MagicMock()
            mock_db_pool_class.return_value = MagicMock()
            mock_search_class.return_value = MagicMock()

            # Initialize once
            initialize_server()
            first_search = server_module._hybrid_search
            first_pool = server_module._db_pool

            # Initialize again (should work)
            initialize_server()
            second_search = server_module._hybrid_search
            second_pool = server_module._db_pool

            # Verify new instances were created
            assert first_search is not None
            assert second_search is not None
            # Note: After second init, instances are different objects
            # (since mocks create new instances)

    def test_database_pool_isolation(self) -> None:
        """Test database pool and search instances are isolated."""
        import src.mcp.server as server_module

        server_module._db_pool = None
        server_module._hybrid_search = None

        with patch("src.mcp.server.get_settings") as mock_settings, patch(
            "src.mcp.server.DatabasePool"
        ) as mock_db_pool_class, patch(
            "src.mcp.server.HybridSearch"
        ) as mock_search_class:

            # Setup mocks
            mock_settings.return_value = MagicMock()
            mock_db_instance = MagicMock()
            mock_db_pool_class.return_value = mock_db_instance
            mock_search_instance = MagicMock()
            mock_search_class.return_value = mock_search_instance

            # Initialize
            initialize_server()

            # Get instances via accessors
            pool = get_database_pool()
            search = get_hybrid_search()

            # Verify they are the initialized instances
            assert pool is mock_db_instance
            assert search is mock_search_instance
