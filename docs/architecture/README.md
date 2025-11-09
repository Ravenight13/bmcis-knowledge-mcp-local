# Architecture Documentation

This directory contains system architecture documentation, data models, and technical design specifications.

## Available Documentation

### Data Models & Flow
- **[DATA_MODELS_ANALYSIS.md](./DATA_MODELS_ANALYSIS.md)** - Comprehensive analysis of all data models and their relationships
- **[DATA_MODELS_QUICK_REFERENCE.md](./DATA_MODELS_QUICK_REFERENCE.md)** - Quick reference guide for data model structures
- **[DATA_FLOW_DIAGRAM.md](./DATA_FLOW_DIAGRAM.md)** - Complete data flow diagrams showing system interactions

## Architecture Overview

The system is organized into several key components:

1. **Embedding Module** - Handles document vectorization and semantic search
2. **Search Module** - Provides vector, keyword, hybrid, and RRF search capabilities
3. **Database Layer** - PostgreSQL with pgvector extension for vector storage
4. **MCP Server** - Model Context Protocol server for AI tool integration

## Quick Reference

For quick lookups of data structures and schemas:
- Start with [DATA_MODELS_QUICK_REFERENCE.md](./DATA_MODELS_QUICK_REFERENCE.md)
- For detailed analysis, see [DATA_MODELS_ANALYSIS.md](./DATA_MODELS_ANALYSIS.md)
- For understanding data movement, see [DATA_FLOW_DIAGRAM.md](./DATA_FLOW_DIAGRAM.md)

## Related Documentation

- **Guides**: See [../guides/](../guides/) for development workflows
- **Reference**: See [../reference/](../reference/) for analysis indexes
- **MCP Integration**: See [../mcp-as-tools/](../mcp-as-tools/) for MCP server details
