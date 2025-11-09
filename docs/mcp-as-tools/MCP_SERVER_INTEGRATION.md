# MCP Server Integration Guide

**Integrating Code Execution with BMCIS Knowledge MCP Server**

---

## Overview

This guide shows how to integrate the code execution capabilities into your existing BMCIS Knowledge MCP server, enabling agents to run code that accesses your search, reranking, and filtering capabilities.

---

## Current Architecture

Your BMCIS Knowledge MCP server currently exposes tools directly:

```
Claude Agent
    ↓
MCP Server
    ├─ Tool: bm25_search
    ├─ Tool: vector_search
    ├─ Tool: rerank_results
    ├─ Tool: filter_results
    └─ Tool: get_document
```

**New Architecture:**

```
Claude Agent
    ↓
MCP Server
    ├─ Traditional Tools (keep for backward compatibility)
    │   ├─ Tool: bm25_search
    │   ├─ Tool: vector_search
    │   └─ ...
    │
    └─ NEW Code Execution Tool
        └─ Tool: execute_code
            ↓
            Sandbox Environment
                ├─ HybridSearchAPI (in-memory)
                ├─ RerankerAPI (in-memory)
                ├─ FilterAPI (in-memory)
                └─ ResultProcessor (in-memory)
```

---

## Implementation Steps

### Step 1: Update `.mcp.json`

Your current `.mcp.json` looks like:

```json
{
  "mcpServers": {
    "task-master-ai": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "task-master-ai"],
      "env": {...}
    }
  }
}
```

Update to include the code execution sandbox if you're running it separately:

```json
{
  "mcpServers": {
    "task-master-ai": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "task-master-ai"],
      "env": {...}
    },
    "bmcis-knowledge": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "/path/to/bmcis-knowledge-mcp"
      }
    }
  }
}
```

### Step 2: Locate Your MCP Server Entry Point

Find your main server file (likely `src/mcp_server.py` or `src/server.py`):

```bash
find src -name "*server*.py" -type f
```

You should have something like:

```python
# src/mcp_server.py
import asyncio
from mcp.server import Server
from mcp.types import Tool

app = Server("bmcis-knowledge-mcp")

@app.list_tools()
async def list_tools():
    # Return list of available tools
    pass

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    # Execute tools
    pass

if __name__ == "__main__":
    asyncio.run(app.run(sys.stdin.buffer, sys.stdout.buffer))
```

### Step 3: Add Code Execution Tool to Server

Modify your server to include the new tool:

```python
# src/mcp_server.py
import asyncio
from mcp.server import Server
from mcp.types import Tool
from src.mcp_tools.code_execution_tool import CodeExecutionTool
from src.code_execution.agent_interface import get_executor

app = Server("bmcis-knowledge-mcp")

# Initialize code execution
code_executor = get_executor()
code_tool = CodeExecutionTool()

@app.list_tools()
async def list_tools():
    """List all available tools"""
    tools = [
        # ... your existing tools ...
        Tool(
            name="execute_code",
            description=code_tool.description,
            inputSchema=code_tool.to_mcp_definition()["inputSchema"]
        )
    ]
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool by name"""

    # Route to appropriate handler
    if name == "execute_code":
        return await handle_code_execution(arguments)
    elif name == "bm25_search":
        return await handle_bm25_search(arguments)
    # ... other tools ...
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_code_execution(arguments: dict):
    """Execute agent-written code"""
    code = arguments.get("code")
    description = arguments.get("description", "")

    result = await code_executor.execute_search_workflow(
        code=code,
        description=description
    )

    return {
        "type": "text",
        "text": str(result)
    }

# ... other handlers ...

if __name__ == "__main__":
    import sys
    asyncio.run(app.run(sys.stdin.buffer, sys.stdout.buffer))
```

### Step 4: Create the Code Execution MCP Tool

Create `src/mcp_tools/code_execution_tool.py`:

```python
"""MCP Tool definition for code execution"""

from typing import Any
from src.code_execution.agent_interface import get_executor


class CodeExecutionTool:
    """MCP tool wrapper for code execution"""

    def __init__(self):
        self.executor = get_executor()

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return """
Execute Python code that uses BMCIS Knowledge APIs for search, reranking, and filtering.

The code execution environment provides access to:

1. **HybridSearchAPI** - Perform BM25 + vector search fusion
   ```python
   from src.code_api.search import HybridSearchAPI
   api = HybridSearchAPI()
   results = await api.hybrid_search(
       query="your question",
       top_k=5,
       bm25_weight=0.5,
       vector_weight=0.5
   )
   ```

2. **RerankerAPI** - Rerank results using cross-encoder models
   ```python
   from src.code_api.reranking import RerankerAPI
   reranker = RerankerAPI()
   reranked = await reranker.rerank(
       query="your question",
       documents=results,
       top_k=5
   )
   ```

3. **FilterAPI** - Filter results by domain, score, or fields
   ```python
   from src.code_api.filtering import FilterAPI
   filter_api = FilterAPI()
   filtered = filter_api.filter_by_domain(results, "specific_domain")
   ```

4. **ResultProcessor** - Format results compactly
   ```python
   from src.code_api.results import ResultProcessor
   processor = ResultProcessor()
   summary = processor.format_for_agent(results)
   ```

IMPORTANT: Assign final results to `__result__` variable:
```python
__result__ = final_results
```

This approach reduces token overhead by 98%, from 150K+ to ~2K tokens.

For examples, see CODE_EXECUTION_WITH_MCP.md in the docs folder.
"""

    def to_mcp_definition(self) -> dict[str, Any]:
        """Convert to MCP tool definition"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Assign result to __result__."
                    },
                    "description": {
                        "type": "string",
                        "description": "What this code does (for logging)"
                    }
                },
                "required": ["code"]
            }
        }

    async def execute(self, code: str, description: str = "") -> dict:
        """Execute code safely"""
        return await self.executor.execute_search_workflow(code, description)
```

### Step 5: Test the Integration

Create a test file to verify the tool works:

```python
# tests/integration/test_mcp_code_execution.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.mcp_tools.code_execution_tool import CodeExecutionTool


@pytest.fixture
def code_tool():
    return CodeExecutionTool()


def test_tool_definition(code_tool):
    """Verify tool definition is properly formatted"""
    definition = code_tool.to_mcp_definition()

    assert definition["name"] == "execute_code"
    assert "inputSchema" in definition
    assert "code" in definition["inputSchema"]["properties"]


@pytest.mark.asyncio
async def test_simple_code_execution(code_tool):
    """Test basic code execution"""
    code = """
x = 42
__result__ = x
"""

    result = await code_tool.execute(code)

    assert result["success"]
    assert result["result"] == 42


@pytest.mark.asyncio
async def test_search_api_in_execution(code_tool):
    """Test that search API is available in execution context"""
    code = """
from src.code_api.search import HybridSearchAPI
api = HybridSearchAPI()
# Would normally call: results = await api.hybrid_search("test", top_k=5)
__result__ = "Search API available"
"""

    result = await code_tool.execute(code)

    assert result["success"]
    assert "Search API available" in str(result["result"])
```

Run the test:

```bash
pytest tests/integration/test_mcp_code_execution.py -v
```

### Step 6: Update Agent System Prompt

Inform agents about the new tool by updating your system prompt:

```markdown
## Available Tools

### execute_code

Execute Python code that uses BMCIS Knowledge APIs.

**Example Usage:**

```python
code = '''
from src.code_api.search import HybridSearchAPI

api = HybridSearchAPI()
results = await api.hybrid_search("What is BMCIS?", top_k=5)

summary = [{
    "title": r.title,
    "score": r.relevance_score
} for r in results]

__result__ = summary
'''

# Then call the tool with this code
```

**Benefits:**
- 98% token reduction compared to traditional tool calls
- Faster execution through in-environment processing
- Better privacy (intermediate results stay in sandbox)
- Support for complex workflows (search → rerank → filter)

**Best Practices:**
1. Use code execution for complex multi-step workflows
2. Keep code simple and readable
3. Always assign final result to `__result__`
4. Return only essential fields (use ResultProcessor.summarize())
```

---

## Backward Compatibility

Your existing tools can coexist with code execution:

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool - supporting both new and old tools"""

    # New code execution tool
    if name == "execute_code":
        return await handle_code_execution(arguments)

    # Old direct-call tools (still supported)
    elif name == "bm25_search":
        return await handle_bm25_search(arguments)
    elif name == "vector_search":
        return await handle_vector_search(arguments)
    elif name == "rerank_results":
        return await handle_rerank(arguments)
    # ... etc
```

This allows gradual migration - agents can use old tools while you test new ones.

---

## Configuration Options

### Adjust Sandbox Limits

Edit `src/code_execution/sandbox.py`:

```python
class CodeExecutionSandbox:
    def __init__(
        self,
        timeout_seconds: int = 30,      # Change timeout
        max_memory_mb: int = 512        # Change memory limit
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
```

### Add Allowed Imports

```python
ALLOWED_IMPORTS = {
    'bmcis_knowledge': ['search', 'reranking', 'filtering'],
    'json': [],
    'datetime': [],
    're': [],
    'statistics': [],
    'numpy': ['array', 'mean'],      # Add if needed
    'pandas': ['DataFrame'],          # Add if needed
}
```

### Change Output Filtering

```python
# In src/code_api/results.py
ResultProcessor.summarize(
    results,
    max_content_length=150,  # Adjust preview length
    fields=['id', 'title', 'score']  # Specify fields to include
)
```

---

## Deployment Checklist

- [ ] Code API modules created (`src/code_api/`)
- [ ] Sandbox implemented (`src/code_execution/`)
- [ ] MCP tool defined (`src/mcp_tools/code_execution_tool.py`)
- [ ] Server integration complete (`src/mcp_server.py`)
- [ ] Unit tests passing (`pytest tests/test_code_api.py`)
- [ ] Security tests passing (`pytest tests/test_sandbox_security.py`)
- [ ] Integration tests passing (`pytest tests/integration/test_mcp_code_execution.py`)
- [ ] System prompt updated with examples
- [ ] Performance benchmarks measured
- [ ] Error handling tested
- [ ] Production deploy plan ready

---

## Monitoring & Logging

Add logging to track code execution:

```python
import logging

logger = logging.getLogger(__name__)

async def handle_code_execution(arguments: dict):
    """Execute agent-written code with logging"""
    code = arguments.get("code")
    description = arguments.get("description", "")

    logger.info(f"Executing code: {description}")
    start_time = time.time()

    result = await code_executor.execute_search_workflow(code, description)

    execution_time = time.time() - start_time
    logger.info(f"Execution completed in {execution_time:.2f}s")

    if not result["success"]:
        logger.warning(f"Execution failed: {result.get('error')}")

    # Log metrics for monitoring
    logger.debug({
        "execution_time": execution_time,
        "success": result["success"],
        "token_overhead": len(str(result["result"])) // 4
    })

    return {"type": "text", "text": str(result)}
```

Configure logging in your server startup:

```python
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stderr',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
        'src.code_execution': {
            'level': 'DEBUG',
            'propagate': True
        }
    }
})
```

---

## Troubleshooting

### Tool not appearing in `list_tools()`

**Issue**: Agent doesn't see the `execute_code` tool

**Solution**:
```python
# Make sure you're returning it in list_tools()
@app.list_tools()
async def list_tools():
    tools = [
        # ... existing tools ...
        Tool(
            name="execute_code",
            description=code_tool.description,
            inputSchema=code_tool.to_mcp_definition()["inputSchema"]
        )
    ]
    return tools
```

### Code execution fails with import errors

**Issue**: `Non-whitelisted import` error

**Solution**: Update `ALLOWED_IMPORTS` in `src/code_execution/sandbox.py`

### Execution timeout

**Issue**: Code takes too long to run

**Solution**: Increase timeout or optimize code:
```python
sandbox = CodeExecutionSandbox(timeout_seconds=60)
```

### Token usage not improving

**Issue**: Still seeing high token counts

**Solution**: Make sure code returns compact results:
```python
# ❌ Wrong: Returning too much
__result__ = raw_search_results  # 50K tokens!

# ✅ Right: Returning summary
__result__ = [{
    'id': r['id'],
    'title': r['title'],
    'score': r['score']
} for r in results[:5]]  # ~100 tokens
```

---

## Performance Metrics

Track these metrics to verify improvement:

```python
# Collect metrics
metrics = {
    'token_reduction': token_count_before / token_count_after,
    'execution_time': end_time - start_time,
    'error_rate': errors / total_executions,
    'memory_peak': peak_memory_mb,
}

# Target values
assert metrics['token_reduction'] > 50  # Expect >98%
assert metrics['execution_time'] < 5    # Expect <3s
assert metrics['error_rate'] < 0.01     # Expect <1%
assert metrics['memory_peak'] < 500     # Expect <512MB
```

---

## Next Steps

1. Implement Phase 1: Code API modules
2. Implement Phase 2: Sandbox
3. **Implement Phase 3: MCP integration (this guide)**
4. Test with sample agent code
5. Measure token reduction
6. Deploy to production
7. Monitor performance

---

## Resources

- **Main Implementation Guide**: `CODE_EXECUTION_WITH_MCP.md`
- **Quick Start**: `CODE_EXECUTION_QUICK_START.md`
- **MCP Specification**: https://modelcontextprotocol.io/
- **Your MCP Server Code**: `src/mcp_server.py`

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Status**: Ready for Implementation
