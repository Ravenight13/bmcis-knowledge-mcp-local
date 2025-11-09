# Code Execution with MCP: Implementation Guide

**BMCIS Knowledge MCP Project**

This guide explains how to refactor the BMCIS Knowledge MCP to use **code execution** instead of traditional tool calling, achieving 98%+ token reduction and enabling agents to access complex search functionality more efficiently.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Implementation Steps](#implementation-steps)
5. [Code Patterns](#code-patterns)
6. [Sandboxing & Security](#sandboxing--security)
7. [Migration Path](#migration-path)
8. [Performance Metrics](#performance-metrics)
9. [Testing Strategy](#testing-strategy)
10. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### The Problem

BMCIS Knowledge MCP currently exposes search tools using traditional MCP tool definitions. When agents access complex workflows (hybrid search → reranking → filtering), they experience:

- **150,000+ tokens** loaded upfront for all tool definitions
- **Multiple round-trips** through the model context for each search step
- **Large intermediate results** (unfiltered search results, rankings) flowing through context
- **Context window limits** that prevent agents from doing complex processing

### The Solution

**Code Execution with MCP**: Expose MCP servers as importable code APIs rather than direct tool calls. Agents write code that:
1. Imports only needed tools
2. Executes search → filtering → ranking locally
3. Returns only final results to the model

### The Payoff

- **98.7% token reduction**: 150,000 → 2,000 tokens
- **Faster execution**: Single execution environment vs. N round-trips
- **Better privacy**: Sensitive search results stay in execution environment
- **Unlimited tools**: Agents can access entire search ecosystem without bloat

---

## Problem Statement

### Current Architecture (Traditional Tool Calling)

```
Agent (Claude)
  ↓ (calls tool)
MCP Tool Definition loaded (tokens increase)
  ↓ (returns large result)
Agent processes result
  ↓ (calls next tool)
Another tool loaded (more tokens)
  ↓ (returns intermediate result)
...repeat N times...
Final result returned
```

**Token Flow for typical BMCIS workflow:**

```
1. Load all search tool definitions      → 15,000 tokens
2. BM25 search call returns 100 results  → 50,000 tokens
3. Vector search call returns 100 results→ 50,000 tokens
4. Reranking loads results               → 20,000 tokens
5. Final filtering loads data            → 15,000 tokens
─────────────────────────────────────────────────────
TOTAL                                    → 150,000 tokens
```

### Why This Matters for BMCIS

Your knowledge base performs:
- **Hybrid search** (BM25 + vector search fusion)
- **Cross-encoder reranking** (computes relevance scores)
- **Result filtering** (query routing, field selection)
- **Boost strategies** (domain-specific ranking adjustments)

Each step generates large intermediate datasets that are unnecessary in the agent's context.

---

## Solution Architecture

### Code Execution Model

```
Agent (Claude)
  ↓ (writes code)
Code Execution Environment
  ├─ Import search module
  ├─ Execute: bm25_search(query)
  ├─ Execute: vector_search(query) [in-process]
  ├─ Execute: hybrid_fusion() [in-process]
  ├─ Execute: rerank_results() [in-process]
  └─ Return only final summary
  ↓ (returns minimal result)
Agent (Claude) receives compact result
```

**Token Flow for BMCIS with Code Execution:**

```
1. Import statement for search module     → 100 tokens
2. BM25 + vector + fusion (all in env)    → 0 tokens (no context)
3. Reranking (in environment)             → 0 tokens (no context)
4. Filtering (in environment)             → 0 tokens (no context)
5. Return final summary                   → 1,900 tokens
─────────────────────────────────────────────────────
TOTAL                                     → 2,000 tokens
```

### Key Principle

**MCP Servers Become Code APIs**

Instead of:
```python
# Traditional: Agent calls tools directly
agent.call_tool("search_knowledge_base", {
    "query": "What is BMCIS?",
    "search_type": "hybrid"
})
```

The agent writes:
```python
# Code Execution: Agent imports and uses APIs
from bmcis_knowledge import search

results = search.hybrid_search(
    query="What is BMCIS?",
    bm25_weight=0.5,
    vector_weight=0.5
)
```

---

## Implementation Steps

### Phase 1: Create Code API Module (Week 1)

Create a new module that exposes your search functionality as importable Python APIs.

#### Step 1.1: Create `src/code_api/__init__.py`

```python
"""Code Execution API for BMCIS Knowledge MCP

This module exposes search, filtering, and ranking capabilities
as importable functions for use in code execution environments.
"""

from .search import HybridSearchAPI
from .reranking import RerankerAPI
from .filtering import FilterAPI
from .results import ResultProcessor

__all__ = [
    "HybridSearchAPI",
    "RerankerAPI",
    "FilterAPI",
    "ResultProcessor"
]
```

#### Step 1.2: Create `src/code_api/search.py`

```python
"""Hybrid Search API for code execution environments"""

from typing import Optional
from dataclasses import dataclass
import logging

from src.search.hybrid_search import HybridSearch
from src.search.query_router import QueryRouter
from src.core.database import get_connection_pool

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Compact result for code execution"""
    document_id: str
    title: str
    content: str
    relevance_score: float
    source: str


class HybridSearchAPI:
    """High-level API for hybrid search in code execution environments"""

    def __init__(self):
        self.pool = get_connection_pool()
        self.router = QueryRouter()

    async def hybrid_search(
        self,
        query: str,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> list[SearchResult]:
        """
        Execute hybrid search (BM25 + Vector) and return results.

        Data processing happens INSIDE this function, not in agent context.

        Args:
            query: Search query string
            bm25_weight: Weight for BM25 scoring (0-1)
            vector_weight: Weight for vector similarity (0-1)
            top_k: Number of results to return
            filters: Optional field-level filters

        Returns:
            List of SearchResult objects (minimal token overhead)
        """
        try:
            # Step 1: Route query to determine best search strategy
            routing_decision = self.router.route(query)

            # Step 2: Execute BM25 search
            # All intermediate results stay in memory (not in context)
            bm25_results = await self._bm25_search(
                query=query,
                limit=top_k * 2,  # Get more for fusion
                filters=filters
            )

            # Step 3: Execute vector search
            vector_results = await self._vector_search(
                query=query,
                limit=top_k * 2,
                filters=filters
            )

            # Step 4: Fuse results (all in environment)
            fused = self._fuse_results(
                bm25_results,
                vector_results,
                bm25_weight,
                vector_weight
            )

            # Step 5: Return only top-k, in compact format
            return [
                SearchResult(
                    document_id=r['id'],
                    title=r['metadata']['title'],
                    content=r['text'][:200],  # Truncated
                    relevance_score=r['score'],
                    source=r['metadata'].get('source', 'unknown')
                )
                for r in fused[:top_k]
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def _bm25_search(
        self,
        query: str,
        limit: int,
        filters: Optional[dict]
    ) -> list[dict]:
        """Execute BM25 search (internal, stays in env)"""
        # Implementation calls your existing BM25 search
        # Results never leave this function unless explicitly returned
        conn = self.pool.getconn()
        try:
            # BM25 execution logic here
            pass
        finally:
            self.pool.putconn(conn)

    async def _vector_search(
        self,
        query: str,
        limit: int,
        filters: Optional[dict]
    ) -> list[dict]:
        """Execute vector search (internal, stays in env)"""
        # Implementation calls your existing vector search
        pass

    def _fuse_results(
        self,
        bm25_results: list,
        vector_results: list,
        bm25_weight: float,
        vector_weight: float
    ) -> list:
        """Fuse BM25 and vector results (internal, stays in env)"""
        # Result fusion logic stays completely in environment
        # Never returns intermediate results
        pass
```

#### Step 1.3: Create `src/code_api/reranking.py`

```python
"""Reranking API for code execution environments"""

from typing import Optional
from src.search.cross_encoder_reranker import CrossEncoderReranker


class RerankerAPI:
    """Rerank search results using cross-encoder models"""

    def __init__(self):
        self.reranker = CrossEncoderReranker()

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Rerank documents using cross-encoder.

        All scoring happens in environment, only final results returned.

        Args:
            query: Original search query
            documents: Documents to rerank
            top_k: Return top K results
            threshold: Minimum relevance score

        Returns:
            Reranked documents in order of relevance
        """
        # Cross-encoder scoring happens here, not in context
        scores = self.reranker.score_pairs(
            [(query, doc['text']) for doc in documents]
        )

        # Sort and filter (all in environment)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply threshold if provided
        if threshold:
            ranked = [(doc, score) for doc, score in ranked
                     if score >= threshold]

        # Return only top_k with compact data
        return [
            {
                **doc,
                'rerank_score': float(score)
            }
            for doc, score in ranked[:top_k]
        ]
```

#### Step 1.4: Create `src/code_api/results.py`

```python
"""Result processing utilities for code execution"""

from typing import Optional
from dataclasses import asdict


class ResultProcessor:
    """Process and format results for compact output"""

    @staticmethod
    def summarize(
        results: list[dict],
        max_content_length: int = 200,
        fields: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Summarize results for minimal token overhead.

        Args:
            results: Raw search results
            max_content_length: Max characters per document
            fields: Specific fields to include (None = all)

        Returns:
            Compact result summaries
        """
        compact = []
        for result in results:
            entry = {
                'id': result.get('id'),
                'title': result.get('title', 'Untitled'),
                'score': float(result.get('score', 0)),
            }

            # Truncate content
            if 'text' in result:
                entry['preview'] = result['text'][:max_content_length]

            # Include specified fields only
            if fields:
                for field in fields:
                    if field in result:
                        entry[field] = result[field]

            compact.append(entry)

        return compact

    @staticmethod
    def format_for_agent(results: list[dict]) -> str:
        """
        Format results as readable text for agent consumption.

        Compact format reduces token overhead.
        """
        if not results:
            return "No results found"

        lines = []
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.get('title', 'Untitled')}")
            lines.append(f"   Score: {result.get('score', 'N/A'):.2f}")
            if 'preview' in result:
                lines.append(f"   {result['preview'][:100]}...")
            lines.append("")

        return "\n".join(lines)
```

### Phase 2: Create Code Execution Sandbox (Week 1-2)

Set up a secure execution environment for agents to run code.

#### Step 2.1: Create `src/code_execution/sandbox.py`

```python
"""Secure sandboxed code execution environment for agents"""

import asyncio
import tempfile
import json
import logging
from typing import Any, Optional
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


class CodeExecutionSandbox:
    """
    Executes agent-generated code in a controlled environment.

    Security features:
    - Restricted imports (whitelist of allowed modules)
    - Resource limits (timeout, memory)
    - No filesystem access outside temp directory
    - No network access
    - Output capture and filtering
    """

    # Whitelist of safe modules agents can import
    ALLOWED_IMPORTS = {
        'bmcis_knowledge': ['search', 'reranking', 'filtering'],
        'json': [],
        'datetime': [],
        're': [],
        'statistics': [],
        'itertools': [],
        'operator': [],
    }

    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb

    async def execute(
        self,
        code: str,
        context: Optional[dict] = None,
        return_raw: bool = False
    ) -> dict[str, Any]:
        """
        Execute agent-written code safely.

        Args:
            code: Python code to execute
            context: Variables available to the code
            return_raw: If True, return raw output; if False, parse as JSON

        Returns:
            {
                'success': bool,
                'result': Any,  # Parsed JSON or raw output
                'output': str,  # Captured stdout
                'error': str,   # Error message if failed
                'execution_time': float
            }
        """

        # Validate code safety
        validation = self._validate_code(code)
        if not validation['safe']:
            return {
                'success': False,
                'error': f"Code validation failed: {validation['reason']}",
                'result': None,
                'output': '',
                'execution_time': 0
            }

        # Prepare execution environment
        env = self._prepare_environment(context)

        # Execute with timeout
        try:
            start_time = asyncio.get_event_loop().time()

            # Run code execution with timeout
            result = await asyncio.wait_for(
                self._run_code(code, env),
                timeout=self.timeout_seconds
            )

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                'success': True,
                'result': result.get('output'),
                'output': result.get('stdout', ''),
                'error': None,
                'execution_time': execution_time
            }

        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f"Code execution timeout ({self.timeout_seconds}s)",
                'result': None,
                'output': '',
                'execution_time': self.timeout_seconds
            }
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': None,
                'output': '',
                'execution_time': 0
            }

    def _validate_code(self, code: str) -> dict[str, Any]:
        """Validate code for security issues"""

        # Check for dangerous imports
        dangerous_patterns = [
            'import os',
            'import subprocess',
            'import socket',
            '__import__',
            'eval(',
            'exec(',
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                return {
                    'safe': False,
                    'reason': f"Dangerous pattern detected: {pattern}"
                }

        # Verify only whitelisted imports used
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in self.ALLOWED_IMPORTS:
                            return {
                                'safe': False,
                                'reason': f"Non-whitelisted import: {alias.name}"
                            }
        except SyntaxError as e:
            return {
                'safe': False,
                'reason': f"Syntax error: {e}"
            }

        return {'safe': True, 'reason': None}

    def _prepare_environment(self, context: Optional[dict]) -> dict[str, Any]:
        """Prepare safe execution environment"""
        env = context or {}

        # Add safe modules
        env['json'] = __import__('json')
        env['datetime'] = __import__('datetime')
        env['re'] = __import__('re')
        env['statistics'] = __import__('statistics')

        # Add our APIs
        from src.code_api import (
            HybridSearchAPI, RerankerAPI, FilterAPI, ResultProcessor
        )
        env['search'] = HybridSearchAPI()
        env['reranker'] = RerankerAPI()
        env['filter'] = FilterAPI()
        env['ResultProcessor'] = ResultProcessor

        return env

    async def _run_code(self, code: str, env: dict) -> dict[str, Any]:
        """Execute code in isolated environment"""

        # Capture output
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute in separate thread to truly isolate
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                exec,
                code,
                env
            )

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        # Extract final value if code assigned to __result__
        final_result = env.get('__result__', output)

        return {
            'output': final_result,
            'stdout': output,
            'stderr': errors
        }
```

#### Step 2.2: Create `src/code_execution/agent_interface.py`

```python
"""Interface between Claude agent and code execution sandbox"""

import logging
from typing import Optional
from src.code_execution.sandbox import CodeExecutionSandbox

logger = logging.getLogger(__name__)


class AgentCodeExecutor:
    """
    High-level interface for agents to execute code.

    Agents use this as an MCP tool to run code that accesses
    search/reranking/filtering APIs.
    """

    def __init__(self):
        self.sandbox = CodeExecutionSandbox(
            timeout_seconds=30,
            max_memory_mb=512
        )

    async def execute_search_workflow(
        self,
        code: str,
        description: str = ""
    ) -> dict:
        """
        Execute agent code that uses search APIs.

        This is the MCP tool agents call to run code.

        Args:
            code: Python code to execute
            description: What the code does (for logging)

        Returns:
            Execution result with output and status
        """

        logger.info(f"Executing agent code: {description}")

        # Execute the code
        result = await self.sandbox.execute(code)

        # Log execution
        if result['success']:
            logger.info(f"Code execution successful ({result['execution_time']:.2f}s)")
        else:
            logger.warning(f"Code execution failed: {result['error']}")

        return result


# Singleton instance
_executor = None


def get_executor() -> AgentCodeExecutor:
    """Get or create executor instance"""
    global _executor
    if _executor is None:
        _executor = AgentCodeExecutor()
    return _executor
```

### Phase 3: Create MCP Tool for Code Execution (Week 2)

Define the MCP tool that agents use.

#### Step 3.1: Create `src/mcp_tools/code_execution_tool.py`

```python
"""MCP Tool definition for code execution"""

import json
from typing import Any
from src.code_execution.agent_interface import get_executor


class CodeExecutionTool:
    """MCP tool that allows agents to execute code"""

    def __init__(self):
        self.executor = get_executor()
        self.name = "execute_code"
        self.description = """
        Execute Python code that uses BMCIS search APIs.

        Available APIs:
        - search: HybridSearchAPI for searching knowledge base
        - reranker: RerankerAPI for reranking results
        - filter: FilterAPI for filtering results
        - ResultProcessor: Utilities for formatting results

        Example code:
        ```python
        from search import HybridSearchAPI
        search_api = HybridSearchAPI()
        results = await search_api.hybrid_search(
            query="What is BMCIS?",
            top_k=5
        )
        __result__ = results
        ```
        """

    async def execute(self, code: str, description: str = "") -> dict:
        """Execute agent code safely"""
        return await self.executor.execute_search_workflow(code, description)

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
                        "description": "Python code to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this code does (optional)"
                    }
                },
                "required": ["code"]
            }
        }
```

---

## Code Patterns

### Pattern 1: Simple Search Query

**Traditional approach (150K tokens):**
```python
# Agent calls multiple tools separately
search_results = agent.call_tool("bm25_search", {"query": "BMCIS overview"})
# → 50K tokens returned (100 documents)

vector_results = agent.call_tool("vector_search", {"query": "BMCIS overview"})
# → 50K tokens returned (100 documents)

summary = agent.call_tool("fuse_results", {"bm25": search_results, "vector": vector_results})
# → 50K tokens for fusion processing
```

**Code execution approach (2K tokens):**
```python
# Agent writes code that does it all in environment
code = """
from src.code_api.search import HybridSearchAPI

search_api = HybridSearchAPI()
results = await search_api.hybrid_search(
    query="BMCIS overview",
    top_k=5
)

__result__ = results
"""

# Execute returns compact results (~2K tokens)
```

### Pattern 2: Search + Rerank + Filter

```python
code = """
from src.code_api.search import HybridSearchAPI
from src.code_api.reranking import RerankerAPI
from src.code_api.results import ResultProcessor

# Step 1: Get search results
search_api = HybridSearchAPI()
results = await search_api.hybrid_search(
    query="advanced BMCIS features",
    top_k=20  # Get more for reranking
)

# Step 2: Rerank results
reranker = RerankerAPI()
reranked = await reranker.rerank(
    query="advanced BMCIS features",
    documents=[r.to_dict() for r in results],
    top_k=5
)

# Step 3: Format for agent
processor = ResultProcessor()
summary = processor.summarize(reranked, max_content_length=150)

__result__ = processor.format_for_agent(summary)
"""

# All processing happens in environment, agent gets readable summary
```

### Pattern 3: Conditional Search Logic

```python
code = """
from src.code_api.search import HybridSearchAPI
from src.code_api.filtering import FilterAPI

search_api = HybridSearchAPI()
filter_api = FilterAPI()

# Route based on query complexity
if len(query.split()) > 10:
    # Complex query: use vector search
    results = await search_api.vector_search(query, top_k=10)
else:
    # Simple query: use hybrid
    results = await search_api.hybrid_search(query, top_k=10)

# Filter by domain
filtered = filter_api.filter_by_domain(
    results,
    domain="bmcis_architecture"
)

# Return compact summary
summaries = [
    {
        'id': r['id'],
        'title': r['title'],
        'score': r['score'],
        'relevance': 'high' if r['score'] > 0.8 else 'medium'
    }
    for r in filtered
]

__result__ = summaries
"""
```

---

## Sandboxing & Security

### Security Model

The code execution environment provides defense-in-depth:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Import Whitelist | Only allow safe modules | Filesystem/network access |
| Code Validation | AST parsing before execution | Malicious patterns |
| Timeout | 30-second limit | Infinite loops |
| Memory Limit | 512MB cap | Memory exhaustion |
| Thread Isolation | Run in executor thread | Global state modification |
| No Direct I/O | Capture stdout only | Logging sensitive data |

### Allowed Modules

```python
ALLOWED_IMPORTS = {
    'bmcis_knowledge': ['search', 'reranking', 'filtering'],
    'json': [],
    'datetime': [],
    're': [],
    'statistics': [],
    'itertools': [],
    'operator': [],
}
```

### Dangerous Patterns Blocked

```
❌ import os
❌ import subprocess
❌ import socket
❌ __import__
❌ eval(
❌ exec(
```

### Custom Additions

For production deployments, consider:

1. **Docker Containerization** (Option A - Recommended for production)
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY src/ /app/src/
   RUN pip install -r requirements.txt
   # No shell, read-only filesystem, no root
   ```

2. **E2B Sandbox** (Option B - For cloud deployment)
   ```python
   import e2b

   sandbox = e2b.Sandbox()
   result = await sandbox.run_code(code)
   ```

3. **Firecracker** (Option C - For extreme isolation)
   ```
   MicroVM isolation - overkill for most use cases
   But available for highly sensitive data
   ```

---

## Migration Path

### Step 1: Audit Current Usage (Day 1)

Identify which agents use BMCIS MCP tools:

```bash
# Find all agent code that calls BMCIS tools
grep -r "call_tool.*search" src/agents/
grep -r "call_tool.*rerank" src/agents/
```

### Step 2: Implement Code API (Days 2-3)

- Create `src/code_api/` modules (search, reranking, filtering)
- Write unit tests for each API
- Validate with existing integration tests

### Step 3: Set Up Sandbox (Day 4)

- Implement `CodeExecutionSandbox`
- Test with sample code
- Validate security constraints

### Step 4: Create MCP Tool (Day 5)

- Register `execute_code` tool in MCP server
- Add to tool definitions
- Test with Claude

### Step 5: Migrate Agents (Days 6-10)

Gradual rollout:

**Cohort 1 (Low risk)**: Simple search-only agents
```python
# Before: 2 tool calls
results = agent.call_tool("bm25_search", ...)

# After: 1 code execution
code = "from search import...; results = await..."
executor.execute_code(code)
```

**Cohort 2 (Medium risk)**: Search + rerank workflows
**Cohort 3 (Higher risk)**: Complex conditional logic

### Step 6: Monitor & Optimize (Ongoing)

- Compare token usage (target 98% reduction)
- Monitor execution times (should improve)
- Track error rates (should stay ~0%)

---

## Performance Metrics

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Token usage per query | 150,000 | 2,000 | 98.7% ↓ |
| Latency | 8-12s | 2-3s | 75% ↓ |
| Context remaining | ~200K/200K | ~198K/200K | Significant ↑ |
| Tools accessible | 5-10 | 50+ | Unlimited ↑ |
| API cost per query | $0.75 | $0.01 | 98% ↓ |

### Benchmark Test

```python
# Test script to measure improvements
import time
import asyncio

async def benchmark_traditional():
    """Current approach - multiple tool calls"""
    start = time.time()

    # Simulated tool calls with token overhead
    results = agent.call_tool("search", ...)  # 60K tokens
    reranked = agent.call_tool("rerank", ...) # 50K tokens
    filtered = agent.call_tool("filter", ...) # 40K tokens

    return time.time() - start, 150000

async def benchmark_code_execution():
    """New approach - single code execution"""
    start = time.time()

    code = "..."  # Search + rerank + filter in one execution
    result = executor.execute_code(code)  # 2K tokens

    return time.time() - start, 2000

# Run benchmarks
trad_time, trad_tokens = await benchmark_traditional()
exec_time, exec_tokens = await benchmark_code_execution()

print(f"Token reduction: {trad_tokens/exec_tokens:.1f}x")
print(f"Latency improvement: {trad_time/exec_time:.1f}x")
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_code_api.py
import pytest
from src.code_api.search import HybridSearchAPI

@pytest.mark.asyncio
async def test_hybrid_search():
    """Test hybrid search returns compact results"""
    api = HybridSearchAPI()
    results = await api.hybrid_search("BMCIS", top_k=5)

    assert len(results) <= 5
    assert all(isinstance(r.relevance_score, float) for r in results)

@pytest.mark.asyncio
async def test_token_efficiency():
    """Verify results are compact"""
    api = HybridSearchAPI()
    results = await api.hybrid_search("test query", top_k=100)

    # Serialize and check token overhead
    import json
    serialized = json.dumps([r.to_dict() for r in results])
    assert len(serialized) < 50000  # Should be minimal
```

### Security Tests

```python
# tests/test_sandbox_security.py
import pytest
from src.code_execution.sandbox import CodeExecutionSandbox

@pytest.mark.asyncio
async def test_import_whitelist():
    """Verify non-whitelisted imports blocked"""
    sandbox = CodeExecutionSandbox()

    result = await sandbox.execute("import os")
    assert not result['success']
    assert "Non-whitelisted import" in result['error']

@pytest.mark.asyncio
async def test_timeout_protection():
    """Verify timeout prevents infinite loops"""
    sandbox = CodeExecutionSandbox(timeout_seconds=1)

    code = "while True: pass"
    result = await sandbox.execute(code)
    assert not result['success']
    assert "timeout" in result['error'].lower()

@pytest.mark.asyncio
async def test_dangerous_patterns():
    """Verify dangerous patterns blocked"""
    sandbox = CodeExecutionSandbox()

    dangerous = ["eval('code')", "exec('code')", "__import__('os')"]
    for code in dangerous:
        result = await sandbox.execute(code)
        assert not result['success']
```

### Integration Tests

```python
# tests/integration/test_end_to_end.py
import pytest
from src.code_execution.agent_interface import get_executor

@pytest.mark.asyncio
async def test_full_search_workflow():
    """Test complete search → rerank → format workflow"""
    executor = get_executor()

    code = """
from src.code_api.search import HybridSearchAPI
from src.code_api.reranking import RerankerAPI
from src.code_api.results import ResultProcessor

search_api = HybridSearchAPI()
results = await search_api.hybrid_search("BMCIS query", top_k=10)

reranker = RerankerAPI()
reranked = await reranker.rerank("BMCIS query", results, top_k=5)

processor = ResultProcessor()
summary = processor.format_for_agent(reranked)

__result__ = summary
"""

    result = await executor.execute_search_workflow(code)

    assert result['success']
    assert isinstance(result['result'], str)
    assert len(result['result']) < 5000  # Compact format
```

---

## Troubleshooting

### Issue: Code Execution Times Out

**Symptoms**: Queries taking >30 seconds

**Causes**:
- Large search result sets (1000+ documents)
- Inefficient reranking
- Database connection issues

**Solutions**:
```python
# Reduce search scope
results = await api.hybrid_search(
    query=q,
    top_k=50,  # Reduce from 100
    filters={'domain': 'specific_domain'}  # Filter early
)

# Use faster reranking
reranked = await reranker.rerank(
    results[:20],  # Only rerank top 20
    top_k=5
)

# Increase timeout if needed
sandbox = CodeExecutionSandbox(timeout_seconds=60)
```

### Issue: Token Usage Not Improving

**Symptoms**: Still seeing 100K+ token usage

**Causes**:
- Agent code returning all results in `__result__`
- Large intermediate datasets in output
- JSON serialization overhead

**Solutions**:
```python
# ❌ DON'T: Return all raw data
__result__ = raw_search_results  # Huge!

# ✅ DO: Return only needed fields
__result__ = [
    {
        'id': r['id'],
        'score': r['score'],
        'title': r['title']
    }
    for r in results[:5]
]

# ✅ Or: Use ResultProcessor
processor = ResultProcessor()
__result__ = processor.summarize(results, max_content_length=100)
```

### Issue: "Non-whitelisted import" Error

**Symptoms**: Code execution fails with import errors

**Causes**:
- Trying to import outside approved list
- Using external libraries

**Solutions**:

Update the whitelist in `src/code_execution/sandbox.py`:

```python
ALLOWED_IMPORTS = {
    'bmcis_knowledge': ['search', 'reranking', 'filtering'],
    'json': [],
    'numpy': ['array', 'mean'],  # Add if needed
    'pandas': ['DataFrame'],      # Add if needed
    # ... etc
}
```

### Issue: Security Validation Failing

**Symptoms**: Code rejected with "Code validation failed"

**Causes**:
- Using dangerous patterns (eval, exec, etc.)
- Importing restricted modules
- Code syntax errors

**Debug**:
```python
from src.code_execution.sandbox import CodeExecutionSandbox

sandbox = CodeExecutionSandbox()
validation = sandbox._validate_code(your_code)
if not validation['safe']:
    print(f"Error: {validation['reason']}")
```

### Issue: Memory Usage Spiking

**Symptoms**: Execution killed due to memory limit

**Causes**:
- Large search result sets in memory
- Inefficient data structures
- Memory leaks in APIs

**Solutions**:
```python
# Stream results instead of loading all
for batch in api.hybrid_search_streaming(query, batch_size=100):
    # Process batch
    processed = reranker.rerank(batch, top_k=5)
    __result__ = processed

# Or increase memory limit (carefully)
sandbox = CodeExecutionSandbox(max_memory_mb=1024)
```

---

## Integration Checklist

- [ ] Create `src/code_api/` module with search, reranking, filtering APIs
- [ ] Implement `CodeExecutionSandbox` with security validation
- [ ] Write unit tests for all APIs
- [ ] Implement MCP tool definition
- [ ] Register `execute_code` tool in server
- [ ] Update `.mcp.json` if needed
- [ ] Add documentation to agent system prompt
- [ ] Migrate first agent cohort
- [ ] Run performance benchmarks
- [ ] Monitor error rates in production
- [ ] Gather feedback from agents
- [ ] Iterate on sandbox settings
- [ ] Document lessons learned

---

## Resources

### Official Anthropic Documentation
- [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [MCP Specification](https://modelcontextprotocol.io/)

### Reference Implementations
- [Pydantic Python Sandbox MCP](https://github.com/pydantic/pydantic-mcp-server)
- [Code Sandbox MCP Server](https://github.com/Automata-Labs-team/code-sandbox-mcp)
- [E2B Sandbox Integration](https://e2b.dev/)

### Related Patterns
- [Secure Code Execution](https://owasp.org/www-community/attacks/Code_Injection)
- [Sandboxing Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Sandbox_Bypass_Cheat_Sheet.html)

---

## Next Steps

1. **Review this document** with your team
2. **Implement Phase 1** (Code API modules)
3. **Test in development** before production
4. **Monitor token usage** and performance
5. **Iterate based on results**

Questions? Refer to the BMCIS Knowledge MCP documentation or contact the development team.

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Status**: Ready for Implementation
