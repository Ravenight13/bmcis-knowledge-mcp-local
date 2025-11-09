# Code Execution with MCP: Quick Start Guide

**For developers implementing code execution in BMCIS Knowledge MCP**

---

## 30-Minute Setup

### 1. Copy Code API Skeleton (5 min)

Create the directory structure:

```bash
mkdir -p src/code_api
mkdir -p src/code_execution
touch src/code_api/__init__.py
touch src/code_api/search.py
touch src/code_api/reranking.py
touch src/code_api/filtering.py
touch src/code_api/results.py
touch src/code_execution/sandbox.py
touch src/code_execution/agent_interface.py
```

### 2. Copy Example Code (10 min)

Use the code snippets from `CODE_EXECUTION_WITH_MCP.md`:
- `SearchAPI` class from Phase 1.2
- `RerankerAPI` class from Phase 1.3
- `CodeExecutionSandbox` from Phase 2.1

Update imports to match your project structure.

### 3. Create MCP Tool (5 min)

Register the `execute_code` tool in your MCP server:

```python
from src.mcp_tools.code_execution_tool import CodeExecutionTool

# In your server initialization
tool = CodeExecutionTool()
server.add_tool(tool.name, tool.to_mcp_definition(), tool.execute)
```

### 4. Test It (10 min)

```bash
# Run unit tests
pytest tests/test_code_api.py -v
pytest tests/test_sandbox_security.py -v

# Manual test
python3 -c "
import asyncio
from src.code_execution.agent_interface import get_executor

async def test():
    executor = get_executor()
    code = '''
from src.code_api.search import HybridSearchAPI
search_api = HybridSearchAPI()
results = await search_api.hybrid_search('test', top_k=5)
__result__ = len(results)
'''
    result = await executor.execute_search_workflow(code)
    print(result)

asyncio.run(test())
"
```

---

## Common Agent Code Patterns

### Pattern 1: Simple Search

```python
code = """
from src.code_api.search import HybridSearchAPI

search_api = HybridSearchAPI()
results = await search_api.hybrid_search(
    query="your question here",
    top_k=5
)

__result__ = [{
    'title': r.title,
    'score': r.relevance_score,
    'preview': r.content[:100]
} for r in results]
"""
```

### Pattern 2: Search + Rerank

```python
code = """
from src.code_api.search import HybridSearchAPI
from src.code_api.reranking import RerankerAPI

search_api = HybridSearchAPI()
results = await search_api.hybrid_search("query", top_k=20)

reranker = RerankerAPI()
reranked = await reranker.rerank(
    query="query",
    documents=[{
        'id': r.document_id,
        'text': r.content,
        'title': r.title
    } for r in results],
    top_k=5
)

__result__ = reranked
"""
```

### Pattern 3: With Filtering

```python
code = """
from src.code_api.search import HybridSearchAPI
from src.code_api.filtering import FilterAPI

search_api = HybridSearchAPI()
filter_api = FilterAPI()

results = await search_api.hybrid_search(
    query="query",
    filters={'domain': 'specific_domain'},
    top_k=10
)

filtered = filter_api.filter_by_score(results, min_score=0.7)

__result__ = [{
    'id': r['id'],
    'title': r['title'],
    'score': f"{r['score']:.2f}"
} for r in filtered]
"""
```

---

## Token Usage Verification

Before/after comparison:

```python
import json

# BEFORE: Traditional tool calling (estimate tokens)
before_tokens = {
    'tool_definitions': 15000,
    'search_results': 50000,
    'rerank_call': 30000,
    'filter_call': 20000,
    'total': 115000
}

# AFTER: Code execution (measure actual)
code = """..."""  # Your code above
result = await executor.execute_search_workflow(code)

# Serialize result to see token overhead
serialized = json.dumps(result['result'])
after_tokens = len(serialized) // 4  # Rough token estimate

print(f"Reduction: {before_tokens['total'] / after_tokens:.1f}x")
```

---

## Security Checklist

- [ ] Only whitelisted imports in code
- [ ] No `import os`, `subprocess`, `socket`, etc.
- [ ] No `eval()`, `exec()`, `__import__()`
- [ ] 30-second timeout configured
- [ ] Memory limit set to 512MB
- [ ] Output capture working
- [ ] Error handling in place

---

## Debugging Commands

### Check if sandbox is working

```bash
python3 << 'EOF'
import asyncio
from src.code_execution.sandbox import CodeExecutionSandbox

async def test():
    sandbox = CodeExecutionSandbox()

    # Test 1: Valid code
    result = await sandbox.execute("x = 1 + 1; __result__ = x")
    print("Test 1 (valid code):", result['success'])

    # Test 2: Blocked import
    result = await sandbox.execute("import os")
    print("Test 2 (blocked import):", not result['success'])

    # Test 3: Timeout
    result = await sandbox.execute("while True: pass", timeout_seconds=1)
    print("Test 3 (timeout):", not result['success'])

asyncio.run(test())
EOF
```

### Check token overhead

```bash
python3 << 'EOF'
import json
from src.code_api.results import ResultProcessor

# Create sample results
sample = [
    {
        'id': 'doc1',
        'title': 'Sample Document',
        'content': 'This is a long piece of content...' * 10,
        'score': 0.95
    }
]

processor = ResultProcessor()
summary = processor.summarize(sample, max_content_length=100)

# Calculate tokens
serialized = json.dumps(summary)
tokens = len(serialized) // 4
print(f"Result tokens: {tokens}")
EOF
```

---

## Integration Testing

### Test 1: API Integration

```bash
pytest tests/test_code_api.py::test_hybrid_search -v
```

### Test 2: Sandbox Security

```bash
pytest tests/test_sandbox_security.py -v
```

### Test 3: End-to-End

```bash
pytest tests/integration/test_end_to_end.py -v
```

### Test 4: Performance Benchmark

```bash
python3 tests/benchmark_tokens.py
```

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Import path wrong | Use absolute imports: `from src.code_api...` |
| `RuntimeError: Event loop closed` | Async issue | Use `asyncio.run()` for standalone scripts |
| `TimeoutError` | Query too slow | Reduce `top_k`, add filters, increase timeout |
| `Non-whitelisted import` | Trying to import restricted module | Update `ALLOWED_IMPORTS` in sandbox.py |
| Token usage still high | Returning too much data | Use `ResultProcessor.summarize()` |

---

## Performance Target

| Metric | Target | Check |
|--------|--------|-------|
| Token reduction | >95% | Measure `before / after` ratio |
| Latency | <5s | Time execution end-to-end |
| Error rate | <1% | Monitor logs for failures |
| Memory usage | <500MB | Watch for spikes |

---

## Next Steps

1. **Week 1**: Implement Phase 1 (Code API)
2. **Week 1**: Implement Phase 2 (Sandbox)
3. **Week 2**: Implement Phase 3 (MCP Tool)
4. **Week 2**: Migrate first agent
5. **Week 3**: Monitor and optimize

---

## Command Cheatsheet

```bash
# Run all tests
pytest tests/ -v

# Test just code execution
pytest tests/test_code_api.py tests/test_sandbox_security.py -v

# Run single test
pytest tests/test_code_api.py::test_hybrid_search -v

# Check coverage
pytest tests/ --cov=src

# Debug specific code
python3 -m pdb -c "b src/code_api/search.py:50" -c "c" tests/test_code_api.py

# Monitor execution
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... run code
"
```

---

## Key Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| `src/code_api/search.py` | Search API | Adding search features |
| `src/code_api/reranking.py` | Reranking API | Changing reranking logic |
| `src/code_execution/sandbox.py` | Sandbox security | Updating whitelist or limits |
| `tests/test_code_api.py` | API tests | After changing APIs |
| `tests/test_sandbox_security.py` | Security tests | After changing sandbox |

---

## Success Criteria

✅ Code execution tool registered in MCP server
✅ All unit tests passing
✅ Sandbox security tests passing
✅ Token usage reduced by >95%
✅ Latency improved by >50%
✅ Error rate <1%
✅ One agent successfully migrated
✅ Performance benchmarks meeting targets

---

**Need help?** Refer to `CODE_EXECUTION_WITH_MCP.md` for detailed explanations.
