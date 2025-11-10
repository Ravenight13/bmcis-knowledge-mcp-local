# Claude Desktop Optimization Guide

## Table of Contents

1. [Overview](#overview)
2. [Desktop Limitations and Constraints](#desktop-limitations-and-constraints)
3. [Response Size Optimization](#response-size-optimization)
4. [Token Budgeting Strategies](#token-budgeting-strategies)
5. [Metadata Requirements](#metadata-requirements)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

Claude Desktop presents unique challenges and opportunities for MCP tool integration. This guide provides comprehensive strategies for optimizing MCP responses specifically for Claude Desktop's context window limitations, UI requirements, and performance characteristics.

### Key Challenges

- **Limited Context Window**: Desktop has stricter token limits than API
- **UI Responsiveness**: Need for quick, incremental responses
- **Visual Presentation**: Responses must be human-readable and UI-friendly
- **Memory Management**: Efficient use of conversation context
- **Error Recovery**: Graceful handling of size limit errors

### Optimization Goals

1. **Minimize Token Usage**: Stay within Desktop's context limits
2. **Maximize Information Density**: Deliver maximum value per token
3. **Enable Progressive Disclosure**: Start minimal, expand on demand
4. **Maintain Responsiveness**: Sub-second response times
5. **Provide Clear Feedback**: Actionable warnings and suggestions

## Desktop Limitations and Constraints

### Context Window Limits

Claude Desktop has specific context window constraints that vary by model and subscription tier:

| Tier | Context Window | Recommended Response Budget |
|------|---------------|----------------------------|
| Free | 8,000 tokens | 2,000 tokens (25%) |
| Pro | 32,000 tokens | 8,000 tokens (25%) |
| Enterprise | 100,000 tokens | 25,000 tokens (25%) |

### Response Size Calculations

```python
class DesktopTokenCalculator:
    """
    Calculate optimal response sizes for Desktop
    """

    def __init__(self, context_limit: int = 32000):
        self.context_limit = context_limit
        self.response_budget = context_limit * 0.25  # 25% for responses
        self.reserved_for_conversation = context_limit * 0.5  # 50% for context
        self.buffer = context_limit * 0.25  # 25% safety buffer

    def calculate_max_results(self, mode: str) -> int:
        """
        Calculate maximum results for a response mode

        Returns:
            Maximum number of results that fit in budget
        """
        tokens_per_result = {
            "ids_only": 10,
            "metadata": 200,
            "preview": 500,
            "full": 1500
        }

        envelope_overhead = 300  # Response envelope tokens
        available = self.response_budget - envelope_overhead

        return int(available / tokens_per_result[mode])

    def recommend_mode(self, desired_results: int) -> str:
        """
        Recommend optimal mode for desired result count
        """
        for mode in ["full", "preview", "metadata", "ids_only"]:
            if desired_results <= self.calculate_max_results(mode):
                return mode
        return "ids_only"  # Fallback
```

### Desktop-Specific Constraints

1. **Synchronous Processing**: Desktop processes responses synchronously
2. **UI Rendering Limits**: Large responses can slow down the UI
3. **Memory Pressure**: Large contexts can cause memory issues
4. **Network Latency**: Desktop adds ~50-100ms overhead
5. **Format Requirements**: Responses must be Markdown-friendly

## Response Size Optimization

### Adaptive Response Sizing

Dynamically adjust response size based on available context:

```python
class AdaptiveResponseFormatter:
    """
    Adapt response format to available context space
    """

    def __init__(self, conversation_tokens: int = 0):
        self.conversation_tokens = conversation_tokens
        self.max_context = 32000  # Desktop Pro limit

    def format_response(self, results: List[SearchResult]) -> dict:
        """
        Format response adaptively based on available space
        """
        available = self.max_context - self.conversation_tokens

        if available > 20000:
            # Plenty of space - use preview or full
            return self.format_rich_response(results)
        elif available > 10000:
            # Moderate space - use metadata
            return self.format_metadata_response(results)
        elif available > 5000:
            # Limited space - use ids with key metadata
            return self.format_compact_response(results)
        else:
            # Very limited - minimal response
            return self.format_minimal_response(results)

    def format_minimal_response(self, results: List[SearchResult]) -> dict:
        """Ultra-compact format for tight contexts"""
        return {
            "found": len(results),
            "top_3": [
                {"id": r.chunk_id, "score": round(r.hybrid_score, 2)}
                for r in results[:3]
            ],
            "warning": "Context limit approaching - minimal response mode"
        }
```

### Compression Techniques

#### 1. Field Truncation

```python
def truncate_fields(result: dict, max_lengths: dict) -> dict:
    """
    Truncate fields to maximum lengths

    Args:
        result: Result dictionary
        max_lengths: Field name -> max length mapping

    Example:
        >>> truncate_fields(
        ...     {"text": "Very long text...", "file": "path/to/file.md"},
        ...     {"text": 100, "file": 20}
        ... )
    """
    truncated = result.copy()

    for field, max_len in max_lengths.items():
        if field in truncated and len(str(truncated[field])) > max_len:
            value = str(truncated[field])
            truncated[field] = value[:max_len-3] + "..."

    return truncated
```

#### 2. Smart Summarization

```python
class SmartSummarizer:
    """
    Intelligently summarize content while preserving key information
    """

    def summarize_chunk(self, text: str, target_length: int = 200) -> str:
        """
        Create intelligent summary preserving key terms

        Strategy:
        1. Extract first and last sentences
        2. Identify key terms (high TF-IDF)
        3. Build summary around key terms
        """
        if len(text) <= target_length:
            return text

        sentences = text.split('. ')

        # Always include first sentence (usually most important)
        summary = sentences[0]

        # Add last sentence if different and space available
        if len(sentences) > 1:
            last = sentences[-1]
            if len(summary) + len(last) + 5 < target_length:
                summary = f"{summary}... {last}"

        # Trim to exact length
        if len(summary) > target_length:
            summary = summary[:target_length-3] + "..."

        return summary
```

#### 3. Response Deduplication

```python
class ResponseDeduplicator:
    """
    Remove redundant information from responses
    """

    def deduplicate_results(self, results: List[dict]) -> List[dict]:
        """
        Remove duplicate or highly similar results

        Returns:
            Deduplicated results with similarity groups
        """
        seen_content = {}
        deduplicated = []

        for result in results:
            content_hash = self.hash_content(result.get('chunk_text', ''))

            if content_hash in seen_content:
                # Mark as duplicate of earlier result
                similar_id = seen_content[content_hash]
                result['duplicate_of'] = similar_id
                result['chunk_text'] = f"[Duplicate of #{similar_id}]"
            else:
                seen_content[content_hash] = result['chunk_id']
                deduplicated.append(result)

        return deduplicated

    def hash_content(self, text: str) -> str:
        """Create hash of normalized content"""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:8]
```

## Token Budgeting Strategies

### Progressive Token Allocation

Allocate tokens progressively based on result quality:

```python
class ProgressiveTokenAllocator:
    """
    Allocate tokens based on result quality scores
    """

    def __init__(self, total_budget: int = 8000):
        self.total_budget = total_budget
        self.envelope_overhead = 300

    def allocate(self, results: List[SearchResult]) -> dict:
        """
        Allocate tokens progressively:
        - Top results get full content
        - Mid results get preview
        - Low results get metadata only

        Returns:
            Dict mapping chunk_id to response_mode
        """
        available = self.total_budget - self.envelope_overhead
        allocations = {}
        used = 0

        # Sort by score
        sorted_results = sorted(results, key=lambda r: r.hybrid_score, reverse=True)

        for i, result in enumerate(sorted_results):
            if result.hybrid_score > 0.8 and used + 1500 < available:
                # High quality - full content
                allocations[result.chunk_id] = "full"
                used += 1500
            elif result.hybrid_score > 0.6 and used + 500 < available:
                # Medium quality - preview
                allocations[result.chunk_id] = "preview"
                used += 500
            elif used + 200 < available:
                # Lower quality - metadata
                allocations[result.chunk_id] = "metadata"
                used += 200
            else:
                # Out of budget - ids only
                allocations[result.chunk_id] = "ids_only"
                used += 10

        return allocations
```

### Context-Aware Budgeting

Adjust token budget based on conversation state:

```python
class ContextAwareBudget:
    """
    Dynamic token budgeting based on conversation context
    """

    def __init__(self):
        self.conversation_history = []
        self.total_limit = 32000

    def calculate_available_budget(self) -> int:
        """
        Calculate available token budget

        Considers:
        - Conversation history size
        - Expected follow-up queries
        - Safety margin
        """
        # Calculate conversation tokens
        conversation_tokens = sum(
            len(msg.get('content', '')) // 4  # Rough token estimate
            for msg in self.conversation_history
        )

        # Reserve space for follow-ups
        follow_up_reserve = 2000

        # Safety margin
        safety_margin = 1000

        available = (
            self.total_limit
            - conversation_tokens
            - follow_up_reserve
            - safety_margin
        )

        return max(available, 1000)  # Minimum 1000 tokens

    def should_clear_context(self) -> bool:
        """
        Determine if context should be cleared

        Returns:
            True if conversation should be reset
        """
        available = self.calculate_available_budget()
        return available < 2000  # Less than 2000 tokens available
```

## Metadata Requirements

### Essential Metadata for Desktop

Desktop requires specific metadata fields for optimal UI rendering:

```python
class DesktopMetadata:
    """
    Essential metadata fields for Claude Desktop
    """

    REQUIRED_FIELDS = {
        "operation",      # Tool operation name
        "version",        # API version
        "timestamp",      # ISO 8601 timestamp
        "request_id",     # Unique request ID
        "status"          # success/partial/error
    }

    OPTIONAL_ENHANCED = {
        "confidence_score",    # Overall confidence
        "relevance_percentile", # Relevance in result set
        "source_reliability",   # Source quality indicator
        "content_preview",      # Brief content preview
        "visual_hints"          # UI rendering hints
    }

    @classmethod
    def format_for_desktop(cls, response: dict) -> dict:
        """
        Format response with Desktop-required metadata

        Example output:
        {
            "_metadata": {
                "operation": "semantic_search",
                "version": "1.0.0",
                "timestamp": "2025-11-09T10:30:00Z",
                "request_id": "req_abc123",
                "status": "success",
                "visual_hints": {
                    "highlight_top": 3,
                    "group_by": "source_file",
                    "show_confidence": true
                }
            }
        }
        """
        return {
            "_metadata": {
                "operation": response.get("operation", "unknown"),
                "version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": str(uuid.uuid4()),
                "status": "success" if response.get("results") else "partial",
                "visual_hints": cls.generate_visual_hints(response)
            },
            **response
        }

    @classmethod
    def generate_visual_hints(cls, response: dict) -> dict:
        """Generate UI rendering hints"""
        results = response.get("results", [])

        # Identify high-confidence results
        high_confidence = [
            r for r in results
            if r.get("hybrid_score", 0) > 0.8
        ]

        return {
            "highlight_top": len(high_confidence),
            "group_by": "source_category" if len(results) > 10 else None,
            "show_confidence": any(r.get("confidence") for r in results),
            "collapse_duplicates": True,
            "expandable_previews": len(results) > 5
        }
```

### Confidence Indicators

Provide clear confidence indicators for Desktop UI:

```python
def generate_confidence_indicators(result: SearchResult) -> dict:
    """
    Generate comprehensive confidence indicators

    Returns:
        Dict with visual confidence indicators
    """
    score = result.hybrid_score

    # Determine confidence level
    if score > 0.9:
        level = "very_high"
        color = "green"
        icon = "✓✓✓"
    elif score > 0.75:
        level = "high"
        color = "blue"
        icon = "✓✓"
    elif score > 0.6:
        level = "medium"
        color = "yellow"
        icon = "✓"
    else:
        level = "low"
        color = "gray"
        icon = "?"

    return {
        "score": round(score, 3),
        "level": level,
        "color": color,
        "icon": icon,
        "percentile": calculate_percentile(score, all_scores),
        "explanation": generate_score_explanation(result)
    }
```

## Error Handling and Recovery

### Graceful Degradation

Handle Desktop limitations gracefully:

```python
class GracefulDegradation:
    """
    Gracefully degrade response quality when hitting limits
    """

    def handle_token_limit_error(
        self,
        original_request: dict,
        error: Exception
    ) -> dict:
        """
        Recover from token limit errors

        Strategy:
        1. Reduce response mode
        2. Decrease page size
        3. Apply field filtering
        4. Return partial results
        """
        # Extract error details
        if "token" in str(error).lower():
            # Try with reduced response mode
            if original_request["response_mode"] == "full":
                return self.retry_with_mode(original_request, "preview")
            elif original_request["response_mode"] == "preview":
                return self.retry_with_mode(original_request, "metadata")
            elif original_request["response_mode"] == "metadata":
                return self.retry_with_mode(original_request, "ids_only")

        # Reduce page size
        if original_request.get("page_size", 10) > 5:
            return self.retry_with_page_size(
                original_request,
                original_request["page_size"] // 2
            )

        # Last resort - return error with suggestions
        return {
            "error": "TOKEN_LIMIT_EXCEEDED",
            "suggestions": [
                "Clear conversation history",
                "Use 'ids_only' response mode",
                "Reduce page_size to 3",
                "Filter specific fields only"
            ],
            "partial_results": self.get_minimal_results(original_request)
        }
```

### Warning System

Proactive warning system for Desktop users:

```python
class DesktopWarningSystem:
    """
    Generate actionable warnings for Desktop users
    """

    def check_response_health(self, response: dict, context: dict) -> List[dict]:
        """
        Check response for potential issues

        Returns:
            List of warning objects
        """
        warnings = []

        # Check token usage
        tokens_used = response.get("execution_context", {}).get("tokens_used", 0)
        tokens_available = context.get("tokens_available", 32000)

        if tokens_used > tokens_available * 0.8:
            warnings.append({
                "level": "warning",
                "code": "APPROACHING_TOKEN_LIMIT",
                "message": f"Using {tokens_used}/{tokens_available} tokens (80%)",
                "suggestion": "Consider using 'metadata' mode or clearing context"
            })

        # Check response time
        execution_time = response.get("execution_context", {}).get("execution_time_ms", 0)
        if execution_time > 1000:
            warnings.append({
                "level": "info",
                "code": "SLOW_RESPONSE",
                "message": f"Response took {execution_time}ms",
                "suggestion": "Consider using cached queries or smaller page_size"
            })

        # Check result quality
        results = response.get("results", [])
        high_quality = [r for r in results if r.get("hybrid_score", 0) > 0.7]
        if len(high_quality) == 0:
            warnings.append({
                "level": "info",
                "code": "LOW_QUALITY_RESULTS",
                "message": "No high-confidence results found",
                "suggestion": "Try rephrasing the query or using different keywords"
            })

        return warnings
```

## Performance Optimization

### Response Streaming

Implement response streaming for better perceived performance:

```python
class StreamingResponseFormatter:
    """
    Stream responses for better Desktop UX
    """

    async def stream_results(
        self,
        results: List[SearchResult],
        mode: str = "metadata"
    ):
        """
        Stream results progressively

        Yields:
            Partial responses for incremental rendering
        """
        # Send metadata immediately
        yield {
            "type": "metadata",
            "total_found": len(results),
            "mode": mode,
            "streaming": True
        }

        # Stream high-priority results first
        high_priority = [r for r in results if r.hybrid_score > 0.8]
        for result in high_priority:
            yield {
                "type": "result",
                "priority": "high",
                "data": self.format_result(result, mode)
            }
            await asyncio.sleep(0.01)  # Small delay for UI updates

        # Stream remaining results
        remaining = [r for r in results if r.hybrid_score <= 0.8]
        for result in remaining:
            yield {
                "type": "result",
                "priority": "normal",
                "data": self.format_result(result, mode)
            }
            await asyncio.sleep(0.02)  # Slightly longer delay

        # Send completion signal
        yield {
            "type": "complete",
            "summary": self.generate_summary(results)
        }
```

### Caching Strategy

Desktop-optimized caching:

```python
class DesktopCache:
    """
    Caching optimized for Desktop usage patterns
    """

    def __init__(self):
        self.cache = {}
        self.access_patterns = {}

    def get_with_prediction(self, key: str) -> tuple[Any, List[str]]:
        """
        Get cached result and predict next queries

        Returns:
            (cached_result, predicted_next_queries)
        """
        result = self.cache.get(key)

        # Track access pattern
        self.access_patterns[key] = time.time()

        # Predict next queries based on patterns
        predictions = self.predict_next_queries(key)

        return result, predictions

    def predict_next_queries(self, current_query: str) -> List[str]:
        """
        Predict likely follow-up queries

        Example:
            "authentication" -> ["JWT authentication", "OAuth authentication"]
        """
        # Common query refinements
        refinements = {
            "auth": ["authentication", "authorization", "JWT", "OAuth"],
            "api": ["REST API", "GraphQL", "endpoints", "documentation"],
            "error": ["error handling", "exceptions", "debugging", "logs"]
        }

        predictions = []
        for pattern, expansions in refinements.items():
            if pattern in current_query.lower():
                predictions.extend([
                    f"{current_query} {exp}"
                    for exp in expansions[:2]
                ])

        return predictions[:3]  # Return top 3 predictions
```

## Best Practices

### 1. Response Mode Selection

```python
def select_optimal_mode(
    query_intent: str,
    available_tokens: int,
    result_count: int
) -> str:
    """
    Select optimal response mode based on context

    Decision tree:
    1. Check available tokens
    2. Consider query intent
    3. Balance detail vs. breadth
    """
    # Token-based selection
    if available_tokens < 5000:
        return "ids_only"

    # Intent-based selection
    if query_intent == "browse":
        return "metadata"  # Browsing needs file info
    elif query_intent == "analyze":
        return "full"  # Analysis needs content
    elif query_intent == "preview":
        return "preview"  # Preview mode for scanning
    else:
        return "metadata"  # Safe default
```

### 2. Error Recovery Patterns

```python
async def resilient_search(query: str, client: MCPClient):
    """
    Resilient search with automatic recovery
    """
    strategies = [
        {"mode": "full", "size": 5},
        {"mode": "preview", "size": 10},
        {"mode": "metadata", "size": 20},
        {"mode": "ids_only", "size": 50}
    ]

    for strategy in strategies:
        try:
            response = await client.semantic_search(
                query=query,
                response_mode=strategy["mode"],
                page_size=strategy["size"]
            )
            return response
        except TokenLimitError:
            continue  # Try next strategy
        except Exception as e:
            logger.error(f"Search failed with {strategy}: {e}")

    # All strategies failed
    return {"error": "Unable to complete search", "strategies_tried": strategies}
```

### 3. Context Management

```python
class ContextManager:
    """
    Manage Desktop conversation context efficiently
    """

    def __init__(self, max_tokens: int = 32000):
        self.max_tokens = max_tokens
        self.messages = []
        self.token_count = 0

    def add_message(self, message: dict) -> bool:
        """
        Add message to context, pruning if necessary

        Returns:
            True if message added, False if context full
        """
        message_tokens = self.estimate_tokens(message)

        # Prune old messages if needed
        while self.token_count + message_tokens > self.max_tokens * 0.8:
            if not self.prune_oldest():
                return False  # Cannot prune more

        self.messages.append(message)
        self.token_count += message_tokens
        return True

    def prune_oldest(self) -> bool:
        """Remove oldest non-essential message"""
        for i, msg in enumerate(self.messages):
            if not msg.get("essential", False):
                removed = self.messages.pop(i)
                self.token_count -= self.estimate_tokens(removed)
                return True
        return False

    def estimate_tokens(self, message: dict) -> int:
        """Rough token estimation"""
        content = str(message.get("content", ""))
        return len(content) // 4  # Rough estimate
```

## Troubleshooting

### Common Desktop Issues

#### Issue: Response Truncated in UI

**Symptom**: Response appears cut off in Desktop interface

**Diagnosis**:
```python
# Check response size
response_size = len(json.dumps(response))
if response_size > 50000:  # 50KB limit for UI rendering
    print("Response too large for UI rendering")
```

**Solution**:
```python
# Split large responses
def split_large_response(response: dict, max_size: int = 45000) -> List[dict]:
    """Split response into UI-friendly chunks"""
    results = response.get("results", [])
    chunks = []
    current_chunk = {"results": [], **response}

    for result in results:
        test_chunk = {**current_chunk, "results": current_chunk["results"] + [result]}
        if len(json.dumps(test_chunk)) > max_size:
            # Save current chunk
            chunks.append(current_chunk)
            current_chunk = {"results": [result], **response}
        else:
            current_chunk["results"].append(result)

    if current_chunk["results"]:
        chunks.append(current_chunk)

    return chunks
```

#### Issue: Slow Desktop Response

**Symptom**: Desktop UI becomes unresponsive with large results

**Solution**:
```python
# Implement progressive loading
class ProgressiveLoader:
    def load_results_progressively(self, results: List[dict]) -> Iterator[dict]:
        """Load results in small batches"""
        batch_size = 3
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            yield {
                "batch": i // batch_size + 1,
                "total_batches": (len(results) + batch_size - 1) // batch_size,
                "results": batch
            }
            time.sleep(0.1)  # Allow UI to update
```

#### Issue: Context Window Overflow

**Symptom**: "Context length exceeded" errors

**Solution**:
```python
# Implement context reset strategy
class ContextResetStrategy:
    def should_reset(self, error: Exception) -> bool:
        """Determine if context reset is needed"""
        error_msg = str(error).lower()
        return any(term in error_msg for term in [
            "context", "token", "limit", "exceeded"
        ])

    def create_summary_and_reset(self, conversation: List[dict]) -> dict:
        """Create summary before resetting"""
        return {
            "action": "context_reset",
            "summary": self.summarize_conversation(conversation),
            "preserved_queries": self.extract_important_queries(conversation),
            "message": "Context reset due to token limit. Summary preserved."
        }
```

## Conclusion

Optimizing for Claude Desktop requires careful attention to token budgets, response formatting, and error handling. By following the strategies and patterns outlined in this guide, you can create MCP tools that work seamlessly within Desktop's constraints while providing maximum value to users.

Key takeaways:
1. Always consider Desktop's context limitations
2. Use progressive disclosure to manage token usage
3. Implement graceful degradation for error recovery
4. Provide clear metadata and visual hints
5. Monitor and optimize performance continuously

For additional resources, see:
- [Response Formatting Guide](./response-formatting-guide.md)
- [API Reference: Response Formats](../api-reference/response-formats.md)
- [MCP Tools API Reference](../api-reference/mcp-tools.md)