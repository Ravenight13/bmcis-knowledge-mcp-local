# MCP Tools Usage Examples

This guide provides practical examples of using the BMCIS Knowledge MCP tools in real-world scenarios. Each example includes the complete workflow, expected responses, and best practices.

## Basic Search Examples

### Example 1: Simple Keyword Search

**Scenario:** Find information about authentication methods in your knowledge base.

```python
# Claude's request
semantic_search("authentication methods")

# Response (default metadata mode)
{
  "results": [
    {
      "chunk_id": 45678,
      "source_file": "docs/security/auth-guide.md",
      "source_category": "documentation",
      "hybrid_score": 0.934,
      "rank": 1,
      "chunk_index": 12,
      "total_chunks": 45
    },
    {
      "chunk_id": 45679,
      "source_file": "guides/oauth-implementation.md",
      "source_category": "guides",
      "hybrid_score": 0.887,
      "rank": 2,
      "chunk_index": 3,
      "total_chunks": 28
    }
    // ... 8 more results
  ],
  "total_found": 10,
  "strategy_used": "hybrid",
  "execution_time_ms": 156.3
}
```

**Key Insights:**
- Default `metadata` mode provides file locations and scores
- Results sorted by hybrid score (combining semantic and keyword relevance)
- Quick response time (<200ms typical)

### Example 2: Natural Language Query

**Scenario:** Ask a question in natural language to find relevant information.

```python
# Claude's request
semantic_search("How do I implement JWT authentication in a REST API?")

# Response with preview mode for snippets
semantic_search(
    "How do I implement JWT authentication in a REST API?",
    response_mode="preview"
)

{
  "results": [
    {
      "chunk_id": 78901,
      "source_file": "tutorials/jwt-rest-api.md",
      "source_category": "tutorials",
      "hybrid_score": 0.956,
      "rank": 1,
      "chunk_snippet": "To implement JWT authentication in a REST API, start by setting up a token generation endpoint. When users log in with valid credentials, generate a JWT token containing user claims and sign it with your secret key...",
      "context_header": "## JWT Implementation Steps",
      "chunk_index": 7,
      "total_chunks": 23
    }
    // ... more results with snippets
  ]
}
```

### Example 3: Exploring a Topic Progressively

**Scenario:** Research cloud providers comprehensively while managing token usage.

```python
# Step 1: Cast a wide net with IDs only
initial_results = semantic_search(
    "cloud providers comparison pricing features",
    top_k=30,
    response_mode="ids_only"
)
# Token cost: ~300 tokens
# Returns 30 chunk IDs with scores

# Step 2: Get metadata for high-scoring results
relevant_results = semantic_search(
    "cloud providers comparison pricing features",
    top_k=15,
    response_mode="metadata"
)
# Token cost: ~3,750 tokens
# Shows which documents contain relevant information

# Step 3: Deep dive into most relevant content
detailed_results = semantic_search(
    "cloud providers comparison pricing features",
    top_k=5,
    response_mode="full"
)
# Token cost: ~7,500 tokens
# Provides complete content for analysis

# Total workflow cost: ~11,550 tokens
# Compared to: ~45,000 tokens if using full mode for all 30 results
```

## Vendor Deep Dive Examples

### Example 1: Basic Vendor Information

**Scenario:** Get an overview of a vendor's presence in the knowledge graph.

```python
# Claude's request
vendor_info = find_vendor_info("Microsoft Corporation")

# Response (default metadata mode)
{
  "vendor_name": "Microsoft Corporation",
  "statistics": {
    "entity_count": 342,
    "relationship_count": 891,
    "entity_type_distribution": {
      "PRODUCT": 124,
      "PERSON": 67,
      "ORG": 45,
      "LOCATION": 38,
      "EVENT": 34,
      "TECHNOLOGY": 34
    },
    "relationship_type_distribution": {
      "PRODUCES": 124,
      "WORKS_FOR": 67,
      "PARTNERS_WITH": 89,
      "LOCATED_IN": 38,
      "ACQUIRED": 23,
      "COMPETES_WITH": 15
    }
  },
  "top_entities": null,
  "last_updated": "2024-11-09T16:45:00Z"
}
```

**Analysis Points:**
- Microsoft has 342 related entities in the graph
- Products (124) and people (67) are the main entity types
- Strong partnership network (89 PARTNERS_WITH relationships)

### Example 2: Vendor Relationship Exploration

**Scenario:** Understand a vendor's ecosystem and key relationships.

```python
# Step 1: Get overview
overview = find_vendor_info("Salesforce", response_mode="metadata")
print(f"Salesforce has {overview['statistics']['entity_count']} related entities")

# Step 2: Preview top relationships
preview = find_vendor_info("Salesforce", response_mode="preview")

# Examine top 5 entities
for entity in preview['entities']:
    print(f"- {entity['name']} ({entity['entity_type']}): {entity['confidence']:.2f}")

# Output:
# - Salesforce CRM (PRODUCT): 0.98
# - Marc Benioff (PERSON): 0.96
# - Tableau Software (ORG): 0.94
# - Slack Technologies (ORG): 0.93
# - MuleSoft (ORG): 0.92

# Step 3: Analyze relationships
for rel in preview['relationships']:
    print(f"Salesforce {rel['relationship_type']} -> {rel['target_id']}")

# Output:
# Salesforce PRODUCES -> Salesforce CRM
# Salesforce ACQUIRED -> Tableau Software
# Salesforce ACQUIRED -> Slack Technologies
# Salesforce ACQUIRED -> MuleSoft
# Salesforce PARTNERS_WITH -> Microsoft Corporation
```

### Example 3: Competitive Analysis Workflow

**Scenario:** Compare multiple vendors in the same space.

```python
# Define vendors to analyze
vendors = ["Amazon Web Services", "Microsoft Azure", "Google Cloud Platform"]

# Collect comparative data
vendor_data = {}
for vendor in vendors:
    try:
        info = find_vendor_info(vendor, response_mode="metadata")
        vendor_data[vendor] = info['statistics']
    except ValueError as e:
        # Handle vendor not found
        print(f"Note: {vendor} not found, trying variations...")
        # Try semantic search to find correct name
        search = semantic_search(f'"{vendor}" cloud provider', top_k=1)
        # Retry with corrected name if found

# Compare results
comparison = {
    "vendor": [],
    "products": [],
    "partnerships": [],
    "total_entities": []
}

for vendor, stats in vendor_data.items():
    comparison["vendor"].append(vendor)
    comparison["products"].append(stats['entity_type_distribution'].get('PRODUCT', 0))
    comparison["partnerships"].append(
        stats['relationship_type_distribution'].get('PARTNERS_WITH', 0)
    )
    comparison["total_entities"].append(stats['entity_count'])

# Analysis output
print("Cloud Provider Comparison:")
print("-" * 50)
for i, vendor in enumerate(comparison["vendor"]):
    print(f"{vendor}:")
    print(f"  Products: {comparison['products'][i]}")
    print(f"  Partnerships: {comparison['partnerships'][i]}")
    print(f"  Total Entities: {comparison['total_entities'][i]}")
```

## Token-Efficient Research Examples

### Example 1: Literature Review Workflow

**Scenario:** Conduct a comprehensive literature review on a technical topic.

```python
# Phase 1: Discovery (find all relevant documents)
all_docs = semantic_search(
    "machine learning model deployment production",
    top_k=50,
    response_mode="ids_only"
)
print(f"Found {len(all_docs['results'])} relevant chunks")
# Cost: ~500 tokens

# Phase 2: Document Identification (understand sources)
doc_metadata = semantic_search(
    "machine learning model deployment production",
    top_k=25,
    response_mode="metadata"
)

# Group by source file
documents = {}
for result in doc_metadata['results']:
    source = result['source_file']
    if source not in documents:
        documents[source] = []
    documents[source].append(result)

print(f"Content found in {len(documents)} unique documents")
# Cost: ~6,250 tokens

# Phase 3: Selective Deep Dive (read most relevant)
top_chunks = [r for r in doc_metadata['results'] if r['hybrid_score'] > 0.85]
chunk_ids = [r['chunk_id'] for r in top_chunks[:5]]

# Get full content for top 5 chunks
detailed = semantic_search(
    "machine learning model deployment production",
    top_k=5,
    response_mode="full"
)
# Cost: ~7,500 tokens

# Total cost: ~14,250 tokens (vs ~75,000 for full mode on all 50)
# Savings: 81%
```

### Example 2: Multi-Step Investigation

**Scenario:** Investigate a security incident by progressively drilling down.

```python
# Step 1: Broad search for security events
security_scan = semantic_search(
    "security breach vulnerability exploit 2024",
    top_k=20,
    response_mode="metadata"
)

# Identify relevant categories
categories = set(r['source_category'] for r in security_scan['results'])
print(f"Found security info in categories: {categories}")
# Output: {'incidents', 'vulnerabilities', 'patches', 'advisories'}

# Step 2: Focus on incidents
incident_search = semantic_search(
    "security incident data breach",
    top_k=10,
    response_mode="preview"
)

# Extract vendor names from snippets
import re
vendor_pattern = r'\b([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)\b'
vendors_mentioned = set()
for result in incident_search['results']:
    snippet = result.get('chunk_snippet', '')
    potential_vendors = re.findall(vendor_pattern, snippet)
    vendors_mentioned.update(potential_vendors)

# Step 3: Investigate specific vendors
for vendor in list(vendors_mentioned)[:3]:
    try:
        vendor_security = find_vendor_info(vendor, response_mode="preview")
        print(f"\n{vendor} Security Profile:")
        print(f"  Total entities: {vendor_security['statistics']['entity_count']}")

        # Look for security-related entities
        for entity in vendor_security.get('entities', []):
            if any(term in entity['name'].lower()
                   for term in ['security', 'breach', 'vulnerability']):
                print(f"  - {entity['name']} ({entity['entity_type']})")
    except ValueError:
        continue  # Vendor not found, skip
```

### Example 3: Building a Knowledge Map

**Scenario:** Create a comprehensive map of a technology domain.

```python
# Define exploration strategy
def explore_domain(domain_query, max_depth=3):
    """Progressively explore a domain using semantic search."""

    knowledge_map = {
        "core_topics": [],
        "documents": {},
        "key_vendors": [],
        "relationships": []
    }

    # Level 1: Identify core topics
    core_search = semantic_search(domain_query, top_k=15, response_mode="metadata")

    # Extract unique documents and categories
    for result in core_search['results']:
        doc = result['source_file']
        if doc not in knowledge_map['documents']:
            knowledge_map['documents'][doc] = {
                'category': result['source_category'],
                'relevance': result['hybrid_score'],
                'chunks': []
            }
        knowledge_map['documents'][doc]['chunks'].append(result['chunk_id'])

    # Level 2: Get preview of most relevant content
    preview_search = semantic_search(domain_query, top_k=5, response_mode="preview")

    # Extract topics from headers and snippets
    for result in preview_search['results']:
        if result.get('context_header'):
            topic = result['context_header'].replace('#', '').strip()
            if topic and topic not in knowledge_map['core_topics']:
                knowledge_map['core_topics'].append(topic)

    # Level 3: Identify key vendors/organizations
    full_search = semantic_search(domain_query, top_k=3, response_mode="full")

    # Simple entity extraction from full text
    for result in full_search['results']:
        text = result['chunk_text']
        # Look for company names (simplified pattern)
        companies = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company)\b', text)
        knowledge_map['key_vendors'].extend(companies)

    # Deduplicate vendors
    knowledge_map['key_vendors'] = list(set(knowledge_map['key_vendors']))

    return knowledge_map

# Use the exploration function
blockchain_map = explore_domain("blockchain distributed ledger cryptocurrency")

print("Blockchain Knowledge Map:")
print(f"Core Topics: {blockchain_map['core_topics']}")
print(f"Documents: {len(blockchain_map['documents'])} sources")
print(f"Key Vendors: {blockchain_map['key_vendors'][:5]}")  # Top 5 vendors
```

## Error Handling Examples

### Example 1: Handling Vendor Not Found

```python
def safe_vendor_lookup(vendor_name):
    """Safely look up vendor with fallback to search."""
    try:
        # Try exact match first
        return find_vendor_info(vendor_name)
    except ValueError as e:
        if "not found" in str(e):
            print(f"Vendor '{vendor_name}' not found. Searching for alternatives...")

            # Search for vendor
            search_results = semantic_search(
                f'"{vendor_name}" company organization vendor',
                top_k=5,
                response_mode="preview"
            )

            # Extract potential vendor names from results
            potential_vendors = []
            for result in search_results['results']:
                # Look in snippet for company names
                snippet = result.get('chunk_snippet', '')
                if vendor_name.lower() in snippet.lower():
                    # Try to extract the correct name
                    # This is simplified - real implementation would be more sophisticated
                    potential_vendors.append(snippet.split('.')[0])

            if potential_vendors:
                print(f"Found potential matches: {potential_vendors[:3]}")
                # Try first match
                return find_vendor_info(potential_vendors[0])
            else:
                print(f"No vendors found matching '{vendor_name}'")
                return None
        else:
            raise  # Re-raise other errors

# Usage
vendor_info = safe_vendor_lookup("Acme")  # Might be "Acme Corp" in database
```

### Example 2: Handling Rate Limits Gracefully

```python
import time

def rate_limited_search(queries, delay=1.0):
    """Execute multiple searches with rate limit handling."""
    results = []

    for i, query in enumerate(queries):
        try:
            result = semantic_search(query)
            results.append(result)

            # Add delay between requests
            if i < len(queries) - 1:
                time.sleep(delay)

        except Exception as e:
            if "Rate limit exceeded" in str(e):
                # Extract wait time from error message
                import re
                wait_match = re.search(r'Try again in (\d+) seconds', str(e))
                if wait_match:
                    wait_time = int(wait_match.group(1))
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time + 1)  # Add 1 second buffer

                    # Retry the request
                    result = semantic_search(query)
                    results.append(result)
                else:
                    print(f"Rate limit error without wait time: {e}")
                    results.append(None)
            else:
                print(f"Error searching for '{query}': {e}")
                results.append(None)

    return results

# Execute batch of searches
search_queries = [
    "authentication methods",
    "authorization patterns",
    "security best practices",
    "vulnerability management",
    "incident response"
]

all_results = rate_limited_search(search_queries)
```

### Example 3: Handling Ambiguous Vendor Names

```python
def resolve_vendor_ambiguity(vendor_query):
    """Resolve ambiguous vendor names intelligently."""
    try:
        return find_vendor_info(vendor_query)
    except ValueError as e:
        if "Ambiguous vendor name" in str(e):
            # Extract suggestions from error message
            import re
            matches = re.search(r"Multiple matches found: \[(.*?)\]", str(e))
            if matches:
                suggestions = [s.strip().strip("'") for s in matches.group(1).split(",")]
                print(f"Multiple vendors found matching '{vendor_query}':")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")

                # Try to pick the best match
                # Strategy 1: Exact match (case-insensitive)
                for suggestion in suggestions:
                    if suggestion.lower() == vendor_query.lower():
                        print(f"Using exact match: {suggestion}")
                        return find_vendor_info(suggestion)

                # Strategy 2: Starts with query
                for suggestion in suggestions:
                    if suggestion.lower().startswith(vendor_query.lower()):
                        print(f"Using prefix match: {suggestion}")
                        return find_vendor_info(suggestion)

                # Strategy 3: Use first suggestion (highest confidence)
                print(f"Using highest confidence match: {suggestions[0]}")
                return find_vendor_info(suggestions[0])
            else:
                print(f"Could not parse suggestions from error: {e}")
                return None
        else:
            raise  # Re-raise other errors

# Usage examples
resolve_vendor_ambiguity("Microsoft")  # Might match "Microsoft Corporation", "Microsoft Azure", etc.
resolve_vendor_ambiguity("AWS")  # Might match "AWS", "Amazon Web Services"
resolve_vendor_ambiguity("Google")  # Might match "Google", "Google LLC", "Google Cloud"
```

## Common Workflows

### Workflow 1: Research Assistant Pattern

```python
def research_topic(topic, max_tokens=50000):
    """Complete research workflow with token budget."""

    research = {
        "topic": topic,
        "overview": None,
        "key_documents": [],
        "detailed_content": [],
        "vendors": [],
        "token_usage": 0
    }

    # Phase 1: Overview (minimal tokens)
    overview = semantic_search(topic, top_k=20, response_mode="ids_only")
    research["overview"] = f"Found {overview['total_found']} relevant results"
    research["token_usage"] += 200  # Approximate

    # Phase 2: Document discovery (moderate tokens)
    docs = semantic_search(topic, top_k=15, response_mode="metadata")
    unique_docs = {}
    for result in docs['results']:
        doc = result['source_file']
        if doc not in unique_docs:
            unique_docs[doc] = result['hybrid_score']

    research["key_documents"] = sorted(
        unique_docs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    research["token_usage"] += 3750  # Approximate

    # Phase 3: Selective deep dive (budget remaining tokens)
    remaining_budget = max_tokens - research["token_usage"]
    num_full_results = min(5, remaining_budget // 3000)  # ~3000 tokens per full result

    if num_full_results > 0:
        detailed = semantic_search(topic, top_k=num_full_results, response_mode="full")
        research["detailed_content"] = detailed['results']
        research["token_usage"] += num_full_results * 3000

    # Phase 4: Extract vendors if budget allows
    if research["token_usage"] < max_tokens - 5000:
        for content in research["detailed_content"]:
            # Simple vendor extraction
            text = content.get('chunk_text', '')
            vendors = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Corp|Inc|Ltd)\b', text)
            research["vendors"].extend(vendors)

        research["vendors"] = list(set(research["vendors"]))[:5]

        # Get vendor info for top vendors
        for vendor in research["vendors"][:2]:
            if research["token_usage"] < max_tokens - 3000:
                try:
                    vendor_info = find_vendor_info(vendor, response_mode="metadata")
                    research["token_usage"] += 3000
                except:
                    pass

    return research

# Execute research
ml_research = research_topic("machine learning production deployment", max_tokens=30000)
print(f"Research complete. Token usage: {ml_research['token_usage']}/{30000}")
```

### Workflow 2: Vendor Due Diligence Pattern

```python
def vendor_due_diligence(vendor_name):
    """Comprehensive vendor analysis workflow."""

    report = {
        "vendor": vendor_name,
        "exists": False,
        "profile": None,
        "products": [],
        "partnerships": [],
        "competitors": [],
        "recent_mentions": [],
        "risk_indicators": []
    }

    # Step 1: Verify vendor exists
    try:
        vendor_info = find_vendor_info(vendor_name, response_mode="full")
        report["exists"] = True
        report["profile"] = vendor_info
    except ValueError as e:
        if "not found" in str(e).lower():
            # Try to find vendor through search
            search = semantic_search(f'"{vendor_name}"', top_k=5)
            if search['total_found'] > 0:
                report["recent_mentions"] = search['results']
            return report  # Vendor doesn't exist in knowledge graph
        else:
            raise

    # Step 2: Analyze products
    for entity in vendor_info.get('entities', []):
        if entity['entity_type'] == 'PRODUCT':
            report["products"].append(entity['name'])

    # Step 3: Identify partnerships and competitors
    for entity in vendor_info.get('entities', []):
        if entity['entity_type'] == 'ORG':
            # Check relationship type (would need relationship data)
            # Simplified version:
            if entity['name'] != vendor_name:
                if entity['confidence'] > 0.8:
                    report["partnerships"].append(entity['name'])
                else:
                    report["competitors"].append(entity['name'])

    # Step 4: Search for recent activity
    recent_search = semantic_search(
        f'{vendor_name} announcement news update',
        top_k=5,
        response_mode="preview"
    )
    report["recent_mentions"] = recent_search['results']

    # Step 5: Risk analysis (search for negative indicators)
    risk_search = semantic_search(
        f'{vendor_name} risk vulnerability breach issue problem',
        top_k=5,
        response_mode="preview"
    )

    for result in risk_search['results']:
        if result['hybrid_score'] > 0.7:  # High relevance to risk terms
            report["risk_indicators"].append({
                "source": result['source_file'],
                "snippet": result.get('chunk_snippet', ''),
                "relevance": result['hybrid_score']
            })

    return report

# Run due diligence
diligence = vendor_due_diligence("Acme Corporation")

# Generate summary
print(f"Vendor Due Diligence Report: {diligence['vendor']}")
print(f"Status: {'Found' if diligence['exists'] else 'Not in database'}")
if diligence['exists']:
    print(f"Products: {len(diligence['products'])}")
    print(f"Partnerships: {len(diligence['partnerships'])}")
    print(f"Risk Indicators: {len(diligence['risk_indicators'])}")
```

## Best Practices Summary

### 1. Progressive Disclosure Strategy

Always start with the minimum information needed:
- Use `ids_only` for filtering and relevance checking
- Use `metadata` (default) for most searches
- Use `preview` when you need content samples
- Reserve `full` mode for final analysis

### 2. Token Budget Management

```python
# Calculate token budget for research task
def calculate_token_budget(num_searches, response_modes):
    """Estimate token usage for planned searches."""

    token_costs = {
        "ids_only": 10,
        "metadata": 250,
        "preview": 500,
        "full": 1500
    }

    total_tokens = 0
    for mode, count in response_modes.items():
        tokens_per_result = token_costs[mode]
        total_tokens += tokens_per_result * count * num_searches

    return total_tokens

# Plan research with 100K token budget
budget = calculate_token_budget(
    num_searches=5,
    response_modes={
        "ids_only": 50,    # 5 searches × 50 results × 10 tokens
        "metadata": 20,     # 5 searches × 20 results × 250 tokens
        "preview": 10,      # 5 searches × 10 results × 500 tokens
        "full": 3          # 5 searches × 3 results × 1500 tokens
    }
)
print(f"Estimated token usage: {budget:,} tokens")
```

### 3. Error Recovery Patterns

Always implement graceful error handling:
- Catch specific errors (vendor not found, rate limits)
- Provide fallback strategies (search when lookup fails)
- Log errors for debugging
- Retry with exponential backoff for transient errors

### 4. Caching Strategy

Implement client-side caching to reduce API calls:

```python
from functools import lru_cache
from hashlib import md5

@lru_cache(maxsize=100)
def cached_search(query_hash):
    """Cache search results by query hash."""
    # Note: In real implementation, decode hash back to query
    return semantic_search(query)

def search_with_cache(query, **kwargs):
    """Search with result caching."""
    # Create cache key from query and parameters
    cache_key = md5(f"{query}{kwargs}".encode()).hexdigest()
    return cached_search(cache_key)
```

---

*These examples demonstrate practical usage patterns for the BMCIS Knowledge MCP tools. Adapt them to your specific use cases and requirements.*