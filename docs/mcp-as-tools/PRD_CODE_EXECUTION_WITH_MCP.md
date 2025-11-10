<div align="center">

[🏠 Home](#) | [📋 ToC](#table-of-contents) | [🚀 Quick Start](#phase-0-foundation--setup) | [⚠️ Risks](#risk-priority-matrix) | [✅ Success](#success-criteria-for-mvp) | [📊 Tests](#test-pyramid) | [🏗️ Roadmap](#development-phases)

</div>

---

# Code Execution with MCP: Product Requirements Document

**Repository Planning Graph (RPG) Format**
**Target Audience**: Engineering Team
**Status**: Ready for Implementation
**Last Updated**: November 2024

---

## Executive Summary

This PRD defines the implementation of **Code Execution with MCP**, a transformative capability that enables Claude agents to execute Python code within a secure sandbox environment, reducing token overhead by **90-95%** while dramatically improving performance for complex search and filtering workflows in BMCIS Knowledge MCP.

**Key Metrics:**
- **Token Reduction**: 150,000 → 7,500-15,000 tokens (90-95% reduction)
- **Latency Improvement**: 1,200ms → 300-500ms (3-4x faster)
- **Cost Savings**: $0.45 → $0.02-$0.05 per complex query (90-95% cheaper)
- **Target Adoption**: >50% of qualifying agents within 90 days, >80% within 180 days

**Value Proposition**: Code execution enables **progressive disclosure** patterns where agents request compact metadata first, then selectively fetch full content only for relevant results. This achieves industry-leading token efficiency (90-95% reduction) while preserving accuracy through on-demand detail retrieval.

---

## Table of Contents

### Quick Links
[Executive Summary](#executive-summary) | [Problem Statement](#problem-statement) | [Implementation Roadmap](#development-phases) | [Risks](#technical-risks) | [Test Strategy](#test-pyramid)

### Detailed Navigation

1. **[Executive Summary](#executive-summary)**
   - Key Metrics
   - Target Adoption

2. **[Overview](#problem-statement)**
   - 2.1 [Problem Statement](#problem-statement)
   - 2.2 [Target Users](#target-users)
   - 2.3 [Success Metrics](#success-metrics)

3. **[Functional Decomposition](#capability-tree)**
   - 3.1 [Capability Tree](#capability-tree)
     - Code Execution Engine
     - Search & Processing APIs
     - MCP Integration

4. **[Structural Decomposition](#repository-structure)**
   - 4.1 [Repository Structure](#repository-structure)
   - 4.2 [Module Definitions](#module-definitions)

5. **[Dependency Graph](#dependency-chain)**
   - 5.1 [Foundation Layer (Phase 0)](#foundation-layer-phase-0)
   - 5.2 [Code Intelligence Layer (Phase 1)](#code-intelligence-layer-phase-1)
   - 5.3 [Security & Execution Layer (Phase 2)](#security--execution-layer-phase-2)
   - 5.4 [Integration Layer (Phase 3)](#integration-layer-phase-3)

6. **[Implementation Roadmap](#development-phases)**
   - 6.1 [Phase 0: Foundation & Setup](#phase-0-foundation--setup)
   - 6.2 [Phase 1: Code Search & Processing APIs](#phase-1-code-search--processing-apis)
   - 6.3 [Phase 2: Sandbox & Execution Engine](#phase-2-sandbox--execution-engine)
   - 6.4 [Phase 3: MCP Integration & Full System Testing](#phase-3-mcp-integration--full-system-testing)

7. **[Test Strategy](#test-pyramid)**
   - 7.1 [Test Pyramid](#test-pyramid)
   - 7.2 [Coverage Requirements](#coverage-requirements)
   - 7.3 [Critical Test Scenarios](#critical-test-scenarios)

8. **[Architecture](#system-components)**
   - 8.1 [System Components](#system-components)
   - 8.2 [Data Models](#data-models)
   - 8.3 [Technology Stack](#technology-stack)
   - 8.4 [Design Decisions](#design-decisions)

9. **[Risks](#technical-risks)**
   - 9.1 [Technical Risks](#technical-risks)
   - 9.2 [Dependency Risks](#dependency-risks)
   - 9.3 [Scope Risks](#scope-risks)
   - 9.4 [Risk Priority Matrix](#risk-priority-matrix)

10. **[Appendix](#references)**
    - 10.1 [References](#references)
    - 10.2 [Glossary](#glossary)
    - 10.3 [Open Questions](#open-questions)
    - 10.4 [Success Criteria](#success-criteria-for-mvp)

---

<overview>

## Problem Statement

Claude agents accessing BMCIS Knowledge MCP through traditional tool calling face severe efficiency bottlenecks when executing complex search workflows. A typical multi-stage search operation (BM25 keyword search → vector similarity search → reranking → metadata filtering) consumes 150,000+ tokens per workflow execution. This creates three critical pain points:

1. **Token Overhead**: Each tool call requires full schema serialization, parameter validation, and result marshaling. A 4-step search workflow requires 4 separate round-trips, with each step adding 30,000-40,000 tokens of overhead.

2. **Latency Cascades**: Sequential tool calls introduce cumulative latency. A search workflow requiring BM25 filtering (200ms) → vector search (300ms) → reranking (400ms) → metadata filtering (100ms) totals 1,000ms+ in network round-trips alone, excluding agent reasoning time between steps.

3. **Cost Explosion**: At current token pricing, complex search workflows cost $0.30-0.45 per execution. For agents performing 100+ searches per session, this translates to $30-45 in token costs, making production deployment economically prohibitive.

Concrete example: An agent researching "enterprise authentication patterns in microservices" must (1) keyword search for "authentication", (2) vector search for semantic matches, (3) rerank by relevance, (4) filter by "microservices" tag. This 4-step workflow consumes 152,000 tokens and takes 1.2 seconds.

**Progressive Disclosure Opportunity**: Traditional tool calling forces agents to receive full content for all results upfront (150K tokens). Code execution enables a progressive disclosure pattern:
1. Initial search returns metadata + signatures (2K tokens)
2. Agent analyzes and identifies relevant subset
3. Selective full-content requests for top 2-3 results (12K tokens)
4. Total: 14K tokens (91% reduction) with same accuracy

This pattern aligns with human research behavior: skim many results, deep-dive on few.

## Target Users

**Primary Persona: Research & Analysis Agents**
- Claude agents performing knowledge discovery, technical research, or multi-document analysis
- Workflows involving 10-50+ search operations per session with complex filtering requirements
- Need to compose search primitives (BM25, vector, reranking) into custom retrieval pipelines
- Current pain: Token budgets exhausted after 3-5 complex searches, forcing workflow simplification

**Secondary Persona: Integration Engineers**
- Developers building agent-powered applications that query BMCIS Knowledge MCP
- Requirements: Sub-500ms search latency, <$0.10 per agent session cost
- Current pain: Traditional tool calling makes production deployment cost-prohibitive and latency-sensitive applications infeasible

**Workflow Context**:
- Agents spawn for 30-90 minute research sessions
- Execute 20-100 search operations with varying complexity
- Require programmable search composition (e.g., "search A OR B, then rerank by C, filter by D")
- Need to iterate on search parameters based on intermediate results

## Success Metrics

**Token Efficiency**:
- **Primary**: 90-95% token reduction for multi-step search workflows (150,000 → 7,500-15,000 tokens)
- **Secondary**: Average tokens per search operation <4,000 (vs. 37,500 baseline)
- **Context**: Realistic reduction accounting for iterative refinement and selective full-content requests

**Performance**:
- **Primary**: 3-4x latency improvement for 4-step workflows (1,200ms → 300-500ms)
- **Secondary**: 95th percentile search latency <500ms for code execution path
- **Context**: Includes subprocess startup overhead (50-100ms)

**Cost Reduction**:
- **Primary**: 90-95% cost savings per complex search workflow ($0.45 → $0.02-$0.05)
- **Secondary**: Agent session cost <$0.30 for 50-search sessions (vs. $22.50 baseline)
- **Context**: Based on Claude Sonnet pricing ($3/million input tokens)

**Adoption**:
- **Primary**: >50% of agents with 10+ search operations adopt code execution within 90 days
- **Secondary**: >80% adoption within 180 days
- **Tertiary**: >70% of integration engineers recommend for new implementations
- **Context**: Realistic timeline accounting for documentation, examples, and ecosystem maturity

**Reliability**:
- **Primary**: Code execution sandbox reliability >99.9% (no crashes, memory leaks, or security violations)
- **Secondary**: Error rate <2% for valid Python code submissions
- **Tertiary**: Token budget compliance >95% (actual usage within 10% of predicted)

### Progressive Disclosure Pattern

The revised metrics assume implementation of a **progressive disclosure pattern** where agents request increasingly detailed content:

**Level 0: IDs Only (100-500 tokens)**
- Use case: Counting matches, checking existence, cache validation
- Token budget: 100-500 tokens for 50 results

**Level 1: Signatures + Metadata (2,000-4,000 tokens)**
- Use case: Understanding what functions exist, where they are, high-level purpose
- Token budget: 200-400 tokens/result = 2,000-4,000 tokens for 10 results

**Level 2: Signatures + Truncated Bodies (5,000-10,000 tokens)**
- Use case: Understanding implementation approach without full details
- Token budget: 500-1,000 tokens/result = 5,000-10,000 tokens for 10 results

**Level 3: Full Content (10,000-50,000+ tokens)**
- Use case: Deep implementation analysis, refactoring, debugging
- Token budget: 10,000-15,000 tokens/result = 30,000-45,000 tokens for 3 results

</overview>

---

<functional-decomposition>

## Capability Tree

### Capability: Code Execution Engine
Core infrastructure for safely executing agent-written Python code with resource isolation, validation, and runtime management.

#### Feature: Code Validation & Security Analysis
- **Description**: Statically analyze code for dangerous patterns before execution, enforcing whitelist of allowed operations
- **Inputs**: Python source code (string), security configuration (whitelist, allowed modules)
- **Outputs**: Validation result (pass/fail), list of blocked operations, risk assessment level
- **Behavior**: Parse code AST, check for eval/exec/import violations, scan for dangerous builtins, return security analysis report

#### Feature: Sandbox Environment & Resource Isolation
- **Description**: Execute code in isolated subprocess with CPU, memory, and execution time limits enforced at OS level
- **Inputs**: Validated Python code, resource limits (memory MB, timeout seconds), execution context (globals/locals)
- **Outputs**: Execution result (output, return value, errors), resource usage metrics, execution metadata
- **Behavior**: Spawn subprocess with resource limits, execute code, capture output, enforce timeout via SIGKILL/TerminateProcess, cleanup resources

#### Feature: Runtime Orchestration & Error Recovery
- **Description**: Manage complete execution lifecycle including setup, validation, execution, cleanup, and error handling
- **Inputs**: Code string, execution context, retry configuration
- **Outputs**: Final execution result with success status, complete execution trace, error details if failed
- **Behavior**: Validate → execute → capture output → handle errors → cleanup, with exponential backoff retry for transient failures

### Capability: Search & Processing APIs
High-level APIs for searching knowledge base and processing results through ranking, filtering, and formatting.

#### Feature: Hybrid Search (BM25 + Vector)
- **Description**: Perform combined keyword and semantic search across code/documentation repository
- **Inputs**: Query string, BM25 weight (0-1), vector weight (0-1), top_k results, optional filters
- **Outputs**: List of SearchResult objects with relevance scores and metadata
- **Behavior**: Execute BM25 search and vector search in parallel, fuse results with weighted scoring, return top-k, all processing stays in-environment

#### Feature: Semantic Reranking
- **Description**: Improve result quality by reranking with cross-encoder models
- **Inputs**: Original query string, document list, top_k to return, optional relevance threshold
- **Outputs**: Reranked document list with updated relevance scores
- **Behavior**: Compute pairwise relevance scores, sort by score, apply threshold filter, return top-k with new rankings

#### Feature: Result Filtering & Selection
- **Description**: Filter search results by domain, score threshold, metadata, or custom predicates
- **Inputs**: Search results, filter criteria (domain, min_score, metadata fields), optional custom filter function
- **Outputs**: Filtered result subset matching all criteria
- **Behavior**: Iterate results, apply each filter criterion, accumulate matching results, preserve original ordering where possible

#### Feature: Result Processing & Compaction
- **Description**: Format results into compact representations suitable for agent consumption while preserving essential information
- **Inputs**: Raw search/execution results, output format (json/markdown/text), max preview length, field selection
- **Outputs**: Formatted results string or structured object, token-efficient representation
- **Behavior**: Truncate long content to preview length, select only specified fields, apply formatting template, return compact output

### Capability: MCP Integration
Tool definitions and server integration for Model Context Protocol, enabling agent access to code execution.

#### Feature: MCP Tool Definition & Schema
- **Description**: Define execute_code and search_code tools with input/output schemas compliant with MCP specification
- **Inputs**: Tool specifications (name, description, input schema, output schema)
- **Outputs**: MCP-compliant tool definitions in JSON schema format
- **Behavior**: Create JSON schema objects, validate against MCP specification, produce tool manifests for server registration

#### Feature: Request Routing & Handling
- **Description**: Route incoming MCP requests to appropriate handler (code execution vs search), validate inputs, invoke tool, format response
- **Inputs**: MCP request object (tool name, arguments), execution context
- **Outputs**: MCP response object (success/error, result data)
- **Behavior**: Parse request → validate inputs against schema → invoke handler → capture result → format response → return to client

#### Feature: Server Integration & Lifecycle
- **Description**: Initialize MCP server, register tools, handle client connections, manage session lifecycle
- **Inputs**: Server configuration (port, handlers, tools), client requests
- **Outputs**: Server runtime, registered tools, client responses
- **Behavior**: Boot server → register all tools → accept connections → dispatch requests → handle errors → graceful shutdown

</functional-decomposition>

---

<structural-decomposition>

## Repository Structure

```
bmcis-knowledge-mcp/
├── src/
│   ├── code_api/
│   │   ├── __init__.py                    # Public API exports
│   │   ├── search.py                      # HybridSearchAPI, SearchResult
│   │   ├── reranking.py                   # RerankerAPI
│   │   ├── filtering.py                   # FilterAPI
│   │   └── results.py                     # ResultProcessor
│   ├── code_execution/
│   │   ├── __init__.py                    # Public exports
│   │   ├── sandbox.py                     # CodeExecutionSandbox
│   │   ├── agent_interface.py             # AgentCodeExecutor
│   │   └── validation.py                  # InputValidator, SecurityChecker
│   ├── mcp_tools/
│   │   ├── __init__.py                    # Public exports
│   │   ├── code_execution_tool.py         # MCP tool definition
│   │   └── server_integration.py          # MCP server integration
│   └── [existing modules remain unchanged]
├── tests/
│   ├── test_code_api.py                   # API functionality tests
│   ├── test_sandbox_security.py           # Security validation tests
│   ├── test_code_execution.py             # Execution engine tests
│   ├── integration/
│   │   ├── test_mcp_code_execution.py     # MCP integration tests
│   │   └── test_end_to_end.py             # Full workflow tests
│   └── fixtures/
│       ├── sample_code.py                 # Test code samples
│       └── test_queries.json              # Test search queries
├── docs/
│   ├── code-execution/
│   │   ├── README.md                      # Feature overview
│   │   ├── implementation-guide.md        # Detailed implementation steps
│   │   └── security-model.md              # Security architecture
│   └── mcp-as-tools/
│       └── [this PRD]
├── pyproject.toml
├── .env.example                           # Environment configuration template
└── README.md
```

## Module Definitions

### Module: src/code_api/
- **Maps to capability**: Search & Processing APIs
- **Responsibility**: Provide high-level Python APIs for search, reranking, filtering, and result processing operations
- **File structure**:
  ```
  code_api/
  ├── __init__.py              # Exports: HybridSearchAPI, RerankerAPI, FilterAPI, ResultProcessor
  ├── search.py                # HybridSearchAPI class
  ├── reranking.py             # RerankerAPI class
  ├── filtering.py             # FilterAPI class
  └── results.py               # ResultProcessor class, SearchResult dataclass
  ```
- **Exports**:
  - `HybridSearchAPI` - Execute BM25 + vector search with configurable fusion
  - `RerankerAPI` - Rerank results using cross-encoder models
  - `FilterAPI` - Filter results by domain, score, metadata
  - `ResultProcessor` - Format and compact results for token efficiency
  - `SearchResult` - Dataclass for individual search results

### Module: src/code_execution/
- **Maps to capability**: Code Execution Engine
- **Responsibility**: Safe execution of untrusted Python code with resource isolation, validation, and error handling
- **File structure**:
  ```
  code_execution/
  ├── __init__.py              # Exports: CodeExecutionSandbox, AgentCodeExecutor, ValidationResult
  ├── sandbox.py               # CodeExecutionSandbox class, execution logic
  ├── agent_interface.py       # AgentCodeExecutor wrapper for agent use
  └── validation.py            # InputValidator, dangerous pattern detection
  ```
- **Exports**:
  - `CodeExecutionSandbox` - Isolated execution environment with resource limits
  - `AgentCodeExecutor` - High-level interface for agents to execute code
  - `InputValidator` - Security validation and dangerous pattern detection
  - `ExecutionResult` - Dataclass containing execution output and metadata
  - `ValidationResult` - Dataclass for validation status and findings

### Module: src/mcp_tools/
- **Maps to capability**: MCP Integration
- **Responsibility**: Define MCP tools, handle request routing, integrate with MCP server
- **File structure**:
  ```
  mcp_tools/
  ├── __init__.py              # Exports: CodeExecutionTool, register_tools
  ├── code_execution_tool.py   # MCP tool definition and handler
  └── server_integration.py    # Server registration and initialization
  ```
- **Exports**:
  - `CodeExecutionTool` - MCP tool wrapper for execute_code functionality
  - `register_tools()` - Register tools with MCP server
  - `tool_execute_code()` - Handler for execute_code tool invocation
  - `MCPToolDefinition` - Tool definition with schema

</structural-decomposition>

---

<dependency-graph>

## Dependency Chain

### Foundation Layer (Phase 0)
No dependencies - these are built first.

- **Base Types & Data Models**: Provides core data structures (SearchResult, ExecutionResult, ValidationResult) and type definitions used across all modules
- **Configuration Management**: Provides environment configuration, resource limit settings, allowed module whitelist, and runtime parameters
- **Logging Infrastructure**: Provides structured logging, error tracking, debug output, and audit trail capabilities
- **Error Handling Framework**: Provides custom exception types, error codes, and standardized error response formatting

### Code Intelligence Layer (Phase 1)
Depends on Foundation Layer (base-types, logging, error-handling).

- **Code Search API**: Depends on [base-types, logging, error-handling]. Provides BM25 and vector search capabilities, hybrid fusion
- **Reranking Engine**: Depends on [base-types, logging, error-handling]. Provides semantic reranking using cross-encoder models
- **Result Filtering**: Depends on [base-types, logging, error-handling]. Provides filtering by domain, score, metadata
- **Result Processing**: Depends on [code-search-api, reranking-engine, result-filtering]. Aggregates and compacts results

### Security & Execution Layer (Phase 2)
Depends on Foundation Layer and Code Intelligence Layer.

- **Input Validator**: Depends on [base-types, logging, error-handling]. Performs AST analysis and dangerous pattern detection
- **CodeExecutionSandbox**: Depends on [base-types, input-validator, logging, error-handling]. Provides isolated code execution
- **AgentCodeExecutor**: Depends on [code-execution-sandbox, input-validator, code-intelligence-layer]. Orchestrates execution workflow

### Integration Layer (Phase 3)
Depends on all previous layers.

- **MCP Tool Definitions**: Depends on [code-executor, result-processing, base-types]. Defines tool schemas and parameters
- **Request Handler**: Depends on [mcp-tool-definitions, code-executor, result-processing]. Routes and processes requests
- **Server Integration**: Depends on [request-handler, configuration, logging, error-handling]. Implements MCP server lifecycle

### Validation Layer (Phase 4)
Depends on all implementation layers for comprehensive testing.

- **Unit Tests**: Tests all foundation and code intelligence modules in isolation
- **Integration Tests**: Tests interactions between security/execution layer and integration layer
- **End-to-End Tests**: Tests complete request-response cycles through MCP server
- **Security Tests**: Penetration testing and validation bypass attempts

</dependency-graph>

---

<implementation-roadmap>

## 🏗️ Development Phases

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 📋 Phase 0: Foundation & Setup
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Duration**: 1.5-2 weeks (10-12 working days)

**Team Composition**:
- 1 Senior Engineer (lead, 100%)
- 1 Mid-Level Engineer (80%)
- 1 Infrastructure Specialist (20%, setup only)
- **Total Effort**: 2.0 FTE-weeks

**Resource Requirements**:
- Development environments (local machines)
- CI/CD pipeline setup (GitHub Actions, free tier)
- Security scanning tools (bandit, safety - open source)
- Pre-commit hooks infrastructure

**Goal**: Establish project infrastructure, core type definitions, testing framework, and security specifications for the Code Execution with MCP system.

**Entry Criteria**:
- PRD specification approved and validated
- Technical architecture plan completed
- Development environment requirements documented
- Security requirements and constraints defined

**Tasks**:
- [ ] **Initialize project structure and development environment** (depends on: none)
  - Acceptance criteria:
    - Directory structure follows constitutional organization policies (src/, tests/, docs/)
    - Git repository initialized with appropriate .gitignore and branch protection rules
    - Development dependencies installed (Python 3.11+, MCP SDK, pytest, security tools)
    - Pre-commit hooks configured for linting, type checking, and security scanning
    - CI/CD pipeline configured and operational
  - Test strategy:
    - Verify directory structure matches specification
    - Run `pytest --collect-only` to verify test discovery
    - Execute pre-commit hooks on sample files
    - Verify CI pipeline triggers and passes on initial commit

- [ ] **Define and document core data models and interfaces** (depends on: none)
  - Acceptance criteria:
    - SearchResult, CodeSearchRequest, ExecutionRequest, ExecutionResult dataclasses defined
    - Type hints and pydantic validation for all models
    - Interface contracts documented with docstrings
    - All models have serialization/deserialization support
  - Test strategy:
    - Unit tests for model validation (valid/invalid inputs)
    - Serialization round-trip tests
    - Type checking with mypy (100% coverage)

- [ ] **Set up testing infrastructure and CI/CD pipeline** (depends on: project initialization)
  - Acceptance criteria:
    - pytest configured with coverage reporting
    - GitHub Actions workflow for automated testing
    - Security scanning tools integrated (bandit, safety)
    - Performance benchmarking framework established
  - Test strategy:
    - Run test suite locally and in CI
    - Verify coverage reporting accuracy
    - Confirm security scan detects intentional vulnerabilities

- [ ] **Create security sandbox specification and constraints** (depends on: none)
  - Acceptance criteria:
    - Resource limits documented (CPU cores, memory MB, execution seconds)
    - Filesystem access restrictions specified
    - Network isolation requirements documented
    - Python module whitelist created with rationale for each allowed module
  - Test strategy:
    - Security architecture review
    - Specification validation against OWASP guidelines
    - Whitelist review for completeness and safety

**Exit Criteria**:
- All development tools installed and verified working
- Core data models defined with passing type checks
- CI/CD pipeline executes successfully on all commits
- Security specification reviewed and approved by security team
- Repository has clean initial commit with foundation code

**Delivers**:
- Developers can clone repository and run tests locally
- CI/CD automatically validates code quality on commits
- Clear security boundaries established for implementation
- Foundation for parallel development of Phase 1-3 components

---

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔍 Phase 1: Code Search & Processing APIs
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Duration**: 2-3 weeks (12-18 working days)

**Team Composition**:
- 2 Senior Engineers (100% each, can work in parallel)
- 1 Mid-Level Engineer (100%)
- Security Specialist (10%, review only)
- **Total Effort**: 3.1 FTE-weeks

**Resource Requirements**:
- Sample knowledge base (100+ code files for testing)
- Vector embedding models (pre-trained, ~500MB download)
- BM25 indexing infrastructure
- Performance testing compute (can use local machines)

**Goal**: Implement core search functionality with hybrid search, semantic reranking, filtering, and result processing.

**Entry Criteria**:
- Phase 0 exit criteria met
- Search APIs architecture approved
- Sample knowledge base available for integration testing

**Tasks**:
- [ ] **Implement HybridSearchAPI with keyword and semantic search** (depends on: Phase 0)
  - Acceptance criteria:
    - BM25 keyword search implemented using rank_bm25 library
    - Vector search implemented with pretrained embeddings
    - Hybrid fusion algorithm combines both approaches with configurable weights (0-1)
    - API returns SearchResult objects with relevance scores (0-1 normalized)
    - Handles edge cases: empty queries, no results, malformed inputs
    - Full code execution stays in-environment (no context leakage)
  - Test strategy:
    - Unit tests: 20+ scenarios (function names, docstrings, imports, etc.)
    - Integration tests: search against sample codebase (100+ files)
    - Performance tests: <500ms for typical queries
    - Edge case validation: empty strings, special characters, unicode

- [ ] **Implement SemanticReranker for result optimization** (depends on: HybridSearchAPI)
  - Acceptance criteria:
    - Cross-encoder model integrated (cross-encoder/ms-marco-MiniLM-L-6-v2)
    - Reranking improves NDCG@10 by >15% on test queries
    - Configurable reranking depth and threshold
    - Efficient batching for multiple results
    - Returns reranked SearchResult objects with updated scores
  - Test strategy:
    - Unit tests: scoring correctness with synthetic queries
    - A/B tests: quality improvement measurement
    - Performance tests: reranking overhead <200ms for top-10

- [ ] **Implement FilterEngine for context-aware filtering** (depends on: HybridSearchAPI)
  - Acceptance criteria:
    - File type filters (by extension, language)
    - Path-based filters (include/exclude patterns regex)
    - Metadata filters (custom field matching)
    - Composite filters with AND/OR logic
    - Returns filtered SearchResult subset
  - Test strategy:
    - Unit tests: 25+ filter combinations
    - Integration tests: filtering on multi-language codebase
    - Performance tests: filtering overhead <50ms

- [ ] **Implement ResultProcessor for output formatting** (depends on: SemanticReranker, FilterEngine)
  - Acceptance criteria:
    - Multiple output formats (JSON, markdown, compact text)
    - Context extraction (surrounding lines, full functions)
    - Syntax highlighting metadata for code snippets
    - Token-efficient compaction (target: <5000 tokens for 10 results)
    - Pagination support for large result sets
  - Test strategy:
    - Unit tests: format correctness for each type
    - Snapshot tests: consistent formatting
    - Token counting: verify compaction targets

- [ ] **Create comprehensive integration test suite for search pipeline** (depends on: all Phase 1 tasks)
  - Acceptance criteria:
    - End-to-end tests: search → rerank → filter → format workflow
    - Test coverage ≥90% for Phase 1 components
    - Performance benchmarks documented
    - Error handling validation for all failure modes
  - Test strategy:
    - 30+ integration scenarios
    - Load testing: 100 concurrent searches
    - Failure injection: network errors, resource exhaustion
    - Regression test suite established

**Exit Criteria**:
- All search APIs pass unit and integration tests
- Code coverage ≥90% for Phase 1
- Performance benchmarks <1s end-to-end search
- Code review and architecture approval
- Documentation complete with API examples

**Delivers**:
- Developers can perform hybrid search on knowledge base
- Results are accurately ranked and filtered
- Multiple output formats support different use cases
- Comprehensive test suite prevents regressions

---

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🔒 Phase 2: Sandbox & Execution Engine
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Duration**: 2.5-3.5 weeks (15-21 working days)

**Team Composition**:
- 2 Senior Engineers (100% each)
- 1 Security Specialist (40%, critical phase)
- 1 Mid-Level Engineer (80%, testing focus)
- **Total Effort**: 3.2 FTE-weeks

**Resource Requirements**:
- RestrictedPython library and dependencies
- Security testing tools (custom scripts + open source scanners)
- Resource monitoring infrastructure (tracemalloc, psutil)
- Isolated testing environments (Docker optional)

**Goal**: Build secure code execution environment with resource isolation, validation, and orchestration.

**Entry Criteria**:
- Phase 0 security specification approved
- Phase 1 search APIs operational
- Subprocess-based execution approach validated and approved
- Security testing framework in place

**Tasks**:
- [ ] **Implement CodeExecutionSandbox with subprocess-based resource isolation** (depends on: Phase 0 security spec)
  - Acceptance criteria:
    - Subprocess-based execution with OS-level process isolation (REQUIRED in v1)
    - Timeout enforcement via SIGKILL (Linux/macOS) or TerminateProcess (Windows)
    - Resource limits enforced: memory <512MB via setrlimit, execution <30s via subprocess timeout
    - Restricted global namespace (only safe builtins) for user code
    - Execution results captured (stdout, stderr, return value, exceptions)
    - Clean subprocess lifecycle: spawn → execute → terminate → cleanup
    - Support for async code execution (async/await) within subprocess
    - 100% timeout reliability (no GIL interference, uncatchable SIGKILL)
  - Test strategy:
    - Security tests: 15+ attack vectors blocked including infinite loops
    - Timeout enforcement: 100% reliability with CPU-bound infinite loops
    - Platform compatibility: Linux, macOS, Windows subprocess termination
    - Resource limit tests: CPU, memory enforcement via OS
    - Stability tests: 1000+ executions without leaks
    - Output capture tests: all streams captured correctly
    - Subprocess overhead: <30ms P95 measured

- [ ] **Implement InputValidator for code safety checking** (depends on: CodeExecutionSandbox)
  - Acceptance criteria:
    - AST-based analysis for dangerous patterns (eval, exec, import violations)
    - Static analysis prevents code injection attacks
    - Configurable safety rules with whitelist/denylist
    - Clear error messages for validation failures
    - Validation completes in <100ms
  - Test strategy:
    - 50+ malicious code samples blocked
    - Legitimate code passes validation
    - Performance: <100ms validation
    - Security audit: penetration testing

- [ ] **Implement AgentCodeExecutor for orchestrated execution** (depends on: CodeExecutionSandbox, InputValidator)
  - Acceptance criteria:
    - Workflow: validate → execute → capture → cleanup
    - Retry logic with exponential backoff (max 3 retries)
    - Execution metadata logging (duration, resource usage, exit status)
    - Error recovery and graceful degradation
    - Support for multi-step execution plans
  - Test strategy:
    - Integration tests: end-to-end workflows
    - Failure injection: validate retry logic
    - Concurrency tests: parallel executions
    - Performance: <5s overhead for typical queries

- [ ] **Implement result validation and output sanitization** (depends on: AgentCodeExecutor)
  - Acceptance criteria:
    - Output sanitization removes XSS/injection vectors
    - Result size limits enforced (max 10MB output)
    - Structured error reporting with stack traces
    - Execution artifacts logged for debugging
  - Test strategy:
    - Security: malicious output sanitization
    - Size limits: large outputs truncated
    - Error format: consistent structure
    - Audit trail: all executions logged

**Exit Criteria**:
- Sandbox successfully isolates code execution
- Security testing: zero isolation breaches
- Resource limits enforced 100%
- Code review and security audit approved
- Execution performance <5s overhead
- All tests pass with ≥90% coverage

**Delivers**:
- Agents can execute untrusted code safely
- System administrators configure resource limits
- Security team can audit all executions
- Users receive validated, sanitized results

---

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 🚀 Phase 3: MCP Integration & Full System Testing
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Duration**: 2-2.5 weeks (12-15 working days)

**Team Composition**:
- 2 Senior Engineers (100% each)
- 1 Mid-Level Engineer (100%)
- Security Specialist (20%, audit)
- Infrastructure Specialist (20%, deployment prep)
- **Total Effort**: 3.4 FTE-weeks

**Resource Requirements**:
- MCP SDK and dependencies
- End-to-end testing infrastructure
- Monitoring and observability tools (logging, metrics)
- Production-like staging environment
- External security audit engagement

**Goal**: Integrate code search and execution into MCP server with complete testing and production readiness.

**Entry Criteria**:
- Phase 1 and Phase 2 complete and validated
- MCP SDK configured
- Tool definitions specified
- Integration test plan approved

**Tasks**:
- [ ] **Define MCP tool schemas for search and execution** (depends on: Phase 1, Phase 2)
  - Acceptance criteria:
    - `execute_code` tool: code (string), description (string), timeout (int, optional)
    - `search_code` tool: query (string), top_k (int), filters (dict, optional)
    - JSON schema validation for all inputs
    - Tool descriptions with realistic examples
    - Error response schemas defined
  - Test strategy:
    - Schema validation: valid/invalid inputs
    - Documentation: examples execute correctly
    - Protocol compliance: validate against MCP spec

- [ ] **Implement MCP server with tool registration** (depends on: MCP tool schemas)
  - Acceptance criteria:
    - Server initializes and registers tools correctly
    - Tools discoverable via MCP list_tools
    - Request routing to search/execution handlers
    - Error handling with MCP-compliant error responses
    - Server lifecycle management (startup, shutdown, cleanup)
  - Test strategy:
    - MCP client tests: tool discovery and invocation
    - Protocol tests: MCP specification compliance
    - Error handling: malformed requests
    - Load: 50 concurrent requests

- [ ] **Create end-to-end integration test suite** (depends on: MCP server)
  - Acceptance criteria:
    - Full workflows: search → execute → results
    - Multi-step scenarios validated
    - Error propagation through MCP boundary
    - Performance: <2s end-to-end latency
    - All integration points tested
  - Test strategy:
    - 20+ realistic usage scenarios
    - Failure modes: network errors, timeouts
    - Load testing: sustained throughput
    - Regression suite: critical paths

- [ ] **Implement monitoring, logging, and observability** (depends on: MCP server)
  - Acceptance criteria:
    - Structured logging for all operations
    - Metrics: request rates, latencies, error rates
    - Health check endpoint
    - Execution audit trail with retention policy
  - Test strategy:
    - Log validation: completeness
    - Metrics accuracy: verification
    - Health checks: various states
    - Audit trail: retention verification

- [ ] **Perform security audit and penetration testing** (depends on: all Phase 3 tasks)
  - Acceptance criteria:
    - Third-party security audit completed
    - All high/critical vulnerabilities remediated
    - Zero isolation breaches in penetration testing
    - Security documentation updated
    - Incident response plan established
  - Test strategy:
    - External audit by security firm
    - Automated vulnerability scanning
    - Manual penetration testing
    - Security checklist (OWASP top 10)

**Exit Criteria**:
- MCP server operational with tools discoverable
- End-to-end tests pass 95%+ success rate
- Security audit: zero unresolved critical issues
- Performance: 99th percentile <3s
- Documentation: complete and approved
- Production deployment checklist validated

**Delivers**:
- Production-ready MCP server with tools
- Users can search and execute code via MCP
- Comprehensive monitoring and observability
- Security-validated system
- Complete documentation


---

## Timeline & Resource Planning

### Project Duration
- **Optimized Timeline**: 7-9 weeks
- **Recommended Timeline**: 8 weeks + 1 week buffer
- **Critical Path**: Foundation → Sandbox → Integration (6-7 weeks)

### Team Composition Summary
- **Core Team**: 2 Senior Engineers, 1 Mid-Level Engineer (3-4 FTE)
- **Specialists**: Security Specialist (2-3 weeks), Infrastructure Specialist (0.5-1 week)
- **Total Effort**: 25-35 FTE-weeks across all phases

### Budget Estimate
- **Engineering**: $120K-160K (with benefits/overhead)
- **Infrastructure**: $2K-5K (compute, tools, environments)
- **Security Audit**: $10K-25K (external penetration testing)
- **Total Project Cost**: $135K-190K

### Key Milestones
1. **Week 2**: Foundation complete, security spec approved
2. **Week 4**: Search APIs operational, sandbox core working
3. **Week 5**: Security validated, zero breaches
4. **Week 7**: MCP integration complete, E2E tests passing
5. **Week 8**: Security audit passed, production ready

### Parallelization Strategy
- **Weeks 3-4**: Phase 1 (Search APIs) || Phase 2a (Sandbox Core) - run in parallel
- **Week 5**: Phase 2b (Security Hardening) - all hands
- **Weeks 6-7**: Phase 3 (MCP Integration) - all hands
- **Savings**: 2-3 weeks vs sequential approach

### Critical Path Analysis
The longest sequential path determines minimum project duration:

```
Phase 0: Foundation (2 weeks)
    ↓
Phase 2a: Core Sandbox (2 weeks)
    ↓
Phase 2b: Security Hardening (1.5 weeks)
    ↓
Phase 3: MCP Integration (2-2.5 weeks)
```

**Total Critical Path**: 6-7 weeks minimum

**Why This Path is Critical**:
- Foundation must complete before any implementation
- Sandbox security is highest risk, cannot be parallelized
- MCP integration requires working sandbox
- Security audit must be last (requires complete system)

**Parallelizable Work** (saves 2-3 weeks):
- Phase 1 (Search APIs) can run parallel to Phase 2a (Sandbox Core)
- Documentation and testing can proceed continuously
- Infrastructure setup can overlap with Phase 0

For detailed phase breakdowns with task estimates, see `SOLUTION_TIME_ESTIMATES.md`.


</implementation-roadmap>

---

<test-strategy>

## Test Pyramid

```
        /\
       /E2E\       ← 5% (End-to-end, full workflows)
      /------\
     /Integration\ ← 25% (Module interactions)
    /------------\
   /  Unit Tests  \ ← 70% (Fast, isolated, deterministic)
  /----------------\
```

## Coverage Requirements
- **Line coverage**: 85% minimum
- **Branch coverage**: 80% minimum
- **Function coverage**: 90% minimum
- **Critical path coverage**: 100% (security, execution, error handling)

## Critical Test Scenarios

### CodeExecutionSandbox Security & Isolation
**Happy path**:
- Execute simple Python code (print, arithmetic)
- Expected: Successful execution, stdout captured, clean exit

**Edge cases**:
- Code execution timeout (exceeds 30s limit)
- Expected: TimeoutError raised, partial output captured if any
- Code with valid allowed imports (json, math, datetime)
- Expected: Successful execution with module available

**Error cases**:
- Code attempts forbidden import (os, subprocess, socket)
- Expected: ImportError blocked, security violation logged
- Code contains dangerous patterns (eval, exec, __import__)
- Expected: Execution blocked, SecurityError raised
- Code with syntax errors
- Expected: SyntaxError caught, error message returned

**Integration points**:
- MCP tool invokes sandbox with code payload
- Expected: Code executes in isolation, results sanitized, returned to MCP layer

### HybridSearchAPI Functionality
**Happy path**:
- Execute search with valid query, returns top_k ranked results
- Expected: Results ordered by relevance score, limited to top_k

**Edge cases**:
- Query with no matching results
- Expected: Empty result set, no errors
- Reranking disabled (rerank=false)
- Expected: Results returned without reranking step
- Large result set (1000+ matches)
- Expected: top_k selected correctly, performance <1s

**Error cases**:
- Empty query string
- Expected: ValidationError, clear message
- Search backend unavailable
- Expected: ServiceError, graceful fallback
- Timeout during search
- Expected: TimeoutError, partial results if available

**Integration points**:
- Sandbox executes code calling HybridSearchAPI
- Expected: Results stay in-environment, compact output returned

### MCP Tool Invocation End-to-End
**Happy path**:
- Client invokes execute_code tool with valid Python code
- Expected: Code executes in sandbox, results formatted, returned to client in <2s

**Edge cases**:
- Tool invoked with missing optional parameters
- Expected: Defaults applied, execution proceeds
- Multiple concurrent tool invocations
- Expected: Each handled independently, no race conditions

**Error cases**:
- Tool invocation with malformed parameters
- Expected: ValidationError with parameter details
- Tool execution fails (timeout, exception)
- Expected: Error message propagated to client

**Integration points**:
- MCP client → MCP server → Tool routing → Module execution → Response formatting
- Expected: End-to-end flow completes, errors propagate correctly

## Test Generation Guidelines

### Unit Test Generation (70%)
- **Target**: Individual functions and classes (SearchAPI, Sandbox, etc.)
- **Mock external dependencies**: Database, filesystem, network calls
- **Focus areas**: Input validation, business logic, edge cases, error handling
- **Coverage**: >90% for tested functions
- **Naming**: `test_<module>_<function>_<scenario>`

### Integration Test Generation (25%)
- **Target**: Module interactions (API + Sandbox, Reranker + Filter, etc.)
- **Test real integrations** with test fixtures
- **Focus areas**: Data flow, API contracts, error propagation
- **Coverage**: >80% for integration paths
- **Naming**: `test_integration_<module1>_<module2>_<scenario>`

### End-to-End Test Generation (5%)
- **Target**: Full workflows through MCP server
- **Test complete paths**: Client → MCP → Tool → Execution → Response
- **Focus areas**: User workflows, performance, observability
- **Coverage**: >90% for critical user paths
- **Naming**: `test_e2e_<workflow>_<scenario>`

### Security Test Generation
- **Target**: Sandbox isolation, input validation, error handling
- **Focus**: Attack vectors, bypass attempts, resource exhaustion
- **Coverage**: 100% of critical security paths
- **Tools**: Bandit for static analysis, custom penetration tests

</test-strategy>

---

<architecture>

## System Components

**1. Sandbox Isolation Layer**
- **Responsibility**: Execute untrusted Python code with strict resource constraints and security controls
- **Key Features**: Whitelist-based module restrictions, timeout enforcement, memory limits, output capture

**2. API Abstraction Layer**
- **Responsibility**: Provide high-level Python APIs for search, filtering, reranking operations
- **Key Features**: Unified search interface, composable operations, in-environment processing

**3. Result Processing Pipeline**
- **Responsibility**: Format and compact execution results for token efficiency
- **Key Features**: Multiple output formats, intelligent truncation, field selection, metadata preservation

**4. MCP Server Integration Layer**
- **Responsibility**: Register tools, handle requests, manage MCP protocol communication
- **Key Features**: Tool schema definition, request routing, error normalization, session management

## System Architecture Diagrams

The following diagrams provide comprehensive visual documentation of the system architecture, illustrating component interactions, execution flows, and deployment topology with security layers.

### Component Interaction Diagram

This diagram shows the static architecture of all system components and their communication patterns.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MCP Agent Environment                               │
│  ┌────────────────┐                                      ┌─────────────────┐   │
│  │  Claude Agent  │──────── MCP Request ────────────────▶│   MCP Client    │   │
│  │                │◀─────── MCP Response ────────────────│                 │   │
│  └────────────────┘                                      └────────┬────────┘   │
└──────────────────────────────────────────────────────────────────┼─────────────┘
                                                                    │
                                                          MCP Protocol (JSON-RPC)
                                                                    │
┌──────────────────────────────────────────────────────────────────▼─────────────┐
│                           MCP Server (bmcis-knowledge-mcp)                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                          Request Router & Handler                         │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │  │
│  │  │ Tool Registry │    │   Validator  │    │  Session Management      │  │  │
│  │  │              │    │              │    │                          │  │  │
│  │  │ • execute_code    │ • Schema     │    │ • Connection tracking   │  │  │
│  │  │ • search_code │    │ • Input      │    │ • Request queuing       │  │  │
│  │  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘  │  │
│  └──────────┼───────────────────┼────────────────────────┼─────────────────┘  │
│             │                   │                        │                     │
│  ┌──────────▼───────────────────▼────────────────────────▼─────────────────┐  │
│  │                        AgentCodeExecutor (Orchestrator)                  │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Workflow: Validate → Execute → Capture → Process → Return         │  │  │
│  │  └────┬───────────────┬────────────────┬─────────────────┬───────────┘  │  │
│  └───────┼───────────────┼────────────────┼─────────────────┼──────────────┘  │
│          │               │                │                 │                  │
│  ┌───────▼────────┐ ┌────▼─────────┐ ┌────▼──────────┐ ┌────▼──────────┐     │
│  │                │ │              │ │               │ │               │     │
│  │ InputValidator │ │ CodeExecutor │ │  Search APIs  │ │ResultProcessor│     │
│  │                │ │  (Sandbox)   │ │               │ │               │     │
│  │ ┌────────────┐ │ │ ┌──────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │     │
│  │ │    AST     │ │ │ │Subprocess│ │ │ │  Hybrid   │ │ │ │  Format   │ │     │
│  │ │  Analysis  │ │ │ │Isolation │ │ │ │  Search   │ │ │ │  Compact  │ │     │
│  │ ├────────────┤ │ │ ├──────────┤ │ │ ├───────────┤ │ │ ├───────────┤ │     │
│  │ │ Dangerous  │ │ │ │ Resource │ │ │ │ Semantic  │ │ │ │ Truncate  │ │     │
│  │ │  Pattern   │ │ │ │  Limits  │ │ │ │  Rerank   │ │ │ │  Fields   │ │     │
│  │ │ Detection  │ │ │ ├──────────┤ │ │ ├───────────┤ │ │ ├───────────┤ │     │
│  │ ├────────────┤ │ │ │  Output  │ │ │ │  Filter   │ │ │ │  Token    │ │     │
│  │ │ Whitelist  │ │ │ │ Capture  │ │ │ │  Engine   │ │ │ │Efficiency │ │     │
│  │ │  Modules   │ │ │ └──────────┘ │ │ └───────────┘ │ │ └───────────┘ │     │
│  │ └────────────┘ │ └──────────────┘ └───────────────┘ └───────────────┘     │
│  └────────────────┘                                                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        External Integration Points                       │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │  │
│  │  │  Codebase  │  │  Vector DB │  │   BM25     │  │  Cross-Encoder │   │  │
│  │  │   Files    │  │            │  │   Index    │  │     Models     │   │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Component Descriptions:**

| Component | Responsibility | Key Interfaces |
|-----------|---------------|----------------|
| **MCP Client** | Protocol communication with server | `call_tool()`, `list_tools()` |
| **Request Router** | Dispatch requests to appropriate handlers | Route by tool name |
| **AgentCodeExecutor** | Orchestrate entire execution workflow | Coordinate validation, execution, processing |
| **InputValidator** | Security analysis of submitted code | AST parsing, pattern detection |
| **CodeExecutor** | Isolated code execution environment | Subprocess with resource limits |
| **Search APIs** | Knowledge base search operations | BM25, vector, rerank, filter |
| **ResultProcessor** | Format results for token efficiency | Truncate, select fields, compact |

**Data Flow Patterns:**

1. **Inbound Flow**: Agent → MCP Client → MCP Server → Request Router → AgentCodeExecutor
2. **Execution Flow**: AgentCodeExecutor → InputValidator → CodeExecutor → Search APIs
3. **Result Flow**: Search APIs → ResultProcessor → AgentCodeExecutor → MCP Server → Agent
4. **Error Flow**: Any component → Error handler → Formatted error → MCP response

### Execution Sequence Diagram

This diagram illustrates the temporal flow of a code execution request, including timeout enforcement at multiple stages and comprehensive error handling.

```
Agent          MCP Client       MCP Server      Request Router    AgentCodeExecutor    InputValidator    CodeExecutor    Search APIs    ResultProcessor
  │                 │                │                 │                  │                   │                │               │                │
  │  execute_code   │                │                 │                  │                   │                │               │                │
  │  (code, timeout)│                │                 │                  │                   │                │               │                │
  ├────────────────▶│                │                 │                  │                   │                │               │                │
  │                 │   JSON-RPC     │                 │                  │                   │                │               │                │
  │                 │    Request     │                 │                  │                   │                │               │                │
  │                 ├───────────────▶│                 │                  │                   │                │               │                │
  │                 │                │  Parse & Route  │                  │                   │                │               │                │
  │                 │                ├────────────────▶│                  │                   │                │               │                │
  │                 │                │                 │   Orchestrate    │                   │                │               │                │
  │                 │                │                 ├─────────────────▶│                   │                │               │                │
  │                 │                │                 │                  │  Validate Code    │                │               │                │
  │                 │                │                 │                  ├──────────────────▶│                │               │                │
  │                 │                │                 │                  │                   │  AST Analysis  │               │                │
  │                 │                │                 │                  │                   ├───────┐        │               │                │
  │                 │                │                 │                  │                   │       │        │               │                │
  │                 │                │                 │                  │                   │◀──────┘        │               │                │
  │                 │                │                 │                  │                   │  Check Patterns│               │                │
  │                 │                │                 │                  │                   ├───────┐        │               │                │
  │                 │                │                 │                  │                   │       │        │               │                │
  │                 │                │                 │                  │   Validation     │◀──────┘        │               │                │
  │                 │                │                 │                  │     Result       │                │               │                │
  │                 │                │                 │                  │◀──────────────────┤                │               │                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │   [If Safe]       │                │               │                │
  │                 │                │                 │                  │   Execute Code    │                │               │                │
  │                 │                │                 │                  ├──────────────────────────────────▶│               │                │
  │                 │                │                 │                  │                   │                │ Create Process│                │
  │                 │                │                 │                  │                   │                ├──────┐        │                │
  │                 │                │                 │                  │                   │                │      │        │                │
  │                 │                │                 │                  │                   │                │◀─────┘        │                │
  │                 │                │                 │                  │                   │                │ Set Limits    │                │
  │                 │                │                 │                  │                   │                ├──────┐        │                │
  │                 │                │                 │                  │                   │                │      │        │                │
  │                 │                │                 │                  │                   │                │◀─────┘        │                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │ Start Timeout │                │
  │                 │                │                 │                  │                   │                ├──────┐        │                │
  │                 │                │                 │                  │                   │                │  30s │        │                │
  │                 │                │                 │                  │                   │                │◀─────┘        │                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │ Execute Code  │                │
  │                 │                │                 │                  │                   │                ├──────────────▶│                │
  │                 │                │                 │                  │                   │                │               │  Search Query  │
  │                 │                │                 │                  │                   │                │               ├───────┐        │
  │                 │                │                 │                  │                   │                │               │  BM25 │        │
  │                 │                │                 │                  │                   │                │               │◀──────┘        │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               ├───────┐        │
  │                 │                │                 │                  │                   │                │               │Vector │        │
  │                 │                │                 │                  │                   │                │               │◀──────┘        │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               ├───────┐        │
  │                 │                │                 │                  │                   │                │               │Rerank │        │
  │                 │                │                 │                  │                   │                │               │◀──────┘        │
  │                 │                │                 │                  │                   │                │  Results      │                │
  │                 │                │                 │                  │                   │                │◀──────────────┤                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │ Capture Output│                │
  │                 │                │                 │                  │                   │                ├──────┐        │                │
  │                 │                │                 │                  │                   │                │      │        │                │
  │                 │                │                 │                  │   Execution       │                │◀─────┘        │                │
  │                 │                │                 │                  │     Result        │                │               │                │
  │                 │                │                 │                  │◀──────────────────────────────────┤               │                │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │  Process Results  │                │               │                │
  │                 │                │                 │                  ├───────────────────────────────────────────────────────────────────▶│
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               │    Compact     │
  │                 │                │                 │                  │                   │                │               │   & Format     │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               │   Truncate     │
  │                 │                │                 │                  │                   │                │               │    Fields      │
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │   Formatted       │                │               │                │
  │                 │                │                 │                  │     Results       │                │               │                │
  │                 │                │                 │                  │◀───────────────────────────────────────────────────────────────────┤
  │                 │                │                 │                  │                   │                │               │                │
  │                 │                │                 │   Final Result   │                   │                │               │                │
  │                 │                │                 │◀─────────────────┤                   │                │               │                │
  │                 │                │  MCP Response   │                  │                   │                │               │                │
  │                 │                │◀────────────────┤                  │                   │                │               │                │
  │                 │   JSON-RPC     │                 │                  │                   │                │               │                │
  │                 │    Response    │                 │                  │                   │                │               │                │
  │                 │◀───────────────┤                 │                  │                   │                │               │                │
  │  Result/Error   │                │                 │                  │                   │                │               │                │
  │◀────────────────┤                │                 │                  │                   │                │               │                │
  │                 │                │                 │                  │                   │                │               │                │

                                                     ERROR HANDLING PATHS

  [Validation Failure]────────────────────────────────▶ Format Error ────▶ Return Error Response
  [Timeout (30s)]─────────────────────────────────────▶ Kill Process ────▶ Return Timeout Error
  [Resource Limit]────────────────────────────────────▶ Terminate ───────▶ Return Resource Error
  [Search API Error]──────────────────────────────────▶ Capture ─────────▶ Return Partial Results
```

**Sequence Phases:**

| Phase | Duration | Key Operations | Timeout Points |
|-------|----------|---------------|----------------|
| **1. Request Intake** | <10ms | Parse JSON-RPC, validate schema | Network timeout (5s) |
| **2. Validation** | <100ms | AST analysis, pattern detection | Validation timeout (1s) |
| **3. Execution Setup** | <50ms | Create subprocess, set limits | Process spawn timeout (2s) |
| **4. Code Execution** | Variable | Run user code, API calls | User-defined timeout (default 30s) |
| **5. Result Processing** | <200ms | Compact, format, truncate | Processing timeout (5s) |
| **6. Response** | <10ms | Serialize, send response | Network timeout (5s) |

**Error Recovery Strategies:**

1. **Validation Failure**: Return immediately with security violation details
2. **Timeout**: Kill subprocess via SIGKILL/TerminateProcess, return partial results if available
3. **Resource Exhaustion**: Terminate process, log metrics, return error
4. **API Failures**: Graceful degradation, return available data
5. **Network Issues**: Retry with exponential backoff (max 3 attempts)

### Deployment Topology Diagram

This diagram shows the runtime topology including process boundaries, thread pools (4-8 workers), defense-in-depth security layers, and resource isolation mechanisms.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Host Operating System                                  │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              MCP Server Process (Python)                            │ │
│  │                               PID: Main, UID: mcp-user                             │ │
│  │                                                                                     │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                           Main Thread (Event Loop)                            │ │ │
│  │  │                                                                                │ │ │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │ │
│  │  │  │ MCP Protocol │  │Request Queue │  │ Tool Registry│  │ Session Manager  │ │ │ │
│  │  │  │   Handler    │  │              │  │              │  │                  │ │ │ │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │ │ │
│  │  └───────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                     │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                         Thread Pool (4-8 workers)                             │ │ │
│  │  │                                                                                │ │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │ │ │
│  │  │  │  Worker Thread 1 │  │  Worker Thread 2 │  │  Worker Thread N │  ...        │ │ │
│  │  │  │                  │  │                  │  │                  │              │ │ │
│  │  │  │ • Request Handler│  │ • Request Handler│  │ • Request Handler│              │ │ │
│  │  │  │ • AgentExecutor  │  │ • AgentExecutor  │  │ • AgentExecutor  │              │ │ │
│  │  │  │ • Validation     │  │ • Validation     │  │ • Validation     │              │ │ │
│  │  │  └─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘              │ │ │
│  │  └────────────┼──────────────────────┼──────────────────────┼──────────────────────┘ │ │
│  │               │                      │                      │                        │ │
│  │               └──────────────────────┴──────────────────────┘                        │ │
│  │                                      │                                               │ │
│  │                            Spawn Subprocess for Execution                            │ │
│  │                                      ▼                                               │ │
│  └──────────────────────────────────────┬───────────────────────────────────────────────┘ │
│                                         │                                                 │
│  ┌──────────────────────────────────────▼───────────────────────────────────────────────┐ │
│  │                          Code Execution Subprocess (Isolated)                        │ │
│  │                           PID: Child, UID: sandbox-user                             │ │
│  │                                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                            Security Layers (Defense in Depth)                    │ │ │
│  │  │                                                                                  │ │ │
│  │  │  Layer 1: AST Validation     ┌──────────────────────────────────────────┐      │ │ │
│  │  │  ├─ Parse code AST           │         User-Submitted Python Code        │      │ │ │
│  │  │  ├─ Check dangerous patterns │                                          │      │ │ │
│  │  │  └─ Whitelist verification   └────────────────┬─────────────────────────┘      │ │ │
│  │  │                                                │                                │ │ │
│  │  │  Layer 2: RestrictedPython   ┌────────────────▼─────────────────────────┐      │ │ │
│  │  │  ├─ Limited builtins         │      RestrictedPython Environment        │      │ │ │
│  │  │  ├─ No eval/exec/compile     │   • Safe builtins only                  │      │ │ │
│  │  │  └─ Module whitelist:        │   • No file/network access              │      │ │ │
│  │  │     • json, math, datetime   │   • Limited introspection               │      │ │ │
│  │  │     • collections, itertools └────────────────┬─────────────────────────┘      │ │ │
│  │  │     • re, string                              │                                │ │ │
│  │  │                                                │                                │ │ │
│  │  │  Layer 3: Process Isolation  ┌────────────────▼─────────────────────────┐      │ │ │
│  │  │  (REQUIRED for MVP)          │         Subprocess Boundaries            │      │ │ │
│  │  │  ├─ Separate process         │   • Memory: 512MB max (rlimit)          │      │ │ │
│  │  │  ├─ Resource limits:         │   • CPU: 30s max execution             │      │ │ │
│  │  │  │  • CPU: 100% single core  │   • No fork/exec permissions           │      │ │ │
│  │  │  │  • Memory: 512MB          │   • Temp directory only               │      │ │ │
│  │  │  │  • Time: 30s              │   • No network access                 │      │ │ │
│  │  │  └─ No network access        └────────────────┬─────────────────────────┘      │ │ │
│  │  │                                                │                                │ │ │
│  │  │  Layer 4: OS Sandboxing      ┌────────────────▼─────────────────────────┐      │ │ │
│  │  │  (Optional - Linux only)     │        seccomp-bpf (if available)        │      │ │ │
│  │  │  ├─ seccomp-bpf filters      │   • Syscall filtering                   │      │ │ │
│  │  │  ├─ Namespace isolation      │   • Block: open, socket, fork          │      │ │ │
│  │  │  └─ Capability dropping      │   • Allow: read, write (pipes only)    │      │ │ │
│  │  │                              └──────────────────────────────────────────┘      │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                           Execution Environment                                  │ │ │
│  │  │                                                                                  │ │ │
│  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐             │ │ │
│  │  │  │   Search APIs     │  │   Local Copies   │  │  Output Capture  │             │ │ │
│  │  │  │                   │  │                  │  │                  │             │ │ │
│  │  │  │ • HybridSearchAPI │  │ • In-memory DB   │  │ • stdout pipe    │             │ │ │
│  │  │  │ • RerankerAPI     │  │ • Cached vectors │  │ • stderr pipe    │             │ │ │
│  │  │  │ • FilterAPI       │  │ • No external    │  │ • return value   │             │ │ │
│  │  │  │ • ResultProcessor │  │   connections    │  │ • exceptions     │             │ │ │
│  │  │  └──────────────────┘  └──────────────────┘  └──────────────────┘             │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          Optional: Docker Container (v1.1/v2)                         │ │
│  │                                                                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │  docker run --rm --memory=512m --cpus=1 --network=none --read-only             │ │ │
│  │  │    --tmpfs /tmp:rw,noexec,nosuid,size=100m                                     │ │ │
│  │  │    --security-opt=no-new-privileges                                            │ │ │
│  │  │    code-execution:latest                                                       │ │ │
│  │  └──────────────────────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Resource Isolation Matrix:**

| Resource | Main Process | Worker Thread | Subprocess | Docker Container |
|----------|-------------|---------------|------------|------------------|
| **CPU** | Unlimited | Shared pool | 1 core max (100%) | 1 core (--cpus=1) |
| **Memory** | System limit | Shared heap | 512MB (rlimit) | 512MB (--memory) |
| **Time** | Unlimited | Request timeout | 30s hard limit | 30s timeout |
| **Network** | Full access | Full access | Blocked (iptables) | None (--network=none) |
| **Filesystem** | Full access | Full access | /tmp only | tmpfs only |
| **Processes** | Can spawn | Can spawn | Cannot fork | Cannot fork |

**Security Layer Effectiveness:**

| Attack Vector | AST Validation | RestrictedPython | Process Isolation | seccomp-bpf | Docker |
|---------------|---------------|------------------|-------------------|-------------|--------|
| **Code Injection** | ✓ Blocks | ✓ Blocks | - | - | - |
| **Import Dangerous Modules** | ✓ Blocks | ✓ Blocks | ✓ No access | ✓ Filtered | ✓ None available |
| **Resource Exhaustion** | - | Partial | ✓ Limits | ✓ Limits | ✓ Hard limits |
| **File System Access** | ✓ Detected | ✓ Blocked | ✓ /tmp only | ✓ Filtered | ✓ tmpfs only |
| **Network Access** | ✓ Detected | ✓ No modules | ✓ Blocked | ✓ Filtered | ✓ None |
| **Process Spawning** | ✓ Detected | ✓ No access | ✓ Cannot fork | ✓ Blocked | ✓ Blocked |
| **Memory Bombs** | Partial | - | ✓ 512MB limit | - | ✓ Kernel enforced |
| **CPU Spinning** | - | - | ✓ 30s timeout | - | ✓ cgroup limits |

**Concurrency Model:**

```
Thread Pool Architecture:
├─ Main Thread: Handles MCP protocol, request intake, response dispatch
├─ Worker Pool: 4-8 threads for request processing
│  ├─ Each worker can handle one request
│  ├─ Subprocess spawning is synchronous per worker
│  └─ Multiple subprocesses can run concurrently (different workers)
└─ Subprocess: One per code execution request
   ├─ Isolated from parent and siblings
   ├─ Resource limits enforced by OS
   └─ Killed after timeout or completion
```

**Implementation Notes for Deployment:**

- **Essential Component**: Subprocess isolation (Layer 3) is REQUIRED for MVP security
- **Thread Pool Sizing**: Start with 4-8 workers, tune based on production load (0.33 req/s sustainable)
- **Docker Integration**: Optional in v1, defer to v1.1 unless required for compliance
- **Security Layers**: All 4 layers provide defense-in-depth, each catches different attack vectors
- **Monitoring Points**: Each component boundary is a metrics collection point for observability

## Data Models

```python
@dataclass
class SearchResult:
    document_id: str
    title: str
    content: str
    relevance_score: float  # 0.0-1.0 normalized
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: int
    resources_used: Dict[str, Any]  # memory_mb, cpu_percent, etc.

@dataclass
class ValidationResult:
    is_safe: bool
    blocked_operations: List[str]
    risk_level: str  # "safe", "medium", "high"
```

## Technology Stack

**Language**: Python 3.11+
- **Rationale**: Rich sandboxing ecosystem, excellent MCP SDK support, async/await for concurrent execution
- **Trade-offs**: Subprocess overhead (+15-30ms), more memory overhead than compiled alternatives
- **Alternatives considered**: Node.js (weaker sandboxing), Go (compilation overhead for dynamic code)

**Sandboxing Approach**: Subprocess-based execution with OS-level process isolation (REQUIRED in v1)
- **Rationale**: Guaranteed timeout via SIGKILL/TerminateProcess, cross-platform (Windows/Linux/macOS), simpler than threading-based approaches, no GIL interference
- **Trade-offs**: +15-30ms subprocess overhead per execution (acceptable for 30s budget), subprocess spawn cost
- **Alternatives considered**: Threading with signal.alarm (GIL interference, unreliable timeouts, catchable signals), RestrictedPython (insufficient for timeout enforcement), Docker only (100-300ms overhead)

**Security Model**: Whitelist-based (explicit allowed imports vs blacklist)
- **Rationale**: Secure by default, easier to audit, impossible to enumerate all dangerous operations
- **Trade-offs**: More restrictive, requires careful API design
- **Alternatives considered**: Blacklist (incomplete, easy to bypass)

**MCP Integration**: Official Python MCP SDK
- **Rationale**: First-party protocol support, guaranteed compliance, active maintenance
- **Trade-offs**: Coupled to Anthropic release cycle
- **Alternatives considered**: Custom protocol implementation (high maintenance)

## Design Decisions

**Decision: Token-First Result Processing**
- **Rationale**: Fundamental goal of this feature is 98% token reduction
- **Implementation**: ResultProcessor truncates content, selects essential fields, uses compact JSON
- **Trade-offs**: Users receive truncated results (design for agent use, not human readability)

**Decision: Thread-Pool with Subprocess Isolation**
- **Rationale**: MCP servers must handle concurrent clients without blocking. Thread pool enables parallel execution while subprocess isolation ensures security.
- **Implementation**:
  - Async MCP server (event loop for I/O multiplexing)
  - ThreadPoolExecutor (10 workers, bounded queue depth 50)
  - Subprocess execution (OS-level isolation, SIGKILL timeout)
- **Concurrency Model**:
  - Request handling: Async (non-blocking)
  - Code execution: Subprocess (isolated, one per thread)
  - Thread pool: Bounded to prevent resource exhaustion
- **Load Shedding**: 429 Too Many Requests when queue full, exponential backoff
- **Throughput**: 0.33 requests/second sustained (20/minute, 1,200/hour)
- **Trade-offs**:
  - Pro: 10x throughput vs single-threaded, fair scheduling, production-ready
  - Pro: Simpler than async subprocess management
  - Con: 10-20ms subprocess spawn overhead (acceptable)
- **Security**: Threads manage subprocesses (not execute code), OS-level isolation preserved
- **Alternatives Considered**:
  - Single-threaded: Blocks all clients during execution (unacceptable UX)
  - Async with asyncio.create_subprocess_exec: More complex, similar performance
  - Process pool: Equivalent to thread pool for this use case

**Decision: Explicit Allowlist for Modules**
- **Rationale**: Impossible to enumerate all dangerous patterns, whitelist is safer
- **Trade-offs**: More restrictive for users
- **Alternatives**: Blacklist (incomplete), full trust (unacceptable risk)

## Concurrency Architecture

**Request Flow**:
1. MCP async server receives `execute_code` requests from multiple clients
2. Async handler validates input (AST analysis, <100ms, non-blocking)
3. Submits execution to ThreadPoolExecutor via `run_in_executor()`
4. Worker thread spawns subprocess for code execution
5. Thread blocks waiting for subprocess (up to 30s timeout)
6. Other worker threads continue processing other requests in parallel
7. Subprocess completes → thread returns result → async handler returns to client

**Layers**:
- **Layer 1 (MCP Server)**: Async event loop, handles all client I/O
- **Layer 2 (Thread Pool)**: 10 worker threads, bounded queue (50 depth)
- **Layer 3 (Subprocess)**: OS-isolated Python interpreter per execution

**Capacity Management**:
- Max concurrent executions: 10 (ThreadPoolExecutor workers)
- Max queue depth: 50 pending requests
- Rejection policy: HTTP 429 when queue full, retry-after guidance
- Sustainable throughput: 0.33 requests/second (1,200/hour)

**Security Isolation**:
- Threads do NOT execute user code directly
- Threads manage subprocess lifecycle (spawn, wait, kill)
- Subprocesses isolated by OS (memory, file descriptors, network)
- No shared state between concurrent executions

## Security Architecture

### Defense-in-Depth Security Model

Code Execution with MCP implements a **multi-layer security architecture** with 8 independent defense layers. No single layer is sufficient; security relies on defense-in-depth.

**Security Layers** (in order of execution):
1. **Input Validation**: AST analysis blocks dangerous patterns (eval, exec, forbidden imports)
2. **Subprocess Isolation**: OS-level process boundary with independent resource limits (MANDATORY)
3. **System Call Filtering**: Kernel-enforced syscall restrictions (seccomp-bpf on Linux)
4. **Resource Constraints**: CPU, memory, file descriptor, and disk limits
5. **Network Isolation**: Network namespace (Linux) or firewall rules
6. **Filesystem Isolation**: chroot/temp directory, read-only stdlib access
7. **Output Sanitization**: XSS prevention, size limits, path scrubbing
8. **Audit & Monitoring**: Structured logging, anomaly detection, rate limiting

**Security Posture**:
- ✅ **Suitable for**: Semi-trusted agent code, internal development, research workloads
- ❌ **NOT suitable for**: Adversarial users, public-facing APIs, untrusted code from external sources

**Subprocess Isolation (Mandatory)**:
All code execution occurs in isolated subprocesses, NOT threads. This provides:
- Guaranteed timeout enforcement via SIGKILL
- Memory isolation (separate address space)
- Crash isolation (subprocess crash doesn't affect parent)
- OS-level resource accounting
- **Overhead**: 10-50ms subprocess startup (acceptable for security benefit)

**Platform Support**:
- **Linux**: Full support (seccomp-bpf, network namespaces, cgroups) - **Primary platform**
- **macOS**: Partial support (sandbox-exec, rlimit) - **Secondary platform**
- **Windows**: Limited support (Job Objects, firewall rules) - **Tertiary platform**

See SOLUTION_ISSUE_1_SECURITY_ARCHITECTURE.md for complete 8-layer specification.

## Threat Model and Security Assumptions

### Explicit Security Assumptions (v1)

**1. Semi-Trusted Code Sources**
- Code is generated by Claude agents, not directly by untrusted users
- Agents provide first line of defense (unlikely to generate malicious code)
- Security focuses on accidental misuse and moderate-effort attacks

**2. Internal Deployment Context**
- System operates in controlled environments (corporate networks, research labs)
- External network protections (firewalls, IDS) exist
- Physical access is controlled

**3. Observable Operations**
- All code execution is logged and auditable
- Security team can investigate incidents
- Anomaly detection alerts on suspicious patterns

**4. Acceptable Risk Tolerance**
- System prioritizes usability over cryptographic security guarantees
- Accepts risk of sophisticated attacks (state-level adversaries)
- Relies on detection and incident response for advanced threats

### Explicit Non-Goals

**What This System Does NOT Defend Against**:
- ❌ State-level adversaries with kernel exploits or zero-days
- ❌ Hardware side-channel attacks (Spectre, Meltdown, cache timing)
- ❌ Social engineering targeting operators
- ❌ Persistent compromises (rootkits, firmware attacks)
- ❌ Physical access attacks

**Use Cases This System Is NOT Designed For**:
- ❌ Public-facing code execution API (e.g., online coding platforms)
- ❌ Multi-tenant SaaS with adversarial users
- ❌ Financial transaction processing
- ❌ PII/PHI data processing (without additional controls)
- ❌ Cryptographic operations

### Threat Actors

| Threat Actor | Skill Level | Defended? | Notes |
|--------------|-------------|-----------|-------|
| Curious Agent | Low | ✅ Yes | Input validation + isolation |
| Buggy Code | N/A | ✅ Yes | Resource limits + timeout |
| Moderate Attacker | Medium | ✅ Yes | Multiple layers required to bypass |
| Advanced Persistent Threat | High | ⚠️ Partial | Detection and response, not prevention |
| State-Level Adversary | Very High | ❌ No | Out of scope |

### Security Boundaries

**Trust Boundary 1**: Agent → MCP Server
- Agent is semi-trusted
- Input validation provides sanity checking
- Not designed to defend against malicious agents

**Trust Boundary 2**: MCP Server → Subprocess
- Subprocess is untrusted
- All 8 security layers enforced
- Primary security boundary

**Trust Boundary 3**: Subprocess → Host OS
- Host OS is trusted
- Kernel vulnerabilities out of scope
- OS updates and patching are operational requirements

### Residual Risks

**Accepted Risks** (documented and monitored):
1. **RestrictedPython Bypasses**: Known history of bypasses, mitigated by subprocess isolation
2. **Kernel Exploits**: Out of scope, rely on OS security updates
3. **Timing Attacks**: Execution time variation may leak information (accepted for v1)
4. **Resource Competition**: Concurrent executions may compete for resources (rate limiting mitigates)

**Risk Mitigation Strategy**: Defense-in-depth ensures that exploitation of one vulnerability doesn't compromise the entire system.

</architecture>

---

<risks>

## Technical Risks

**Risk: Timeout Mechanism Failure**
- **Status**: RESOLVED via subprocess-based execution in v1
- **Implementation**: subprocess.Popen with proc.kill() (SIGKILL on Linux/macOS, TerminateProcess on Windows)
- **Reliability**: 100% (OS-level process termination, uncatchable by user code)
- **Why Threading Failed**: GIL interference makes threading.Timer unreliable for CPU-bound infinite loops, signal.alarm catchable by user code
- **Platform Support**: Cross-platform (Windows, Linux, macOS) via subprocess stdlib
- **Overhead**: +15-30ms subprocess spawn overhead (acceptable for 30s execution budget)

**Risk: Memory Exhaustion**
- **Impact**: High (OOM kills server process, affects all users)
- **Likelihood**: Medium (large allocations in user code or memory leaks)
- **Mitigation**:
  - Set `resource.setrlimit()` memory caps
  - Monitor with tracemalloc during execution
  - Reject code with obvious memory bombs (very large literals)
- **Fallback**: Docker-based execution with `--memory` flag for hard kernel limits

**Risk: Security Validation Bypass**
- **Impact**: Critical (arbitrary code execution, data exfiltration)
- **Likelihood**: Low (conservative whitelist approach)
- **Mitigation**:
  - Regular security audits of allowed builtins
  - Community review of security model
  - Penetration testing by external firm
  - Rapid patching for any discovered bypasses
- **Fallback**: Disable feature immediately if bypass confirmed, patch and re-enable

**Risk: Token Usage Not Improving**
- **Impact**: Medium (feature doesn't deliver promised value)
- **Likelihood**: Medium (compaction logic may be insufficient)
- **Mitigation**:
  - A/B testing with real queries baseline vs code execution
  - Metrics tracking token reduction % in production
  - User feedback loop during beta
  - Iterative refinement of ResultProcessor
- **Fallback**: Add user controls for verbosity level, provide raw output option

**Risk: Performance Overhead**
- **Impact**: Medium (feature adoption blocked by latency)
- **Likelihood**: Low (sandboxing overhead typically <50ms)
- **Mitigation**:
  - Benchmark critical paths with realistic workloads
  - Lazy initialization of sandbox environment
  - Performance budgets in tests (<100ms overhead)
- **Fallback**: Make code execution opt-in, provide bypass for trusted environments

## Dependency Risks

**Risk: MCP SDK Breaking Changes**
- **Impact**: High (feature breaks with new SDK versions)
- **Likelihood**: Medium (SDK in active development)
- **Mitigation**:
  - Pin SDK version with tested compatibility range
  - Subscribe to SDK release notes
  - Integration tests covering SDK interfaces
- **Fallback**: Fork SDK or implement minimal protocol layer

**Risk: RestrictedPython Maintenance**
- **Impact**: Medium (security patches delayed, Python incompatibility)
- **Likelihood**: Low (mature project, but small team)
- **Mitigation**:
  - Monitor project activity and security advisories
  - Maintain fork capability
  - Fallback to manual AST filtering
- **Fallback**: Switch to custom sandbox using `ast` module

**Risk: Docker Unavailability**
- **Impact**: Low (optional advanced feature degrades gracefully)
- **Likelihood**: High (many users won't have Docker)
- **Mitigation**:
  - Make Docker fully optional with clear documentation
  - Detect Docker availability at runtime
  - Provide helpful error messages with setup links
- **Fallback**: Threading-based sandbox always available as baseline

## Scope Risks

**Risk: Scope Creep to Full REPL**
- **Impact**: High (timeline delays, complexity explosion)
- **Likelihood**: High (natural feature expansion pressure)
- **Mitigation**:
  - Strict v1 scope: read-only operations, no persistent state
  - Defer REPL features, package installation to v2+
  - Clear documentation of intentional limitations
  - Timebox v1 implementation to 2-3 weeks
- **Fallback**: Cut advanced features, ship minimal viable sandbox

**Risk: Underestimation of Security Hardening**
- **Impact**: High (delays release or ships with vulnerabilities)
- **Likelihood**: Medium (security is hard, edge cases numerous)
- **Mitigation**:
  - Allocate 30-40% of timeline to security review
  - Engage security specialists early
  - Comprehensive attack scenario test suite
  - Limited beta before general release
- **Fallback**: Launch with explicit "beta" label, gather field data

**Risk: Integration Complexity**
- **Impact**: Medium (refactoring existing code, regression risks)
- **Likelihood**: Medium (depends on current server architecture coupling)
- **Mitigation**:
  - Design as isolated module with clear interfaces
  - Feature flag for gradual rollout
  - Comprehensive regression test suite
  - Incremental integration with rollback points
- **Fallback**: Ship as separate MCP server initially, merge later

**Risk: Documentation Insufficiency**
- **Impact**: Medium (low adoption, high support burden)
- **Likelihood**: High (common failure mode)
- **Mitigation**:
  - Allocate 20% of timeline to docs and examples
  - Include 10+ realistic use case examples
  - Video walkthrough for setup and patterns
  - FAQ from beta user feedback
- **Fallback**: Comprehensive README-only approach


## Risk Priority Matrix

### 🔴 Critical Priority (Immediate Action Required)
| Risk | Impact | Likelihood | Timeline | Mitigation Status |
|------|--------|------------|----------|-------------------|
| **Security Validation Bypass** | 🔴 Critical | 🟡 Low | Blocks Launch | ⏳ Testing Phase |

### 🟠 High Priority (Address in Current Phase)
| Risk | Impact | Likelihood | Timeline | Mitigation Status |
|------|--------|------------|----------|-------------------|
| **Timeout Mechanism Failure** | 🟠 High | 🟡 Medium | Delays Testing | ✅ Multi-layer design |
| **Memory Exhaustion** | 🟠 High | 🟡 Medium | Delays Phase 2 | ⏳ Resource limits planned |
| **MCP SDK Breaking Changes** | 🟠 High | 🟡 Medium | Blocks Phase 3 | ✅ Version pinning |
| **Scope Creep to Full REPL** | 🟠 High | 🔴 High | Delays All Phases | ✅ Strict v1 scope |
| **Security Hardening Underestimation** | 🟠 High | 🟡 Medium | Delays Launch | ⏳ 40% timeline allocated |

### 🟡 Medium Priority (Monitor & Prepare)
| Risk | Impact | Likelihood | Timeline | Mitigation Status |
|------|--------|------------|----------|-------------------|
| **Token Usage Not Improving** | 🟡 Medium | 🟡 Medium | Post-Launch | ⏳ A/B testing planned |
| **Performance Overhead** | 🟡 Medium | 🟢 Low | Affects Adoption | ✅ Benchmarks in place |
| **RestrictedPython Maintenance** | 🟡 Medium | 🟢 Low | Long-term | ⏳ Fork capability ready |
| **Integration Complexity** | 🟡 Medium | 🟡 Medium | Phase 3 | ✅ Isolated module design |
| **Documentation Insufficiency** | 🟡 Medium | 🔴 High | Affects Adoption | ⏳ 20% timeline allocated |

### 🟢 Low Priority (Acceptable Risk)
| Risk | Impact | Likelihood | Timeline | Mitigation Status |
|------|--------|------------|----------|-------------------|
| **Docker Unavailability** | 🟢 Low | 🔴 High | Optional Feature | ✅ Graceful fallback |

### Legend
- **Impact Levels**: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low
- **Likelihood**: 🔴 High (>70%) | 🟡 Medium (30-70%) | 🟢 Low (<30%)
- **Mitigation Status**: ✅ Complete | ⏳ In Progress | ❌ Not Started

### Risk Response Strategy
1. **Critical Risks**: Daily monitoring, immediate escalation, stop-work triggers
2. **High Risks**: Weekly review, active mitigation, fallback plans ready
3. **Medium Risks**: Bi-weekly review, monitoring metrics, preparation phase
4. **Low Risks**: Monthly review, accept and monitor

</risks>

---

<appendix>

## References

### Official Documentation
- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Claude API Documentation](https://docs.claude.com/)

### Research & Best Practices
- [Python Security Best Practices](https://owasp.org/www-community/attacks/Code_Injection)
- [Sandbox Design Patterns](https://cheatsheetseries.owasp.org/cheatsheets/Sandbox_Bypass_Cheat_Sheet.html)
- RestrictedPython Documentation
- Pydantic Security Validation

### Related Systems
- [Pydantic Python Sandbox MCP](https://github.com/pydantic/pydantic-mcp-server)
- [Code Sandbox MCP Server](https://github.com/Automata-Labs-team/code-sandbox-mcp)
- [E2B Sandbox](https://e2b.dev/)

## Glossary

**Code Execution**: Running Python code dynamically within a controlled sandbox environment
**Sandbox**: Isolated execution environment with resource limits and security restrictions
**Token**: Unit of text processed by Claude models; approximately 4 characters
**Whitelist**: Explicit list of allowed operations (secure by default)
**MCP**: Model Context Protocol - standardized interface for AI tool integration
**Hybrid Search**: Combination of keyword (BM25) and semantic (vector) search
**Reranking**: Re-scoring and reordering search results using more sophisticated models

## Open Questions

1. **Docker Support**: Should v1 include Docker as a sandbox backend option, or defer to v2?
   - Decision: Defer Docker to v2, focus on subprocess-based baseline for v1 (provides sufficient isolation)

2. **Result Caching**: Should we cache frequent search queries or execution results?
   - Decision: No caching in v1 to avoid stale results, implement cache-busting strategy if added

3. **State Persistence**: Should executed code be able to save state between invocations?
   - Decision: No persistent state in v1 (security risk), each execution starts clean

4. **Package Installation**: Should users be able to install packages (pip install) dynamically?
   - Decision: No package installation in v1 (security risk), whitelist of pre-installed packages only

5. **Async Support**: Should we support async/await in user code?
   - Decision: Yes, use asyncio.run() to execute async code, makes APIs more flexible

## Implementation Notes for Developers

### Phase 0 - Foundation
- Create branch `work/mcp-code-execution-phase-0`
- Coordinate with existing BMCIS Knowledge MCP team
- Review existing code organization and patterns

### Phase 1 - Search APIs
- Integrate with existing search infrastructure (BM25, vector search)
- Ensure compatibility with current database layer
- Performance test against production data

### Phase 2 - Sandbox
- Implement subprocess-based execution (REQUIRED in v1)
- Plan for optional Docker upgrade in v1.1 or v2 for enhanced filesystem isolation
- Security audit early in phase

### Phase 3 - MCP Integration
- Register new tools in existing MCP server
- Update system prompts to inform agents about code execution
- Beta test with friendly agent implementations

## Success Criteria for MVP

✅ Token reduction: 90-95% for test workflows (150K → 7.5K-15K)
✅ Token budget compliance: >95% (actual ≤ predicted + 10%)
✅ Latency improvement: 3-4x faster execution (1,200ms → 300-500ms)
✅ Cost reduction: 90-95% cheaper per query ($0.45 → $0.02-$0.05)
✅ Security: Zero isolation breaches in penetration testing
✅ Reliability: >99.9% uptime, <2% error rate
✅ Adoption: >50% of agents with 10+ searches adopt within 90 days
✅ Documentation: Complete API docs, 10+ examples, progressive disclosure guide, setup guide
✅ Validation: A/B test confirms ≥85% token reduction with statistical significance (p<0.01)

</appendix>

---

## Task Master Integration

This PRD is formatted for `task-master parse-prd` parsing with explicit dependency graphs and phased breakdown:

1. **Functional decomposition** defines WHAT capabilities the system delivers
2. **Structural decomposition** defines WHERE code lives (module structure)
3. **Dependency graph** defines HOW to sequence development (topological order)
4. **Implementation roadmap** defines TASKS in each phase with acceptance criteria
5. **Test strategy** defines VALIDATION approach at each level
6. **Architecture & risks** document DECISIONS and CONSTRAINTS

Run: `task-master parse-prd docs/mcp-as-tools/PRD_CODE_EXECUTION_WITH_MCP.md --research`

---

**Document Version**: 1.0
**Created**: November 2024
**Status**: Ready for Task Master Parsing
**Audience**: Engineering Team
**Next Step**: Parse with Task Master → Create task graph → Begin Phase 0
