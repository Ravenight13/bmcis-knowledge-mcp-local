#!/usr/bin/env python3
"""
Script to apply solution integrations to PRD_CODE_EXECUTION_WITH_MCP.md
Integrates SOLUTION_ISSUE_1_SECURITY_ARCHITECTURE.md and SOLUTION_ISSUE_2_TOKEN_METRICS.md
"""

import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def apply_issue_2_executive_summary(content):
    """Update Executive Summary with revised token metrics"""
    old = """This PRD defines the implementation of **Code Execution with MCP**, a transformative capability that enables Claude agents to execute Python code within a secure sandbox environment, reducing token overhead by 98% while dramatically improving performance for complex search and filtering workflows in BMCIS Knowledge MCP.

**Key Metrics:**
- **Token Reduction**: 150,000 → 2,000 tokens (98.7% reduction)
- **Latency Improvement**: 8-12s → 2-3s (4x faster)
- **Cost Savings**: $0.45 → $0.01 per complex query (98% cheaper)
- **Target Adoption**: >80% of qualifying agents within 30 days"""

    new = """This PRD defines the implementation of **Code Execution with MCP**, a transformative capability that enables Claude agents to execute Python code within a secure sandbox environment, reducing token overhead by **90-95%** while dramatically improving performance for complex search and filtering workflows in BMCIS Knowledge MCP.

**Key Metrics:**
- **Token Reduction**: 150,000 → 7,500-15,000 tokens (90-95% reduction)
- **Latency Improvement**: 1,200ms → 300-500ms (3-4x faster)
- **Cost Savings**: $0.45 → $0.02-$0.05 per complex query (90-95% cheaper)
- **Target Adoption**: >50% of qualifying agents within 90 days, >80% within 180 days

**Value Proposition**: Code execution enables **progressive disclosure** patterns where agents request compact metadata first, then selectively fetch full content only for relevant results. This achieves industry-leading token efficiency (90-95% reduction) while preserving accuracy through on-demand detail retrieval."""

    return content.replace(old, new)

def apply_issue_2_success_metrics(content):
    """Update Success Metrics section with revised targets"""
    old = """## Success Metrics

**Token Efficiency**:
- **Primary**: 98%+ token reduction for multi-step search workflows (150,000 → 2,000 tokens)
- **Secondary**: Average tokens per search operation <5,000 (vs. 37,500 baseline)

**Performance**:
- **Primary**: 3x latency improvement for 4-step workflows (1,200ms → 400ms)
- **Secondary**: 95th percentile execution latency <500ms (includes subprocess overhead)

**Cost Reduction**:
- **Primary**: 98%+ cost savings per complex search workflow ($0.45 → $0.01)
- **Secondary**: Agent session cost <$0.10 for 50-search sessions

**Adoption**:
- **Primary**: >80% of agents with 10+ search operations adopt code execution within 30 days
- **Secondary**: >60% of integration engineers choose code execution for new implementations

**Reliability**:
- **Primary**: Code execution sandbox reliability >99.9% (no crashes, memory leaks, or security violations)
- **Secondary**: Error rate <2% for valid Python code submissions"""

    new = """## Success Metrics

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
- Token budget: 10,000-15,000 tokens/result = 30,000-45,000 tokens for 3 results"""

    return content.replace(old, new)

def apply_issue_2_problem_statement(content):
    """Add progressive disclosure opportunity to Problem Statement"""
    # Find line 34 (after the concrete example)
    pattern = r"(Concrete example: An agent researching.*takes 1\.2 seconds\.)"

    addition = r"""\1

**Progressive Disclosure Opportunity**: Traditional tool calling forces agents to receive full content for all results upfront (150K tokens). Code execution enables a progressive disclosure pattern:
1. Initial search returns metadata + signatures (2K tokens)
2. Agent analyzes and identifies relevant subset
3. Selective full-content requests for top 2-3 results (12K tokens)
4. Total: 14K tokens (91% reduction) with same accuracy

This pattern aligns with human research behavior: skim many results, deep-dive on few."""

    return re.sub(pattern, addition, content)

def insert_security_architecture_after_architecture(content):
    """Insert Security Architecture section after existing architecture section"""

    # Find the end of architecture section (before risks section)
    marker = "</architecture>\n\n---\n\n<risks>"

    security_section = """
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

<risks>"""

    return content.replace(marker, security_section)

def update_success_criteria_appendix(content):
    """Update Success Criteria in appendix"""
    old = """✅ Token reduction: 98%+ for test workflows (150K → 2K)
✅ Latency improvement: 4x faster execution (1.2s → 300ms)
✅ Cost reduction: 98% cheaper per query ($0.45 → $0.01)
✅ Security: Zero isolation breaches in penetration testing
✅ Reliability: >99.9% uptime, <1% error rate
✅ Adoption: >80% of agents with 10+ searches adopt feature
✅ Documentation: Complete API docs, 10+ examples, setup guide"""

    new = """✅ Token reduction: 90-95% for test workflows (150K → 7.5K-15K)
✅ Token budget compliance: >95% (actual ≤ predicted + 10%)
✅ Latency improvement: 3-4x faster execution (1,200ms → 300-500ms)
✅ Cost reduction: 90-95% cheaper per query ($0.45 → $0.02-$0.05)
✅ Security: Zero isolation breaches in penetration testing
✅ Reliability: >99.9% uptime, <2% error rate
✅ Adoption: >50% of agents with 10+ searches adopt within 90 days
✅ Documentation: Complete API docs, 10+ examples, progressive disclosure guide, setup guide
✅ Validation: A/B test confirms ≥85% token reduction with statistical significance (p<0.01)"""

    return content.replace(old, new)

def main():
    prd_path = "docs/mcp-as-tools/PRD_CODE_EXECUTION_WITH_MCP.md"

    print("Reading PRD file...")
    content = read_file(prd_path)

    print("Applying Issue 2 - Executive Summary updates...")
    content = apply_issue_2_executive_summary(content)

    print("Applying Issue 2 - Success Metrics updates...")
    content = apply_issue_2_success_metrics(content)

    print("Applying Issue 2 - Problem Statement additions...")
    content = apply_issue_2_problem_statement(content)

    print("Applying Issue 1 - Security Architecture section...")
    content = insert_security_architecture_after_architecture(content)

    print("Applying Issue 2 - Success Criteria appendix updates...")
    content = update_success_criteria_appendix(content)

    print("Writing updated PRD...")
    write_file(prd_path, content)

    print("Integration complete!")

if __name__ == "__main__":
    main()
