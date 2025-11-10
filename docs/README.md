# Documentation Index

Welcome to the bmcis-knowledge-mcp-local documentation. This directory contains all project documentation organized by category.

## Directory Structure

### 📚 [guides/](./guides/)
Comprehensive guides for development, workflow, and tooling.
- Development setup and best practices
- Claude Code integration
- Git branch masking strategy

### 🏗️ [architecture/](./architecture/)
System architecture, data models, and technical design.
- Data model specifications
- System architecture diagrams
- Data flow documentation

### 📖 [reference/](./reference/)
Quick reference materials and lookup guides.
- Analysis indexes
- Configuration references
- Algorithm guides

### 🔄 [task-refinements/](./task-refinements/)
Task refinement plans and implementation summaries.
- Master refinement plans
- Task-specific refinements
- Execution checklists

### 📜 [project-history/](./project-history/)
Historical documentation and completion reports.
- Milestone completion reports
- Team retrospectives
- Project evolution

### 📋 [planning/](./planning/)
Future planning, roadmaps, and proposals.
- Feature proposals
- Enhancement roadmaps
- Architecture decision records

### 🔍 [subagent-reports/](./subagent-reports/)
AI-generated analysis reports organized by analysis type.
- API analysis
- Architecture reviews
- Code quality reports
- Performance analysis
- Security analysis
- Testing reports

### 📝 [refinement-plans/](./refinement-plans/)
Detailed task implementation plans.
- Task 1-5 implementation plans
- Verification guides
- Master implementation guides

### 🔌 [mcp-as-tools/](./mcp-as-tools/)
MCP (Model Context Protocol) server integration documentation.
- MCP server setup
- Code execution guides
- Integration patterns

## Quick Start Paths

### For New Contributors
1. Start: [guides/DEVELOPMENT.md](./guides/DEVELOPMENT.md)
2. Review: [architecture/DATA_MODELS_QUICK_REFERENCE.md](./architecture/DATA_MODELS_QUICK_REFERENCE.md)
3. Understand: [guides/CLAUDE.md](./guides/CLAUDE.md)

### For System Understanding
1. Architecture: [architecture/DATA_FLOW_DIAGRAM.md](./architecture/DATA_FLOW_DIAGRAM.md)
2. Data Models: [architecture/DATA_MODELS_ANALYSIS.md](./architecture/DATA_MODELS_ANALYSIS.md)
3. Implementation: [task-refinements/REFINEMENTS_MASTER_PLAN.md](./task-refinements/REFINEMENTS_MASTER_PLAN.md)

### For Task Implementation
1. Index: [task-refinements/REFINEMENT-PLANS-INDEX.md](./task-refinements/REFINEMENT-PLANS-INDEX.md)
2. Plans: [refinement-plans/](./refinement-plans/)
3. Verification: [refinement-plans/VERIFICATION.md](./refinement-plans/VERIFICATION.md)

## Additional Resources

### Performance & Optimization
- [performance_optimization_roadmap.md](./performance_optimization_roadmap.md)
- [performance_quick_reference.md](./performance_quick_reference.md)
- [search_performance_optimization.md](./search_performance_optimization.md)

### Search Module
- [search-config-reference.md](./search-config-reference.md)
- [rrf-algorithm-guide.md](./rrf-algorithm-guide.md)
- [boost-strategies-guide.md](./boost-strategies-guide.md)

### Task-Specific Documentation
- [task-5-4-implementation-summary.md](./task-5-4-implementation-summary.md)
- [task-5-refinement-orchestration.md](./task-5-refinement-orchestration.md)

## Navigation Tips

- Each subdirectory has its own README.md with detailed contents
- Use the reference/ directory for quick lookups
- Check subagent-reports/ for detailed analysis on specific components
- Review project-history/ to understand decision rationale

## Contributing

When adding new documentation:
1. Place files in the appropriate subdirectory
2. Update the subdirectory's README.md
3. Add entry to this index if it's a major document
4. Use descriptive filenames with dates when applicable

## Project Root Documentation

Some key files remain in the project root for tooling requirements:
- **CLAUDE.md** - Auto-loaded by Claude Code (also in guides/)
- **.taskmaster/** - Task Master AI configuration
- **.claude/** - Claude Code configuration
- **session-handoffs/** - Session-by-session work logs
