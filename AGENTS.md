# PROJECT KNOWLEDGE BASE

**Generated:** 2025-12-29 01:43:00 AM
**Commit:** 97654e6
**Branch:** main

## OVERVIEW

Multi-agent AI dev platform. Python backend (Poetry), React frontend (Vite), containerized deployment (Docker). Supports CLI, local GUI, cloud, enterprise.

## STRUCTURE

```
openhands/
├── openhands/        # Python core: agents, runtime, LLM, controller
├── frontend/         # React SPA (Vite): chat UI, agent management
├── enterprise/        # SaaS features: auth, storage, billing
├── evaluation/        # Benchmarks (30+): SWE-Bench, GAIA, etc.
├── tests/            # Python tests: unit, runtime, e2e
├── containers/        # Dockerfiles: app, runtime, enterprise
├── .github/workflows/ # CI/CD: multi-platform builds, 19+ workflows
└── skills/           # Microagent prompts (markdown)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| Add agent | `openhands/agenthub/*/agent.py` | Register in `__init__.py` |
| Fix runtime bug | `openhands/runtime/` | Sandbox exec, file ops, jupyter |
| Frontend feature | `frontend/src/components/features/` | React + TypeScript |
| Add integration | `openhands/integrations/*/service/` | GitHub, GitLab, Bitbucket |
| New benchmark | `evaluation/benchmarks/` | Follow existing pattern |
| Storage layer | `enterprise/storage/` | SQLAlchemy models, PostgreSQL |
| Server routes | `openhands/server/routes/` | FastAPI endpoints |

## CONVENTIONS

**Multi-entry architecture:**
- CLI: `python -m openhands.core.main`
- Server: `python -m openhands.server` (uvicorn, port 3000)
- Frontend: `npm run dev` (Vite, port 3001)
- Docker: `docker run -p 3000:3000`

**Python:**
- Poetry for deps (not pip/setuptools)
- `python -m` module invocation (not setuptools entry_points)
- Pytest with `-p no:warnings`, `asyncio_mode = auto`
- Pre-commit hooks: ruff, mypy, black

**Frontend:**
- Vite (not webpack/rollup)
- Vitest (not Jest)
- MSW for API mocking
- Testing Library for component testing

**Container-first:**
- All components run in Docker
- Multi-platform builds (AMD64/ARM64)
- GHCR for image registry

**Anti-patterns (project-specific):**
- "NEVER ASK FOR HUMAN HELP" in evaluation scripts
- 131 TODO/FIXME markers (technical debt)
- Deprecated V0 types in frontend (V1 migration in progress)

## ANTI-PATTERNS (THIS PROJECT)

- **Hardcoded browsing restrictions**: Evaluation scripts disable web browsing ("SHOULD NEVER attempt to browse") - limits agent autonomy
- **Legacy code**: Frontend has deprecated V0 types, TODOs for V1 migration
- **Technical debt**: 131 TODO/FIXME markers across 85 files
- **Production safety**: Logger.py warns "NEVER be enabled in production"
- **Over-restrictive prompts**: "ALWAYS" and "DO NOT" directives reduce agent flexibility

## UNIQUE STYLES

**Agent delegation**: Multi-agent system with delegate levels, shared iteration counters
**Memory condensation**: Long conversations condensed to manage LLM context
**MCP tools**: External tool integration via Model Context Protocol
**Microagents**: Prompt-based enhancements (markdown files), not code
**Runtime plugins**: Jupyter, AgentSkills, Browser as optional plugins

## COMMANDS

```bash
# Development
make lint           # Pre-commit hooks (ruff, mypy)
make test            # Python tests
make test-frontend   # Frontend tests
make build           # Docker images

# CLI
python -m openhands.core.main -t "task"  # Direct agent execution

# Server
python -m openhands.server              # Start web API

# Frontend
cd frontend && npm run dev            # Vite dev server
cd frontend && npm run build && npm start  # Production

# Docker
docker run -p 3000:3000 ghcr.io/open-hands/openhands:latest
```

## NOTES

**Multi-component design**: Separate entry points for different use cases (CLI agent, web server, frontend)
**Docker migration**: Many TODOs indicate ongoing containerization work
**V0 → V1**: Frontend migrating from legacy types, @deprecated markers in TypeScript
**Enterprise**: Source-available (visible), license required for production use
**CI/CD**: Production-grade, 19+ workflows, automated testing, multi-platform builds
**Benchmarks**: 30+ evaluation frameworks, extensive testing infrastructure
