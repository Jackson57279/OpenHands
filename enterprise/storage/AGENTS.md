# AGENTS.md - enterprise/storage

## OVERVIEW
Unified database storage layer for OpenHands SaaS/Enterprise using SQLAlchemy and PostgreSQL.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| DB Engine/Sessions | `database.py` | Sync/Async engines, pool config, GCP Connector support |
| CRUD Operations | `*_store.py` | Store classes (e.g. `SaasConversationStore`) implementing data logic |
| SQLAlchemy Models | `*.py` (non-store) | Table definitions (e.g. `stored_repository.py`, `api_key.py`) |
| Base Definition | `base.py` | Shared `Base` class for all SQLAlchemy models |
| Integration Stores | `jira_*`, `linear_*`, `slack_*` | Third-party integration-specific metadata storage |

## CONVENTIONS
- **Dual Dialects**: Uses `postgresql+pg8000` for sync and `postgresql+asyncpg` for async.
- **Store Interface**: Store classes implement interfaces from `openhands.storage` to maintain compatibility with core.
- **Sync-in-Async**: Uses `call_sync_from_async` to execute synchronous SQLAlchemy calls within async Store methods.
- **Data Mapping**: `_to_external_model` and `save_*` methods handle conversion between SQLAlchemy models and core dataclasses.
- **User-Centric**: Most queries are strictly filtered by `user_id` or `github_user_id` for multi-tenancy.

## ANTI-PATTERNS
- **Global Session Leak**: Don't use `session_maker` directly in business logic; use a Store instance.
- **Untracked Versions**: Versions (e.g., 'V0', 'V1') are often hardcoded in filters; verify version compatibility when querying.
- **Manual Mapping**: Avoid manual attribute copying; use `dataclasses.asdict()` or helper methods where possible.
