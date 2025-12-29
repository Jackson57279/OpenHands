# INTEGRATIONS KNOWLEDGE BASE

## OVERVIEW

Multi-service bridge for Git providers (GitHub, GitLab, Bitbucket, Azure DevOps, Forgejo) and productivity tools (Jira, Slack, VS Code).

## STRUCTURE

```
openhands/integrations/
├── github/, gitlab/, ... # Provider-specific implementations
│   ├── service/           # Logic for repos, PRs, branches, resolver
│   └── *_service.py       # Main service implementation entry point
├── templates/             # Jinja2 templates for prompts and instructions
│   ├── resolver/          # Issue/PR resolution prompts by provider
│   └── suggested_task/    # Failing checks, merge conflicts, etc.
├── protocols/             # Shared communication (HTTP client)
├── vscode/                # VS Code extension source
├── provider.py            # ProviderHandler (central service factory)
└── service_types.py       # Common interfaces (GitService, Repository, etc.)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| Add new Git provider | `openhands/integrations/` | Implement `GitService`, register in `provider.py` |
| Modify PR/Issue logic | `*/service/prs.py` | Provider-specific API handling |
| Update Agent prompts | `templates/resolver/` | Jinja2 templates for issue instructions |
| Add suggested task | `templates/suggested_task/` | Logic in `service_types.py` |
| Fix auth/token issues | `provider.py` | Token refreshing and environment masking |
| VS Code Extension | `vscode/src/` | TypeScript extension code |

## ANTI-PATTERNS

- **Hardcoded URLs**: Use `base_domain` from `ProviderToken` for self-hosted instances (GHE, GitLab Self-managed)
- **Bypassing GitService**: Use `ProviderHandler.get_service()` instead of direct API calls
- **Mixed Logic**: Keep API-specific parsing in `service/` modules, not in the main agent loop
- **Inconsistent Types**: Always use models from `service_types.py` for repository and PR data
