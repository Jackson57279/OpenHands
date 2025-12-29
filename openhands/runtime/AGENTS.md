# RUNTIME KNOWLEDGE BASE

## OVERVIEW

Sandboxed execution environment for agent actions. Manages lifecycle of isolated containers/environments and handles action execution (Bash, Python, Browser, File Ops).

## STRUCTURE

```
openhands/runtime/
├── impl/      # Runtime implementations (Docker, K8s, Local, Remote)
├── plugins/   # Agent capabilities (AgentSkills, Jupyter, Browser, VSCode)
├── builder/   # Logic for building sandbox environments and images
├── utils/     # Low-level helpers (Bash, Git, Files, Monitoring)
├── base.py    # Primary interface (Runtime class)
└── action_execution_server.py  # REST API running INSIDE the sandbox
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|--------|
| Add new runtime | `impl/` | Inherit from `ActionExecutionClient` or `Runtime` |
| New agent skill/tool | `plugins/agent_skills/` | Part of the default toolbox for agents |
| Fix bash/command exec | `action_execution_server.py` | Core engine inside the sandbox |
| Runtime image/build | `builder/` | Uses templates in `utils/runtime_templates/` |
| File viewing logic | `file_viewer_server.py` | Browser-based file viewing inside sandbox |
| Low-level sandbox ops | `utils/` | Git diffing, file searching, memory monitoring |

## CONVENTIONS

- **Isolated Execution**: All agent actions MUST run inside the runtime (never on host directly).
- **Client-Server Architecture**: The backend communicates with `ActionExecutionServer` (FastAPI) via HTTP even in local Docker.
- **Asyncio-First**: Heavy use of `asyncio` for non-blocking I/O across networked components.
- **Stateful Shell**: Bash environments are stateful; state persists across `CmdRunAction` calls.

## ANTI-PATTERNS

- **Bypassing Sandbox**: Avoid implementing file/shell operations using host-side Python libraries. Use `Runtime.run()` or `Runtime.read()`.
- **Direct Shell Calls**: Don't use `subprocess.run` on the host; always route through the runtime's bash execution engine.
- **Blocking Calls**: Never use synchronous network or file I/O in the main runtime loop; use `httpx.AsyncClient` and `ainit`.
