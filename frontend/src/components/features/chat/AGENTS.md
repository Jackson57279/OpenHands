# CHAT FEATURE KNOWLEDGE BASE

## OVERVIEW
Main chat UI for agent-user communication, event stream rendering, and task management.

## WHERE TO LOOK
- `ChatInterface`: Orchestrates chat layout, message history, and input.
- `InteractiveChatBox`: Handles user input, multi-modal file uploads, and stop/play controls.
- `EventMessage`: Core component for rendering backend Actions and Observations.
- `event-content-helpers/`: Logic for translating raw events into user-friendly UI content.
- `PlanPreview`: UI for visualizing and monitoring the agent's multi-step plan.
- `TaskTracking`: Components for displaying task lists and status badges.
- `GitControlBar`: Integration UI for repository operations (push, PR, branch).
- `microagent/`: Microagent interaction UI, including status indicators and modals.

## ANTI-PATTERNS
- **Direct Event Rendering**: NEVER render raw events directly; use `getEventContent` helper.
- **V0/V1 Mix**: Avoid mixing legacy `V0Messages` and new `V1Messages` without clear transition.
- **In-component Parsing**: Keep event-to-string parsing logic in `event-content-helpers`.
- **Hardcoded Strings**: Always use `i18n` with `ACTION_MESSAGE$` or `OBSERVATION_MESSAGE$` prefixes.
