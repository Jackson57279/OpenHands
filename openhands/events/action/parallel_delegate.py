from dataclasses import dataclass, field
from openhands.events.action.action import Action
from openhands.core.schema import ActionType


@dataclass
class ParallelDelegateAction(Action):
    """Action to delegate multiple tasks to parallel agents."""
    agent: str
    tasks: list[dict[str, any]]
    max_concurrent: int = 10
    thought: str = ''
    action: str = ActionType.PARALLEL_DELEGATE