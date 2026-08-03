from __future__ import annotations

from log_agent.application.executor import CommandExecutor
from log_agent.domain.models import Investigation, Phase
from log_agent.domain.state_machine import StartRequested, transition


class InvestigationRunner:
    """Drive a new investigation until its state machine reaches a terminal phase."""

    def __init__(self, executor: CommandExecutor) -> None:
        self._executor = executor

    async def run(self, initial: Investigation) -> Investigation:
        if initial.phase is not Phase.NEW:
            raise ValueError("runner requires a new investigation")

        step = transition(initial, StartRequested())
        while step.commands:
            if len(step.commands) != 1:
                raise RuntimeError("the current runner supports exactly one command per step")
            command = step.commands[0]
            event = await self._executor.execute(step.state, command)
            step = transition(step.state, event)
        return step.state
