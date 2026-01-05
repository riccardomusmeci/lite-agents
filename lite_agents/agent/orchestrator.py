


from __future__ import annotations
from typing import List, Any
from lite_agents.agent._base import BaseAgent
from lite_agents.core.message import ChatMessage

class Orchestrator(BaseAgent):
    
    def __init__(
        self,
        name: str,
        description: str,
        agents: List[BaseAgent],
        **kwargs
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.agents = agents

    def run(self, messages: List[ChatMessage]) -> Any:
        raise NotImplementedError("Orchestrator run method not implemented yet.")