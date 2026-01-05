from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, List

from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import LLMUsage
from lite_agents.agent.memory import AgentMemory
from lite_agents.llm.lite import LiteLLM

class BaseAgent(ABC):
    """Base abstract class for all agents.
    
    Args:
        name (str): the agent name
        description (str): the agent description
        llm (LiteLLM | None, optional): the LiteLLM instance to use. Defaults to None.
        system_prompt (str | None, optional): the system prompt to use. Defaults to None.
        memory (AgentMemory | None, optional): the agent memory instance. Defaults to None.
        stream (bool, optional): whether to stream the responses. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[LiteLLM] = None,
        system_prompt: Optional[str] = None,
        memory: Optional[AgentMemory] = None,
        stream: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = memory or AgentMemory()
        self.stream = stream
        self.usage: List[LLMUsage] = []

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Prepare the messages for the LLM.
        
        By default, this adds the system prompt if it exists.
        
        Args:
            messages (List[ChatMessage]): the input messages

        Returns:
            List[ChatMessage]: the prepared messages
        """
        if self.system_prompt:
            messages = [ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)] + messages
            self.memory.add_system_step(messages[0])
        return messages

    @abstractmethod
    def run(self, messages: List[ChatMessage]) -> Any:
        """Run the agent with the given messages.
        
        Args:
            messages (List[ChatMessage]): the input messages
            
        Returns:
            Any: the agent response
        """
        pass