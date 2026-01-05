from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union, Generator

from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import LLMUsage
from lite_agents.agent.memory import AgentMemory
from lite_agents.llm.lite import LiteLLM
from lite_agents.core.response import (
    TextResponse, 
    ToolCall,
    ToolResult, 
    TextResponseDelta, 
    LLMUsage,
    AgentReachedMaxSteps,
)

# AgentEvent (a tagged union)
AgentEvent = Union[
    TextResponseDelta,
    TextResponse,
    ToolCall,
    ToolResult,
    AgentReachedMaxSteps,
]

# Streaming Type
AgentEventStream = Generator[AgentEvent, None, None]

# Response Type
AgentResponse = list[AgentEvent]

class BaseAgent(ABC):
    """Base abstract class for all agents.
    
    Args:
        name (str): the agent name
        description (str): the agent description
        llm (LiteLLM): the LiteLLM instance to use
        system_prompt (str | None, optional): the system prompt to use. Defaults to None.
        memory (AgentMemory | None, optional): the agent memory instance. Defaults to None.
        stream (bool, optional): whether to stream the responses. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        description: str,
        llm: LiteLLM,
        system_prompt: str | None = None,
        memory: AgentMemory | None = None,
        stream: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = memory or AgentMemory()
        self.stream = stream
        self.usage: list[LLMUsage] = []

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Prepare the messages for the LLM.
        
        By default, this adds the system prompt if it exists.
        
        Args:
            messages (list[ChatMessage]): the input messages

        Returns:
            list[ChatMessage]: the prepared messages
        """
        if self.system_prompt:
            messages = [ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)] + messages
            self.memory.add_system_step(messages[0])
        return messages

    @abstractmethod
    def run(self, messages: list[ChatMessage]) -> Any:
        """Run the agent with the given messages.
        
        Args:
            messages (list[ChatMessage]): the input messages
            
        Returns:
            Any: the agent response
        """
        pass