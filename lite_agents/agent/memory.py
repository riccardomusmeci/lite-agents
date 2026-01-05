from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import LLMUsage, TextResponse
from lite_agents.llm import LiteLLM
from lite_agents.prompts.memory import SUMMARIZE_MEMORY_PROMPT
from lite_agents.logger import setup_logger
from dataclasses import dataclass, field
from typing import Any
import json

logger = setup_logger()

@dataclass
class AnswerStep:
    response: ChatMessage
    usage: LLMUsage
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response.to_dict(),
            "usage": self.usage.to_dict(),
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AnswerStep":
        return AnswerStep(
            response=ChatMessage.from_dict(data["response"]),
            usage=LLMUsage.from_dict(data["usage"]),
        )

@dataclass  
class HumanStep:
    message: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HumanStep":
        return HumanStep(
            message=data["message"]
        )

@dataclass
class RetryStep:
    reason: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RetryStep":
        return RetryStep(
            reason=data["reason"]
        )

@dataclass
class ToolStep:
    name: str
    kwargs: dict
    result: str
    usage: LLMUsage | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kwargs": self.kwargs,
            "result": self.result,
            "usage": self.usage.to_dict() if self.usage else None,
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ToolStep":
        return ToolStep(
            name=data["name"],
            kwargs=data["kwargs"],
            result=data["result"],
            usage=LLMUsage.from_dict(data["usage"]) if data["usage"] else None,
        )

@dataclass
class SytemStep:
    prompt: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SytemStep":
        return SytemStep(
            prompt=data["prompt"]
        )

@dataclass
class ChiefStep:
    reason: str
    agent: str
    raw_response: str
    expanded_query: str
    usage: LLMUsage | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "agent": self.agent,
            "raw_response": self.raw_response,
            "expanded_query": self.expanded_query,
            "usage": self.usage.to_dict() if self.usage else None,
        }
        
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChiefStep":
        return ChiefStep(
            reason=data["reason"],
            agent=data["agent"],
            raw_response=data["raw_response"],
            expanded_query=data.get("expanded_query", ""),
            usage=LLMUsage.from_dict(data["usage"]) if data.get("usage") else None,
        )

@dataclass
class AgentMemory:
    """Agent memory class
    
    Attributes:
        steps (list[SytemStep | AnswerStep | HumanStep | ToolStep | ChiefStep]): list of interaction steps.
        summary (str): concise summary of the memory. It is filled when summarize() is called.
        usage (LLMUsage | None): usage information of the memory. It is filled when summarize() is called.
    """
    steps: list[SytemStep | AnswerStep | HumanStep | ToolStep | ChiefStep] = field(default_factory=list)
    summary: str = "" 
    usage: LLMUsage | None = None
    
    def add_system_step(self, message: ChatMessage) -> None:
        if message.role != ChatRole.SYSTEM:
            raise ValueError("SystemStep message must have role SYSTEM.")
        self.steps.append(SytemStep(prompt=message.content))
    
    def add_human_step(self, message: ChatMessage) -> None:
        if message.role != ChatRole.USER:
            raise ValueError("HumanStep message must have role USER.")
        self.steps.append(HumanStep(message=message.content))
        
    def add_agent_step(self, response: ChatMessage | TextResponse, usage: LLMUsage) -> None:
        if isinstance(response, TextResponse):
            response = ChatMessage(role=ChatRole.ASSISTANT, content=response.content)
        else:
            if response.role != ChatRole.ASSISTANT:
                raise ValueError("AnswerStep response must have role ASSISTANT.")
        self.steps.append(AnswerStep(response=response, usage=usage))
        
    def add_tool_step(self, response: ChatMessage, usage: LLMUsage | None = None) -> None:
        if response.role != ChatRole.TOOL:
            raise ValueError("ToolStep response must have role TOOL.")
        self.steps.append(
            ToolStep(
                name=response.name or "",
                kwargs=response.tool_kwargs or {},
                result=response.content or "",
                usage=usage
            )
        )
        
    def add_retrieval_step(self, chunks: list[dict[str, Any]]) -> None:
        """Add a retrieval step to the memory.
        
        Args:
            chunks (list[dict[str, str | float]]): list of retrieved chunks with 'content', 'metadata', and 'similarity'.
        """
        self.steps.append(
            ToolStep(
                name="retrieval",
                kwargs={},
                result=chunks,
                usage=None
            )
        )

    def add_chief_step(self, reason: str, agent: str, raw_response: str, expanded_query: str, usage: LLMUsage | None = None) -> None:
        """Add a chief step to the memory.
        
        Args:
            reason (str): the reasoning behind the routing.
            agent (str): the name of the agent routed to.
            raw_response (str): the output received from the agent.
            expanded_query (str): the expanded query used for routing.
            usage (LLMUsage | None, optional): the usage of the chief LLM. Defaults to None.
        """
        self.steps.append(ChiefStep(
            reason=reason,
            agent=agent,
            raw_response=raw_response,
            expanded_query=expanded_query,
            usage=usage
        ))
        
    def add_retry_step(self, message: ChatMessage) -> None:
        """Add a retry step to the memory as a HumanStep.
        
        Args:
            message (ChatMessage): the retry message from the user.
        """
        if message.role != ChatRole.USER:
            raise ValueError("RetryStep message must have role USER.")
        self.steps.append(RetryStep(reason=message.content))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the memory to a dictionary representation."""
        return {
            "steps": [
                {
                    "type": step.__class__.__name__,
                    "data": step.to_dict()
                }
                for step in self.steps
            ]
        }
        
    def to_json(self, filepath: str) -> None:
        """Save the memory to a JSON file.
        
        Args:
            filepath (str): path to the JSON file.
        """
        import json
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AgentMemory":
        """Create an AgentMemory instance from a dictionary representation.
        
        Args:
            data (dict[str, Any]): dictionary representation of the memory.
            
        Returns:
            AgentMemory: the AgentMemory instance.
        """
        memory = AgentMemory()
        
        for step_data in data["steps"]:
            step_type = step_data["type"]
            step_content = step_data["data"]
            
            if step_type == "SytemStep":
                memory.steps.append(SytemStep.from_dict(step_content))
            elif step_type == "AnswerStep":
                memory.steps.append(AnswerStep.from_dict(step_content))
            elif step_type == "HumanStep":
                memory.steps.append(HumanStep.from_dict(step_content))
            elif step_type == "ToolStep":
                memory.steps.append(ToolStep.from_dict(step_content))
            elif step_type == "ChiefStep":
                memory.steps.append(ChiefStep.from_dict(step_content))
            elif step_type == "RetryStep":
                memory.steps.append(RetryStep.from_dict(step_content))
            else:
                raise ValueError(f"Unknown step type: {step_type}")
            
        return memory
            
    @staticmethod
    def load_json(filepath: str) -> "AgentMemory":
        """Load the memory from a JSON file.
        
        Args:
            filepath (str): path to the JSON file.
            
        Returns:
            AgentMemory: the AgentMemory instance.
        """
        import json
        with open(filepath, "r") as f:
            data = json.load(f)
        return AgentMemory.from_dict(data)
    
    def __add__(self, other: "AgentMemory") -> "AgentMemory":
        """Combine two AgentMemory instances.
        
        Args:
            other (AgentMemory): another AgentMemory instance.
            
        Returns:
            AgentMemory: the combined AgentMemory instance.
        """
        combined = AgentMemory()
        combined.steps = self.steps + other.steps
        return combined
    
    def summarize(self, llm: LiteLLM) -> str:
        """Summarize the agent memory.
        
        Args:
            llm (LiteLLM): the LLM instance to use for summarization.
            
        Returns:
            str: the summary of the memory.
        """
        
        summary_steps = []
        for step in self.steps:
            if isinstance(step, AnswerStep):
                summary_steps.append({
                    "type": "AnswerStep",
                    "data": {
                        "response": {
                            "role": step.response.role,
                            "content": step.response.content
                        }
                    }
                })
            elif isinstance(step, HumanStep):
                summary_steps.append({
                    "type": "HumanStep",
                    "data": step.to_dict()
                })
            elif isinstance(step, ToolStep):
                summary_steps.append({
                    "type": "ToolStep",
                    "data": {
                        "name": step.name,
                        "kwargs": step.kwargs,
                        "result": step.result,
                    }
                })
            else:
                # Skip other step types for summarization
                logger.debug(f"Skipping step type {type(step)} in summarization")
                continue
                
        message = ChatMessage(
            role=ChatRole.USER,
            content=SUMMARIZE_MEMORY_PROMPT.format(steps=json.dumps({"steps": summary_steps}, indent=2))
        )
        
        response = llm.generate(messages=[message])
        
        self.summary = response.content
        self.usage = llm.usage
        
        return response.content