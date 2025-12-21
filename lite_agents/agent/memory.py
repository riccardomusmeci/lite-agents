from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import LLMUsage, TextResponse
from dataclasses import dataclass, field

@dataclass
class AnswerStep:
    response: ChatMessage
    usage: LLMUsage

@dataclass  
class HumanStep:
    message: ChatMessage
    
@dataclass
class ToolStep:
    name: str
    kwargs: dict
    result: str
    usage: LLMUsage | None = None

@dataclass
class SytemStep:
    prompt: str

@dataclass
class AgentMemory:
    """Agent memory class"""
    steps: list[SytemStep | AnswerStep | HumanStep | ToolStep] = field(default_factory=list)
    
    def add_system_step(self, message: ChatMessage) -> None:
        if message.role != ChatRole.SYSTEM:
            raise ValueError("SystemStep message must have role SYSTEM.")
        self.steps.append(SytemStep(prompt=message.content))
    
    def add_human_step(self, message: ChatMessage) -> None:
        if message.role != ChatRole.USER:
            raise ValueError("HumanStep message must have role USER.")
        self.steps.append(HumanStep(message=message))
        
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
    