from dataclasses import dataclass
import json

@dataclass
class LLMUsage:
    """LLM usage statistics
    
    Args:
        model (str | None): the model name
        input_tokens (int | None): number of input tokens
        output_tokens (int | None): number of output tokens
        time (float | None): time taken in seconds
    """
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    time: float | None = None

@dataclass
class TextResponseDelta:
    """Streaming text chunk
    
    Args:
        delta (str | None): the text delta
    """
    delta: str | None = None

@dataclass
class ToolResponseDelta:
    """Streaming tool chunk
    
    Args:
        name (str | None): the tool name
        kwargs (str | None): the tool arguments as JSON string
        id (str | None): the tool call id
    """
    name: str | None = None
    kwargs: str | None = None
    id: str | None = None

@dataclass
class TextResponse:
    """Text response from LLM
    
    Args:
        content (str | None): the text content
    """
    content: str | None = None
    
    @staticmethod
    def from_deltas(deltas: list[TextResponseDelta]) -> "TextResponse":
        """Return TextResponse filled with data from streaming deltas
        
        Args:
            deltas (list[TextResponseDelta]): list of text response deltas
            
        Returns:
            TextResponse: text response object
        
        """
        content = ''.join([delta.delta or "" for delta in deltas])
        return TextResponse(content=content)

@dataclass
class ToolResponse:
    """Tool response from LLM
    
    Args:
        name (str | None): the tool name
        kwargs (dict | None): the tool arguments
        id (str | None): the tool call id
    """
    name: str | None = None
    kwargs: dict | None = None
    id: str | None = None
    
    @staticmethod
    def from_deltas(deltas: list[ToolResponseDelta]) -> "ToolResponse":
        """Return ToolResponse filled with data from streaming

        Args:
            deltas (list[ToolResponseDelta]): list of tool response deltas

        Returns:
            ToolResponse: tool response object
        """
        name = ""
        kwargs = ""
        tool_id = None
        for delta in deltas:
            if delta.name is not None:
                name += delta.name
            if delta.kwargs is not None:
                kwargs += delta.kwargs
            if delta.id is not None:
                tool_id = delta.id
        if kwargs:
            kwargs = json.loads(kwargs)
        return ToolResponse(name=name, kwargs=kwargs, id=tool_id)

@dataclass
class AgentReachedMaxSteps:
    """Indicates that the agent has reached the maximum number of steps allowed.
    
    Args:
        message (str | None): the message indicating the max steps reached.
    """
    content: str | None = None

    