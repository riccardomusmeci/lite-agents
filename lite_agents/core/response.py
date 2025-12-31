from dataclasses import dataclass
import json
from typing import Any

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
class ToolCallDelta:
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
class ToolCall:
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
    def from_deltas(deltas: list[ToolCallDelta]) -> "ToolCall":
        """Return ToolCall filled with data from streaming

        Args:
            deltas (list[ToolCallDelta]): list of tool response deltas

        Returns:
            ToolCall: tool response object
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
        return ToolCall(name=name, kwargs=kwargs, id=tool_id)
    
@dataclass
class ToolResult:
    """Result of a tool execution.
    
    Args:
        success (bool): whether the tool execution was successful
        result (Any | None): the result of the tool execution
        error (str | None): the error message if execution failed
    """
    success: bool
    result: Any | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        """Convert the ToolResult to a dictionary.

        Returns:
            dict: the dictionary representation of the ToolResult.
        """
        # Normalize the tool return to string: if it's dict/list serialize it to JSON.
        if self.result is None:
            return ""
        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, ensure_ascii=False)
                
        return {
            "success": self.success,
            "result": str(self.result),
            "error": self.error
        }
        
    def to_str(self) -> str:
        """Convert the ToolResult to a string.

        Returns:
            str: the string representation of the ToolResult.
        """
        # Normalize the tool return to string: if it's dict/list serialize it to JSON.
        tool_result = self.to_dict()
        return str(tool_result)
    
@dataclass
class AgentReachedMaxSteps:
    """Indicates that the agent has reached the maximum number of steps allowed.
    
    Args:
        message (str | None): the message indicating the max steps reached.
    """
    content: str | None = None

    