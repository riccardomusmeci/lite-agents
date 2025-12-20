from dataclasses import dataclass
import json

@dataclass
class LLMUsage:
    """LLM usage statistics"""
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    time: float | None = None

@dataclass
class TextResponseDelta:
    """Streaming text chunk"""
    delta: str | None = None

@dataclass
class ToolResponseDelta:
    """Streaming tool chunk"""
    name: str | None = None
    kwargs: str | None = None

@dataclass
class TextResponse:
    """Text response from LLM"""
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
    """Tool response from LLM"""
    name: str | None = None
    kwargs: dict | None = None
    
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
        for delta in deltas:
            if delta.name is not None:
                name += delta.name
            if delta.kwargs is not None:
                kwargs += delta.kwargs
        if kwargs:
            kwargs = json.loads(kwargs)
        return ToolResponse(name=name, kwargs=kwargs)



    