from dataclasses import dataclass
from strenum import StrEnum

class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    

@dataclass
class ChatMessage:
    """Chat message object
    
    Args:
        role (ChatRole): the role of the message sender.
        content (str | None, optional): the message content. Defaults to None.
        name (str | None, optional): the name of the tool (for tool messages). Defaults to None.
        tool_calls (list | None, optional): the tool calls (for assistant messages). Defaults to None.
        tool_call_id (str | None, optional): the tool call id (for tool messages). Defaults to None.
        tool_kwargs (dict | None, optional): the tool arguments (for tool messages). Defaults to None.
    """
    role: ChatRole
    content: str | None = None
    name: str | None = None
    tool_calls: list | None = None
    tool_kwargs: dict | None = None
    tool_call_id: str | None = None
    
    def to_dict(self) -> dict:
        """Convert the ChatMessage to a dictionary.

        Returns:
            dict: the dictionary representation of the ChatMessage.
        """
        message = {
            "role": self.role.value, 
            "content": self.content
        }
        if self.name:
            message["name"] = self.name
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id    
        return message
        

