from dataclasses import dataclass
from strenum import StrEnum

class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    

@dataclass
class ChatMessage:
    """Chat message object
    
    Args:
        role (ChatRole): the role of the message sender.
        content (str | None, optional): the message content. Defaults to None.
    """
    role: ChatRole
    content: str | None = None
    
    def to_dict(self) -> dict:
        """Convert the ChatMessage to a dictionary.

        Returns:
            dict: _description_
        """
        return {
            "role": self.role.value, 
            "content": self.content or ""
        }
        

