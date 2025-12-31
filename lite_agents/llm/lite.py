from dotenv import load_dotenv
from litellm import completion
from typing import Generator
from litellm.utils import (
    ModelResponse
)
import time
import json
from lite_agents.core.message import ChatMessage
from lite_agents.core.response import (
    TextResponse, 
    ToolCall, 
    TextResponseDelta, 
    ToolCallDelta,
    LLMUsage
)
from lite_agents.core.tool import Tool

load_dotenv()

class LiteLLM:
    """LiteLLM class to handle LLM generation with litellm.
    
    Args:
        model (str): the model name
        api_key (str, optional): the API key for authentication. Defaults to None.
    """
    def __init__(
        self,
        model: str,
        api_key: str = None,
    ) -> None:
        """Initialize the LiteLLM class."""
        self.model = model
        self.api_key = api_key
        self.usage = LLMUsage()
               
    def generate(
        self, 
        messages: list[ChatMessage],
        tools: list[Tool] = None,
    ) -> TextResponse | ToolCall:
        """Generate a response from the model given the messages and optional tools.

        Args:
            messages (list[ChatMessage]): the chat messages
            tools (list[Tool], optional): the list of tools to use. Defaults to None.

        Returns:
            TextResponse | ToolCall: the model's response (either text or tool call)
        """

        start_time = time.time()
        
        response: ModelResponse = completion(
            model=self.model,
            api_key=self.api_key,
            tools=[tool.to_dict() for tool in tools],
            messages=[message.to_dict() for message in messages],
            stream=False
        )
        
        self.usage = LLMUsage(
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            time=time.time() - start_time
        )
        
        if response.choices[0].finish_reason == "stop":
            return TextResponse(
                content=response.choices[0].message.content
            )
        # tool calls handling
        elif response.choices[0].finish_reason == "tool_calls":
            tool_call = response.choices[0].message.tool_calls[0]
            name = tool_call.function.name
            kwargs = json.loads(tool_call.function.arguments)  # to be parsed as JSON
            return ToolCall(
                name=name,
                kwargs=kwargs,
                id=tool_call.id
            )
            
        raise ValueError("Unknown finish reason.")
    
    def stream(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] = None,
    ) -> Generator[TextResponseDelta | ToolCallDelta]:
        """Stream a response from the model given the messages and optional tools.

        Args:
            messages (list[ChatMessage]): the chat messages
            tools (list[Tool], optional): the list of tools to use. Defaults to None.
        
        Yields:
            Generator[TextResponseDelta | ToolCallDelta]: the model's response deltas (either text or tool call)
        """    
        
        stream = completion(
            model=self.model,
            api_key=self.api_key,
            tools=[tool.to_dict() for tool in tools],
            messages=[message.to_dict() for message in messages],
            stream=True,
            stream_options={"include_usage": True}
        )
        start_time = time.time() 
        for chunk in stream:
            content = chunk.choices[0].delta.get("content")
            tool_name, tool_args = None, None
            if chunk.choices[0].delta.tool_calls:
                tool_name = chunk.choices[0].delta.tool_calls[0].function.name
                tool_args = chunk.choices[0].delta.tool_calls[0].function.arguments
                tool_id = chunk.choices[0].delta.tool_calls[0].id
            if content:
                yield TextResponseDelta(
                    delta=content
                )
            if tool_name or tool_args:
                yield ToolCallDelta(
                    name=tool_name,
                    kwargs=tool_args,
                    id=tool_id
                )
            else:
                if hasattr(chunk, "usage") and chunk.usage:
                    self.usage = LLMUsage(
                        model=self.model,
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        time=time.time() - start_time
                    )
     