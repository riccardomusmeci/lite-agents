from __future__ import annotations
import json
from typing import Any, Generator, Union
from lite_agents.llm.lite import LiteLLM
from lite_agents.core.tool import Tool
from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import (
    TextResponse, 
    ToolCall,
    ToolResult, 
    TextResponseDelta, 
    ToolCallDelta, 
    LLMUsage,
    AgentReachedMaxSteps
)
from lite_agents.agent.memory import AgentMemory


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

class Agent:
    """Base agent class that loops over question -> tool call -> tool execution -> answer.
    
    Args:
        llm (LiteLLM): the LiteLLM instance to use
        name (str): the agent name
        description (str): the agent description
        system_prompt (str | None, optional): the system prompt to use. Defaults to None.
        tools (list[Tool] | None, optional): the list of tools available to the agent. Defaults to None.
        stream (bool, optional): whether to stream the responses. Defaults to False.
        max_iterations (int, optional): the maximum number of iterations to perform. Defaults to 12.
        memory (AgentMemory | None, optional): the agent memory instance. Defaults to None.
    """
    def __init__(
        self,
        llm: LiteLLM,
        name: str,
        description: str,
        system_prompt: str | None = None,
        tools: list[Tool] | None = None,
        stream: bool = False,
        max_iterations: int = 12,
        memory: AgentMemory | None = None,
    ) -> None:
        """Initialize the Agent class."""
        self.llm = llm
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else []
        self.stream = stream
        self.max_iterations = max_iterations
        self.usage: list[LLMUsage] = []
        self.memory: AgentMemory = memory or AgentMemory()

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Prepare the messages for the LLM

        Args:
            messages (list[ChatMessage]): the input messages

        Returns:
            list[ChatMessage]: the prepared messages with system prompt if any
        """
        if self.system_prompt:
            messages = [ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)] + messages
            self.memory.add_system_step(messages[0])            
        return messages

    def _format_tool_response(self, tool_response: ToolCall) -> dict[str, Any]:
        """Format the ToolCall into a tool call dictionary.

        Args:
            tool_response (ToolCall): the tool response to format
        Returns:
            dict[str, Any]: the formatted tool call dictionary
        """
        # OpenAI-style tool call payload
        return {
            "id": tool_response.id,
            "type": "function",
            "function": {
                "name": tool_response.name,
                "arguments": json.dumps(tool_response.kwargs, ensure_ascii=False),
            },
        }

    def _tool_result_as_str(self, result: Any) -> str:
        """Make the tool result into a string for the tool message.

        Args:
            result (Any): the tool result

        Returns:
            str: the stringified tool result
        """
        # Normalize the tool return to string: if it's dict/list serialize it to JSON.
        if result is None:
            return ""
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def _find_tool(self, name: str) -> Tool | None:
        """Look for a tool by name.

        Args:
            name (str): the name of the tool to find

        Returns:
            Tool | None: the tool if found, None otherwise
        """
        return next((t for t in self.tools if t.name == name), None)
                    
    def _run_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute the tool based on the LLM answer

        Args:
            tool_call (ToolCall): tool call from LLM

        Returns:
            Any: the tool result if successful, error dict otherwise
        """
        tool = self._find_tool(tool_call.name)

        if tool is None:
            tool_result = ToolResult(
                success=False,
                result=None,
                error=f"ToolNotFound: '{tool_call.name}'. Available tools: {[t.name for t in self.tools]}",
            )
        else:
            try:
                tool_output = tool.execute(**tool_call.kwargs)
                tool_result = ToolResult(
                    success=True,
                    result=tool_output,
                    error=None,
                )
            except Exception as e:
                tool_result = ToolResult(
                    success=False,
                    result=None,
                    error=f"ToolExecutionError ({tool_call.name}): {type(e).__name__}: {str(e)}",
                )
        return tool_result

    def _run_loop(self, messages: list[ChatMessage]) -> AgentEventStream:
        """Run the agentic loop with possible function calling. It works like this:
            1) calls LLM with messages + tools
            2) if TextResponse -> done
            3) if ToolCall -> execute tool, append tool call + tool result, repeat
        The loop continues until a TextResponse is returned or max_iterations is reached.
        
        Args:
            messages (list[ChatMessage]): the initial messages
            
        Yields:
            AgentEventStream: the streaming text deltas, tool calls, tool results, or final text response
            
        Raises:
            ValueError: if an unknown response type is received from the LLM
        """
        # Saving the initial messages for memory
        self.memory.add_human_step(messages[-1])
        for step in range(1, self.max_iterations + 1):
            text_response = None
            tool_response = None
            if self.stream:
                streamer = self.llm.stream(messages=messages, tools=self.tools)
                text_deltas, tool_deltas = [], []                
                for chunk in streamer:
                    if isinstance(chunk, TextResponseDelta):
                        text_deltas.append(chunk)
                        yield chunk
                    elif isinstance(chunk, ToolCallDelta):
                        tool_deltas.append(chunk)
                    else:
                        raise TypeError(f"Unsupported chunk type from LLM stream: {type(chunk)}")
                # If only text response was received, stop here
                if text_deltas:
                    self.memory.add_agent_step(
                        response=TextResponse.from_deltas(text_deltas), 
                        usage=self.llm.usage
                    )
                if text_deltas and not tool_deltas:
                    return
                # If tool deltas was received, construct ToolCall and continue iterating
                if tool_deltas:
                    # reconstruct ToolCall from deltas
                    tool_response = ToolCall.from_deltas(tool_deltas)
                    yield tool_response
            else:
                llm_response = self.llm.generate(messages=messages, tools=self.tools)
                if isinstance(llm_response, TextResponse):
                    text_response = llm_response
                    self.memory.add_agent_step(text_response, self.llm.usage)                    
                    yield text_response
                    return
                elif isinstance(llm_response, ToolCall):
                    tool_response = llm_response
                    yield tool_response
                else:
                    raise TypeError(f"Unsupported response type from LLM: {type(llm_response)}")
            
            # 2) Tool call requested by the LLM
            if tool_response:
                tool_result = self._run_tool(tool_response)
                # Append: assistant message that contains tool_calls
                messages.append(
                    ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=None,
                        tool_calls=[self._format_tool_response(tool_response)],
                    )
                )
                
                # Append: tool message with the result
                messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL,
                        content=tool_result.to_str(),
                        tool_call_id=tool_response.id,
                        name=tool_response.name,
                        tool_kwargs=tool_response.kwargs,
                    )
                )
                self.memory.add_tool_step(
                    response=messages[-1],
                    usage=self.llm.usage
                )
                yield tool_result
                
            # 3) Unexpected response type
            else:
                raise ValueError("Unknown response type from LLM.")

        # Max iterations reached
        yield AgentReachedMaxSteps(
            content=f"Agent {self.name} reached the maximum number of iterations for answering the query."
        )

    def run(self, messages: list[ChatMessage]) -> AgentEventStream | AgentResponse:
        """Run the agent with the given messages.
    
        Args:
            messages (list[ChatMessage]): the input messages
            
        Returns:
            AgentEventStream | AgentResponse: the streaming generator or full response list
        """
        messages_for_run = self._prepare_messages(messages)
        if not self.stream:
            responses = []
            for step in self._run_loop(messages_for_run):
                responses.append(step)
            return responses
        return self._run_loop(messages_for_run)