from __future__ import annotations
import logging

from lite_agents.agent._base import (
    BaseAgent, 
    AgentEventStream, 
    AgentResponse,
)
from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.agent.memory import AgentMemory
from lite_agents.prompts.chief import (
    CHIEF_SYSTEM_PROMPT_WITH_EXPANSION, 
    CHIEF_SYSTEM_PROMPT_NO_EXPANSION
)
from lite_agents.llm.lite import LiteLLM
from lite_agents.utils.parse import parse_json_from_keys

logger = logging.getLogger(__name__)

OUTPUT_JSON_KEYS_WITH_EXPANSION = [
    "route_to",
    "reason",
    "context",
    "expanded_query",
]

OUTPUT_JSON_KEYS_NO_EXPANSION = [
    "route_to",
    "reason",
]

### FAIL REASON CONSTANTS
FAIL_IVALID_JSON = "INVALID_JSON"
FAIL_UNKNOWN_AGENT = "UNKNOWN_AGENT"

class AgentChief(BaseAgent):
    """AgentChief agent that routes requests to other agents.
    
    Args:
        agents (list[BaseAgent]): the list of agents to manage.
        llm (LiteLLM | None, optional): the LiteLLM instance to use. Defaults to None.
        memory (AgentMemory | None, optional): the agent memory instance. Defaults to None.
        name (str, optional): the agent name. Defaults to "AgentChief".
        description (str, optional): the agent description. Defaults to "Orchestrates and routes requests to other agents.".
        max_retries (int, optional): maximum number of retries for JSON parsing. Defaults to 3.
        stream (bool, optional): whether to stream the responses. Defaults to False.
        output_json_keys (list[str] | None, optional): the list of JSON keys to parse from the LLM response. If None, uses default keys. Defaults to None.
        query_expansion (bool, optional): whether to use query expansion in routing. Defaults to False.
    """
    def __init__(
        self,
        agents: list[BaseAgent],
        llm: LiteLLM | None = None,
        memory: AgentMemory | None = None,
        name: str = "AgentChief",
        description: str = "Orchestrates and routes requests to other agents.",
        system_prompt: str | None = None,
        max_retries: int = 3,
        stream: bool = False,
        output_json_keys: list[str] | None = None,
        query_expansion: bool = False
    ) -> None:
        """Initialize the AgentChief."""
        
        if system_prompt is None:
            system_prompt = CHIEF_SYSTEM_PROMPT_WITH_EXPANSION if query_expansion else CHIEF_SYSTEM_PROMPT_NO_EXPANSION
            
        super().__init__(
            name=name, 
            description=description, 
            llm=llm, 
            system_prompt=system_prompt,
            memory=memory, 
            stream=stream
        )
        # Agents map by name
        self.agents = {agent.name: agent for agent in agents}
        self.agents_info = "\n".join([f"- {name}: {agent.description}" for name, agent in self.agents.items()])
        self.max_retries = max_retries
        
        if output_json_keys is not None:
            self.output_json_keys = output_json_keys
        else:
            self.output_json_keys = OUTPUT_JSON_KEYS_WITH_EXPANSION if query_expansion else OUTPUT_JSON_KEYS_NO_EXPANSION
        
    def _prepare_messages(self, messages):
        """Prepare the messages for the LLM.
        
        By default, this adds the system prompt if it exists.
        
        Args:
            messages (list[ChatMessage]): the input messages

        Returns:
            list[ChatMessage]: the prepared messages
        """
        # Routing Logic
        system_prompt = self.system_prompt.format(agents_info=self.agents_info)
        # Prepare messages for routing (prepend system prompt)
        messages = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)] + messages
        return messages

    def run(self, messages: list[ChatMessage]) -> AgentEventStream | AgentResponse:
        """Run the AgentChief with the given messages.
        
        Args:
            messages (list[ChatMessage]): the input messages.
            
        Raises:
            ValueError: if the AgentChief fails to route after max retries.
            
        Returns:
            AgentEventStream | AgentResponse: the response from the routed agent.
        """
        # Routing Logic
        chief_messages = self._prepare_messages(messages)
        
        self.memory.add_human_step(messages[-1])

        fail = None
        # Retry loop
        for attempt in range(self.max_retries):
            if fail:
                logger.info(f"AgentChief retrying due to previous failure: {fail} (attempt {attempt + 1}/{self.max_retries})")
            # Generate answer from LLM
            response = self.llm.generate(messages=chief_messages)
            
            # Parse JSON output
            chief_output = parse_json_from_keys(
                text=response.content, keys=self.output_json_keys
            )
            
            # If parsing succeeded
            if chief_output:
                # Get routing info
                agent_name = chief_output.get("route_to")
                reason = chief_output.get("reason")
                expanded_query = chief_output.get("expanded_query")
                
                self.memory.add_chief_step(
                    reason=reason,
                    agent=agent_name,
                    raw_response=str(response),
                    expanded_query=expanded_query,
                    usage=self.llm.usage
                )
                
                if agent_name and agent_name in self.agents:
                    target_agent = self.agents[agent_name]
                    # Delegate
                    return self._delegate(
                        agent=target_agent, 
                        messages=messages, 
                        expanded_query=expanded_query, 
                    )
                else:
                    logger.warning(f"AgentChief routed to unknown agent: {agent_name}")
                    fail = FAIL_UNKNOWN_AGENT
            else:
                logger.warning("AgentChief received invalid JSON from LLM: {response.content}")
                fail = FAIL_IVALID_JSON
                        
            # If we are here, parsing failed or agent not found.
            # Add a user message to force JSON structure for the next attempt
            if fail == FAIL_UNKNOWN_AGENT:
                retry_user_message = f"Unknown agent '{agent_name}'. Please choose a valid agent from the list."
            if fail == FAIL_IVALID_JSON:
                retry_user_message = f"Invalid JSON format. Please provide a valid JSON object with {self.output_json_keys} keys."
            
            chief_messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=response.content))
            self.memory.add_agent_step(response=chief_messages[-1], usage=response.usage)
            chief_messages.append(ChatMessage(role=ChatRole.USER, content=retry_user_message))
            self.memory.add_retry_step(chief_messages[-1])
            
        raise ValueError(f"AgentChief failed to route after {self.max_retries} attempts.")

    def _delegate(
        self, 
        agent: BaseAgent, 
        messages: list[ChatMessage], 
        expanded_query: str | None, 
    ) -> AgentEventStream | AgentResponse:
        """Delegate execution to the target agent, handling streaming if enabled and then returning the response. Finally, merges the agent memory into the chief memory.
        
        Args:
            agent (BaseAgent): the target agent to delegate to.
            messages (list[ChatMessage]): the original messages.
            expanded_query (str | None): the expanded query to use for the agent. If None, uses the original message.
            
        Returns:
            AgentEventStream | AgentResponse: the response from the agent based on streaming setting.
        """
        
        # Ensure the sub-agent respects the stream setting of the Chief
        agent.stream = self.stream
        agent.memory = self.memory
        
        # Use expanded query for the agent if available, otherwise use original messages
        if expanded_query:
            agent_messages = messages[:-1] + [ChatMessage(role=ChatRole.USER, content=expanded_query)]
        else:
            agent_messages = messages
        
        # if streaming, use the streaming and recording method
        return agent.run(agent_messages)