from __future__ import annotations
from typing import Generator, Union, List, Callable
from lite_agents.llm.lite import LiteLLM
from lite_agents.core.message import ChatMessage, ChatRole
from lite_agents.core.response import (
    TextResponse, 
    TextResponseDelta, 
    LLMUsage
)
from lite_agents.agent.memory import AgentMemory
from lite_agents.agent._base import BaseAgent
from lite_agents.db.db import VectorDB
from lite_agents.logger import setup_logger

logger = setup_logger()

# RAGAgentEvent (simplified compared to AgentEvent for now)
RAGAgentEvent = Union[
    TextResponseDelta,
    TextResponse,
]
# Streaming Type
RAGAgentEventStream = Generator[RAGAgentEvent, None, None]

class RAGAgent(BaseAgent):
    """RAG Agent that retrieves context from a VectorDB before answering.
    
    Args:
        llm (LiteLLM): the LiteLLM instance to use
        vector_db (VectorDB): the vector database instance
        embedding_function (Callable[[str], List[float]]): function to embed queries
        name (str): the agent name
        description (str): the agent description
        system_prompt (str | None, optional): the system prompt to use. Defaults to None.
        stream (bool, optional): whether to stream the responses. Defaults to False.
        memory (AgentMemory | None, optional): the agent memory instance. Defaults to None.
        k (int, optional): number of documents to retrieve. Defaults to 5.
        threshold (float, optional): similarity threshold for retrieval. Defaults to 0.8.
    """
    def __init__(
        self,
        llm: LiteLLM,
        vector_db: VectorDB,
        embedding_function: Callable[[str], List[float]],
        name: str,
        description: str,
        system_prompt: str | None = None,
        stream: bool = False,
        memory: AgentMemory | None = None,
        k: int = 5,
        threshold: float = 0.8,
    ) -> None:
        """Initialize the RAGAgent class."""
        super().__init__(
            name=name,
            description=description,
            llm=llm,
            system_prompt=system_prompt,
            memory=memory,
            stream=stream
        )
        self.vector_db = vector_db
        self.embedding_function = embedding_function
        self.k = k
        self.threshold = threshold

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the vector database based on the query.

        Args:
            query (str): The user's query string.

        Returns:
            str: A formatted string containing the retrieved documents and their metadata.
        """
        query_embedding = self.embedding_function(query)
        results = self.vector_db.query(
            query_embeddings=query_embedding,
            n_results=self.k,
            threshold=self.threshold
        )
        
        # add retrieval step to memory
        self.memory.add_retrieval_step(
            chunks=results or {}
        )
        
        if not results:
            return ""
        
        context_parts = []
        for i, res in enumerate(results):
            content = res.get("content", "")
            # metadata = res.get("metadata", {})
            # Format metadata if present
            context_parts.append(f"<item_{i+1}>\n{content}\n</item_{i+1}>")
        return "\n".join(context_parts)

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Prepare the messages for the LLM by injecting retrieved context.

        If the last message is from the user, it retrieves relevant context and 
        augments the user message with it.

        Args:
            messages (list[ChatMessage]): The original list of chat messages.

        Returns:
            list[ChatMessage]: The list of messages with context injected.
        """
        
        # Get the last user message to use as query
        last_message = messages[-1]
        if last_message.role == ChatRole.USER:
            query = last_message.content
            context = self._retrieve_context(query)
            
            # Augment the message with context
            augmented_content = f"## **Context**\n{context if context else 'EMPTY'}\n\n ## **User Question**\n{query}"
            # Create a new list to avoid modifying the original objects in place if they are reused
            messages = messages[:-1] + [ChatMessage(role=ChatRole.USER, content=augmented_content)]
            if not context:
                logger.warning("No relevant context retrieved for the query.")
                        
        if self.system_prompt:
            messages = [ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)] + messages
            self.memory.add_system_step(messages[0])
            
        return messages

    def run(self, messages: list[ChatMessage]) -> RAGAgentEventStream | TextResponse:
        """Run the RAG agent with the given messages.
    
        Args:
            messages (list[ChatMessage]): the input messages
            
        Returns:
            RAGAgentEventStream | TextResponse: the streaming generator or full response list
        """
        # Save original message to memory
        self.memory.add_human_step(messages[-1])
        
        messages_for_run = self._prepare_messages(messages)
           
        if self.stream:
            return self._stream_response(messages_for_run)
        else:
            return self._generate_response(messages_for_run)

    def _stream_response(self, messages: list[ChatMessage]) -> RAGAgentEventStream:
        """Stream the response from the LLM.

        Args:
            messages (list[ChatMessage]): the list of messages to send to the LLM.

        Yields:
            RAGAgentEventStream: a generator yielding response deltas.
        """
        streamer = self.llm.stream(messages=messages)
        text_deltas = []
        for chunk in streamer:
            if isinstance(chunk, TextResponseDelta):
                text_deltas.append(chunk)
                yield chunk
            else:
                # Ignore tool calls or other types for now in RAGAgent
                logger.warning("RAGAgent received unexpected chunk type during streaming: %s", type(chunk))
                pass
        
        if text_deltas:
            self.memory.add_agent_step(
                response=TextResponse.from_deltas(text_deltas), 
                usage=self.llm.usage
            )

    def _generate_response(self, messages: list[ChatMessage]) -> TextResponse:
        """Generate a non-streaming response from the LLM.

        Args:
            messages (list[ChatMessage]): the list of messages to send to the LLM.

        Returns:
            TextResponse: the full text response.

        Raises:
            ValueError: if the LLM returns an unexpected response type.
        """
        llm_response = self.llm.generate(messages=messages)
        if isinstance(llm_response, TextResponse):
            self.memory.add_agent_step(llm_response, self.llm.usage)
            return llm_response
        else:
            # Handle unexpected response types
            raise ValueError(f"Unexpected response type: {type(llm_response)}")
