# lite-agents 🤖☁️

A lightweight agent framework using lite-llm. 

## 1. Configuration ⚙️

Ensure your API keys are configured in your environment (e.g., in a `.env` file).

## 2. Basic Usage (LiteLLM) ⚡️

If you just need a simple completion without the agentic loop:

```python
import os
from dotenv import load_dotenv
from lite_agents.llm import LiteLLM
from lite_agents.core.message import ChatRole, ChatMessage

load_dotenv()

llm = LiteLLM(model="gpt-5-nano-2025-08-07", api_key=os.getenv("OPENAI_API_KEY"))

response = llm.generate(
    messages=[ChatMessage(role=ChatRole.USER, content="Hello, who are you?")]
)
print(response.content)
```

## 3. Agent & Tools (The Loop) 🔄

The `Agent` class handles the "reasoning loop": it calls the LLM, executes tools if requested, and feeds the result back to the LLM until a final answer is reached.

### Define a Tool 🛠️

```python
from lite_agents.core.tool import Tool

def book_parking_spot(parking_id: str, spot_id: str, user_id: str, date: str) -> str:
    """Book a parking spot by calling an external API.
    
    Args:
        parking_id (str): the parking lot ID.
        spot_id (str): the spot ID.
        date (str): the date to book the spot for.
        
    Returns:
        str: the booking confirmation message.
    """
    return f"✅ Booked spot {spot_id} at parking {parking_id} for date {date} - User: {user_id}"
```

### Initialize the Agent 🤖

```python
from lite_agents.agent import Agent

agent = Agent(
    llm=llm,
    name="Booking Agent",
    description="An agent that can perform bookings.",
    system_prompt="You are a helpful assistant.",
    tools=[Tool(book_parking_spot)],
    stream=False # Default
)
```

### Run (Standard) 🏃

```python
response = agent.run(
    messages=[ChatMessage(role=ChatRole.USER, content="Book spot 45 at PARK123 for me (User A407031) on Aug 15th 2026.")]
)
print("🤖 Agent:", response.content)
```

### Run (Streaming) 🌊

```python
# Enable streaming
agent.stream = True

streamer = agent.run(
    messages=[
        ChatMessage(
            role=ChatRole.USER, 
            content="Book spot 45 at PARK123 for me (User A407031) on Aug 15th 2026."
        )
    ]
)

print("🤖 Agent: ", end="")
for chunk in streamer:
    print(chunk.delta, end="", flush=True)
print()
```


## 4. RAG Agent 🗂️

Before going into the RAG agent, go check the [example](examples/rag.md) on how to easily ingest documents with `lite-agents` with ChromaDB.

Once you have your knowledge base ready, you can spin up a `RAGAgent`.

### Setup ⚙️

You need to define your embedding function (must match the one used for ingestion) and connect to the existing Vector DB.

```python
from lite_agents.llm import LiteLLM
from lite_agents.agent import RAGAgent
from lite_agents.db import ChromaDB
from litellm import embedding
from pathlib import Path
import os

# 1. Setup LLM & DB
llm = LiteLLM(model="gpt-5-nano-2025-08-07", api_key=os.getenv("OPENAI_API_KEY"))
vector_db = ChromaDB(
    collection_name="YOUR_COLLCECTION_NAME",
    path=Path("PATH/TO/CHROMA_DB/FOLDER")
)

# 2. Define Embedding Function
def create_embeddings(text: str) -> list[float]:
    response = embedding(model="text-embedding-3-small", input=[text])
    return response.data[0]['embedding']
```

### Initialize RAG Agent 🤖

```python
rag_agent = RAGAgent(
    llm=llm,
    vector_db=vector_db,
    embedding_function=create_embeddings,
    name="Company Policies RAG Agent",
    description="An agent that answers questions about company policies with RAG technique.",
    system_prompt="...", # Optional: Use default RAG prompt
    stream=True, # or False
    k=5, # number of chunks to retrieve
    threshold=0.7, # min similarity threhsold between query and retrieved chunks
)
```

### Run 🏃

```python
from lite_agents.core.message import ChatMessage, ChatRole

response = rag_agent.run(
    messages=[
        ChatMessage(
            role=ChatRole.USER, 
            content="How do I request remote work?"
        )
    ]
)

# If streaming is enabled
print("🤖 RAGAgent: ", end="")
for chunk in response:
    print(chunk.delta, end="", flush=True)

# If streaming is not enabled
print(response.content)
```


