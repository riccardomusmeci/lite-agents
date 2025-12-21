# lite-agents 🤖☁️

A lightweight agent framework using lite-llm.

## 🚀 Quick Start

Here is a simple example of how to use `lite-agents` to create an agent with custom tools (Function Calling).

### 1. Configuration ⚙️

Ensure your API keys are configured in your environment (e.g., in a `.env` file).

### 2. Basic Usage (LiteLLM) ⚡️

If you just need a simple completion without the agentic loop:

```python
import os
from dotenv import load_dotenv
from lite_agents.llm.lite import LiteLLM
from lite_agents.core.message import ChatRole, ChatMessage

load_dotenv()

llm = LiteLLM(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

response = llm.generate(
    messages=[ChatMessage(role=ChatRole.USER, content="Hello, who are you?")]
)
print(response.content)
```

### 3. Agent & Tools (The Loop) 🔄

The `Agent` class handles the "reasoning loop": it calls the LLM, executes tools if requested, and feeds the result back to the LLM until a final answer is reached.

#### Define a Tool 🛠️

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

#### Initialize the Agent 🤖

```python
from lite_agents.agent.agent import Agent

agent = Agent(
    llm=llm,
    name="Booking Agent",
    description="An agent that can perform bookings.",
    system_prompt="You are a helpful assistant.",
    tools=[Tool(book_parking_spot)],
    stream=False # Default
)
```

#### Run (Standard) 🏃

```python
response = agent.run(
    messages=[ChatMessage(role=ChatRole.USER, content="Book spot 45 at PARK123 for me (User A407031) on Aug 15th 2026.")]
)
print("🤖 Agent:", response.content)
```

#### Run (Streaming) 🌊

```python
# Enable streaming
agent.stream = True

generator = agent.run(
    messages=[ChatMessage(role=ChatRole.USER, content="Book spot 45 at PARK123 for me (User A407031) on Aug 15th 2026.")]
)

print("🤖 Agent: ", end="")
for chunk in generator:
    print(chunk.delta, end="", flush=True)
print()
```

### How it works 🧠

1.  **Automatic Tools**: `lite-agents` inspects your Python function, reads types and docstrings, and automatically generates the JSON definition for the LLM.
2.  **Agent Loop**: The `Agent` automatically handles the `ToolResponse` from the LLM, executes the Python function, and sends the result back to the model.
3.  **Streaming**: In streaming mode, the agent yields text chunks as they arrive, while handling tool executions silently in the background.
