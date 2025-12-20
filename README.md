# lite-agents 🤖☁️

A lightweight agent framework using lite-llm.

## 🚀 Quick Start

Here is a simple example of how to use `lite-agents` to create an agent with custom tools (Function Calling).

### 1. Configuration ⚙️

Ensure your API keys are configured in your environment (e.g., in a `.env` file).

### 2. Code Example 💻

```python
import os
from dotenv import load_dotenv
from lite_agents.llm.lite import LiteLLM
from lite_agents.core.tool import Tool
from lite_agents.core.message import ChatRole, ChatMessage

# Load environment variables
load_dotenv()

# 1. Define a pure Python function to use as a tool
def add_from_list(numbers: list[int]) -> int:
    """Add a list of integers and return the result.
    
    Args:
        numbers (list[int]): the list of integers to add.
        
    Returns:
        int: the sum of the integers.
    """
    return sum(numbers)

# 2. Initialize the model (e.g., Claude 3 Haiku)
llm = LiteLLM(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    system_prompt="You are a helpful assistant with access to tools."
)

# 3. Execute a call (streaming) passing the tools
response = llm.generate(
    messages=[ChatMessage(role=ChatRole.USER, content="How much is 10 + 20 + 5?")],
    tools=[Tool(add_from_list)] # The tool is automatically converted
)

print(response)

# 4. Token usage
print(f"\n\nToken Usage: {llm.usage}")
```

### How it works 🧠

1.  **Automatic Tools**: `lite-agents` inspects your Python function (`add_from_list`), reads types and docstrings, and automatically generates the JSON definition for the LLM.
2.  **Seamless Integration**: Simply pass the list of `Tool(...)` to the `generate` or `stream` method.
3.  **Autonomous Execution**: The model decides **if** and **when** to use the tool to answer the user's query.
