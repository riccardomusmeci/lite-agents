from lite_agents.llm.lite import LiteLLM
from lite_agents.core.tool import Tool
from lite_agents.core.message import ChatRole, ChatMessage
from lite_agents.core.response import TextResponse, ToolResponse, TextResponseDelta, ToolResponseDelta
import os 
from dotenv import load_dotenv

load_dotenv()

# api_key = os.getenv("ANTHROPIC_API_KEY", "test")
# print(api_key)


def add_from_list(numbers: list[int]) -> int:
    """Add a list of integers and return the result.
    
    Args:
        numbers (list[int]): the list of integers to add.
        
    Returns:
        int: the sum of the integers.
    """
    return sum(numbers)

def test_function(a: str, b: float | None, c: list[int] | None) -> str:
    return f"a: {a}, b: {b}, c: {c}"

def add(a: int, b: int | None = None) -> int:
    """Add two integers and return the result.
    
    Args:
        a (int): the first integer.
        b (int, optional): the second integer. Defaults to None.
        
    Returns:
        int: the sum of the two integers.
    """
    return a + b

tool = Tool(add_from_list)
# print(tool.to_dict())
# quit()
    
# quit()

llm = LiteLLM(
    model="claude-3-haiku-20240307",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    system_prompt="You are a helpful assistant with access to tools. If you need to use a tool, do not write other than the call to use and its arguments.",
)

# response = llm.generate(
#     messages=[ChatMessage(role=ChatRole.USER, content="How much is 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1?")],
#     tools=[Tool(add_from_list)]
# )

# quit()

stream = llm.stream(
    messages=[ChatMessage(role=ChatRole.USER, content="How much is 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1?")],
    tools=[Tool(add_from_list)]
)

for chunk in stream:
    print(chunk)

print(llm.usage)
quit()

response = llm.generate(
    messages=[ChatMessage(role=ChatRole.USER, content="ciao, come stai?")],
    tools=[Tool(add_from_list)]
)

print(response)