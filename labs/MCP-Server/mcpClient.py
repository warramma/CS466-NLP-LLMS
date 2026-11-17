import asyncio
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# Get the llm model
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")

# Unused models commented out per lab instructions
# model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")
# gemma = init_chat_model(model="gemma4:latest", model_provider="ollama")
# gpt = init_chat_model(model="gpt-5.4-mini", model_provider="openai")


async def main():
    # Create an MCP Client
    client = MultiServerMCPClient({
        "course_assistant": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "server.py"],
            "cwd": str(Path(__file__).resolve().parent),
        }
    })

    # Discover and print server tools
    tools = await client.get_tools()
    for tool in tools:
        print(tool.name)

    # Create langchain agent and add tools to the agent
    prompt = """You are a helpful chatbot assistant. 
For course related query use MCP course_assistant server tools. 
For other general enquiry respond of your own."""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
    )

    user_query = (
        "What are the title and credit hours for cs 111?"
    )

    try:
        response = await agent.ainvoke({
            "messages": [
                HumanMessage(content=user_query)
            ]
        })
        print("\nAgent Response:")
        print(response["messages"][-1].content)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())