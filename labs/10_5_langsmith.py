import os
import requests
import warnings
from uuid import uuid4
from typing import Any
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
Weather_API_Key = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.chat_models import init_chat_model

llm_groq_1 = init_chat_model(
    model="openai/gpt-oss-120b", 
    model_provider="groq",
    metadata={
        "ls_provider": "groq", 
        "ls_model_name": "openai/gpt-oss-120b"
    }
)
llm_groq_2 = init_chat_model(
    model="qwen/qwen3.8-27b", 
    model_provider="groq",
    metadata={
        "ls_provider": "groq", 
        "ls_model_name": "qwen/qwen3.8-27b"
    }
)

# Create a weather tool
from langchain.tools import tool
url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def getweather(location: str) -> str:
    """Take the location and return the current weather."""
    params = {
        "q": location, 
        "appid": Weather_API_Key, 
        "units": "imperial"
    }

    try: 
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception:
        return "The requested weather information is unavailable."

# Search tools
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun

web_search = TavilySearch(
    max_results=3, 
    topic="general", 
    search_depth="basic"
)
duck_search = DuckDuckGoSearchRun()

tools = [getweather, web_search, duck_search]

# Memory and Middleware setup
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.agents.middleware import after_model, before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Trim messages to manage the context window."""
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    keep = [messages[0]] + messages[-3:]
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES), 
            *keep
        ]
    }

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Delete old messages."""
    messages = state["messages"]
    if len(messages) > 5:
        return {
            "messages": [
                RemoveMessage(id=m.id) for m in messages[1:3]
            ]
        }

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware, 
    ModelCallLimitMiddleware, 
    ModelRetryMiddleware, 
    PIIMiddleware
)

system_prompt = """
You are a helpful chatbot assistant. Use getweather tool for answering weather related queries. Use web_search tool for recent events, finance, news, pop culture and internet search. Use web_search for any queries involving today or the current year (2026). Use duck_search when web_search is not available or returns no results. For normal chat conversation, don't use a tool, just answer on your own.
"""

agent_groq_1 = create_agent(
    model=llm_groq_1, 
    tools=tools, 
    name="Groq-GPT-OSS-Agent",
    checkpointer=InMemorySaver(),
    middleware=[
        delete_old_messages,
        SummarizationMiddleware(
            model=llm_groq_1, trigger=("tokens", 8000), keep=("messages", 10)
        ),
        ModelRetryMiddleware(max_retries=5, backoff_factor=1.5, initial_delay=1, max_delay=60),
        ModelCallLimitMiddleware(thread_limit=10, run_limit=4, exit_behavior="end"),
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_input=True),
        PIIMiddleware(pii_type="credit_card", strategy="redact", apply_to_input=True), 
        PIIMiddleware(pii_type="ip", strategy="mask", apply_to_input=True),
        PIIMiddleware(pii_type="api_key", strategy="mask", detector=r"g?sk[a-zA-Z0-9_-]{16,32}", apply_to_input=True),
        PIIMiddleware(pii_type="offensive_words", strategy="redact", detector=r"(?i)\b(?:crap|heck|darn)\b", apply_to_input=True)
    ],
    system_prompt=system_prompt
)

agent_groq_2 = create_agent(
    model=llm_groq_2, 
    tools=tools, 
    name="Groq-Qwen3.8-27B-Agent",
    checkpointer=InMemorySaver(),
    middleware=[
        delete_old_messages,
        SummarizationMiddleware(
            model=llm_groq_2, trigger=("tokens", 8000), keep=("messages", 10)
        ),
        ModelRetryMiddleware(max_retries=5, backoff_factor=1.5, initial_delay=1, max_delay=60),
        ModelCallLimitMiddleware(thread_limit=10, run_limit=4, exit_behavior="end"),
        PIIMiddleware(pii_type="email", strategy="redact", apply_to_input=True),
        PIIMiddleware(pii_type="credit_card", strategy="redact", apply_to_input=True), 
        PIIMiddleware(pii_type="ip", strategy="mask", apply_to_input=True),
        PIIMiddleware(pii_type="api_key", strategy="mask", detector=r"g?sk[a-zA-Z0-9_-]{16,32}", apply_to_input=True),
        PIIMiddleware(pii_type="offensive_words", strategy="redact", detector=r"(?i)\b(?:crap|heck|darn)\b", apply_to_input=True)
    ],
    system_prompt=system_prompt
)

# Chat Loop
thread_id = str(uuid4())
thread_config = {"configurable": {"thread_id": thread_id}}

from langchain.messages import HumanMessage

while True:
    messages = []
    user_input = input("What is your query? ")

    if user_input.lower() in ["bye", "exit"]:
        print("Bye!")
        exit()

    if user_input.lower().strip() == "/new":
        thread_id = str(uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}
        print("Welcome to the new session...")
        continue

    messages.append(HumanMessage(content=user_input))

    try:
        print("\n--- Model 1 Output (Groq: GPT-OSS 120B) ---")
        config_groq_1 = {
            **thread_config,
            "tags": ["gpt-oss-120b"],
            "metadata": {"model_provider": "groq", "model_name": "openai/gpt-oss-120b"}
        }
        final_result_1 = agent_groq_1.invoke({"messages": messages}, config_groq_1)
        for message in final_result_1["messages"]:
            message.pretty_print()

        print("\n--- Model 2 Output (Groq: Qwen 3.8 27B) ---")
        config_groq_2 = {
            **thread_config,
            "tags": ["qwen3.8-27b"],
            "metadata": {"model_provider": "groq", "model_name": "qwen/qwen3.8-27b"}
        }
        final_result_2 = agent_groq_2.invoke({"messages": messages}, config_groq_2)
        for message in final_result_2["messages"]:
            message.pretty_print()

    except Exception as e:
        print(f"Error: {e}")