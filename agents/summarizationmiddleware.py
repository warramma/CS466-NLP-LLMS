import os
import requests
from dotenv import load_dotenv
from groq import RateLimitError as GroqRateLimitError
from openai import RateLimitError as OPenAIRateLimitError

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
Weather_API_Key = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.agents.middleware import SummarizationMiddleware

class LoggedSummarizationMiddleware(SummarizationMiddleware):
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update is not None:
            print("\n SUMMARIZATION")
        return update
    def after_model(self, state, runtie):
        latest_result = state["messages"]
        print("AI") ltest_resut["messages]"][-1].content)
        print(\nUsage Metadata:", latet_result["messages"][-1].usage_metadata")

from langchain.chat_models import init_chat_model
llm = init_chat_model(model="openai/gpt-oss-120B", model_provider="groq")
llm_sum = init_chat_model(model="openai/gpt-oss-120B", model_provider="Groq", profile={"max_input_tokens":8000})
#print("llm profile:", llm.profile["max_input_tokens"])

# Create a weather tool
from langchain.tools import tool
url = "https://api.openweathermap.org/data/2.5/weather?"
@tool
def getweather(location:str)->str:
    """Take the location and return the current weather. """
    params={
        "q": location, 
        "appid" : Weather_API_Key, 
        "units" : "imperial"
    }

    try: 
        response = requests.get(url, params=params, timeout = 10)
        return response.json()
    except:
        return f"The requested weather information is unavailable."
# Search tools
from langchain_tavily import TavilySearch
web_search = TavilySearch(
    max_results=3, 
    topic="general", 
    search_depth="basic"
)
#duck duck go search
from langchain_community.tools import DuckDuckGoSearchRun

duck_search = DuckDuckGoSearchRun()

# add memory
from langgraph.checkpoint.memory import InMemorySaver

#create llm agent
from langchain.agents import create_agent
tools = [getweather, web_search, duck_search]
agent = create_agent(
    model = llm, 
    tools = tools, 
    checkpointer=InMemorySaver(),
    middleware=[
        LoggedSummarizationMiddleware(
            model=llm_sum, trigger=("fraction", 0.7), keep=("fraction", 0.20)
        )
    ],
    system_prompt="""
    You are a helpful chatbot assistant. Use getweather tool for answering weather related queries. Use web_search tool for anything that needs up to date information, including but not limited to news, finances, stocks, sports, and pop culture. Use web_search for any queries involving today or the current year (2026). Use duck_search when web_search is not available or returns no results. For normal chat conversation, don't use a tool, just answer on your own. 
"""
)

#create a thread ID
from uuid import uuid4
thread_id = str(uuid4())
thread_config = {"configurable":{"thread_id":thread_id }}

# Create the weather app messages
from langchain.messages import HumanMessage, SystemMessage
while True:
    messages = []
    user_input = input("What is your query? ")

    if user_input.lower() in ["bye", "exit"]:
        print("Bye!")
        exit()
    if user_input.lower().strip() == "/new":
        thread_id = str(uuid4())
        thread_config = {"configurable":{"thread_id":thread_id}}
        print("Welcome to the new session...")
        continue
    messages.append(HumanMessage(content=user_input))
    try:
        final_result = agent.invoke({"messages": messages}, thread_config)
            # agent is stateful - it actually looks at different states, takes a dictionary
        print(final_result["messages"][-1].content)
        print("Meta information: ", final_result["messages"][-1].usage_metadata)
    except (GroqRateLimitError, OPenAIRateLimitError):
        print(f"{llm.profile.name} rate limit has exceeded, please try again later")
    except Exception as e: 
        print(f"Error occured: {type(e).__name__}")
        print(f"Error details: {e}")

