import os
import requests
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
Weather_API_Key = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain.chat_models import init_chat_model
llm = init_chat_model(model="openai/gpt-oss-120B", model_provider="groq")

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

#create llm agent
from langchain.agents import create_agent
tools = [getweather, web_search]
agent = create_agent(
    model = llm, 
    tools = tools, 
    system_prompt="""
    You are a helpful chatbot assistant. Use getweather tool for answering weather related queries. Use web_search tool for recent event, finance, news, pop culture and internet search. For normal chat conversation, don't use a tool, just answer on your own. 
"""
)

# Create the weather app messages
from langchain.messages import HumanMessage, SystemMessage
messages = []
user_input = input("What is your query? ")
messages.append(HumanMessage(content=user_input))

final_result = agent.invoke({"messages": messages})


# agent is stateful - it actually looks at different states, takes a dictionary

print(final_result)