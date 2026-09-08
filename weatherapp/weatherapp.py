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

llms_with_tools = llm.bind_tools([getweather, web_search])

# Create the weather app messages
from langchain.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage(
        content=(
            "You are a helpful agent. Use the getweather tool for weather information, "
            "web_search for internet search/news, or both if a query requires multiple sources."
        )
    )
]

user_input = input("What is your query? ")
messages.append(HumanMessage(content=user_input))

tools_map = {getweather.name: getweather, web_search.name: web_search}
executed_tools = []

while True:
    aiMessage = llms_with_tools.invoke(messages)
    messages.append(aiMessage)
    
    if not aiMessage.tool_calls:
        final_response = aiMessage
        break
        
    for tool_call in aiMessage.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools_map:
            print(f"Executing tool: {tool_name} with args: {tool_call['args']}")
            tool_message = tools_map[tool_name].invoke(tool_call)
            messages.append(tool_message)
            executed_tools.append(tool_name)

print("\nExecuted Tool(s):", ", ".join(executed_tools) if executed_tools else "None")
print("Final response:\n", final_response.content)