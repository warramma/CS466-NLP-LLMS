
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")



# %%
from langchain.chat_models import init_chat_model
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
#model = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")


# %%
result = model.invoke("Hi")
print(result.content)



# %%
from langchain.tools import tool
import requests
url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def get_weather(location:str)->str:
    """take the location and return the current weather"""

    params={
        "q":location,
        "appid":OPENWEATHER_API_KEY
    }

    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except Exception as e:
        return f"Weather fetching is failed. {e}"


# %%
from langchain_tavily import TavilySearch 
primary_internet_search = TavilySearch(
    topic="news",
    search_depth="basic",
    max_results=5,
    time_range="week"
)


# %%
from langchain_community.tools import DuckDuckGoSearchResults
ddg_search = DuckDuckGoSearchResults()



# %%
available_tools = [get_weather,primary_internet_search,ddg_search]


# %%
prompt = """You are a helpful assistant. 
    For regular conversion use your own knowledge.
    For weather related query use getWeather tool. 
    For news, stocks, current affairs and internet search use primary_internet_search tool. 
    Use ddg_search tool when primary_internet_search tool is failed or not available
    """


# %%
from langgraph.checkpoint.memory import InMemorySaver

# %%
from langchain.agents import create_agent
agent = create_agent(
    model=model,
    #ToDo add checkpointer
    checkpointer=InMemorySaver(),
    tools=available_tools,
    system_prompt=prompt
)



# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from groq import RateLimitError


from uuid import uuid4 
#ToDo generate a thread id
thread_id = str(uuid4())
print(thread_id)

thread_configure  = {"configurable":{"thread_id":thread_id}}

# create an infinite loop
while True:
    messages = [] 
    user_query = input("User: ")
    if(user_query.lower().strip() in ["bye","exit"]):
        print("Bye!")
        exit()

    if(user_query.lower().strip() == '/new'):
        thread_id = str(uuid4())
        print("New thread ID: ",thread_id)
        thread_configure  = {"configurable":{"thread_id":thread_id}}
        continue

    messages.append(HumanMessage(content=user_query))

    try: 
        ##ToDo Add thread_configure
        result = agent.invoke({"messages":messages}, thread_configure)
        print("AI:", result["messages"][-1].content)
    except RateLimitError as Err:
        print(Err)
    except Exception as e:
        print(e)


