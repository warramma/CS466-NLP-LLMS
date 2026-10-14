# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model


model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")


#print(f"LLM Model profile: {model.profile.get('max_input_tokens')}")


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



from langchain_community.tools import DuckDuckGoSearchResults
ddg_search = DuckDuckGoSearchResults()

# %%
ddg_search.invoke("US top news today")


available_tools = [get_weather,primary_internet_search,ddg_search]
#available_tools = [get_weather,ddg_search]

# %%
prompt = """You are a helpful assistant. 
    For regular conversion use your own knowledge.
    For weather related query use getWeather tool. 
    For news, stocks, current affairs and internet search use primary_internet_search tool. 
    Use ddg_search tool when primary_internet_search tool is failed or not available
    """


#Create agent 
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


class LoggedSummarizationMiddleware(SummarizationMiddleware):
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)

        if update is not None: 
            print("SUMMARIZATION happened\n")
            #print(update)

        return update

    def after_model(self, state, runtime):
        latest_message = state["messages"][-1]
        print(f"AI: {latest_message.content}")
        print(f"Meta data: {latest_message.usage_metadata}")

        return None
        # messages = state.get("messages", [])
        # if messages:
        #     latest_message = messages[-1]
        #     print(f"AI: {getattr(latest_message, 'content', '')}")
        #     print(f"Meta data: {getattr(latest_message, 'usage_metadata', None)}")
        # return None



agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    middleware=[
        LoggedSummarizationMiddleware(
            ## ToDd add llm model, replace and add fraction for trigger and keep to 80% and 30% repectively
            model=model, trigger=("fraction", 0.1), keep=("fraction", 0.3),
        ),
    ],
    tools=available_tools,
    system_prompt=prompt
)

# %% [markdown]
# ## add messages 

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from groq import RateLimitError


from uuid import uuid4 
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
        result = agent.invoke({"messages":messages},thread_configure)
        #print("AI:", result["messages"][-1].content)
        #print(f"Metadata: {result["messages"][-1].usage_metadata}")
    except RateLimitError as Err:
        print(Err)
    except Exception as e:
        print(e)


