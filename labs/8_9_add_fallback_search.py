# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
#model = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")




# ## Add an external tool to the model

# %%
from langchain.tools import tool
import requests
url = "https://api.openweathermap.org/data/2.5/weather?"

@tool
def get_weather(location:str)->str:
    """take the location and return the current weather"""

    params={
        "q":location,
        "appid":WEATHER_API_KEY
    }

    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except Exception as e:
        return f"Weather fetching is failed. {e}"

# %% [markdown]
# ## Test the tool

# %%
#get_weather.invoke("Peoria, USA")

# %% [markdown]
# ## Add internet search tool

# %%
from langchain_tavily import TavilySearch 
primary_internet_search = TavilySearch(
    topic="news",
    search_depth="basic",
    max_results=5,
    time_range="week"
)

# %%
#primary_internet_search.invoke("US top news today")

# %% ToDO
# # Add fall back search mechanism

# ToDo create ddg_search%%
from langchain_community.tools import DuckDuckGoSearchResults
#ddg_search = DuckDuckGoSearchResults(output_format="list")
ddg_search = DuckDuckGoSearchResults()

# %%
ddg_search.invoke("US top news today")

# %% [markdown]
# ## Add list of tools to the llm

# combined internet tool
@tool
def internet_search(query:str)->dict:
    """Search the internet using Tavily first and then DuckDuckGo if needed."""
    try: 
        results = primary_internet_search.invoke(query)
        if results and not isinstance(results, Exception):
            return {"status": "success", "provider": "tavily", "results": results}
    except Exception: 
        pass
    try:
        results = ddg_search.invoke(query)
        if results:
            return {"status":"success", "provider":"duckduckgo", "results":results}
    except Exception as e:
        return {"status":"error", "message": f"All searches failed:{e}"}

# %%
available_tools = [get_weather,primary_internet_search, ddg_search]

#ToDo add ddg_search in the available_tools list


# %%
prompt = """You are a helpful assistant. 
    For regular conversion use your own knowledge.
    For weather related query use getWeather tool. 
    For news, stocks, current affairs and internet search use primary_internet_search tool.
    If primary_internet_search is not available or raises an error, use ddg_search tool instead. 
    """
# ToDo Modify the prompt
#Use ddg_search tool when primary_internet_search tool is failed or not available

# %%
from langchain.agents import create_agent
agent = create_agent(
    model=model,
    tools=available_tools,
    system_prompt=prompt
)

# %% [markdown]
# ## add messages 

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
messages = [] 

# %%
#user_query = input("User query: ")

# %%
#user_query = "US top news today"

# %%
user_query = "US top news today. Weather in chicago today. Write a summary on the advancement of AI after Transformers invented."

# %%
messages.append(HumanMessage(content=user_query))

# %%
#messages



# %%
from groq import RateLimitError
try:
    result = agent.invoke({"messages":messages})
    print(result)
    print(result["messages"][-1].content)
except RateLimitError as rateErr:
    print(rateErr)
except Exception as e:
    print(e)


#Do not remove below code checking for unit testing 

try:
    # Assumes ddg_search uses output_format="list".
    user_query = input()
    response = ddg_search.invoke(user_query)
    is_successful = isinstance(response, str) and len(response) > 0
except Exception as e:
    print(f"Fallback search failed: {e}")
    is_successful = False

if is_successful:
    print("ddg_search is successful.")
    #print(response)
else:
    print("ddg_search is failed.")