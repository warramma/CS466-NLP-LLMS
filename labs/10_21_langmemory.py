

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model


model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")

# %% [markdown]
# # Create a tool node 

# %% [markdown]
# ## create weather tool node 

# %%
from langchain.tools import tool
import requests

url= "https://api.openweathermap.org/data/2.5/weather?"
@tool
def getWeather(location:str)->str:
    """Extract the weather information based on the city and return"""
    params={
        "q":location,
        "appid":OPENWEATHER_API_KEY
    }
    try:
        response = requests.get(url,params=params,timeout=10)
        return response.json()
    except Exception as e:
        return f"Fetching error. {e}"



# %% [markdown]
# ## internet search tool 

# %%
from langchain_tavily import TavilySearch

web_search_primary = TavilySearch()

# %%
tools = [getWeather, web_search_primary]
tool_names = {tool.name: tool for tool in tools}
tool_names

# %% [markdown]
# ## Create tool_node 

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END


# %% [markdown]
# ## Create an llm node
# 
# Start -> LLM_Node -> END

# %%
def llm_node(state:MessagesState)->MessagesState:
    """LLM_Node which will take the previous state and return the next state"""
    #currentState = state["messages"]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(
        [
            SystemMessage("You are a helpful assistant"),
            *state["messages"]
        ]
    )
    #print(response)
    return {"messages":[response]}


# %% [markdown]
# ## condition node 




# %% [markdown]
# ## Create the graph with memory

# %%
## built-in tool node
from langgraph.prebuilt import ToolNode, tools_condition

# %%
#ToDO create tool_node using prebuilt ToolNode
#ToDO #don't have to make it from scratch this time.
tool_node = ToolNode(tools)

# %%
from langgraph.checkpoint.memory import InMemorySaver
#ToDO: create checkpointer
#ToDO
checkpointer = InMemorySaver()

# %%
graph = StateGraph(MessagesState)
graph.add_node("llm_node",llm_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"llm_node")
graph.add_conditional_edges("llm_node",tools_condition,["tools",END])
graph.add_edge("tools","llm_node")

graph = graph.compile(checkpointer = checkpointer) #ToDo: add checkpointer




# %%
#ToDO:create configure with a thread id
#ToDO
configure = {"configurable":{"thread_id":1}}

# %%
from groq import RateLimitError as groqRateLimitError

# %%
try:
    response = graph.invoke({
    "messages":
    [
     HumanMessage(content="My name is Wardiyah and I live in Illinois, USA")
    ]
    },configure)
    print(response["messages"][-1])

    response = graph.invoke({
        "messages":
        [
         HumanMessage(content="Where do I live?")
        ]
},config=configure)
    print(response["messages"][-1])

#ToDO:add exceptopn for groqRateLimitError
#ToDO
except groqRateLimitError as limitErr:    
    print(f"Groq rate limit reached. Try again later. {limitErr}")
except Exception as e:
    print(f"Error: {e}")

# %%
states = graph.get_state(config=configure).values["messages"]

#ToDo: print each state 
#ToDO
for state in states:
    state.pretty_print()


