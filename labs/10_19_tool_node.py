# %% [markdown]
# # Create a sample Langgraph apps

# %%
# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model


#model = init_chat_model(model="gemma4:latest", model_provider="ollama")
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
model_fallback2 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")
#gpt = init_chat_model(model="gpt-5.4-mini", model_provider="openai")
#print(f"LLM Model profile: {gpt.profile.get("max_input_tokens")}")

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

# %%
def tool_node(state:MessagesState)->MessagesState:
    """Return tool messages"""
    
    #ToDO: get the last message from the state variable
    #last_message = # ToDO
    last_message = state["messages"][-1]

    if last_message.tool_calls: 
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            selected_tool = tool_names[tool_name]
            tool_message = selected_tool.invoke(tool_call)
            tool_messages.append(tool_message)
   
        return {"messages":tool_messages}     

    return {"messages":f"No tool messages available"}

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

# %%
def should_continue(state:MessagesState)->str:
    """Return the tool_node or END node"""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"
    
    #ToDO: Add the default return
    # ToDO
    return END

# %%
# Test the tool node
initial_state = {
    "messages":[
        HumanMessage(content="Weather in Peoria")
    ]
}

# %%
response = llm_node(initial_state)
response

# %% [markdown]
# ## Create the graph

# %%
graph = StateGraph(MessagesState)
graph.add_node("llm_node",llm_node)
graph.add_node("tool_node",tool_node)

graph.add_edge(START,"llm_node")

# TODO Complete Adding conditional edge 
#graph.add_conditional_edges()
graph.add_conditional_edges(
    "llm_node", should_continue,["tool_node", END]
)

graph.add_edge("tool_node","llm_node")

graph = graph.compile()
graph

# %%
try:
    response = graph.invoke({
    "messages":
    [
     HumanMessage(content="Top news today in USA")
    ]
})
except Exception as e:
    print(f"Error: {e}")

# %%
for m in response["messages"]:
    m.pretty_print()

# %%



