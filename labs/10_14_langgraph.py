# %% [markdown]
# # Create a sample Langgraph apps

# %%
# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model


#model = init_chat_model(model="gemma4:latest", model_provider="ollama")
#model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")



# %% [markdown]
# ## Create an llm node
# 
# Start -> LLM_Node -> END

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END

# %%
def llm_node(state:MessagesState)->MessagesState:
    """LLM_Node which will take the previous state and return the next state"""
    #currentState = state["messages"]

    response = model.invoke(
        [
            SystemMessage("You are a helpful assistant"),
            *state["messages"]
        ]
    )

    #ToDO: return the respone in a list format with a dictionary keyword "messages"
    return {"messages": [response]}


# %%
## Test the llm_node 
initial_state ={
    "messages":
    [
    HumanMessage(content="What is AI?")
    ]
}


# %%
response = llm_node(initial_state)
print(response)

# %% [markdown]
# ## Create the graph

# %%
graph = StateGraph(MessagesState)
graph.add_node("llm_node",llm_node)

graph.add_edge(START,"llm_node")
graph.add_edge("llm_node",END)

#ToDo: Compile the graph and assign to graph
graph = graph.compile()
#ToDo
print(graph)


# %%
try:
    #ToDo: Invoke the graph in the format of {"messages":[list of messages]} 
    response = graph.invoke({
        "messages":[
            HumanMessage(content="What is a metereologist? Give the answer in 3 bullet points.")
        ]
    })
except Exception as e:
    print(f"Error: {e}")

# %%
for m in response["messages"]:
    m.pretty_print()

# %%



