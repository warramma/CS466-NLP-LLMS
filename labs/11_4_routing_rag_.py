# %% [markdown]
# # Langraph routing and Orchestration
# ## Research Assistant  

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
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


# %% [markdown]
# ## Create  Research Assistant state 

# %%
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal 

class ResearchAssistant(TypedDict):
    user_query:str
    routing:list[str]

    weather_generation:str
    web_generation:str
    # TODO: Add RAG_generation state
    synthesizer:str
    RAG_generation:str

# %%
initial_state = {
    "user_query":"""Weather in Peoria. Top news today in USA. What is Langgraph? """, 
}

# %% [markdown]
# ## create routing node 

# %%
class RoutingDecision(BaseModel):
    routes:list[Literal['web','weather', 'RAG']] = Field(description="""Return web it is an current affairs, news, internet 
    search related queries. Return weather if it is weather related query. Return RAG if it is agent, 
    langchain and langgraph related query.""")

# %%
model_routing = model.with_structured_output(RoutingDecision)

# %%
from langchain.messages import HumanMessage, SystemMessage

# %%
model_routing.invoke([
SystemMessage(content="You are a helpful router"),
HumanMessage(content="What is a langgraph?")
])

# %%
def routing(state:ResearchAssistant)->ResearchAssistant:
    """Return a route or list of routes based on the user query"""
    
    result = model_routing.invoke([
            SystemMessage(content="You are a helpful router"),
            HumanMessage(content=state["user_query"])
            ]
            )
    #print(result.routes)
    return {"routing":result.routes}


# %%
routing(initial_state)

# %% [markdown]
# ## Create weather_generation node 

# %%
#Define weather tool 

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


# %%
class Location(BaseModel):
    location:str = Field(description="Extract the location from the query")

# %%
model_location = model.with_structured_output(Location)

# %%
#define weather node 

def weather_generation(state:ResearchAssistant)->ResearchAssistant:
    """Return weather information based on location given"""
    result = model_location.invoke([
            SystemMessage(content="You are a helpful parser"),
            HumanMessage(content=state["user_query"])
        ])
    weather = getWeather.invoke(result.location)

    return {"weather_generation":weather}
    
    

# %%
weather_generation(initial_state)

# %% [markdown]
# ## web_generation node 

# %%
from langchain_tavily import TavilySearch
web_search = TavilySearch() 

# %%
#web_search.invoke({"query":'Todays top news in USA'})

# %%
def web_generation(state:ResearchAssistant)->ResearchAssistant:
    """Generate web search result based on the user query""" 
    result = web_search.invoke({"query":state["user_query"]})

    return {"web_generation":result}

# %%
## test the node 
web_generation(initial_state)

# %% [markdown]
# ## RAG_generation node

# %%
##TODO
##Get embedding model
#Creating embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True,
            'batch_size':32
                   },
    
)

# %%
##TODO
## Get a vectore store 
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="Example_collection",
    embedding_function=embeddings,
    persist_directory="./langchain_documentation_db",
)

# %%
#vectorstore.similarity_search("RAG orhcetration with langchain")

# %%
## Reteiever 
retriever = vector_store.as_retriever(search_kwargs={"k":1})

# %%
from langchain.tools import tool
import json 
@tool
##TODO: define rag_tool
def rag_tool(query: str) -> str:
    """Retrieve context about LangChain, LangGraph, or agents."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) or "No passages retrieved."

# %%
## Create agent with rag_tool
from langchain.agents import create_agent 
RAG_agent = create_agent(
    model=model,
    tools=[rag_tool],
    system_prompt="""You are a helpful assistant. Use rag_tool for answering 
    any question related to langchain or langgraph, attention and transformer.
    For other questions you may answer of your own"""
)

# %%
## create RAG_generation node 
from langchain.messages import HumanMessage
def RAG_generation(state:ResearchAssistant)->ResearchAssistant:
    """Based on the user query use rag_tool to generate context and return"""
    ##TODO: complete RAG_generation node
    result = RAG_agent.invoke({
        "messages": [HumanMessage(content=state["user_query"])]
    })
    return {"RAG_generation": result["messages"][-1].text}


# %%
## test RAG_generation node
#RAG_generation(initial_state)

# %% [markdown]
# ## select specialist 

# %%
from langgraph.graph import StateGraph, START, END

# %%
def selectSpecialist(state: ResearchAssistant) -> list[str]:
    node_for_route = {
        "weather": "weather_generation",
        "web": "web_generation",
        "RAG": "RAG_generation",
    }


    return [node_for_route[route] for route in dict.fromkeys(state["routing"])] or [
        "synthesizer"
    ]

# %% [markdown]
# ## create synthesizer node 

# %%
def synthesizer(state:ResearchAssistant)->ResearchAssistant:
    """Generate final answer based on the evidence and user query"""
    evidence = [] 
    
    if state.get("weather_generation"):
        evidence.append(f"Weather evidence: \n\n {state['weather_generation']}")
    if state.get("web_generation"):
        evidence.append(f"Web evidence: \n\n {state['web_generation']}")
    if state.get("RAG_generation"):
        evidence.append(f"RAG evidence: \n\n {state['RAG_generation']}")
    

    response = model.invoke([
        SystemMessage(content=f"""You are a final synthesizer. Based on the user query: {state['user_query']}
            and information from the specialist node, generate the final answer. Evidence: {evidence}. 
            In case of no evidence, generate answer of your own.             
    """)
    ])
    
    return {"synthesizer":response.content}

# %% [markdown]
# # Build the graph 

# %%
graph = StateGraph(ResearchAssistant)

#Nodes 
graph.add_node(routing)
graph.add_node(weather_generation)
graph.add_node(web_generation)
graph.add_node(RAG_generation)
graph.add_node(synthesizer)
 

#edges 

graph.add_edge(START,"routing")
graph.add_conditional_edges("routing",selectSpecialist, ["weather_generation","web_generation","RAG_generation","synthesizer"])
graph.add_edge("weather_generation","synthesizer")
graph.add_edge("web_generation","synthesizer")
graph.add_edge("RAG_generation","synthesizer")
graph.add_edge("synthesizer",END)

graph = graph.compile()
graph 
               

# %% [markdown]
# ## Test the graph 

# %%
initial_state

# %%

try:
    response = graph.invoke(initial_state)
    print(response["synthesizer"])
except Exception as e: 
    print(f"Error. {e}")


