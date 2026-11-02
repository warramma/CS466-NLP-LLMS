# %% [markdown]
# # Langgraph with Prompt Chaining

# %%
# %% [markdown]
# ### Lets start wotking with LLM

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
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
# # Create the graph state 

# %%
from typing_extensions import TypedDict

class ProgrammingAssistant(TypedDict):
    user_query:str
    programming_language:str 

    code_generation: str 
    code_explanation:str 
    complexity_analysis:str
    code_review:str
    code_improved:str 
    final_report:str 



# %% [markdown]
# ## Create initial state: user_query and programming_language

# %%
initial_state = {
    "user_query":"""Write binary search code, which will return the index position if the 
    searched item is found, otherwise return -1""",
    
    "programming_language":"python"
    
}

# %% [markdown]
# ## Create code_generation node

# %%
def code_generation(state:ProgrammingAssistant)->ProgrammingAssistant:
    """Generate code based on user query and a given programming language"""

    prompt = f"""Generate code based on the {state['user_query']} in
            {state['programming_language']}

            requirements: 
             - generate just source code
        """

    response = model.invoke(prompt)

    return {"code_generation":response.content}

# %%
## test the code_generation node 
result = code_generation(initial_state)
print(result)


# %% [markdown]
# ## Create code_explanation node 

# %%
def code_explanation(state:ProgrammingAssistant)->ProgrammingAssistant:
    """Generate code explnation based on generated code and user query"""

    prompt = f"""Explain code based on the {state['user_query']} and generated code
            {state['code_generation']}

            requirements: 
             - Give at least 2 examples for code explanation
             - Alteast one for successful search
             - Atleast one for unsuccessful search 
        """

    response = model.invoke(prompt)

    return {"code_explanation":response.content}

# %% [markdown]
# ## Create complexity_analysis node 

# %%
def complexity_analysis(state:ProgrammingAssistant)->ProgrammingAssistant:
    """Do the complexity analysis based on generated code and user query"""

    prompt = f"""Perform complexity analysis based on the {state['user_query']} and generated code
            {state['code_generation']}

            requirements: 
             - Perform complexity analysis for best case with omega notation
             - Perform complexity analysis for worst case with big-O notation
             - Perform complexity analysis for average case with theta notation
        """

    response = model.invoke(prompt)

    return {"complexity_analysis":response.content}

# %% [markdown]
# ## Create final report node  

#ToDO: Add code_review node 
def code_review(state:ProgrammingAssistant)->ProgrammingAssistant:
    """review the code for correctness, bugs, readability, and how well it meets the stated requirements"""
    prompt = f"""Review the following generated code based on the user query and programming language.
        User query: {state['user_query']}
        Programming Language: {state['programming_language']}
        Generated Code: {state['code_generation']}
      Specifically, make sure to check edge cases including:
            1. An empty list input
            2. An unsuccessful search
    make sure to also discuss binary search prerequisite/requirement for sorted input.
"""
    response = model.invoke(prompt)
    return {"code_review":response.content}

#ToDO: Add code_improved node
def code_improved(state:ProgrammingAssistant)->ProgrammingAssistant:
    """	Produces an improved implementation using the original code and review feedback."""
    prompt = f"""Generate an improved version of the code that addresses the review findings.
            User Query: {state['user_query']}
            Programming Language: {state['programming_language']}
            Original Code:{state['code_generation']}

            Review Feedback:{state['code_review']}

            Requirements:
            - Preserve the core requirement: return the index when the target item is found and -1 otherwise.
            - Address all bugs, edge cases, and improvements raised in the code review.
            - Output source code ONLY.
            """
    response = model.invoke(prompt)
    return {"code_improved":response.content}

# %%
def final_report(state:ProgrammingAssistant)->ProgrammingAssistant:
    """Generate the final report based on all the previous states and user query"""

    prompt = f"""Generate a final report, which will include the following states, make the report in section format

             - {state['user_query']}
             - {state['programming_language']}
             - {state['code_generation']}
             - {state['code_explanation']}
             - {state['complexity_analysis']}
             - {state['code_review']}
             - {state['code_improved']}
        """

    response = model.invoke(prompt)

    return {"final_report":response.content}

# %% [markdown]
# ##

# %% [markdown]
# # Generate the Langgraph

# %%
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END

# %% [markdown]
# ## Create the graph

# %%
graph = StateGraph(ProgrammingAssistant)
graph.add_node(code_generation)
graph.add_node(code_explanation)
graph.add_node(complexity_analysis)
graph.add_node(code_review)
graph.add_node(code_improved)
graph.add_node(final_report)

#ToDo: Add code_review and code_improved nodes and coressponding edges 

graph.add_edge(START,"code_generation")
graph.add_edge("code_generation","code_explanation")
graph.add_edge("code_explanation","complexity_analysis")
graph.add_edge("complexity_analysis", "code_review")
graph.add_edge("code_review", "code_improved")
graph.add_edge("code_improved","final_report")
graph.add_edge("final_report",END)


graph = graph.compile()
graph

# %%
initial_state

# %%
try:
    response = graph.invoke(initial_state)
except Exception as e:
    print(f"Error: {e}")

# %%
print(response['final_report'])

# %%

#ToDo: Append the final report here  

final_report_text = response.get("final_report", "")
with open("final_report.md", "a", encoding="utf-8") as f:
    f.write("\n\n" + "=" * 50 + "\n")
    f.write(final_report_text)

print("Final report successfully appended to final_report.md")