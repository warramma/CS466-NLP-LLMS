# %% [markdown]
# # Introducing MCP

#absolute path - C:\Users\ramma\Documents\YTLLMS\labs\Devs

# %%
import asyncio
import os 
from langchain.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()


import warnings 
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# %% [markdown]
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")

model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")
gemma = init_chat_model(model="gemma4:latest", model_provider="ollama")
llm = init_chat_model("gemini-3.6-flash", model_provider="google_genai")
#gpt = init_chat_model(model="gpt-5.4-mini",model_provider="openai")


# %% [markdown]
# ## Create an MCP Client 

# %%
from langchain_mcp_adapters.client import MultiServerMCPClient
devs_directory = "C:/Users/ramma/Documents/YTLLMS/labs/Devs"

# %%
client = MultiServerMCPClient({
    "open_meteo_weather":{
        "transport":"stdio",
        "command":"npx",
        "args":[
            "-y",
            "open-meteo-mcp-server"
        ]
    },
    "tavily_mcp": {   
        "transport": "stdio",    
        "command": "npx",    
        "args": [        
        "-y",        
        "tavily-mcp@latest"    
        ],    
        },
    "filesystem_mcp": { 
        "transport": "stdio",    
        "command": "npx",    
        "args": [        "-y",        
        "@modelcontextprotocol/server-filesystem",        
        devs_directory    ]
        }
})

# %%
async def main():
    tools = await client.get_tools()

# %%
    for tool in tools:
        print(tool.name)

# %% [markdown]
# ## Create langchain agent and add tools to the agent 

# %%
    from langchain.agents import create_agent 
    prompt = fprompt = f"""You are a helpful chatbot assistant.Use the available weather tools for weather and forecast questions.Use Tavily tools for web searches and current information.Use filesystem tools to list and read files inside:{devs_directory}Use tools when a request requires external information.For web searches, include source URLs.If a tool fails, report the failure instead of inventing a result.For general knowledge questions, answer directly."""
    agent = create_agent(
        model=llm,
        tools=tools, 
        system_prompt=prompt
    )

# %% [markdown]
# ## Test the agent 

# %%
    user_query = """Answer the following queries in brief. 
    What is the Weather in Dunlap, Peoria, IL zipcode: 61525 today? 
    What is AI? Write in 3 bullets only."""

# %%
    from langchain.messages import HumanMessage

    # %%
    # try:
    #     response = await agent.ainvoke({
    #         "messages":[
    #             HumanMessage(content=user_query)
    #         ]
    #     })
    # except Exception as e: 
    #     raise RuntimeError(f"Error: {e}")

# %%
    #print(response["messages"][-1].content)

    async def test_agent(user_query):
        print("\nQUERY:", user_query)
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=user_query)]
        })


        print("\nTOOL CALLS:")
        for message in response["messages"]:
            for call in getattr(message, "tool_calls", []) or []:
                print("Tool:", call["name"])
                print("Arguments:", call["args"])


        print("\nTOOL RESULTS:")
        for message in response["messages"]:
            if isinstance(message, ToolMessage):
                print("Tool:", message.name)
                print("Status:", getattr(message, "status", "not provided"))
                print(message.content)


        print("\nFINAL ANSWER:")
        print(response["messages"][-1].content)


        return response
    filesystem_response = await test_agent(
        f"List the files in {devs_directory}. "
        "Then read readme.md and report the student name "
        "and verification code exactly as written."
    )
    print(filesystem_response)
    tavily_response = await test_agent(
        "Use Tavily to search for recent news about artificial intelligence "
        "in higher education. Summarize two results and include each "
        "article's title, source URL, and publication date if available."
    )
    print(tavily_response)
    weather_response = await test_agent(
        "What is today's weather in Dunlap, Illinois, ZIP code 61525? "
        "Use the weather tools and state the temperature units."
    )
    print(weather_response)




if __name__ == "__main__":
    asyncio.run(main())


