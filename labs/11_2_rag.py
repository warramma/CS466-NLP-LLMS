# %% [markdown]
# ## RAG Fundametals

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


model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
model_fallback1 = init_chat_model(model="qwen/qwen3.8-27b", model_provider="groq")


# %% [markdown]
# ## Get the RAG Documents

# %%
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests 

# %%
BASE_URL = "https://docs.langchain.com"
DOC_PATHS = [
    "oss/python/langchain/agents",
    "oss/python/deepagents/rag",
    "oss/python/langchain/tools",
    "oss/python/langchain/models",
    "oss/python/deepagents/retrieval",
    "oss/python/langchain/knowledge-base",
    "oss/python/langchain/middleware",
    "oss/python/deepagents/overview",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langgraph/overview",
    "oss/python/langgraph/quickstart",
]



# %%
   
## ToDO: Using langchain community PyPDFLoader package to load pdf documents 
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_folder = Path(".")
pdf_files = sorted(pdf_folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found. Add at least two PDF files to your working folder.")

docs_pdf = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(
        file_path=str(pdf_file),
        mode="single",
        pages_delimiter="\n\n"
    )
    for doc in loader.lazy_load():
        docs_pdf.append(doc)

print(f"Total PDF files: {len(pdf_files)}")
print(f"Total PDF documents: {len(docs_pdf)}")

# %% [markdown]
# ### Load all the documents

# %%
def loadDocument()->list[Document]:
    "Fetch and return list of documents"
    docs:list[Document]=[]
    for path in DOC_PATHS: 
        url = f"{BASE_URL}/{path}.md"
        try:
            response = requests.get(url,timeout=200)
            response.raise_for_status()
        except requests.RequestException:
            continue
        docs.append(
            Document(page_content = response.text, metadata={"source":f"{BASE_URL}/{path}"})
            )
    return docs


# %%
docs = loadDocument()
#print(docs[0])
print(len(docs))

# %% [markdown]
# ## split the document 

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

# %%
all_split_documents = splitter.split_documents(docs)
len(all_split_documents)

# %%
#print(docs_pdf[0].page_content)

# %%
## ToDo: Split pdf documents 
pdf_split = splitter.split_documents(docs_pdf)

print(f"Total PDF chunks: {len(pdf_split)}")
# %% [markdown]
# ## Load your embeddings model 

# %%
from langchain_huggingface import  HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# %%
result = embeddings.embed_query("Hi")
len(result)

# %% [markdown]
# ## Get a vectore store where you want to save the embeddings 

# %%
from langchain_chroma import Chroma
vectorstore = Chroma(
    collection_name="Example_collection",
    embedding_function=embeddings,
    persist_directory="./langchain_documentation_db"
)

# %% [markdown]
# ToDo: Add splitted documents to vectore store 
vectorstore.add_documents(all_split_documents)
print(f"{len(all_split_documents)} chunked indexed")
# %%
vectorstore.add_documents(all_split_documents)
print(f"{len(all_split_documents)} chunked indexed")

# %%
## ToDO: add pdf_split document 
pdf_ids = vectorstore.add_documents(pdf_split)
print(f"Indexed {len(pdf_ids)} PDF chunks")

# %% [markdown]
# ## test the vectore store 

# %%
vectorstore.similarity_search_with_score(query="How to create a langgraph agent?",k=4)

# %%
## test vectoresore 
vectorstore.similarity_search_with_score("What is attention mechanism in transformer architecture?",k=1)

# %% [markdown]
# ## Create a retriever inference 

# %%
retriever = vectorstore.as_retriever(search_kwargs={"k":1})

# %%
##test 
retriever.invoke(input="How to add MCP tools to Langgraph agent?")

# %%
## batch retriever 
retriever.batch(inputs=[
    "What is langchain agent?", "What is an MCP Server?"
])

# %% [markdown]
# ## Create retriver as a tool

# %%
from langchain.tools import tool
import json 
@tool
def rag_tool(query:str)->str:
    """Return context from vectore store based on user query"""
    docs = retriever.invoke(input=query)
    response = "\n\n".join(doc.page_content for doc in docs)
    return json.dumps(response)


# %%
rag_tool.invoke("What is Langchain agent?")

# %% [markdown]
# ## Create a langchain agent and add the rag_tool

# %%
from langchain.agents import create_agent 
agent = create_agent(
    model=model,
    tools=[rag_tool],
    system_prompt="""You are a helpful assistant. Use rag_tool for answering 
    any question related to langchain or langgraph, attention and transformer.
    For other questions you may answer of your own"""
)

# %%
## test the agent 
from langchain.messages import HumanMessage
resonse = agent.invoke({
    "messages":[
        HumanMessage(content="How to create a langgraph agent? Write in 5 bullets. What is AI? What is attention mechanism in Transformer?")
    ]
})

resonse["messages"][-1].pretty_print()


# %%
for m in resonse["messages"]:
    m.pretty_print()


