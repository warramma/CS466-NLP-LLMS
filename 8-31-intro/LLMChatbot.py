import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GROQ_API_KEY"]== os.getenv("GROQ_API_KEY")

#model = init_chat_model(model="", model_provider="google-genai")
model = init_chat_model(model="openai/gpt-oss-120B", model_provider="groq")

from langchain.messages import HumanMessage, AIMessage, SystemMessage
conversations=[SystemMessage("You are a helpful assistant that always includes a pun in your reply.")]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["done"]:
        print("Terminating conversation...")
        exit()
    conversations.append(HumanMessage(content=user_input))
    response = model.invoke(conversations)
    conversations.append(AIMessage(content=response.content))
    print("AI: ", response.content)