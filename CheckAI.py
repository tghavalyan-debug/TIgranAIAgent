import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables from the .env file
load_dotenv()

# Verify the API key is being read correctly
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError(
        "Could not find GOOGLE_API_KEY. "
        "Make sure your .env file is in the same directory as this script "
        "and looks like: GOOGLE_API_KEY=your_actual_key_here"
    )
print("🚀 .env loaded successfully! Starting LangChain + Gemini Connection Test...\n")
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)
response = llm.invoke("hello gemini")
print(response.content)

