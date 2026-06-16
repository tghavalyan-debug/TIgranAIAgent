import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

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

# 1. Test Chat Model
print("--- Testing Chat Model ---")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
response = llm.invoke("What is the capital of France?")
print(f"Question: What is the capital of France?\nResponse: {response.content}\n")

# 2. Test Embeddings
print("--- Testing Embeddings ---")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

sentences = [
    "The cat is sitting on the mat.",
    "A feline is resting on a rug.",
    "The weather is very sunny today."
]

vectors = embeddings_model.embed_documents(sentences)

if len(vectors) == 3 and all(isinstance(v, list) for v in vectors):
    print(f"Successfully generated {len(vectors)} vectors.")
else:
    print("Failed to generate vectors correctly.")

# 3. Compute Cosine Similarity
print("--- Computing Cosine Similarity ---")
# sklearn.metrics.pairwise.cosine_similarity expects 2D arrays
sim_12 = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
sim_13 = cosine_similarity([vectors[0]], [vectors[2]])[0][0]

print(f"Similarity (Cat on mat vs Feline on rug): {sim_12:.4f}")
print(f"Similarity (Cat on mat vs Sunny weather): {sim_13:.4f}")

similarity_check = sim_12 > sim_13
print(f"Similarity check (sim_12 > sim_13): {similarity_check}")

if similarity_check:
    print("\n✅ Step 4 Verified: Vectors generated and similarity check passed!")
else:
    print("\n❌ Step 4 Failed: Similarity check did not pass.")

