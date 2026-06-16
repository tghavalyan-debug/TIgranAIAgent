import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

def inspect(query, persist_directory="chromadb_store", k=5):
    print(f"\n🔍 Inspecting Retrieval for: '{query}'")
    print(f"📂 Store: {persist_directory} | k: {k}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if not os.path.exists(persist_directory):
        print(f"❌ Error: Store at {persist_directory} not found.")
        return

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # search_with_score returns (Document, score)
    # Chroma scores are distances (lower is better/more similar)
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Chunk {i+1} [Distance: {score:.4f}] ---")
        print(f"Content: {doc.page_content[:300]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_query = "What did Tigran do during his internship?"
    inspect(test_query)
