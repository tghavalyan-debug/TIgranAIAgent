import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

FROZEN_QUERIES = [
    "What was Tigran Ghavalyan's role and responsibilities during his internship at SoftAcademy LLC?",
    "How did Tigran contribute to his team while working as a web programmer?",
    "When did Tigran receive his three diplomas in regional and national Olympiads in Math and Informatics?",
    "At what age did he see his first major successes in competitive math and informatics?",
    "What did the horizontal bar in his room teach Tigran about his dreams and goals?",
    "How did a piece of metal in his bedroom influence his mindset towards challenges and achieving goals?",
    "Describe Tigran's volunteering experience at the Sevan Startup Summit 2024 and 2025.",
    "What kind of volunteering work did he do at the tech event by the lake?",
    "Why does Tigran believe Sorbonne Abu Dhabi is the ideal environment for his academic journey in Data Science?",
    "What specific features of SUAD appeal to his career goals in AI and research?"
]

STORES = {
    "baseline": "chromadb_store",
    "lab_small": "lab_small",
    "lab_large": "lab_large",
    "lab_no_overlap": "lab_no_overlap"
}

def run_evaluation():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Results will be a dict: {query: {store: distance}}
    results = {}

    print("📊 Starting Full Experiment Evaluation...")
    
    for store_name, path in STORES.items():
        print(f"\n📂 Evaluating Store: {store_name} ({path})")
        if not os.path.exists(path):
            print(f"⚠️ Warning: Store path {path} not found. Skipping.")
            continue
            
        vectorstore = Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
        
        for q in FROZEN_QUERIES:
            # We use k=4 as per Step 3 instructions
            # search_with_score returns (doc, distance)
            search_results = vectorstore.similarity_search_with_score(q, k=4)
            
            top_distance = search_results[0][1] if search_results else 999
            
            if q not in results:
                results[q] = {}
            results[q][store_name] = top_distance
            
            # Print a quick summary for the user to see progress
            print(f"  - Query: {q[:40]}... | Top Distance: {top_distance:.4f}")
            # Small sleep to avoid any lingering rate limits during search
            time.sleep(0.5)

    # Print a markdown-ready table for the report
    print("\n--- FINAL RESULTS TABLE (MARKDOWN) ---")
    header = "| Query | baseline | lab_small | lab_large | lab_no_overlap |"
    divider = "|-------|----------|-----------|-----------|----------------|"
    print(header)
    print(divider)
    
    for q, store_data in results.items():
        row = f"| {q[:30]}... "
        for name in STORES.keys():
            dist = store_data.get(name, "N/A")
            val = f"{dist:.4f}" if isinstance(dist, float) else dist
            row += f"| {val} "
        row += "|"
        print(row)

if __name__ == "__main__":
    run_evaluation()
