import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

import time

def build_store(splitter, persist_directory):
    """
    Builds a Chroma vector store using the provided splitter and persistence directory.
    """
    # 1. Load documents from data/
    print("📂 Loading documents from data/...")
    
    txt_loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    md_loader = DirectoryLoader('data/', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    pdf_loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    docs.extend(txt_loader.load())
    docs.extend(md_loader.load())
    docs.extend(pdf_loader.load())
    
    if not docs:
        print("⚠️ No documents found in data/.")
        return None

    print(f"📄 Loaded {len(docs)} documents.")

    # 2. Split into chunks using the provided splitter
    print(f"✂️ Splitting documents into chunks using {splitter.__class__.__name__}...")
    chunks = splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks.")

    # 3. Embed and store in ChromaDB
    if os.path.exists(persist_directory):
        print(f"🧹 Clearing existing store at {persist_directory}...")
        shutil.rmtree(persist_directory)

    print(f"🧠 Embedding and storing in ChromaDB at {persist_directory}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Process in batches to avoid rate limits
    batch_size = 10
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"🚀 Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vectorstore.add_documents(batch)
        
        if i + batch_size < len(chunks):
            print("⏳ Waiting 65 seconds to avoid rate limits...")
            time.sleep(65)
    
    print(f"✅ Successfully stored {len(chunks)} chunks in {persist_directory}.")
    return vectorstore
 
if __name__ == "__main__":
    # Default behavior: replicate HW1 (500/50)
    hw1_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    build_store(hw1_splitter, "chromadb_store")
