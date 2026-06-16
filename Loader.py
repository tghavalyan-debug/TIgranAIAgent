import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

def ingest_data():
    # 1. Load documents from data/
    # Supporting .txt, .md, and .pdf
    print("📂 Loading documents from data/...")
    
    # Text and Markdown loader
    txt_loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    md_loader = DirectoryLoader('data/', glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    pdf_loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    docs.extend(txt_loader.load())
    docs.extend(md_loader.load())
    docs.extend(pdf_loader.load())
    
    if not docs:
        print("⚠️ No documents found in data/.")
        return

    print(f"📄 Loaded {len(docs)} documents.")

    # 2. Split into chunks
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} chunks.")

    # 3. Embed and store in ChromaDB
    persist_directory = "chromadb_store"
    
    # To avoid duplication on re-run, we clear the existing store
    if os.path.exists(persist_directory):
        print(f"🧹 Clearing existing store at {persist_directory}...")
        shutil.rmtree(persist_directory)

    print("🧠 Embedding and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ Successfully stored {len(chunks)} chunks in {persist_directory}.")
    return vectorstore

if __name__ == "__main__":
    ingest_data()
