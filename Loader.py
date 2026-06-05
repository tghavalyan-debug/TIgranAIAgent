import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings



def load_text_file(file_path: str, encoding: str = "utf-8"):
    """
    Loads a text file using LangChain's TextLoader.

    Args:
        file_path (str): Path to the .txt file.
        encoding (str): File encoding (default: utf-8).

    Returns:
        list: List of Document objects containing text and metadata.
    """
    try:
        loader = DirectoryLoader(
    "data/",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except UnicodeDecodeError:
        print(f"Error: Could not decode file '{file_path}' with encoding '{encoding}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def build_store(documents):

    splitter = CharacterTextSplitter(chunk_size=200, separator=".")

    chunks = splitter.split_documents(documents)

    load_dotenv()

    for chunk in chunks:
        print(len(chunk.page_content))


    embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004")

    db = Chroma.from_documents(
      documents=chunks,
      embedding=embeddings,
      persist_directory="data/chroma"
)



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Joins it cleanly to 'data/privatedata.txt'
    file_path = os.path.join(script_dir, "data", "privatedata.txt")
    print(file_path)

    docs = load_text_file(file_path)


    if docs:
        print(f"Loaded {len(docs)} document(s).")
        print("First document content preview:")
        print(docs[0].page_content[:200])  # Show first 200 characters
        print("Metadata:", docs[0].metadata)

    build_store(docs)

