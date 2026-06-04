from langchain_community.document_loaders import TextLoader
import os

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
        loader = TextLoader(file_path, encoding=encoding)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except UnicodeDecodeError:
        print(f"Error: Could not decode file '{file_path}' with encoding '{encoding}'.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Joins it cleanly to 'data/privatedata.txt'
    file_path = os.path.join(script_dir, "data", "privatedata.txt")
    docs = load_text_file(file_path)

    if docs:
        print(f"Loaded {len(docs)} document(s).")
        print("First document content preview:")
        print(docs[0].page_content[:200])  # Show first 200 characters
        print("Metadata:", docs[0].metadata)

#from langchain_community.document_loaders import DirectoryLoader

#loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
#docs = loader.load()
#print(f"Loaded {len(docs)} text files.")