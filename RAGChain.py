import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def run_rag_chain():
    # 1. Load the vectorstore
    persist_directory = "chromadb_store"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # 3. Create the prompt with hallucination guard
    template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer or the information is not in the context, just say "I don't have information about that." 
    Do not try to make up an answer.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Build the LCEL chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Run test questions
    questions = [
        # Questions from data
        "Who is Tigran Ghavalyan and what was his internship about?",
        "What is Tigran's relationship with mathematics?",
        "What does the horizontal bar represent in Tigran's story?",
        # Question not in data
        "When was the Eiffel Tower built?"
    ]

    print(" Starting RAG Chain Tests...\n")
    for i, q in enumerate(questions, 1):
        print(f"--- Question {i}: {q} ---")
        response = rag_chain.invoke(q)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    run_rag_chain()
