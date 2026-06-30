import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

persist_directory = "chromadb_store"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer or the information is not in the context, just say "I don't have information about that." 
Do not try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = rag_chain.invoke(question)
    return answer, context

def main():
    questions = [
        "Who is Tigran Ghavalyan and what was his internship about?",
        "What is Tigran's relationship with mathematics?",
        "What does the horizontal bar represent in Tigran's story?",
        "When was the Eiffel Tower built?"
    ]
    print("Starting RAG Chain Tests...\n")
    for i, q in enumerate(questions, 1):
        print(f"--- Question {i}: {q} ---")
        answer, context = answer_question(q)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
