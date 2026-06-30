import os
import sqlite3
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
VECTORSTORE = Chroma(persist_directory="chromadb_store", embedding_function=EMBEDDINGS)
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 5})
LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

MAX_RETRIEVALS = 2


class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    retrieval_count: int
    messages: list
    decision_reasoning: str
    judge_reasoning: str


DECIDE_TEMPLATE = """You are an agent that decides whether a question can be answered from the existing internal document store or needs external knowledge.

Available internal documents contain information about: Tigran Ghavalyan (internship at SoftAcademy LLC, mathematics achievements, olympiad diplomas, horizontal bar story, Sevan Startup Summit volunteering, Sorbonne Abu Dhabi goals, personal essays, recommendation letters).

Rules:
- If the question is about Tigran Ghavalyan, his experiences, education, volunteering, or anything in the internal docs -> say "retrieve"
- If the question is about facts not related to Tigran (world knowledge, current events, math calculations, general facts) -> say "answer_directly"
- If unsure, say "retrieve"

Output ONLY:
retrieve <reasoning>
OR
answer_directly <reasoning>

Question: {question}

Decision:"""

ANSWER_TEMPLATE = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer or the information is not in the context, just say "I don't have information about that."
Do not try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

JUDGE_TEMPLATE = """You are a judge evaluating a RAG answer. Rate whether the answer is satisfactory.

Criteria:
- The answer must be faithful to the context (no hallucination)
- The answer must actually answer the question (be complete)
- If the answer says "I don't have information" or seems insufficient -> should retry

Output ONLY:
satisfactory <reasoning>
OR
retry <reasoning>

Question: {question}
Context: {context}
Answer: {answer}

Judgment:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def decide_node(state: GraphState) -> dict:
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(DECIDE_TEMPLATE)
    chain = prompt | LLM
    result = chain.invoke({"question": question})
    decision_text = result.content if hasattr(result, "content") else str(result)
    if isinstance(decision_text, list):
        texts = [p.get("text", "") for p in decision_text if isinstance(p, dict) and p.get("type") == "text"]
        decision_text = " ".join(texts)
    decision_text = decision_text.strip().lower()
    if decision_text.startswith("retrieve"):
        return {"decision_reasoning": decision_text}
    return {"decision_reasoning": decision_text}


def decide_router(state: GraphState) -> Literal["retrieve", "answer_directly"]:
    decision = state.get("decision_reasoning", "")
    if decision.startswith("retrieve"):
        return "retrieve"
    return "answer_directly"


def retrieve_node(state: GraphState) -> dict:
    question = state["question"]
    query = question
    if state.get("judge_reasoning") and "retry" in state.get("judge_reasoning", "").lower():
        query = f"{question} {state['judge_reasoning']}"
    docs = RETRIEVER.invoke(query)
    context = format_docs(docs)
    return {
        "context": context,
        "retrieval_count": state.get("retrieval_count", 0) + 1,
    }


def answer_node(state: GraphState) -> dict:
    question = state["question"]
    context = state.get("context", "No context retrieved.")
    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    chain = prompt | LLM
    result = chain.invoke({"question": question, "context": context})
    answer_text = result.content if hasattr(result, "content") else str(result)
    if isinstance(answer_text, list):
        texts = [p.get("text", "") for p in answer_text if isinstance(p, dict) and p.get("type") == "text"]
        answer_text = " ".join(texts)
    return {"answer": answer_text}


def answer_directly_node(state: GraphState) -> dict:
    question = state["question"]
    msg = LLM.invoke(f"Answer this question concisely. If you don't know, say so.\n\nQuestion: {question}")
    answer_text = msg.content if hasattr(msg, "content") else str(msg)
    if isinstance(answer_text, list):
        texts = [p.get("text", "") for p in answer_text if isinstance(p, dict) and p.get("type") == "text"]
        answer_text = " ".join(texts)
    return {"answer": answer_text, "context": "(answered directly, no retrieval needed)"}


def judge_node(state: GraphState) -> dict:
    question = state["question"]
    context = state.get("context", "")
    answer = state.get("answer", "")
    prompt = ChatPromptTemplate.from_template(JUDGE_TEMPLATE)
    chain = prompt | LLM
    result = chain.invoke({"question": question, "context": context, "answer": answer})
    judgment_text = result.content if hasattr(result, "content") else str(result)
    if isinstance(judgment_text, list):
        texts = [p.get("text", "") for p in judgment_text if isinstance(p, dict) and p.get("type") == "text"]
        judgment_text = " ".join(texts)
    return {"judge_reasoning": judgment_text.strip().lower()}


def judge_router(state: GraphState) -> Literal["human_approval", "retrieve"]:
    judgment = state.get("judge_reasoning", "")
    retrieval_count = state.get("retrieval_count", 0)
    if "retry" in judgment and retrieval_count < MAX_RETRIEVALS:
        return "retrieve"
    return "human_approval"


def human_approval_node(state: GraphState) -> dict:
    return {}


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("decide", decide_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("answer_directly", answer_directly_node)
    graph.add_node("judge", judge_node)
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("decide")
    graph.add_conditional_edges("decide", decide_router)
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "judge")
    graph.add_edge("answer_directly", "human_approval")
    graph.add_conditional_edges("judge", judge_router)
    graph.add_edge("human_approval", END)

    return graph


def compile_agent(checkpointer=None):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)


def run_agentic_rag(question, thread_id="default", checkpointer=None):
    app = compile_agent(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    for event in app.stream({"question": question, "retrieval_count": 0}, config):
        node_name = list(event.keys())[0]
        if node_name == "human_approval":
            state = app.get_state(config)
            return {
                "status": "awaiting_approval",
                "question": question,
                "answer": state.values.get("answer", ""),
                "context": state.values.get("context", ""),
                "retrieval_count": state.values.get("retrieval_count", 0),
                "judge_reasoning": state.values.get("judge_reasoning", ""),
                "config": config,
                "app": app,
            }
    state = app.get_state(config)
    return {
        "status": "complete",
        "question": question,
        "answer": state.values.get("answer", ""),
        "context": state.values.get("context", ""),
        "retrieval_count": state.values.get("retrieval_count", 0),
        "app": app,
        "config": config,
    }


def resume_after_approval(result, approved=True, feedback=""):
    app = result["app"]
    config = result["config"]
    if approved:
        app.update_state(config, {"answer": f"{result['answer']}\n\n[Approved by human]"})
    else:
        app.update_state(config, {"answer": f"[Human rejected. Feedback: {feedback}]"})
    for event in app.stream(None, config):
        pass
    state = app.get_state(config)
    return state.values.get("answer", "")


def print_graph_structure():
    graph = build_graph()
    print("=== Agentic RAG Graph Structure ===")
    for node_name in graph.nodes:
        print(f"  Node: {node_name}")
    print(f"  Entry: decide")
    print(f"  Edges:")
    print(f"    decide -> retrieve (if about Tigran)")
    print(f"    decide -> answer_directly (if world knowledge)")
    print(f"    retrieve -> answer -> judge")
    print(f"    answer_directly -> human_approval")
    print(f"    judge -> retrieve (if retry needed, max {MAX_RETRIEVALS}x)")
    print(f"    judge -> human_approval (if satisfactory or max retries)")
    print(f"    human_approval -> END (checkpoint, waits for human)")
    print()


if __name__ == "__main__":
    print_graph_structure()

    import sqlite3
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    queries = [
        "What did Tigran do during his internship at SoftAcademy?",
        "What is the capital of France?",
    ]
    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUESTION: {q}")
        print(f"{'='*60}")
        result = run_agentic_rag(q, thread_id=f"test_{q[:10]}", checkpointer=checkpointer)
        print(f"Status: {result['status']}")
        print(f"Answer: {result['answer'][:200]}")
        print(f"Retrievals: {result['retrieval_count']}")
        if result.get("judge_reasoning"):
            print(f"Judge: {result['judge_reasoning'][:100]}")
