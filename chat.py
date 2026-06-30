import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import TOOLS
from memory import ConversationMemory

load_dotenv()

TOOL_MAP = {t.name: t for t in TOOLS}


def extract_text(content):
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(texts)
    return str(content)


SYSTEM_PROMPT = """You are a helpful assistant with access to tools. Follow these rules:
1. Use a tool when you need information or to compute something.
2. Always use the correct tool based on the question.
3. After getting tool results, provide a clear final answer.
4. If you don't have enough information, say so.
5. For math, use the calculator tool.
6. For employee info, use get_employee or get_team.
7. For questions about Tigran or internal docs, use search_docs.
8. For external/world knowledge, use search_web."""

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def run_chat_turn(question, memory, max_steps=5):
    ctx = memory.context()
    history = memory.to_history()

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if ctx and ctx != "No prior conversation.":
        messages.append(SystemMessage(content=f"Prior conversation context:\n{ctx}"))
    for h in history:
        messages.append(HumanMessage(content=h["user"]))
        messages.append(AIMessage(content=h["assistant"]))
    messages.append(HumanMessage(content=question))
    trajectory = []

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            answer = extract_text(response.content)
            memory.add_turn(question, answer)
            return answer, trajectory, ctx

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result = f"Error: tool '{tool_name}' not found"
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as e:
                    result = f"Error: {e}"
            trajectory.append({"tool": tool_name, "args": tool_args})
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    answer = "Max steps reached."
    memory.add_turn(question, answer)
    return answer, trajectory, ctx


def run_conversation(turns):
    memory = ConversationMemory(max_buffer_turns=4)
    transcript = []
    for i, q in enumerate(turns):
        print(f"\n--- Turn {i+1}: {q} ---")
        answer, trajectory, ctx = run_chat_turn(q, memory)
        transcript.append({
            "turn": i + 1,
            "question": q,
            "answer": answer,
            "trajectory": trajectory,
            "assembled_context": ctx
        })
        print(f"Tools used: {[t['tool'] for t in trajectory] or '[none]'}")
        print(f"Answer: {answer[:150]}...")
    return transcript, memory


if __name__ == "__main__":
    turns = [
        "Who manages the Platform team?",
        "What is her email?",
        "What is 18% of 2450?",
        "What did Tigran do during his internship?",
    ]
    transcript, mem = run_conversation(turns)
    print("\n\n=== FINAL MEMORY CONTEXT ===")
    print(mem.context())
    print("\n\n=== TRANSCRIPT ===")
    print(json.dumps(transcript, indent=2))
