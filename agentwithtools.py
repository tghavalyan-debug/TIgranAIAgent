import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tools import TOOLS

load_dotenv()

TOOL_MAP = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are a helpful assistant with access to tools. Follow these rules:
1. ALWAYS select a tool by reading its DESCRIPTION carefully. The description is the ONLY authority on what each tool does.
2. IGNORE the tool name — the description tells you the true purpose.
3. Use a tool when you need information or to compute something.
4. After getting tool results, provide a clear final answer.
5. If you don't have enough information, say so."""

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)


def get_bound_llm():
    return llm.bind_tools(TOOLS)


def extract_text(content):
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(texts)
    return str(content)


def run_agent(question, max_steps=5):
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    trajectory = []
    llm_with_tools = get_bound_llm()

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return extract_text(response.content), trajectory

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
            trajectory.append({"tool": tool_name, "args": tool_args, "result": str(result)[:200]})
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Max steps reached.", trajectory


def run_agent_with_history(question, history, max_steps=5):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        for h in history:
            messages.append(HumanMessage(content=h["user"]))
            messages.append(AIMessage(content=h["assistant"]))
    messages.append(HumanMessage(content=question))
    trajectory = []
    llm_with_tools = get_bound_llm()

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return extract_text(response.content), trajectory

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
            trajectory.append({"tool": tool_name, "args": tool_args, "result": str(result)[:200]})
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return "Max steps reached.", trajectory


if __name__ == "__main__":
    q = "What is 18% of 2450?"
    answer, trajectory = run_agent(q)
    print(f"Question: {q}")
    print(f"Answer: {answer}")
    print(f"Trajectory: {json.dumps(trajectory, indent=2)}")
