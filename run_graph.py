import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from graph_agent import run_agentic_rag, resume_after_approval, print_graph_structure

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

print_graph_structure()
print("Type 'quit' to exit, 'resume <thread>' to resume a paused conversation.")
print()

thread_id = "default"

while True:
    q = input("\n>>> ").strip()
    if not q:
        continue
    if q.lower() == "quit":
        break
    if q.lower().startswith("resume "):
        thread_id = q.split(" ", 1)[1].strip()
        print(f"Resuming thread: {thread_id}")
        continue

    thread_id = f"live_{thread_id}" if thread_id == "default" else thread_id
    result = run_agentic_rag(q, thread_id=thread_id, checkpointer=checkpointer)

    print(f"\n[Status] {result['status']}")
    print(f"[Retrievals] {result['retrieval_count']}")
    print(f"[Judge] {result.get('judge_reasoning', 'N/A')[:150]}")
    print(f"[Answer] {result['answer'][:300]}")

    if result["status"] == "awaiting_approval":
        choice = input("\nApprove? (y/n/edit): ").strip().lower()
        if choice == "y":
            final = resume_after_approval(result, approved=True)
            print(f"[Final] {final[:300]}")
        elif choice == "edit":
            feedback = input("Feedback/revision: ").strip()
            final = resume_after_approval(result, approved=False, feedback=feedback)
            print(f"[Final] {final[:300]}")
        else:
            final = resume_after_approval(result, approved=False, feedback="Rejected by user")
            print(f"[Rejected] {final[:300]}")
