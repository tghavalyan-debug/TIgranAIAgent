import json
import time
from agentwithtools import run_agent

ROUTING_QUERIES = [
    {
        "query": "What is 18% of 2450?",
        "expected_tool": "calculator",
        "description": "Simple math"
    },
    {
        "query": "What is the square root of 144?",
        "expected_tool": "calculator",
        "description": "Math with sqrt"
    },
    {
        "query": "What is Sarah Kim's email address?",
        "expected_tool": "get_employee",
        "description": "Employee lookup by name for email"
    },
    {
        "query": "Who works on the Platform team?",
        "expected_tool": "get_team",
        "description": "Team member listing"
    },
    {
        "query": "What did Tigran do during his internship at SoftAcademy?",
        "expected_tool": "search_docs",
        "description": "Internal doc about Tigran"
    },
    {
        "query": "When was the Eiffel Tower built?",
        "expected_tool": "search_web",
        "description": "External world knowledge"
    },
    {
        "query": "What is Tigran's educational background and volunteering experience?",
        "expected_tool": "search_docs",
        "description": "OVERLAP: asks about Tigran's education and volunteering (internal docs)"
    },
    {
        "query": "What is the capital of France?",
        "expected_tool": "search_web",
        "description": "OVERLAP: simple world fact - could be in docs or web"
    },
]


def evaluate_routing(queries, label):
    correct = 0
    results = []
    for i, item in enumerate(queries):
        q = item["query"]
        expected = item["expected_tool"]
        answer, trajectory = run_agent(q, max_steps=3)
        chosen = trajectory[0]["tool"] if trajectory else "none"
        time.sleep(4)
        is_correct = chosen == expected
        if is_correct:
            correct += 1
        results.append({
            "query": q,
            "expected": expected,
            "chosen": chosen,
            "correct": is_correct,
        })
        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] Q{i+1}: expected={expected}, got={chosen}")
    accuracy = correct / len(queries) if queries else 0
    print(f"\n{label}: {accuracy:.2f} ({correct}/{len(queries)})")
    return accuracy, results


def main():
    print("=" * 60)
    print("Routing Evaluation -- Baseline")
    print("=" * 60)
    baseline_acc, baseline_results = evaluate_routing(ROUTING_QUERIES, "Baseline accuracy")

    from tools import search_docs, search_web, calculator, get_employee, get_team
    orig = {
        "docs": search_docs.description,
        "web": search_web.description,
        "calc": calculator.description,
        "emp": get_employee.description,
        "team": get_team.description,
    }

    print("\n" + "=" * 60)
    print("Broken descriptions -- search_docs and search_web swapped")
    print("=" * 60)
    search_docs.description = "Search the public internet for current events, world facts, news, and general knowledge. Use this for world facts like capitals, history, science."
    search_web.description = "Search internal documents about Tigran Ghavalyan (internship, education, volunteering, personal life). Use this for questions about Tigran."
    get_employee.description = "List ALL members of a team by team name. Returns everyone on that team."
    get_team.description = "Look up a single person by name. Returns their role, team, and email."
    broken_acc, broken_results = evaluate_routing(ROUTING_QUERIES, "Broken accuracy")

    print("\n" + "=" * 60)
    print("Fixed descriptions -- sharpened for disambiguation")
    print("=" * 60)
    search_docs.description = "ONLY for questions about Tigran Ghavalyan (internship, education, volunteering, personal stories). Do NOT use for world facts, history, capitals, or general knowledge."
    search_web.description = "ONLY for external/world knowledge: capitals, history, science, current events, weather, famous people. Do NOT use for questions about Tigran."
    get_employee.description = "Look up ONE employee by their PERSON NAME (e.g., 'Sarah Kim'). Returns role, team, email. Do NOT use for team listings."
    get_team.description = "List ALL members of a team by TEAM NAME (e.g., 'Platform', 'Product', 'AI/ML'). Do NOT use for individual people lookups."
    fixed_acc, fixed_results = evaluate_routing(ROUTING_QUERIES, "Fixed accuracy")

    for name, val in orig.items():
        {"docs": search_docs, "web": search_web, "calc": calculator, "emp": get_employee, "team": get_team}[name].description = val

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline accuracy:  {baseline_acc:.2f}")
    print(f"Broken accuracy:    {broken_acc:.2f}")
    print(f"Fixed accuracy:     {fixed_acc:.2f}")
    arrow = " -> "
    print(f"\nBaseline{arrow}Broken:  {baseline_acc:.2f}{arrow}{broken_acc:.2f} (delta: {broken_acc - baseline_acc:+.2f})")
    print(f"Broken{arrow}Fixed:     {broken_acc:.2f}{arrow}{fixed_acc:.2f} (delta: {fixed_acc - broken_acc:+.2f})")

    broken_failing = [r for r in broken_results if not r["correct"]]
    if broken_failing:
        print(f"\nFailing cases with broken descriptions:")
        for r in broken_failing:
            print(f"  - Q{r['query']}")
            print(f"    Expected: {r['expected']}, Got: {r['chosen']}")

    return baseline_acc, broken_acc, fixed_acc


if __name__ == "__main__":
    main()
