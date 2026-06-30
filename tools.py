import os
import math
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

EMPLOYEES = {
    "sarah kim": {"name": "Sarah Kim", "role": "Engineering Manager", "team": "Platform", "email": "sarah.kim@company.com"},
    "james park": {"name": "James Park", "role": "Senior Engineer", "team": "Platform", "email": "james.park@company.com"},
    "lisa chen": {"name": "Lisa Chen", "role": "Product Manager", "team": "Product", "email": "lisa.chen@company.com"},
    "mike johnson": {"name": "Mike Johnson", "role": "Data Scientist", "team": "AI/ML", "email": "mike.johnson@company.com"},
    "anna novak": {"name": "Anna Novak", "role": "Team Lead", "team": "AI/ML", "email": "anna.novak@company.com"},
    "david wang": {"name": "David Wang", "role": "Backend Developer", "team": "Platform", "email": "david.wang@company.com"},
}

TEAMS = {
    "platform": [
        {"name": "Sarah Kim", "role": "Engineering Manager"},
        {"name": "James Park", "role": "Senior Engineer"},
        {"name": "David Wang", "role": "Backend Developer"},
    ],
    "product": [
        {"name": "Lisa Chen", "role": "Product Manager"},
    ],
    "ai/ml": [
        {"name": "Mike Johnson", "role": "Data Scientist"},
        {"name": "Anna Novak", "role": "Team Lead"},
    ],
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use standard math operators (+, -, *, /, **, %). Supports parentheses. Examples: '18% of 2450' should be written as '0.18 * 2450', 'sqrt(144)' for square root."""
    allowed = set("0123456789.+-*/%()[] ")
    if any(c not in allowed for c in expression):
        try:
            result = eval(expression, {"__builtins__": None}, {"sqrt": math.sqrt, "pow": pow, "pi": math.pi})
            return str(result)
        except:
            return f"Error evaluating: {expression}"
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_employee(name: str) -> str:
    """Look up an employee by name (full or partial). Returns their role, team, and email. Use this when someone asks about a person's details like 'who is X', 'what is X's email', 'who manages Y'."""
    name_lower = name.lower().strip()
    if name_lower in EMPLOYEES:
        e = EMPLOYEES[name_lower]
        return f"{e['name']} — Role: {e['role']}, Team: {e['team']}, Email: {e['email']}"
    for key, e in EMPLOYEES.items():
        if name_lower in key:
            return f"{e['name']} — Role: {e['role']}, Team: {e['team']}, Email: {e['email']}"
    return f"No employee found matching '{name}'."


@tool
def get_team(team_name: str) -> str:
    """List all members of a team. Teams: Platform, Product, AI/ML. Use this when someone asks 'who works on X team', 'list the X team', 'who is on the X team'."""
    team_lower = team_name.lower().strip()
    if team_lower in TEAMS:
        members = TEAMS[team_lower]
        lines = [f"Team: {team_name.title()}", f"Members ({len(members)}):"]
        for m in members:
            lines.append(f"  - {m['name']} ({m['role']})")
        return "\n".join(lines)
    return f"Team '{team_name}' not found. Available teams: Platform, Product, AI/ML."


@tool
def search_docs(query: str) -> str:
    """Search internal documentation and knowledge base about Tigran Ghavalyan. Use this for questions about Tigran's background, internship, education, achievements, volunteering, or personal stories. For any questions about 'Tigran' or information found in the private data files."""
    from rag import answer_question
    answer, _ = answer_question(query)
    return answer


@tool
def search_web(query: str) -> str:
    """Search the public internet for current information. Use this for questions about external topics, world facts, recent events, or general knowledge NOT found in internal documents. For weather, news, famous people, scientific facts, etc."""
    return f"[Web search disabled in sandbox] Cannot search the web for '{query}'. Try search_docs for internal information instead."


TOOLS = [calculator, get_employee, get_team, search_docs, search_web]
