# Week 3 Report — Tool Use & Memory

## Step 1 — Structured Tool Calling

**Question:** "What is 18% of 2450?"

**Trajectory row:**
```json
{"tool": "calculator", "args": {"expression": "0.18 * 2450"}, "result": "441.0"}
```

**Final answer:** 18% of 2450 is 441.

The model selected `calculator` via a structured `tool_call` (not text parsing), received the result `441.0`, and produced the final answer. This proves native function calling works.

---

## Step 2 — Routing Accuracy

### Golden Set (8 questions, including 2 overlap cases)

| # | Query | Expected Tool | Description |
|---|-------|---------------|-------------|
| 1 | What is 18% of 2450? | calculator | Simple math |
| 2 | What is the square root of 144? | calculator | Math with sqrt |
| 3 | What is Sarah Kim's email address? | get_employee | Employee lookup by name |
| 4 | Who works on the Platform team? | get_team | Team member listing |
| 5 | What did Tigran do during his internship? | search_docs | Internal doc about Tigran |
| 6 | When was the Eiffel Tower built? | search_web | External world knowledge |
| 7 | What is Tigran's educational background and volunteering experience? | search_docs | Overlap: asks about Tigran (docs vs web) |
| 8 | What is the capital of France? | search_web | Overlap: world fact (web vs docs) |

### Three accuracy numbers

| Phase | Accuracy | Delta |
|-------|----------|-------|
| **Baseline** (correct descriptions) | **1.00** (8/8) | — |
| **Broken** (descriptions swapped) | **0.25** (2/8) | **−0.75** |
| **Fixed** (sharpened disambiguation) | **1.00** (8/8) | **+0.75** |

### Description before/after fix

**Before (broken)** — `search_docs` described as a web search tool and `search_web` described as a doc search tool; `get_employee` and `get_team` swapped:

```
search_docs.description = "Search the public internet for current events, world facts, news..."
search_web.description  = "Search internal documents about Tigran Ghavalyan..."
get_employee.description = "List ALL members of a team by team name..."
get_team.description     = "Look up a single person by name..."
```

**After (fixed):**
```
search_docs.description = "ONLY for questions about Tigran Ghavalyan (internship, education, volunteering, personal stories). Do NOT use for world facts, history, capitals, or general knowledge."
search_web.description  = "ONLY for external/world knowledge: capitals, history, science, current events, weather, famous people. Do NOT use for questions about Tigran."
get_employee.description = "Look up ONE employee by their PERSON NAME (e.g., 'Sarah Kim'). Returns role, team, email. Do NOT use for team listings."
get_team.description     = "List ALL members of a team by TEAM NAME (e.g., 'Platform', 'Product', 'AI/ML'). Do NOT use for individual people lookups."
```

The fix added explicit negative instructions ("Do NOT use for...") and clarified which input type each tool expects (person name vs team name).

---

## Step 3 — Memory: Without vs With

### Without memory (two separate `run_agent` calls)

```
Q1: Who manages the Platform team?
A1: Sarah Kim is the Engineering Manager for the Platform team.

Q2: What is her email?
A2: I'm sorry, but I couldn't find an employee named "her."
     Could you please provide the name of the person you are looking for?
```

The second call fails — "her" is unknown because no context was carried over.

### With memory (`run_agent_with_history` with prior history)

```
History includes: Who manages the Platform team? -> Sarah Kim...
Follow-up: What is her email?
A3: Sarah Kim's email is sarah.kim@company.com.
```

The follow-up succeeds because the first turn's answer ("Sarah Kim") is injected into the conversation history.

### Assembled context log (proof of injection)

Before the second turn, the memory's `context()` returned:

```
Conversation so far:
User: Who manages the Platform team?
Assistant: Sarah Kim is the Engineering Manager for the Platform team.
```

The string **"Sarah Kim"** appears in this context, proving the code injected it — the model did not "remember" it on its own.

---

## Step 4 — Integration (≥4 turns)

### Transcript

| Turn | Question | Tool Chosen | Expected Tool | Correct? | Memory Needed? |
|------|----------|-------------|---------------|----------|----------------|
| 1 | Who manages the Platform team? | get_team | get_team | Yes | No (first turn) |
| 2 | What is her email? | get_employee | get_employee | Yes | **Yes** — needs "Sarah Kim" from turn 1 |
| 3 | What is 18% of 2450? | calculator | calculator | Yes | No |
| 4 | What did Tigran do during his internship? | search_docs | search_docs | Yes | No |

**Routing accuracy for this conversation:** 4/4 = 1.00

Turn 2 was memory-dependent — without the prior context carrying "Sarah Kim", the follow-up "her" would have failed.

---

## Step 5 — One Analyzed Failure

### Failure: Bad-argument routing — team name passed to `get_employee`

**Root cause:** When descriptions of `get_employee` and `get_team` were swapped during the "broken" phase, the model faithfully followed the descriptions. `get_employee` was described as "List ALL members of a team by team name" and `get_team` as "Look up a single person by name." For Q4 ("Who works on the Platform team?"), the model called `get_employee("Platform")` — a team name passed to a tool now described as a team-listing tool. This returned an error because `get_employee` internally searches an employee dict keyed by person names.

**Fix:** Restore and sharpen the descriptions to explicitly state input types: "person name (e.g., 'Sarah Kim')" for `get_employee` and "team name (e.g., 'Platform')" for `get_team`. Also added negative instructions ("Do NOT use for team listings" / "Do NOT use for individual people lookups").

---

## 6 — Which layer broke more often?

**Routing broke more often.** In the broken-description phase, routing accuracy dropped from 1.00 to 0.25 (6/8 failures). Memory never failed — when the assembled context contained the right information, the model always used it correctly. The agent's vulnerability is in tool **selection**, not memory retention: the model follows misleading descriptions when instructed to, and overlap between tools (get_employee vs get_team, search_docs vs search_web) is the hardest case.
