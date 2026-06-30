import json
import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import answer_question
from eval_set import GOLDEN_QUERIES

load_dotenv()

judge_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

faithfulness_template = """You are an expert judge evaluating a RAG system. Rate the FAITHFULNESS of the answer on 1-5.

Faithfulness: every claim in the answer must be directly supported by the retrieved context.

1 = Most claims unsupported (severe hallucination)
2 = Several unsupported claims
3 = Mixed — some supported, some unsupported
4 = Mostly supported, minor unsupported details
5 = Fully supported — every claim is in the context

Output ONLY valid JSON: {{"reasoning": "...", "faithfulness_score": <1-5>}}

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}"""

correctness_template = """You are an expert judge evaluating a RAG system. Rate the CORRECTNESS of the answer on 1-5.

Correctness: does the answer match the reference answer in meaning? Paraphrasing is fine, but key info should be present.

1 = Completely wrong
2 = Mostly wrong, misses most key points
3 = Partially correct
4 = Mostly correct, minor omissions
5 = Fully correct — captures all key information

Output ONLY valid JSON: {{"reasoning": "...", "correctness_score": <1-5>}}

QUESTION: {question}

GENERATED ANSWER:
{answer}

REFERENCE ANSWER:
{reference}"""

faithfulness_prompt = ChatPromptTemplate.from_template(faithfulness_template)
correctness_prompt = ChatPromptTemplate.from_template(correctness_template)

faithfulness_chain = faithfulness_prompt | judge_llm | StrOutputParser()
correctness_chain = correctness_prompt | judge_llm | StrOutputParser()

CACHE_FILE = "eval_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def parse_score(raw_output, key):
    if raw_output is None:
        return None, "No output"
    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        cleaned = cleaned.strip()
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data[key], data.get("reasoning", "")
        data = json.loads(cleaned)
        return data[key], data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError):
        score_match = re.search(r'(?:faithfulness_score|correctness_score)[:\s]*(\d+)', raw_output, re.IGNORECASE)
        if score_match:
            return int(score_match.group(1)), raw_output[:300]
        return None, raw_output[:300]

def invoke_with_retry(chain, inputs, max_retries=5, delay=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            if "RESOURCE_EXHAUSTED" in str(e):
                wait = 60
                print(f"  Quota hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                print(f"  Retry {attempt+1}/{max_retries} after: {e}")
                time.sleep(delay)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return None
    print(f"  All retries exhausted: {last_error}")
    return None

def evaluate():
    print("=" * 60)
    print("LLM Judge Evaluation")
    print("=" * 60)

    cache = load_cache()
    all_results = []

    for i, item in enumerate(GOLDEN_QUERIES):
        q = item["query"]
        ref = item["reference"]
        qid = f"q{i}"

        if qid in cache:
            print(f"\n--- Query {i+1}/{len(GOLDEN_QUERIES)} (cached) ---")
            result = cache[qid]
        else:
            print(f"\n--- Query {i+1}/{len(GOLDEN_QUERIES)} ---")
            print(f"Q: {q}")
            answer, context = answer_question(q)
            print(f"A: {answer[:120]}...")

            raw_faith = invoke_with_retry(faithfulness_chain, {
                "question": q, "context": context, "answer": answer
            })
            faith_score, faith_reason = parse_score(raw_faith, "faithfulness_score")
            print(f"  Faithfulness: {faith_score}/5")

            raw_correct = invoke_with_retry(correctness_chain, {
                "question": q, "answer": answer, "reference": ref
            })
            correct_score, correct_reason = parse_score(raw_correct, "correctness_score")
            print(f"  Correctness:  {correct_score}/5")

            result = {
                "query": q,
                "answer": answer,
                "context": context,
                "reference": ref,
                "faithfulness_score": faith_score,
                "faithfulness_reasoning": faith_reason,
                "correctness_score": correct_score,
                "correctness_reasoning": correct_reason
            }
            cache[qid] = result
            save_cache(cache)

        all_results.append(result)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    faith_scores = [r["faithfulness_score"] for r in all_results if r["faithfulness_score"] is not None]
    correct_scores = [r["correctness_score"] for r in all_results if r["correctness_score"] is not None]

    faith_pass = sum(1 for s in faith_scores if s >= 4)
    correct_pass = sum(1 for s in correct_scores if s >= 4)

    faith_rate = faith_pass / len(faith_scores) if faith_scores else 0
    correct_rate = correct_pass / len(correct_scores) if correct_scores else 0

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    avg_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0

    print(f"Faithfulness pass-rate (>=4): {faith_rate:.2f} ({faith_pass}/{len(faith_scores)})")
    print(f"Correctness pass-rate (>=4):  {correct_rate:.2f} ({correct_pass}/{len(correct_scores)})")
    print(f"Average faithfulness score:    {avg_faith:.2f}")
    print(f"Average correctness score:     {avg_correct:.2f}")

    return all_results, faith_rate, correct_rate

if __name__ == "__main__":
    results, fr, cr = evaluate()
