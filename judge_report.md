# LLM-as-Judge & Calibration Report

## 1. Baseline Scores (Step 3)

| Metric | Pass Rate | Average Score |
|--------|-----------|---------------|
| Faithfulness (>=4/5) | **1.00** (10/10) | 5.00 |
| Correctness (>=4/5)  | **0.90** (9/10)  | 4.30 |

The judge (gemini-3.1-flash-lite-preview at temperature 0) scored every answer as fully faithful (5/5) since all claims in generated answers were grounded in the retrieved context. Correctness was slightly lower: most answers captured the reference meaning well, but some omitted secondary details.

## 2. Judge ↔ Human Agreement Rate (Step 4)

I manually graded each of the 10 answers on faithfulness (yes/no) and correctness (yes/no), using the same pass threshold (score >= 4 for the judge, semantic match for me).

| Query | Judge F | Human F | Agree? | Judge C | Human C | Agree? |
|-------|---------|---------|--------|---------|---------|--------|
| 1 — Internship role | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 2 — Team contributions | Pass | Pass | ✓ | Pass | **Fail** | ✗ |
| 3 — Diplomas year | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 4 — Age of success | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 5 — Horizontal bar lesson | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 6 — Metal bar mindset | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 7 — Sevan Summit experience | Pass | Pass | ✓ | **Fail** | Fail | ✓ |
| 8 — Volunteering work type | Pass | Pass | ✓ | Pass | Pass | ✓ |
| 9 — Why SUAD ideal | Pass | Pass | ✓ | Pass | **Fail** | ✗ |
| 10 — SUAD features | Pass | Pass | ✓ | Pass | Pass | ✓ |

**Agreement rate: 8/10 = 0.80**

Two disagreements, both on correctness where the judge was more lenient than I was.

## 3. Analyzed Disagreement (Step 5)

### Disagreement 1: Query 2 — "How did Tigran contribute to his team?"

- **Judge:** Correctness 4/5 (pass) — "captures core technical contribution and soft skills"
- **My label:** Fail — the answer omits **mentoring younger students**, which was a major team contribution described in both the context and reference
- **Cause:** The judge weighs partial coverage as a pass, while I consider omission of a substantive responsibility a meaningful gap. **Leniency bias** — the judge accepts "good enough" answers.

### Disagreement 2: Query 9 — "Why does Tigran believe SUAD is ideal?"

- **Judge:** Correctness 4/5 (pass) — "captures the core reason regarding the academic program"
- **My label:** Fail — the answer discusses being a "globally-minded scholar" and "contributing to academic life" but omits the **Career Center** and **research center** that were central to the reference
- **Cause:** The answer is well-written but shifts emphasis away from the reference's specific points. **Verbosity bias** — the judge was swayed by the answer's polished framing rather than checking every key fact.

## 4. Verbosity Stress-Test (Step 6 — Not Done)

Skipped due to API rate-limit constraints (free-tier quota exhausted). The disagreement analysis above already surfaced a likely verbosity bias in Query 9.

## 5. Do I trust this judge's numbers?

**No.** The 0.80 agreement rate shows meaningful gaps: the judge is too lenient on correctness (passing answers that miss key points) and may be swayed by well-phrased but incomplete answers. Until the calibration agreement reaches at least 0.90, the judge's "objective" scores cannot be taken at face value.
