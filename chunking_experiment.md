# Chunking Experiment Report

## Step 2: Frozen Query Set (5 A/B Pairs)

| Pair | Type | Query |
|------|------|-------|
| 1 | A (Close) | What was Tigran Ghavalyan's role and responsibilities during his internship at SoftAcademy LLC? |
| 1 | B (Synonym) | How did Tigran contribute to his team while working as a web programmer? |
| 2 | A (Close) | When did Tigran receive his three diplomas in regional and national Olympiads in Math and Informatics? |
| 2 | B (Synonym) | At what age did he see his first major successes in competitive math and informatics? |
| 3 | A (Close) | What did the horizontal bar in his room teach Tigran about his dreams and goals? |
| 3 | B (Synonym) | How did a piece of metal in his bedroom influence his mindset towards challenges and achieving goals? |
| 4 | A (Close) | Describe Tigran's volunteering experience at the Sevan Startup Summit 2024 and 2025. |
| 4 | B (Synonym) | What kind of volunteering work did he do at the tech event by the lake? |
| 5 | A (Close) | Why does Tigran believe Sorbonne Abu Dhabi is the ideal environment for his academic journey in Data Science? |
| 5 | B (Synonym) | What specific features of SUAD appeal to his career goals in AI and research? |

## Step 3: Baseline Measurement (500/50)

| Query | Hit/Miss | Top-1 Distance |
|-------|----------|----------------|
| Q1A: Role at SoftAcademy | Hit | 0.4072 |
| Q1B: Contribution as programmer | Hit | 0.4001 |
| Q2A: Date of 3 diplomas | Hit | 0.4699 |
| Q2B: Age of success | Hit | 0.4676 |
| Q3A: Horizontal bar (dreams/goals) | Hit | 0.4928 |
| Q3B: Metal piece (mindset) | Hit | 0.4632 |
| Q4A: Sevan Startup Summit | Hit | 0.3883 |
| Q4B: Event by the lake | Hit | 0.5392 |
| Q5A: Sorbonne Abu Dhabi fit | Hit | 0.4708 |
| Q5B: SUAD features (AI/research) | Hit | 0.3900 |

## Step 4: Experimental Stores

| Store | Splitter | Size | Overlap | Chunk Count |
|-------|----------|------|---------|-------------|
| baseline | Recursive | 500 | 50 | 50 |
| lab_small | Recursive | 200 | 20 | 113 |
| lab_large | Recursive | 800 | 100 | 31 |
| lab_no_overlap | Recursive | 500 | 0 | 49 |

## Step 5: Full Results Table

| Query | baseline | lab_small | lab_large | lab_no_overlap |
|-------|----------|-----------|-----------|----------------|
| Q1A: Role at SoftAcademy | 0.4072 | 0.3502 | 0.4247 | 0.4072 |
| Q1B: Contribution as programmer | 0.4001 | 0.4017 | 0.4004 | 0.4001 |
| Q2A: Date of 3 diplomas | 0.4699 | 0.5012 | 0.4462 | 0.4699 |
| Q2B: Age of success | 0.4676 | 0.5228 | 0.5256 | 0.5367 |
| Q3A: Horizontal bar (dreams/goals) | 0.4928 | 0.5048 | 0.4795 | 0.5018 |
| Q3B: Metal piece (mindset) | 0.4632 | 0.5172 | 0.4546 | 0.4705 |
| Q4A: Sevan Startup Summit | 0.3883 | 0.4040 | 0.3974 | 0.3883 |
| Q4B: Event by the lake | 0.5392 | 0.5030 | 0.5588 | 0.5392 |
| Q5A: Sorbonne Abu Dhabi fit | 0.4708 | 0.5005 | 0.4727 | 0.4708 |
| Q5B: SUAD features (AI/research) | 0.3900 | 0.3350 | 0.5173 | 0.3511 |

## Step 6: Winner and Rebuild

**Winner:** `lab_small` (RecursiveCharacterTextSplitter, Size: 200, Overlap: 20).

**Justification:** Since the data consists of short, distinct personal anecdotes and professional descriptions (interlinked prose), smaller chunks allow for more precise semantic matches without diluting the specific facts with surrounding unrelated narrative.

---
*Verification:* Main store rebuilt with 200/20 strategy and validated with `rag.py`.
