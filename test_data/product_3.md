# Project Borealis — SR&ED Technical Narrative (Draft v0.3)  

> INTERNAL — do not share
> Owner: R&D / Finance hybrid
> Date: 2026-01-14
> Status: Draft (contradictions inside on purpose)

## 1. Executive summary (TL;DR)
We attempted to build a **document triage + relevance filter** for SR&ED submissions.
- Goal: reduce manual review time by 70%
- Hypothesis: keyword-only filters fail; need embeddings + rules + signals
- Result: partial success (precision up, recall down) — more below


Line breaks    
with trailing spaces.  
And double spaces.  

Hard wrap vs soft wrap confusion:
This paragraph is intentionally broken
across lines like an old PDF export.

## 2. Background / contexte
Le défi: les documents entrants sont hétérogènes (PDF scannés, emails, RFCs, KEPs).
We need to detect **experimental development** claims vs routine engineering.

### 2.1 Definitions 
- *Routine engineering:* incremental changes, known methods, expected results.
- *Experimental development:* technological uncertainty + systematic investigation.
- *Garbage:* marketing PDFs, pricing sheets, job postings, random meeting notes.

## 3. Scope
In-scope docs:
1) system design notes
2) architecture decision records
3) experiment logs
4) model eval results

Out-of-scope docs:
- sales decks
- customer contracts
- invoices / receipts (unless linked to experimentation)

> NOTE: the pipeline currently **does not OCR** (or does it? see §7.2)

## 4. Work performed (chronological)
### 4.1 Week 1 — naive filter (failed)
We used `contains("SRED") OR contains("uncertainty") OR contains("experiment")`.
**Problem:** false positives on templates, blog posts, legal docs.

### 4.2 Week 2 — embeddings
We added embeddings similarity search with a reference corpus:
- prior SR&ED narratives
- internal R&D notes
- public technical proposals

We saw:
- Precision@10 improved from 0.20 → 0.65
- Recall dropped (unknown; labeling incomplete)

### 4.3 Week 3 — hybrid scoring
Final score = (0.55 * embedding) + (0.25 * rules) + (0.20 * metadata)

Rules included:
- density of technical terms per 1,000 chars
- presence of structured test evidence
- presence of uncertainty language ("unknown", "fails", "edge case")

**BUT**: this punished concise notes + rewarded long verbose docs.

## 5. Experimental evidence 
### 5.1 Experiment table
| Experiment ID | Change | Expected | Observed | Pass? | Notes |
|---|---|---|---|---|---|
| EXP-001 | chunk_size=800 | better retrieval | hallucinations down | ✅ | latency +20% |
| EXP-002 | remove stopwords | higher similarity | worse | ❌ | domain terms removed |
| EXP-003 | add regex “OCR noise” detector | filter junk | mixed | ⚠️ | false positives |

### 5.2 Inline pseudo-math
We estimated relevance as:

RelevanceScore = sigmoid( 0.7 * sim + 0.2 * rules + 0.1 * meta - penalty )

Where penalty includes:
- boilerplate ratio
- repetition ratio

## 6. Raw logs 
2026-01-14T03:14:15Z INFO ingest file=RFC-0123.md bytes=38422 sha=9f2...
2026-01-14T03:14:16Z WARN chunker "overlap too large" overlap=700 size=800
2026-01-14T03:14:17Z ERROR vector_db timeout after 12000ms
2026-01-14T03:14:17Z RETRY #1
2026-01-14T03:14:18Z INFO success

## 7. Contradictions & edge cases
### 7.1 Contradiction test
- This document says we do NOT OCR (§3)
- But this section implies we DO OCR (§7.2)
Your pipeline should not crash; it should tolerate contradictions.

### 7.2 OCR noise sample
Th1s l1ne 1s n0isy.
lIlI1I|!  O0o0  rn m
S R & E D  ( spaced letters )
“smart quotes” and ‘single quotes’ and "normal quotes"
Broken ligatures: ﬁ ﬂ

## 8. Attachments 
- [diagram](assets/arch_v1.png)
- [exported pdf](exports/sred_narrative_draft.pdf)
- [data](data/eval_results.csv)

## 9. TODO
- [ ] add unit tests
- [ ] add eval harness
- [x] ship POC anyway

Footnote test: this is a claim.[^1]

[^1]: Footnote content with **bold** and `code` and a URL-like string: example.com/not-a-real-link
