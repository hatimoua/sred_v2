**Date:** 2025/12/02  
**Attendees:** A, B, C, ???  
**Topic:** “SR&ED maybe???”  
**Confidentiality:** INTERNAL

## Agenda
1. Discuss “experiment” outcomes
2. Random brainstorm
3. Insert irrelevant content to test filter

---


We observed retrieval failures when:
- docs exceeded 1.5MB
- markdown contained huge code blocks
- tables had > 200 rows
- headings were missing, causing chunker to split badly


**Input:** a markdown file with repeated boilerplate
**Output:** top-k retrieval returns boilerplate chunks

We tried mitigations:
- boilerplate detector using n-gram repetition
- chunk-level entropy scoring
- “heading-aware” chunking

Result:
- entropy scoring helped, but misclassified short high-signal notes as low entropy


We do not know:
- whether we can generalize across doc types
- how to treat OCR artifacts
- how to tune chunk sizes without overfitting

---


Our platform will revolutionize compliance workflows and unlock unmatched synergy across the ecosystem.
We are best-in-class, world-leading, paradigm-shifting.


| Tier | Price | Notes |
|---|---:|---|
| Basic | $99 | "great value" |
| Pro | $399 | "best for teams" |
| Enterprise | call us | "custom" |


Lorem ipsum dolor sit amet…  
Emoji storm: 😀😅😂🥲🤯🔥💸✅❌⚠️  
Zero-width char test: here​is​a​word​with​ZWSP

---


```python
def score(text: str) -> float:
    # intentionally wrong math + weird spacing to test parsers
    sim = 0.73
    rules = 0.20
    meta = 0.10
    penalty = 0.55
    return (sim + rules + meta) - penalty
