
## Intro
This appendix is intentionally long and repetitive to stress:
- chunk overlap logic
- deduplication
- heading detection
- table parsing
- code fences inside lists

## 1) Architecture (rough)
- ingest
  - normalize encoding (UTF-8)
  - strip null bytes
  - detect language (fr/en)
- parse markdown
  - headings
  - bullet lists
  - tables
  - code fences
- chunk
  - heading-aware
  - overlap=15%
- embed
- index
- retrieve
- score
- classify

### 1.1 Edge-case list nesting 
- item 1
  - subitem 1.1
    - subsubitem 1.1.1 with `inline code`
      - code fence below should remain intact:
        ```json
        {
          "doc_id": "DOC-00077",
          "lang": "fr",
          "source": "github",
          "tags": ["rfc", "experiment", "sred?"],
          "confidence": 0.42
        }
        ```
- item 2
- item 3

## 2) Evaluation results (tiny + huge numbers)
| metric | value |
|---|---:|
| docs_ingested | 150 |
| avg_chunks_per_doc | 34.7 |
| p95_chunk_tokens | 820 |
| embed_latency_ms | 187 |
| vector_db_timeout_rate | 0.031 |
| boilerplate_ratio_avg | 0.62 |
| weird_ratio | 999999999999 |


We tested on public technical proposals.
We tested on public technical proposals.
We tested on public technical proposals.
We tested on public technical proposals.
We tested on public technical proposals.


- technological uncertainty
- systematic investigation
- hypothesis → test → observation
- failures and unexpected results
- constraints due to unknown behaviour

But also false positives:
- “experimental pricing”
- “A/B test marketing”
- “innovation” with no technical details


<div>
  <p>This is <b>HTML</b> inside markdown.</p>
  <p data-test="x">Should remain text, not crash.</p>
</div>


| col1 | col2 |
|---|---|
| a | b
| c | d |

