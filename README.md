# VTU CSE RAG API

A portfolio-grade backend for answering questions over VTU CSE/BIT syllabi and past papers. It is deliberately an API rather than a chatbot UI: the interesting engineering is provenance-aware ingestion, retrieval, observability, caching, and graceful degradation.

> Corpus status: `data/fixtures/vtu_cse_sample.txt` is a synthetic smoke-test fixture, not an official source. Put official PDFs in `data/raw/` before publishing. The [official VTU scheme/syllabus page](https://vtu.ac.in/en/b-e-scheme-syllabus/) is the recommended source. Never present fixture results as VTU facts.

## Architecture

```text
PDFs -> PyPDF text extraction -> page-aware, module-aware chunks -> embeddings -> FAISS
                                                                        |
POST /query -> query embedding -> top-k cosine retrieval -> Groq LLM -> cited JSON response
                                    |                       |
                                    +---- semantic cache ---+-- failure => raw retrieved chunks
```

An embedding is a numeric vector whose location encodes semantic meaning: text about `normal forms` should be close to a query about `BCNF` even when words differ. FAISS performs nearest-neighbour search over those vectors. Here vectors are L2-normalized, so FAISS inner-product search equals cosine similarity.

## Run locally

Requires Python 3.11+.

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
python -m app.ingest data/fixtures
python run.py
```

The checked-in `.env.example` uses a dependency-free hash embedding only to make smoke tests runnable everywhere; it is lexical and not suitable for a serious demo. For the actual portfolio deployment, install `sentence-transformers`, set `EMBEDDING_BACKEND=sentence_transformer`, and use `all-MiniLM-L6-v2` (the first run downloads the model). Add your free Groq API key to `.env` for generated answers. Without it, the service still returns the retrieved, cited source chunks in `retrieval_fallback` mode.

```powershell
# fresh retrieval + intentionally triggered resilience path
Invoke-RestMethod http://localhost:5000/query -Method POST -ContentType 'application/json' -Body '{"question":"What is in Module 3?", "force_fallback":true}'

# normal path (requires GROQ_API_KEY)
Invoke-RestMethod http://localhost:5000/query -Method POST -ContentType 'application/json' -Body '{"question":"Has BCNF appeared in past papers?"}'
```

Responses include `citations` (`document`, `page`, optional `heading`) and a `latency_ms` breakdown for embedding, FAISS retrieval, LLM, and end-to-end time. `force_fallback` exists only to make the outage path demonstrable in a review; do not expose it publicly in a production API.

## Chunking decision

This is **structure-aware fixed-window chunking**, not naive fixed-size or fully semantic chunking:

- Never cross a PDF page: citations remain exact.
- Split at `Module N` headings first, then use 900-character windows with 120-character overlap only if a section is long.
- Retain heading, source type, source hash and page in chunk metadata.

Academic syllabi have strong explicit structure, so headings are more reliable and cheaper than an LLM/embedding-based semantic splitter. Question papers often have poor PDF structure, so page windows plus overlap avoid losing an entire question at a boundary. A later enhancement is layout-aware question detection and subject/semester metadata supplied from a manifest.

## API contract

`POST /query`

```json
{"question":"What topics come under Module 3 of DBMS?"}
```

Returns `answer` when Groq succeeds; otherwise `answer: null`, `mode: "retrieval_fallback"`, `fallback_reason`, and raw `retrieved_chunks`. This is intentional: a grounded source excerpt is more useful and safer than an invented answer during an LLM outage.

`GET /health` reports whether the local FAISS index is available.

## Measured run

Run the commands below on your machine and record the emitted JSON in the PR/README. Do not substitute estimates: cold model loading, hardware, corpus size, and a remote LLM make timing environment-specific.

```powershell
python -m app.ingest data/fixtures
python -m pytest -q
# Start run.py in another terminal, then issue the same request twice.
```

| Scenario | Evidence field | What it should demonstrate |
| --- | --- | --- |
| First identical query | `cache.hit=false`, `latency_ms` | embedding + FAISS + LLM/fallback cost |
| Second identical query | `cache.hit=true`, `latency_ms` | skips FAISS and LLM; embedding remains for semantic-cache lookup |
| `force_fallback=true` | `mode=retrieval_fallback` | explicit API failure behaviour, with citations |

Verified on 2026-08-26 in this Windows workspace using the 4-chunk synthetic fixture and the `hash` smoke-test backend: ingestion embedding **0.95 ms**; first question **80.94 ms** (embedding 0.28, FAISS 80.61, LLM 0.01); identical cache hit **0.16 ms** (embedding 0.12); forced fallback **2.16 ms** (embedding 0.08, FAISS 2.05). The first call had no Groq key, so its 0.01 ms LLM figure is the local missing-key failure—not an LLM inference measurement.

Real Groq verification: `openai/gpt-oss-20b` returned a grounded answer in **942.17 ms** end-to-end (embedding 0.42 ms, FAISS 11.06 ms, LLM 930.66 ms); the repeat cache hit took **0.13 ms**. This is a warm, local run against the official 36-chunk PDF corpus. Deployment uses `sentence_transformer` with MiniLM rather than this fixture-only hash backend, so re-measure its cold and warm timings after deploying.

I also downloaded the official `csesch.pdf` source into the intentionally gitignored `data/raw/` and ingested it successfully: **36 chunks**, **15.94 ms** corpus embedding, then a forced-fallback query in **39.97 ms** (embedding 0.28, FAISS 39.65). The PDF is not committed because source distribution should be checked before publishing; re-run the documented download/ingestion step on deploy.

## Flask framing for a Java developer

- `@bp.post("/query")` is the small Flask equivalent of a Spring `@PostMapping`; a Blueprint groups routes much like a controller module.
- `create_app()` is an application factory. Flask deliberately does not enforce a Spring Boot-style project layout, so this project chooses `routes -> service -> store/llm` to keep HTTP concerns out of retrieval logic.
- Python type hints improve readability but are not Java compile-time guarantees. Dataclasses (`Chunk`, `Settings`) make the key data shapes explicit; tests compensate for the weaker static safety. Add mypy/ruff in a production iteration.
- Flask’s development server is not production-grade. The Procfile uses Gunicorn; on Windows development, use `python run.py`.

## Tradeoffs and why

| Decision | Gain | Cost / mitigation |
| --- | --- | --- |
| FAISS `IndexFlatIP` | free, local, exact cosine search, no service dependency | one-process RAM index; no metadata filtering, replicas, auth, backups, or distributed scaling. Store metadata separately; migrate to Qdrant/Pinecone/Weaviate when needed. |
| Local MiniLM embeddings | no per-query embedding bill; data stays local | model download/CPU latency and less quality than paid embeddings. Version the model and re-index on change. A hash embedder exists only for zero-download smoke tests, never as the production-quality setting. |
| Groq generation | cheap/free fast inference | external outage/rate limits and provider dependency. Timeout and return retrieved chunks. |
| In-memory semantic cache | very simple latency win | disappears on restart and is per worker; use Redis plus TTL, user/tenant scope, and cache invalidation in production. |
| Top-k=5, no reranker | transparent and fast | irrelevant chunks can occupy context. Add metadata filters and a cross-encoder reranker after measuring quality. |
| Source hash in every chunk | detects changed bytes | current command rebuilds whole index; next step is manifest-driven incremental re-embedding. |

## Interview questions I expect

1. **How do you handle stale embeddings when documents change?** Each chunk stores its source SHA-256. Compare an ingestion manifest of `(path, hash, parser version, chunker version, embedding model)` to the next run; delete/re-embed only changed/deleted sources. Rebuild the FAISS index atomically, then swap versions. This starter has the hash but deliberately rebuilds all documents for simplicity.
2. **How do citations remain trustworthy?** Page is assigned before chunking and chunks never cross pages. The LLM sees numbered retrieved chunks, but citations returned by the API are computed by the backend—not invented by the model. For production, render/verify PDF page text and include a source-file download URL.
3. **Why is FAISS not a vector database?** It is an in-process similarity-search library. It gives fast local search, but not filtering, persistence workflows, concurrency management, HA, backups, multi-tenancy, or observability. That is the deliberate zero-budget trade.
4. **What if retrieval is wrong?** Generation cannot repair absent context. Track a labelled evaluation set (question, relevant page), recall@k and citation accuracy; improve metadata, chunking and reranking based on failed queries, not vibes.
5. **What if Groq fails or hallucinates?** Failure returns source chunks, rather than an ungrounded error. Hallucination is constrained by a source-only prompt, citations, temperature 0, and an explicit `not found` instruction; it is reduced, not eliminated.
6. **How would this scale?** Move cache to Redis, use a managed/vector DB with metadata filters and replicas, queue ingestion, store document/version metadata in Postgres, add auth/rate limits, tracing, and a background index swap.
7. **What is the cache correctness risk?** Similarity can match questions with materially different scope. The conservative threshold is 0.97, response includes the hit/similarity, and production should scope by corpus version plus query filters and use TTLs.

## Deploy (Render free tier)

1. `render.yaml` downloads the public VTU CSE scheme PDF from its official URL during build. Review source terms before changing this to another corpus.
2. Push to GitHub and create a Render Blueprint deployment; `render.yaml` installs dependencies and runs ingestion during build.
3. Set `GROQ_API_KEY` in Render’s encrypted environment variables. Do not commit `.env`.
4. Test `GET /health` and `POST /query` against the Render URL. Free instances cold-start; record that separately from warm latency.

For serious use, store source PDFs in object storage and persist the built FAISS artifact. Free Render filesystems are ephemeral, so a restart may require re-ingestion.
