# 🎓 AI Career RAG — Arabic AI Graduate Career Assistant

A bilingual (Arabic/English) Retrieval-Augmented Generation system that answers AI career questions using real, grounded sources — not hallucinated advice.

**🔗 Live Demo:** [huggingface.co/spaces/khir1232/ai-career-rag](https://huggingface.co/spaces/khir1232/ai-career-rag)

## Why this project

Most "AI Engineer" job postings today require RAG experience, not just chatbot wrappers. This project demonstrates the full RAG lifecycle — data curation, retrieval evaluation, deduplication, embedding upgrades, and re-ranking — applied to a real problem: helping Arabic-speaking AI graduates navigate their career path.

## Architecture

```
Query → E5-large embedding → FAISS retrieval (top-10)
      → Cross-Encoder re-ranking (top-3) → Qwen2.5-1.5B-Instruct → Grounded answer + sources
```

## Evaluation journey (honest results, not cherry-picked)

Two test sets were used: a **direct** set (20 questions phrased close to document titles) and a **hard** set (16 questions phrased indirectly/scenario-based, the way real users actually ask).

| Approach | Direct P@1 | Direct P@3 | Hard P@1 | Hard P@3 |
|---|---|---|---|---|
| MiniLM (baseline) | 80% | 100% | 25% | 25% |
| + E5-large embeddings | 90% | 100% | 12.5% | 50% |
| + Cross-encoder re-ranking | 85% | 100% | **31.2%** | 43.8% |

**Takeaway:** retrieval on direct, keyword-aligned questions is a solved problem for this knowledge base (100% P@3). Indirect, scenario-based questions remain genuinely hard — no single technique tried here fully solved it, and each approach trades off differently between top-1 precision and top-3 recall. This gap is documented rather than hidden, because knowing where a RAG system breaks is as important as knowing where it works.

## Key engineering decisions

- **Data source:** [roadmap.sh](https://roadmap.sh) developer-roadmap content (AI Engineer + Machine Learning roadmaps), cleaned into 281 unified documents after deduplication
- **Deduplication:** Found and merged 15 near-duplicate documents using embedding similarity (cosine ≥ 0.88) — TF-IDF-based matching failed to catch these, but semantic embeddings did. This alone improved direct-question Precision@1 by 15 points (65%→80%).
- **Embedding upgrade:** Swapped a lightweight multilingual MiniLM for `intfloat/multilingual-e5-large`, improving direct-question accuracy and doubling hard-question Precision@3.
- **Re-ranking:** Added a `BAAI/bge-reranker-v2-m3` cross-encoder stage — retrieves 10 candidates via embeddings, then re-scores each against the query directly for a sharper top-1 result on ambiguous questions.
- **Grounding:** The system is prompted to explicitly decline answering when retrieved context is insufficient, reducing hallucination risk.

## Tech stack

`sentence-transformers` · `FAISS` · `multilingual-e5-large` · `bge-reranker-v2-m3` · `Qwen2.5-1.5B-Instruct` · `Gradio` · Hugging Face Spaces

## Known limitations

- Indirect/scenario-phrased questions still underperform direct questions significantly — a known open problem in retrieval, not fully solved here
- Knowledge base currently covers general AI/ML roadmap topics only (no real job postings data yet — planned for v2)
- Generation model is small (1.5B) for free-tier CPU deployment; occasional minor grammatical artifacts in Arabic output
- No multi-turn conversation support yet

## Run locally

```bash
pip install -r requirements.txt
python build_knowledge_base.py      # clean & structure raw markdown sources
python build_embeddings.py          # baseline embeddings + FAISS index
python dedupe_and_rebuild.py        # find & merge near-duplicate documents
python upgrade_embeddings.py        # rebuild with stronger E5-large embeddings
python add_reranking.py             # evaluate with cross-encoder re-ranking
python evaluate_retrieval.py        # direct-question evaluation
python evaluate_hard.py             # indirect-question evaluation
python app.py                       # launch Gradio interface
```

## Author

**Mohamad Khir Alhomsi** — AI Engineer, GenAI & LLM Applications
[LinkedIn](https://linkedin.com/in/mohammad-khir-alhomsi) · [GitHub](https://github.com/khir00)
