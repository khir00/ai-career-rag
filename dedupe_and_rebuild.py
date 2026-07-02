"""
dedupe_and_rebuild.py — يشتغل على Kaggle بعد build_embeddings.py مباشرة
(محتاج kb_index.faiss + kb_documents.json من الخطوة السابقة)

بيعمل أوتوماتيك:
1. يلاقي الوثائق شبه المكررة بالـembeddings الحقيقية (دقة عالية)
2. يدمجهن
3. يعيد بناء الـindex
4. يعيد تشغيل التقييم
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DOCS_PATH = Path("kb_documents.json")
SIMILARITY_THRESHOLD = 0.88  # عتبة تشابه عالية = واثقين إنهن نفس الموضوع فعلاً

NEW_INDEX_OUT = Path("kb_index_v2.faiss")
NEW_DOCS_OUT = Path("kb_documents_v2.json")

# نفس أسئلة evaluate_retrieval.py منسوخة هون مباشرة (بدل الاستيراد، ما بيشتغل عالـKaggle)
TEST_SET = [
    {"query": "What is a vector database used for?", "expected_id": "ai-engineer__vector-database"},
    {"query": "How does chunking work in RAG systems?", "expected_id": "ai-engineer__chunking"},
    {"query": "What is the difference between prompt engineering and context engineering?", "expected_id": "ai-engineer__prompt-vs-context-engineering"},
    {"query": "Explain how large language models work internally", "expected_id": "ai-engineer__how-llms-work"},
    {"query": "What is k-fold cross validation?", "expected_id": "machine-learning__k-fold-cross-validation"},
    {"query": "What are convolutional neural networks used for?", "expected_id": "machine-learning__convolutional-neural-network"},
    {"query": "How do I use FAISS for similarity search?", "expected_id": "ai-engineer__faiss"},
    {"query": "What is multi-head attention in transformers?", "expected_id": "machine-learning__multi-head-attention"},
    {"query": "What sampling parameters control LLM output randomness?", "expected_id": "ai-engineer__sampling-parameters"},
    {"query": "What ethical concerns exist around AI safety?", "expected_id": "ai-engineer__ai-safety-and-ethics"},
    {"query": "شو هو الـvector database وليش بنستخدمه؟", "expected_id": "ai-engineer__vector-database"},
    {"query": "كيف بيشتغل الـchunking بأنظمة الـRAG؟", "expected_id": "ai-engineer__chunking"},
    {"query": "شو الفرق بين prompt engineering و context engineering؟", "expected_id": "ai-engineer__prompt-vs-context-engineering"},
    {"query": "كيف بتشتغل نماذج اللغة الكبيرة من الداخل؟", "expected_id": "ai-engineer__how-llms-work"},
    {"query": "شو هو الـk-fold cross validation؟", "expected_id": "machine-learning__k-fold-cross-validation"},
    {"query": "شو استخدامات الشبكات العصبية الالتفافية CNN؟", "expected_id": "machine-learning__convolutional-neural-network"},
    {"query": "كيف بستخدم FAISS للبحث عن التشابه؟", "expected_id": "ai-engineer__faiss"},
    {"query": "شو هو الـmulti-head attention بالـtransformers؟", "expected_id": "machine-learning__multi-head-attention"},
    {"query": "شو هي المعايير يلي بتتحكم بعشوائية جواب الـLLM؟", "expected_id": "ai-engineer__sampling-parameters"},
    {"query": "شو الهواجس الأخلاقية المتعلقة بسلامة الذكاء الاصطناعي؟", "expected_id": "ai-engineer__ai-safety-and-ethics"},
]


def merge_cluster(docs, indices):
    if len(indices) == 1:
        return docs[indices[0]]
    members = [docs[i] for i in indices]
    base = max(members, key=lambda d: len(d["text"]))
    others = [d for d in members if d is not base]
    combined = base["text"]
    for o in others:
        if o["text"].strip() not in combined:
            combined += "\n\n" + o["text"]
    return {
        "id": base["id"], "source": base["source"], "category": base["category"],
        "title": base["title"], "text": combined,
        "merged_from": [d["id"] for d in members],
    }


def main():
    print("📥 تحميل الوثائق والنموذج...")
    docs = json.load(open(DOCS_PATH, encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️  حساب الـembeddings...")
    texts = [f"{d['title']}. {d['text']}" for d in docs]
    embeddings = np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True), dtype="float32")

    print(f"🔍 إيجاد التكرار (عتبة {SIMILARITY_THRESHOLD})...")
    sim = embeddings @ embeddings.T
    n = len(docs)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim[i][j] >= SIMILARITY_THRESHOLD and docs[i]["category"] == docs[j]["category"]:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)

    merged_docs = [merge_cluster(docs, c) for c in clusters]
    merge_log = [
        {"merged_into": m["title"], "count": len(c)}
        for m, c in zip(merged_docs, clusters) if len(c) > 1
    ]

    print(f"✅ {len(docs)} → {len(merged_docs)} وثيقة")
    for e in merge_log:
        print(f"   • {e['count']} وثائق دمجن بـ \"{e['merged_into']}\"")

    print("\n🗂️  إعادة بناء الـindex من الوثائق المدموجة...")
    new_texts = [f"{d['title']}. {d['text']}" for d in merged_docs]
    new_embeddings = np.array(model.encode(new_texts, normalize_embeddings=True, show_progress_bar=True), dtype="float32")
    new_index = faiss.IndexFlatIP(new_embeddings.shape[1])
    new_index.add(new_embeddings)

    faiss.write_index(new_index, str(NEW_INDEX_OUT))
    json.dump(merged_docs, open(NEW_DOCS_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"💾 {NEW_INDEX_OUT} و {NEW_DOCS_OUT} جاهزين")

    # ---------- إعادة التقييم (نفس أسئلة evaluate_retrieval.py) ----------
    print("\n📊 إعادة التقييم على نفس الأسئلة...")

    def search(query, top_k=3):
        vec = np.array(model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, idx = new_index.search(vec, top_k)
        return [merged_docs[i]["id"] for i in idx[0]]

    hits, hits1 = 0, 0
    for case in TEST_SET:
        # ملاحظة: بعد الدمج، الـid الأصلي ممكن يكون اندمج بوثيقة تانية إلها id مختلف
        # فمنتحقق: هل الـid المتوقع لسا موجود، أو هل هو ضمن merged_from لأي وثيقة بالنتائج
        results = search(case["query"])
        matched = False
        for doc_id in results:
            doc = next(d for d in merged_docs if d["id"] == doc_id)
            candidates = [doc["id"]] + doc.get("merged_from", [])
            if case["expected_id"] in candidates:
                matched = True
                break
        if matched:
            hits += 1
            if results[0] == case["expected_id"] or case["expected_id"] in (
                next(d for d in merged_docs if d["id"] == results[0]).get("merged_from", [])
            ):
                hits1 += 1

    n_q = len(TEST_SET)
    print(f"Precision@1: {hits1/n_q:.1%}  ({hits1}/{n_q})")
    print(f"Precision@3: {hits/n_q:.1%}  ({hits}/{n_q})")


if __name__ == "__main__":
    main()
