"""
evaluate_retrieval.py — يشتغل بعد build_embeddings.py (يحتاج نفس الـkb_index.faiss و kb_documents.json)

يشغّل مجموعة أسئلة اختبار (عربي + إنجليزي) على نظام الـretrieval،
ويحسب Precision@k: من كل الأسئلة، كم مرة الوثيقة الصحيحة طلعت ضمن أفضل k نتيجة.

طريقة الاستخدام: نفس بيئة build_embeddings.py (Kaggle notebook بعد ما تشغل السكريبت الأول)
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path("kb_index.faiss")
DOCS_PATH = Path("kb_documents.json")
TOP_K = 3

# ---------- مجموعة أسئلة الاختبار ----------
# expected_id لازم يطابق id الوثيقة بالضبط (من knowledge_base.jsonl)
TEST_SET = [
    # --- إنجليزي ---
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
    # --- عربي ---
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


def load_index_and_docs():
    index = faiss.read_index(str(INDEX_PATH))
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def search(query: str, model: SentenceTransformer, index, docs, top_k: int = TOP_K):
    vec = model.encode([query], normalize_embeddings=True)
    vec = np.array(vec, dtype="float32")
    scores, indices = index.search(vec, top_k)
    return [(docs[i]["id"], docs[i]["title"], float(s)) for s, i in zip(scores[0], indices[0])]


def main():
    print("📥 تحميل الـindex والنموذج...")
    index, docs = load_index_and_docs()
    model = SentenceTransformer(MODEL_NAME)

    hits = 0
    hits_at_1 = 0
    failures = []

    print(f"\n🔍 تشغيل {len(TEST_SET)} سؤال اختبار (top-{TOP_K})...\n")
    for case in TEST_SET:
        results = search(case["query"], model, index, docs, top_k=TOP_K)
        result_ids = [r[0] for r in results]
        found = case["expected_id"] in result_ids
        found_at_1 = results[0][0] == case["expected_id"]

        if found:
            hits += 1
        if found_at_1:
            hits_at_1 += 1
        else:
            failures.append({
                "query": case["query"],
                "expected": case["expected_id"],
                "got": [f"{r[1]} ({r[0]}, score={r[2]:.3f})" for r in results],
            })

        status = "✅" if found else "❌"
        print(f"{status} {case['query']}")
        if not found:
            print(f"   المتوقع: {case['expected_id']}")
            print(f"   طلع: {result_ids}")

    n = len(TEST_SET)
    precision_at_k = hits / n
    precision_at_1 = hits_at_1 / n

    print("\n" + "=" * 50)
    print("📊 النتائج النهائية")
    print("=" * 50)
    print(f"Precision@1: {precision_at_1:.1%}  ({hits_at_1}/{n})")
    print(f"Precision@{TOP_K}: {precision_at_k:.1%}  ({hits}/{n})")

    if failures:
        print(f"\n⚠️  {len(failures)} حالة فشل — تفاصيل:")
        for fail in failures:
            print(f"\n  السؤال: {fail['query']}")
            print(f"  المتوقع: {fail['expected']}")
            print(f"  طلع بدلها: {fail['got']}")

    # حفظ تقرير كامل لاستخدامه بالـREADME
    report = {
        "total_questions": n,
        "precision_at_1": round(precision_at_1, 3),
        f"precision_at_{TOP_K}": round(precision_at_k, 3),
        "failures": failures,
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n💾 التقرير الكامل محفوظ بـ evaluation_report.json")


if __name__ == "__main__":
    main()
