"""
build_embeddings.py — يشتغل على Kaggle أو Colab (محتاج إنترنت لتحميل النموذج من Hugging Face)

الخطوات:
1. تحميل knowledge_base.jsonl (الوثائق المنظفة)
2. تحويل كل وثيقة لـembedding عبر نموذج multilingual (يدعم عربي + إنجليزي)
3. بناء FAISS index للبحث السريع
4. حفظ كل شي (index + الوثائق) لملفات تقدر تنزلها وتستخدمها لاحقاً

طريقة الاستخدام على Kaggle:
1. أنشئ notebook جديد
2. ارفع ملف knowledge_base.jsonl عبر "Add Data" -> "Upload"
3. تأكد إن الـ Internet مفعّل من إعدادات الـ notebook (يمين الشاشة -> Session options -> Internet: On)
4. الصق هاد الكود بخلية وشغله
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------- الإعدادات ----------
KB_PATH = Path("knowledge_base.jsonl")   # عدّل المسار حسب مكان الملف بعد الرفع
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # يدعم عربي وإنجليزي معاً
INDEX_OUT = Path("kb_index.faiss")
DOCS_OUT = Path("kb_documents.json")


def load_documents(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_embeddings(docs: list[dict], model: SentenceTransformer) -> np.ndarray:
    # منركب العنوان مع النص عشان الـembedding ياخد سياق أوضح
    texts = [f"{d['title']}. {d['text']}" for d in docs]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # مهم لسهولة البحث بـcosine similarity
    )
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity (لأن الvectors منورملة)
    index.add(embeddings)
    return index


def search(query: str, model: SentenceTransformer, index: faiss.Index,
           docs: list[dict], top_k: int = 3) -> list[dict]:
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = docs[idx]
        results.append({
            "title": doc["title"],
            "category": doc["category"],
            "score": float(score),
            "text": doc["text"][:300] + ("..." if len(doc["text"]) > 300 else ""),
        })
    return results


def main():
    print("📥 تحميل الوثائق...")
    docs = load_documents(KB_PATH)
    print(f"   {len(docs)} وثيقة")

    print("\n🧠 تحميل نموذج الـembedding (أول مرة بياخد وقت لتنزيل النموذج)...")
    model = SentenceTransformer(MODEL_NAME)

    print("\n⚙️  توليد الـembeddings...")
    embeddings = build_embeddings(docs, model)
    print(f"   شكل المصفوفة: {embeddings.shape}")

    print("\n🗂️  بناء الـFAISS index...")
    index = build_faiss_index(embeddings)

    print("\n💾 حفظ النتائج...")
    faiss.write_index(index, str(INDEX_OUT))
    with DOCS_OUT.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"   {INDEX_OUT} و {DOCS_OUT} جاهزين")

    # ---------- اختبار سريع ----------
    print("\n🔍 اختبار البحث بأسئلة تجريبية:\n")
    test_queries = [
        "What is prompt engineering?",
        "شو الفرق بين supervised و unsupervised learning؟",
        "How do I evaluate an LLM application?",
    ]
    for q in test_queries:
        print(f"❓ السؤال: {q}")
        results = search(q, model, index, docs, top_k=2)
        for r in results:
            print(f"   → [{r['score']:.3f}] {r['title']} ({r['category']})")
        print()


if __name__ == "__main__":
    main()
