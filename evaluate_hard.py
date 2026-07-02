"""
evaluate_hard.py — يشتغل على Kaggle بعد dedupe_and_rebuild.py
(محتاج kb_index_v2.faiss + kb_documents_v2.json)

مجموعة اختبار "صعبة": أسئلة مصاغة بشكل غير مباشر، بعيدة عن عناوين الوثائق،
متل ما مستخدم حقيقي غامض بيسأل بدون ما يعرف المصطلح التقني الدقيق.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DOCS_PATH = Path("kb_documents_v2.json")
INDEX_PATH = Path("kb_index_v2.faiss")
TOP_K = 3

# أسئلة غير مباشرة — ما فيها نفس كلمات عنوان الوثيقة المستهدفة
HARD_TEST_SET = [
    {
        "query": "لو عندي مليون مستند وبدي دور عن أقرب معنى مشابه بسرعة، وين بخزن هاد النوع من المعلومات؟",
        "expected_id": "ai-engineer__vector-database",
    },
    {
        "query": "How do I split a huge PDF into smaller pieces before feeding it to a search system?",
        "expected_id": "ai-engineer__chunking",
    },
    {
        "query": "شو الفرق العملي بين إني أعطي الموديل تعليمات واضحة، وبين إني أعطيه معلومات خلفية عن الموقف؟",
        "expected_id": "ai-engineer__prompt-vs-context-engineering",
    },
    {
        "query": "What actually happens inside ChatGPT when it predicts the next word?",
        "expected_id": "ai-engineer__how-llms-work",
    },
    {
        "query": "بدي أتأكد إنه الموديل تبعي ما حفظ الداتا عن ظهر قلب، شو الطريقة المعتمدة لهيك تحقق؟",
        "expected_id": "machine-learning__k-fold-cross-validation",
    },
    {
        "query": "What kind of neural network architecture is best for recognizing objects in photos?",
        "expected_id": "machine-learning__convolutional-neural-network",
    },
    {
        "query": "عندي آلاف الـembeddings وبدي دور عن أقرب واحد لسؤال المستخدم بأقل وقت ممكن، شو المكتبة المناسبة؟",
        "expected_id": "ai-engineer__faiss",
    },
    {
        "query": "How can a transformer model pay attention to different parts of a sentence at the same time?",
        "expected_id": "machine-learning__multi-head-attention",
    },
    {
        "query": "لما بدي الموديل يجاوب بشكل عشوائي أكتر أو أقل، شو الإعداد يلي بتحكم فيه؟",
        "expected_id": "ai-engineer__sampling-parameters",
    },
    {
        "query": "What are the risks we should think about before deploying an AI system to the public?",
        "expected_id": "ai-engineer__ai-safety-and-ethics",
    },
    {
        "query": "إذا عندي موديل جاهز بس بدي يصير خبير بمجال محدد جداً (مثلاً طب)، هل بغير أوزانه أو بس بعطيه مصادر خارجية؟",
        "expected_id": "ai-engineer__rag-vs-fine-tuning",
    },
    {
        "query": "بعد ما درّبت موديلي، كيف بعرف فعلياً هل هو منيح ولا لأ قبل ما أنشره؟",
        "expected_id": "ai-engineer__what-is-model-evaluation",
    },
    {
        "query": "How do computers turn a sentence into a list of numbers that captures its meaning?",
        "expected_id": "ai-engineer__embeddings",
    },
    {
        "query": "شو الأداة يلي بتساعدني أربط بين الموديل وقاعدة المعرفة والذاكرة بدون ما أكتب كل شي يدوياً من الصفر؟",
        "expected_id": "ai-engineer__langchain",
    },
    {
        "query": "Why would training an agent through trial-and-error rewards work better than showing it labeled examples?",
        "expected_id": "machine-learning__reinforcement-learning",
    },
    {
        "query": "شو الفرق بين إني أعطي الموديل بيانات مصنّفة بعنوان واضح لكل مثال، وبين إني بس أخليه يلاقي الأنماط لحاله؟",
        "expected_id": "machine-learning__supervised-learning",
    },
]


def load_index_and_docs():
    index = faiss.read_index(str(INDEX_PATH))
    docs = json.load(open(DOCS_PATH, encoding="utf-8"))
    return index, docs


def search(query, model, index, docs, top_k=TOP_K):
    vec = np.array(model.encode([query], normalize_embeddings=True), dtype="float32")
    scores, idx = index.search(vec, top_k)
    return [docs[i] for i in idx[0]]


def matches_expected(doc, expected_id):
    """يتحقق: هل الوثيقة المسترجعة هي المتوقعة، أو أصلاً اندمجت فيها الوثيقة المتوقعة"""
    candidates = [doc["id"]] + doc.get("merged_from", [])
    return expected_id in candidates


def main():
    print("📥 تحميل الـindex والنموذج...")
    index, docs = load_index_and_docs()
    model = SentenceTransformer(MODEL_NAME)

    hits, hits1 = 0, 0
    failures = []

    print(f"\n🔍 تشغيل {len(HARD_TEST_SET)} سؤال صعب (top-{TOP_K})...\n")
    for case in HARD_TEST_SET:
        results = search(case["query"], model, index, docs)
        found = any(matches_expected(d, case["expected_id"]) for d in results)
        found_at_1 = matches_expected(results[0], case["expected_id"])

        if found:
            hits += 1
        if found_at_1:
            hits1 += 1
        else:
            failures.append({
                "query": case["query"],
                "expected": case["expected_id"],
                "got": [d["title"] for d in results],
            })

        status = "✅" if found else "❌"
        print(f"{status} {case['query'][:70]}")

    n = len(HARD_TEST_SET)
    print("\n" + "=" * 50)
    print("📊 نتائج مجموعة الاختبار الصعبة")
    print("=" * 50)
    print(f"Precision@1: {hits1/n:.1%}  ({hits1}/{n})")
    print(f"Precision@{TOP_K}: {hits/n:.1%}  ({hits}/{n})")

    if failures:
        print(f"\n⚠️ {len(failures)} حالة فشل:")
        for f in failures:
            print(f"\n  السؤال: {f['query']}")
            print(f"  المتوقع: {f['expected']}")
            print(f"  طلع: {f['got']}")

    json.dump(
        {"total": n, "precision_at_1": round(hits1/n, 3), f"precision_at_{TOP_K}": round(hits/n, 3), "failures": failures},
        open("evaluation_report_hard.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )
    print("\n💾 evaluation_report_hard.json جاهز")


if __name__ == "__main__":
    main()
