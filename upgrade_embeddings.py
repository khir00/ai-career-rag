"""
upgrade_embeddings.py — يشتغل على Kaggle (GPU on)
(محتاج kb_documents_v2.json من dedupe_and_rebuild.py)

يبني embeddings جديدة بموديل أقوى (multilingual-e5-large)، يعيد بناء الـindex،
ويقارن النتيجة على مجموعتي الاختبار (السهلة والصعبة) مع النسخة القديمة.

ملاحظة: موديل E5 بيحتاج بادئة خاصة لكل نص:
- "query: " قبل أسئلة البحث
- "passage: " قبل نصوص الوثائق
هاد جزء من طريقة تدريب الموديل، وتجاهله بيقلل الدقة كتير.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large"
DOCS_PATH = Path("kb_documents_v2.json")
NEW_INDEX_OUT = Path("kb_index_v3.faiss")
TOP_K = 3

# ---------- نفس المجموعتين (السهلة + الصعبة) للمقارنة المباشرة ----------
EASY_TEST_SET = [
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

HARD_TEST_SET = [
    {"query": "لو عندي مليون مستند وبدي دور عن أقرب معنى مشابه بسرعة، وين بخزن هاد النوع من المعلومات؟", "expected_id": "ai-engineer__vector-database"},
    {"query": "How do I split a huge PDF into smaller pieces before feeding it to a search system?", "expected_id": "ai-engineer__chunking"},
    {"query": "شو الفرق العملي بين إني أعطي الموديل تعليمات واضحة، وبين إني أعطيه معلومات خلفية عن الموقف؟", "expected_id": "ai-engineer__prompt-vs-context-engineering"},
    {"query": "What actually happens inside ChatGPT when it predicts the next word?", "expected_id": "ai-engineer__how-llms-work"},
    {"query": "بدي أتأكد إنه الموديل تبعي ما حفظ الداتا عن ظهر قلب، شو الطريقة المعتمدة لهيك تحقق؟", "expected_id": "machine-learning__k-fold-cross-validation"},
    {"query": "What kind of neural network architecture is best for recognizing objects in photos?", "expected_id": "machine-learning__convolutional-neural-network"},
    {"query": "عندي آلاف الـembeddings وبدي دور عن أقرب واحد لسؤال المستخدم بأقل وقت ممكن، شو المكتبة المناسبة؟", "expected_id": "ai-engineer__faiss"},
    {"query": "How can a transformer model pay attention to different parts of a sentence at the same time?", "expected_id": "machine-learning__multi-head-attention"},
    {"query": "لما بدي الموديل يجاوب بشكل عشوائي أكتر أو أقل، شو الإعداد يلي بتحكم فيه؟", "expected_id": "ai-engineer__sampling-parameters"},
    {"query": "What are the risks we should think about before deploying an AI system to the public?", "expected_id": "ai-engineer__ai-safety-and-ethics"},
    {"query": "إذا عندي موديل جاهز بس بدي يصير خبير بمجال محدد جداً (مثلاً طب)، هل بغير أوزانه أو بس بعطيه مصادر خارجية؟", "expected_id": "ai-engineer__rag-vs-fine-tuning"},
    {"query": "بعد ما درّبت موديلي، كيف بعرف فعلياً هل هو منيح ولا لأ قبل ما أنشره؟", "expected_id": "machine-learning__what-is-model-evaluation"},
    {"query": "How do computers turn a sentence into a list of numbers that captures its meaning?", "expected_id": "ai-engineer__embeddings"},
    {"query": "شو الأداة يلي بتساعدني أربط بين الموديل وقاعدة المعرفة والذاكرة بدون ما أكتب كل شي يدوياً من الصفر؟", "expected_id": "ai-engineer__langchain"},
    {"query": "Why would training an agent through trial-and-error rewards work better than showing it labeled examples?", "expected_id": "machine-learning__reinforcement-learning"},
    {"query": "شو الفرق بين إني أعطي الموديل بيانات مصنّفة بعنوان واضح لكل مثال، وبين إني بس أخليه يلاقي الأنماط لحاله؟", "expected_id": "machine-learning__supervised-learning"},
]


def matches_expected(doc, expected_id):
    candidates = [doc["id"]] + doc.get("merged_from", [])
    return expected_id in candidates


def evaluate(test_set, model, index, docs, label):
    hits, hits1 = 0, 0
    for case in test_set:
        query = f"query: {case['query']}"  # بادئة E5 المطلوبة
        vec = np.array(model.encode([query], normalize_embeddings=True), dtype="float32")
        scores, idx = index.search(vec, TOP_K)
        results = [docs[i] for i in idx[0]]

        if any(matches_expected(d, case["expected_id"]) for d in results):
            hits += 1
        if matches_expected(results[0], case["expected_id"]):
            hits1 += 1

    n = len(test_set)
    print(f"\n📊 {label}")
    print(f"   Precision@1: {hits1/n:.1%}  ({hits1}/{n})")
    print(f"   Precision@{TOP_K}: {hits/n:.1%}  ({hits}/{n})")
    return hits1 / n, hits / n


def main():
    print("📥 تحميل الوثائق والموديل الأقوى (بياخد وقت أطول شوي، ~2GB)...")
    docs = json.load(open(DOCS_PATH, encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️  توليد embeddings جديدة (مع بادئة 'passage: ')...")
    texts = [f"passage: {d['title']}. {d['text']}" for d in docs]
    embeddings = np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True), dtype="float32")

    print("🗂️  بناء index جديد...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(NEW_INDEX_OUT))
    print(f"💾 {NEW_INDEX_OUT} جاهز")

    print("\n" + "=" * 60)
    print("🔬 المقارنة: multilingual-e5-large")
    print("=" * 60)
    evaluate(EASY_TEST_SET, model, index, docs, "مجموعة سهلة (٢٠ سؤال)")
    evaluate(HARD_TEST_SET, model, index, docs, "مجموعة صعبة (١٦ سؤال)")


if __name__ == "__main__":
    main()
