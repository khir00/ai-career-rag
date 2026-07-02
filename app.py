"""
app.py — واجهة Gradio لمشروع Arabic AI Career RAG (نسخة نهائية)
يشتغل على Hugging Face Spaces (SDK: Gradio)

Pipeline: E5-large retrieval (top-10) → Cross-Encoder re-ranking → top-3 → Qwen2.5 generation

محتاج بنفس المجلد: kb_index_v3.faiss و kb_documents_v2.json
"""

import json
from pathlib import Path

import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
GEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
INDEX_PATH = Path("kb_index_v3.faiss")
DOCS_PATH = Path("kb_documents_v2.json")
INITIAL_K = 10
FINAL_K = 3

SYSTEM_PROMPT = (
    "You are a career assistant for Arabic-speaking AI graduates. "
    "Answer ONLY using the provided context. If the context doesn't contain "
    "enough information, say so honestly instead of guessing. "
    "Always mention which topic/source your answer is based on. "
    "Reply in the same language as the question (Arabic or English)."
)

print("📥 تحميل النماذج (بيصير مرة وحدة بس عند بدء تشغيل الـSpace)...")
index = faiss.read_index(str(INDEX_PATH))
docs = json.load(open(DOCS_PATH, encoding="utf-8"))
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, dtype=torch.float32)
print("✅ جاهز")


def retrieve_and_rerank(query, initial_k=INITIAL_K, final_k=FINAL_K):
    q_vec = np.array(embed_model.encode([f"query: {query}"], normalize_embeddings=True), dtype="float32")
    _, idx = index.search(q_vec, initial_k)
    candidates = [docs[i] for i in idx[0]]

    pairs = [(query, f"{d['title']}. {d['text'][:400]}") for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:final_k]]


def build_prompt(query, retrieved_docs):
    context = "\n\n".join(
        f"[Source: {d['title']} — {d['category']}]\n{d['text'][:600]}"
        for d in retrieved_docs
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context above:"


def answer_question(query):
    if not query.strip():
        return "اكتب سؤال أول 🙂", ""

    retrieved = retrieve_and_rerank(query)
    prompt = build_prompt(query, retrieved)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    output = gen_model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    sources_md = "\n\n".join(
        f"**{d['title']}** _({d['category']})_\n\n{d['text'][:250]}..."
        for d in retrieved
    )
    return response, sources_md


with gr.Blocks(title="AI Career RAG Assistant") as demo:
    gr.Markdown("# 🎓 مساعد المسار المهني لخريجي AI")
    gr.Markdown("نظام RAG مبني على مصادر حقيقية — مو إجابات مختلقة")

    query_box = gr.Textbox(
        label="اسأل سؤالك (عربي أو إنجليزي)",
        placeholder="شو بدي أتعلم لأوصل AI Engineer؟",
    )
    submit_btn = gr.Button("🔍 اسأل", variant="primary")

    answer_box = gr.Markdown(label="الجواب")
    with gr.Accordion("📚 المصادر المستخدمة", open=False):
        sources_box = gr.Markdown()

    submit_btn.click(fn=answer_question, inputs=query_box, outputs=[answer_box, sources_box])
    query_box.submit(fn=answer_question, inputs=query_box, outputs=[answer_box, sources_box])

    gr.Markdown("---")
    gr.Markdown("مبني بواسطة Mohamad Khir Alhomsi | roadmap.sh كمصدر بيانات | E5-large + Re-ranking + Qwen2.5-1.5B")

if __name__ == "__main__":
    demo.launch()
