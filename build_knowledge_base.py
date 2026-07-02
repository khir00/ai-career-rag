"""
build_knowledge_base.py

يقرأ ملفات markdown من مجلدات roadmap.sh (ai-engineer + machine-learning)،
يشيل التكرار، ينظف المحتوى، وبحوله لصيغة JSON موحدة جاهزة لمرحلة الـchunking/embedding.

المخرج: knowledge_base.jsonl
كل سطر = {"id": ..., "source": ..., "category": ..., "title": ..., "text": ...}
"""

import json
import re
from pathlib import Path

BASE = Path("developer-roadmap/src/data/roadmaps")
SOURCE_FOLDERS = {
    "ai-engineer": "AI Engineer Roadmap",
    "machine-learning": "Machine Learning Roadmap",
}

OUTPUT_FILE = Path("knowledge_base.jsonl")


def slug_to_title(slug: str) -> str:
    """يحول 'prompt-engineering' -> 'Prompt Engineering'"""
    slug = slug.split("--")[0]  # شيل التكرارات متل 'image--video-recognition'
    words = slug.replace("-", " ").split()
    return " ".join(w.capitalize() for w in words)


def clean_markdown(text: str) -> str:
    """يشيل عناصر markdown الزايدة وياخد بس المحتوى النصي المفيد"""
    # شيل الـheader الأول (# Title) لأنه رح ناخد العنوان من اسم الملف
    text = re.sub(r"^#\s+.+\n", "", text, count=1)
    # شيل قسم "Visit the following resources" وكل الروابط بعده
    text = re.split(r"Visit the following resources", text)[0]
    # شيل markdown links [text](url) واحتفظ بالنص بس
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # شيل أسطر فاضية زايدة
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    documents = []
    seen_slugs = set()  # لإزالة التكرار بين المجلدات
    skipped_short = 0
    skipped_dup = 0

    for folder, category in SOURCE_FOLDERS.items():
        content_dir = BASE / folder / "content"
        if not content_dir.exists():
            print(f"⚠️  المجلد مو موجود: {content_dir}")
            continue

        for md_file in sorted(content_dir.glob("*.md")):
            # اسم الملف: slug@hash.md -> ناخد الـslug بس
            slug = md_file.stem.split("@")[0]

            if slug in seen_slugs:
                skipped_dup += 1
                continue

            raw_text = md_file.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_markdown(raw_text)

            # تجاهل الملفات القصيرة جداً (أقل من ٣٠ كلمة) — مو مفيدة كوثيقة مستقلة
            if len(cleaned.split()) < 30:
                skipped_short += 1
                continue

            seen_slugs.add(slug)
            documents.append({
                "id": f"{folder}__{slug}",
                "source": "roadmap.sh",
                "category": category,
                "title": slug_to_title(slug),
                "text": cleaned,
            })

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✅ تم إنشاء {len(documents)} وثيقة نظيفة")
    print(f"   (تجاهلنا {skipped_dup} تكرار و {skipped_short} ملف قصير جداً)")
    print(f"📄 الملف: {OUTPUT_FILE.resolve()}")

    # عرض مثال
    if documents:
        print("\n--- مثال على وثيقة ---")
        print(json.dumps(documents[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
