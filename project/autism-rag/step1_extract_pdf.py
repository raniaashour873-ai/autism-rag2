import pymupdf as fitz
import re
import json
from collections import Counter


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        pages.append({
            "page_number": page_index + 1,
            "raw_text": text
        })
    doc.close()
    return pages


def find_boilerplate_lines(pages: list[dict], min_repeat_ratio: float = 0.3) -> set:
    """
    بيدور تلقائيًا على أي سطر بيتكرر في نسبة كبيرة من الصفحات (زي الفوتر/الهيدر)
    ويرجعهم كـ set عشان نشيلهم من كل الصفحات. ده بيغنينا عن كتابة regex يدوي
    لكل جملة متكررة.
    """
    line_counter = Counter()
    for page in pages:
        lines = [l.strip() for l in page["raw_text"].split("\n") if l.strip()]
        unique_lines_in_page = set(lines)
        for line in unique_lines_in_page:
            line_counter[line] += 1

    total_pages = len(pages)
    boilerplate = {
        line for line, count in line_counter.items()
        if count / total_pages >= min_repeat_ratio and len(line) > 3
    }
    return boilerplate


def clean_text(text: str, boilerplate_lines: set) -> str:
    lines = text.split("\n")
    cleaned_lines = [l for l in lines if l.strip() not in boilerplate_lines]
    cleaned = "\n".join(cleaned_lines)

    # شيل أي رقم صفحة لوحده في سطر (زي "44" أو "Page 2 of 44")
    cleaned = re.sub(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    cleaned = re.sub(r"^\s*\d{1,3}\s*$", "", cleaned, flags=re.MULTILINE)

    # تنظيف الأسطر والمسافات الفاضية الزيادة
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def run_step1(pdf_path: str, output_path: str = "step1_output.json"):
    print(f"[1/3] بنفتح الملف: {pdf_path}")
    pages = extract_pdf_pages(pdf_path)
    print(f"    عدد الصفحات المستخرجة: {len(pages)}")

    print("[2/3] بنكتشف الأسطر المتكررة (headers/footers)...")
    boilerplate = find_boilerplate_lines(pages)
    print(f"    عدد الأسطر المتكررة اللي هتتشال: {len(boilerplate)}")
    for line in list(boilerplate)[:10]:
        print(f"      - {line[:80]}")

    print("[3/3] بننظف كل الصفحات...")
    cleaned_pages = []
    for p in pages:
        cleaned = clean_text(p["raw_text"], boilerplate)
        if cleaned:  # نتجاهل الصفحات اللي بقت فاضية بالكامل بعد التنظيف
            cleaned_pages.append({
                "page_number": p["page_number"],
                "text": cleaned
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)

    print(f"\n✅ تم الحفظ في: {output_path}")
    print(f"عدد الصفحات بعد التنظيف: {len(cleaned_pages)}")
    print("\n--- عينة من صفحة 2 بعد التنظيف ---")
    sample = next((p for p in cleaned_pages if p["page_number"] == 2), cleaned_pages[0])
    print(sample["text"][:600])
    return cleaned_pages


if __name__ == "__main__":
    run_step1("nice_cg142.pdf")