import json
import re


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def split_into_sections(page_text: str) -> list[dict]:
    """
    بندور على عناوين بالشكل: رقم.رقم (زي 1.1, 1.2, 1.5.1) متبوعة بعنوان
    """
    section_pattern = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+([A-Z].{3,100})$", re.MULTILINE)
    matches = list(section_pattern.finditer(page_text))

    if not matches:
        return [{"section_title": "General", "text": page_text}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        section_title = f"{match.group(1)} {match.group(2)}".strip()
        section_text = page_text[start:end].strip()
        sections.append({"section_title": section_title, "text": section_text})

    return sections


def chunk_section(section_text: str, min_tokens: int = 400, max_tokens: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [section_text]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if estimate_tokens(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [section_text]


def run_step2(step1_output_path: str = "step1_output.json",
              output_path: str = "step2_chunks.json",
              min_tokens: int = 400,
              max_tokens: int = 800):
    with open(step1_output_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_number = page["page_number"]
        sections = split_into_sections(page["text"])

        for section in sections:
            sub_chunks = chunk_section(section["text"], min_tokens, max_tokens)
            for text_chunk in sub_chunks:
                chunk_counter += 1
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "page_number": page_number,
                    "section_title": section["section_title"],
                    "text": text_chunk,
                    "estimated_tokens": estimate_tokens(text_chunk)
                })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ عدد الـ chunks الناتجة: {len(all_chunks)}")
    print(f"تم الحفظ في: {output_path}")
    print("\n--- عينة من أول 3 chunks ---")
    for c in all_chunks[:3]:
        print(f"\n[{c['chunk_id']}] صفحة {c['page_number']} | {c['section_title']} | {c['estimated_tokens']} token")
        print(c['text'][:150])

    return all_chunks


if __name__ == "__main__":
    run_step2()