import json
import re

from config import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _token_prefix(text: str, token_count: int) -> str:
    char_limit = max(0, token_count * 4)
    if len(text) <= char_limit:
        return text
    return text[:char_limit].rsplit(" ", 1)[0]


def _token_suffix(text: str, token_count: int) -> str:
    char_limit = max(0, token_count * 4)
    if len(text) <= char_limit:
        return text
    return text[-char_limit:].split(" ", 1)[-1]


def split_into_sections(page_text: str) -> list[dict]:
    """
    Detect NICE-style numbered headings (1.1, 1.2.11) even when the title wraps.
    """
    section_pattern = re.compile(
        r"^(\d+\.\d+(?:\.\d+)*)(?:\s+([A-Z][^\n]{0,120}))?\s*$",
        re.MULTILINE,
    )
    matches = []
    for match in section_pattern.finditer(page_text):
        rest = match.group(2) or ""
        # Table-of-contents dotted leaders are not real section headings.
        if ".." in rest or re.search(r"\.{4,}", rest):
            continue
        matches.append(match)

    if not matches:
        return [{"section_title": "General", "text": page_text}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        title_rest = (match.group(2) or "").strip()
        section_title = f"{match.group(1)} {title_rest}".strip()
        section_text = page_text[start:end].strip()
        if section_text:
            sections.append({"section_title": section_title, "text": section_text})

    return sections or [{"section_title": "General", "text": page_text}]


def chunk_section(
    section_text: str,
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if not paragraphs:
        paragraphs = [section_text]

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if estimate_tokens(candidate) <= max_tokens:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            overlap = _token_suffix(current_chunk, overlap_tokens)
            current_chunk = (overlap + "\n\n" + para).strip() if overlap else para
            if estimate_tokens(current_chunk) > max_tokens:
                chunks.append(para if estimate_tokens(para) <= max_tokens else _token_prefix(para, max_tokens))
                current_chunk = ""
        else:
            chunks.append(_token_prefix(para, max_tokens) if estimate_tokens(para) > max_tokens else para)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return _merge_tiny_chunks(chunks, min_tokens, max_tokens) or [section_text]


def _merge_tiny_chunks(chunks: list[str], min_tokens: int, max_tokens: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < min_tokens:
            combined = (merged[-1] + "\n\n" + chunk).strip()
            if estimate_tokens(combined) <= max_tokens:
                merged[-1] = combined
                continue
        merged.append(chunk)

    if len(merged) >= 2 and estimate_tokens(merged[0]) < min_tokens:
        combined = (merged[0] + "\n\n" + merged[1]).strip()
        if estimate_tokens(combined) <= max_tokens:
            merged = [combined] + merged[2:]

    return merged


def _merge_page_chunks(page_chunks: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    """Merge heading-only leftovers with the previous chunk on the same page."""
    merged: list[dict] = []
    for chunk in page_chunks:
        if merged and chunk["estimated_tokens"] < min_tokens:
            combined_text = (merged[-1]["text"] + "\n\n" + chunk["text"]).strip()
            if estimate_tokens(combined_text) <= max_tokens:
                keep_title = chunk["section_title"]
                if merged[-1]["section_title"] != "General":
                    keep_title = merged[-1]["section_title"]
                merged[-1]["text"] = combined_text
                merged[-1]["section_title"] = keep_title
                merged[-1]["estimated_tokens"] = estimate_tokens(combined_text)
                continue
        merged.append(chunk)
    return merged


def run_step2(
    step1_output_path: str = "step1_output.json",
    output_path: str = "step2_chunks.json",
    min_tokens: int = CHUNK_MIN_TOKENS,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
):
    with open(step1_output_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    last_section_title = "General"

    for page in pages:
        page_number = page["page_number"]
        sections = split_into_sections(page["text"])
        if sections and sections[0]["section_title"] == "General":
            first_line = next((ln.strip() for ln in page["text"].splitlines() if ln.strip()), "")
            unnumbered = (
                "Overview", "Introduction", "Contents", "Your responsibility",
                "Update information", "Finding more information", "Key priorities",
                "Recommendations", "Who is it for",
            )
            if any(first_line.lower().startswith(h.lower()) for h in unnumbered):
                sections[0]["section_title"] = first_line[:120]
            else:
                sections[0]["section_title"] = last_section_title

        page_chunks = []
        for section in sections:
            last_section_title = section["section_title"]
            sub_chunks = chunk_section(
                section["text"],
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            for text_chunk in sub_chunks:
                page_chunks.append({
                    "page_number": page_number,
                    "section_title": section["section_title"],
                    "text": text_chunk,
                    "estimated_tokens": estimate_tokens(text_chunk),
                })

        page_chunks = _merge_page_chunks(page_chunks, min_tokens, max_tokens)
        all_chunks.extend(page_chunks)

    numbered = []
    for i, chunk in enumerate(all_chunks, start=1):
        numbered.append({
            "chunk_id": f"chunk_{i:04d}",
            "page_number": chunk["page_number"],
            "section_title": chunk["section_title"],
            "text": chunk["text"],
            "estimated_tokens": chunk["estimated_tokens"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(numbered, f, ensure_ascii=False, indent=2)

    tiny = sum(1 for c in numbered if c["estimated_tokens"] < min_tokens)
    print(f"[ok] chunks written: {len(numbered)}")
    print(f"min/max/overlap tokens: {min_tokens}/{max_tokens}/{overlap_tokens}")
    print(f"chunks still below min_tokens: {tiny}")
    print(f"[ok] saved: {output_path}")

    return numbered


if __name__ == "__main__":
    run_step2()
