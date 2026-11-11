import os
import re
import json
import pymupdf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import PAPERS, PAPERS_DIR, INDEX_DIR

SECTION_REGEX = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\s+[A-Z][A-Za-z0-9\s\-]+|[A-Z\s]{4,30})$")

def clean_text(text: str) -> str:
    # Normalize whitespaces, remove excessive linebreaks
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"-\n([a-z])", r"\1", text) # fix hyphenation
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def extract_pages(pdf_path: Path, paper_id: str, paper_title: str) -> List[Dict[str, Any]]:
    doc = pymupdf.open(pdf_path)
    pages = []
    current_section = "Introduction"

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        blocks = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
        page_text_parts = []

        for b in blocks:
            if b[6] == 0: # text block
                b_text = b[4].strip()
                if not b_text:
                    continue
                first_line = b_text.split("\n")[0].strip()
                # Check for section heading
                if len(first_line) < 60 and SECTION_REGEX.match(first_line):
                    current_section = first_line
                page_text_parts.append(b_text)

        full_page_text = clean_text("\n\n".join(page_text_parts))
        pages.append({
            "paper_id": paper_id,
            "paper_title": paper_title,
            "page": page_num,
            "section": current_section,
            "text": full_page_text
        })
    doc.close()
    return pages

def token_count(text: str) -> int:
    # Approximate token count: ~4 chars per token or ~1.3 tokens per word
    words = text.split()
    return max(1, int(len(words) * 1.3))

def chunk_text(pages: List[Dict[str, Any]], target_tokens: int) -> List[Dict[str, Any]]:
    chunks = []
    chunk_idx = 0

    for p in pages:
        page_text = p["text"]
        paragraphs = [p_text.strip() for p_text in page_text.split("\n\n") if p_text.strip()]
        
        current_chunk_words = []
        current_tokens = 0
        section = p["section"]

        for para in paragraphs:
            para_tokens = token_count(para)
            if current_tokens + para_tokens > target_tokens and current_chunk_words:
                chunk_str = " ".join(current_chunk_words)
                chunks.append({
                    "chunk_id": f"{p['paper_id']}_p{p['page']}_c{chunk_idx}",
                    "paper_id": p["paper_id"],
                    "paper_title": p["paper_title"],
                    "page": p["page"],
                    "section": section,
                    "text": chunk_str,
                    "tokens": token_count(chunk_str)
                })
                chunk_idx += 1
                current_chunk_words = []
                current_tokens = 0

            # If a single paragraph is larger than target_tokens, split into subchunks
            if para_tokens > target_tokens:
                words = para.split()
                sub_size = max(1, int(target_tokens / 1.3))
                for i in range(0, len(words), sub_size):
                    sub_words = words[i:i + sub_size]
                    sub_str = " ".join(sub_words)
                    chunks.append({
                        "chunk_id": f"{p['paper_id']}_p{p['page']}_c{chunk_idx}",
                        "paper_id": p["paper_id"],
                        "paper_title": p["paper_title"],
                        "page": p["page"],
                        "section": section,
                        "text": sub_str,
                        "tokens": token_count(sub_str)
                    })
                    chunk_idx += 1
            else:
                current_chunk_words.append(para)
                current_tokens += para_tokens

        if current_chunk_words:
            chunk_str = " ".join(current_chunk_words)
            chunks.append({
                "chunk_id": f"{p['paper_id']}_p{p['page']}_c{chunk_idx}",
                "paper_id": p["paper_id"],
                "paper_title": p["paper_title"],
                "page": p["page"],
                "section": section,
                "text": chunk_str,
                "tokens": token_count(chunk_str)
            })
            chunk_idx += 1

    return chunks

def build_indices():
    all_pages = []
    for paper_id, info in PAPERS.items():
        pdf_path = PAPERS_DIR / info["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing {pdf_path}. Run downloader.py first.")
        pages = extract_pages(pdf_path, paper_id, info["title"])
        all_pages.extend(pages)
        print(f"Extracted {len(pages)} pages from {info['title']}")

    # Build 3 chunk settings: 300, 600, 800 tokens
    settings = [300, 600, 800]
    for target in settings:
        chunks = chunk_text(all_pages, target_tokens=target)
        out_file = INDEX_DIR / f"chunks_{target}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {out_file} (target tokens: {target})")

if __name__ == "__main__":
    build_indices()
