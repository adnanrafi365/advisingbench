"""
Chunk UIC source documents for AdvisingBench retrieval.

This script:
1. Reads local source text files from data/raw/source_texts/
2. Cleans repeated navigation-like text lightly
3. Splits documents into overlapping text chunks
4. Saves chunks to data/processed/source_chunks.csv

Raw source text is kept local.
Processed chunks are used for retrieval experiments.
"""

from pathlib import Path
import re
import pandas as pd


SOURCE_TEXT_DIR = Path("data/raw/source_texts")
SOURCE_MANIFEST_FILE = Path("data/raw/source_manifest.csv")
OUTPUT_FILE = Path("data/processed/source_chunks.csv")

CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def clean_text(text: str) -> str:
    """Clean scraped source text and remove common navigation noise."""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    skip_exact_phrases = {
        "Skip to Content",
        "AZ Index",
        "Catalog Home",
        "UIC Home",
        "Academic Catalog",
        "Catalog Navigation",
        "Down arrow icon",
        "Undergraduate Catalog",
        "Graduate and Professional Catalog",
        "All Course Descriptions",
        "Home",
        "Archive & Links",
        "On This Page",
    }

    skip_contains_phrases = [
        "University of Illinois Chicago Academic Catalog",
        "2026–2027 Undergraduate Catalog",
        "2025–2026 Undergraduate Catalog",
        "Admissions, Registration & Finances",
        "Degree Programs and Requirements",
        "Colleges & Departments",
    ]

    lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if stripped in skip_exact_phrases:
            continue

        if any(phrase in stripped for phrase in skip_contains_phrases):
            continue

        # Skip very short navigation-like lines.
        if len(stripped.split()) <= 2 and len(stripped) < 25:
            continue

        lines.append(stripped)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def is_low_value_chunk(chunk: str) -> bool:
    """Return True if a chunk is mostly navigation or low-value text."""
    navigation_terms = [
        "Catalog Home",
        "UIC Home",
        "AZ Index",
        "Skip to Content",
        "Catalog Navigation",
        "Archive & Links",
        "Down arrow icon",
        "Colleges & Departments",
        "Course Descriptions",
        "Admissions, Registration & Finances",
    ]

    matches = sum(1 for term in navigation_terms if term in chunk)

    # Skip chunks with many navigation terms.
    if matches >= 3:
        return True

    # Skip chunks that are too short.
    if len(chunk.split()) < 50:
        return True

    return False


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word chunks and skip low-value chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if len(chunk) > 100 and not is_low_value_chunk(chunk):
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def main() -> None:
    if not SOURCE_TEXT_DIR.exists():
        raise FileNotFoundError(f"Missing source text directory: {SOURCE_TEXT_DIR}")

    if not SOURCE_MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Missing source manifest: {SOURCE_MANIFEST_FILE}")

    manifest = pd.read_csv(SOURCE_MANIFEST_FILE)

    rows = []

    for _, source in manifest.iterrows():
        source_id = source["source_id"]
        use_for_rag = source["use_for_rag"]

        if use_for_rag not in ["yes", "review"]:
            continue

        source_file = SOURCE_TEXT_DIR / f"{source_id}.txt"

        if not source_file.exists():
            print(f"Skipping missing file: {source_file}")
            continue

        raw_text = source_file.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)
        chunks = chunk_words(cleaned_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

        print(f"{source_id}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{source_id}_C{i+1:03d}",
                "source_id": source_id,
                "title": source["title"],
                "url": source["url"],
                "category": source["category"],
                "quality": source["quality"],
                "chunk_index": i + 1,
                "text": chunk,
            })

    chunks_df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(OUTPUT_FILE, index=False)

    print("\nChunking complete.")
    print(f"Saved chunks to: {OUTPUT_FILE}")
    print(f"Total chunks: {len(chunks_df)}")

    print("\nChunks by source:")
    print(chunks_df["source_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()
