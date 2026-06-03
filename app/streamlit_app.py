"""
AdvisingBench Streamlit Demo

This demo:
1. Accepts an academic advising question
2. Retrieves relevant UIC source chunks using FAISS
3. Shows a placeholder grounded answer
4. Displays retrieved official sources
5. Adds a safety disclaimer

This is not an official UIC advising tool.
"""

from pathlib import Path
import pickle

import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/processed/vector_index/advisingbench.faiss")
METADATA_FILE = Path("data/processed/vector_index/chunk_metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_FILE}. "
            "Run src/retrieval/build_faiss_index.py first."
        )

    return faiss.read_index(str(INDEX_FILE))


@st.cache_data
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_FILE}. "
            "Run src/retrieval/build_faiss_index.py first."
        )

    with METADATA_FILE.open("rb") as f:
        return pickle.load(f)


def retrieve(question: str, model, index, metadata, top_k: int = TOP_K):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def make_placeholder_answer(question: str, results: list[dict]) -> str:
    if not results:
        return (
            "I could not retrieve enough official UIC source context. "
            "Please check with the appropriate UIC office or advisor."
        )

    source_ids = []
    seen = set()

    for item in results:
        source_id = item["source_id"]
        if source_id not in seen:
            seen.add(source_id)
            source_ids.append(source_id)

    source_text = ", ".join(source_ids)

    return (
        "This is a demo placeholder answer. In the final version, an LLM will answer "
        "using only the retrieved official UIC sources. "
        f"The most relevant retrieved sources are: {source_text}."
    )


def main():
    st.set_page_config(
        page_title="AdvisingBench Demo",
        page_icon="🎓",
        layout="wide",
    )

    st.title("AdvisingBench")
    st.subheader("Groundedness Benchmark for LLM Academic Advising Answers")

    st.warning(
        "This is an independent research-style demo, not an official UIC advising tool. "
        "Students should verify important academic decisions with official UIC advisors, "
        "departments, the Registrar, or the Office of International Services."
    )

    st.markdown(
        """
        **What this demo does:**  
        It retrieves relevant official UIC source chunks for an advising question and shows
        which sources a future citation-grounded LLM answer should use.
        """
    )

    question = st.text_input(
        "Ask an academic advising question:",
        value="Can an F-1 international student take only online classes?",
    )

    if st.button("Retrieve UIC Sources"):
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Loading retrieval system and searching sources..."):
            model = load_embedding_model()
            index = load_faiss_index()
            metadata = load_metadata()
            results = retrieve(question, model, index, metadata, TOP_K)

        st.markdown("## Placeholder Answer")
        st.write(make_placeholder_answer(question, results))

        st.markdown("## Retrieved Official UIC Sources")

        if not results:
            st.info("No source chunks retrieved.")
            return

        for rank, item in enumerate(results, start=1):
            with st.expander(
                f"Rank {rank}: {item['source_id']} — {item['title']} "
                f"(score={item['score']:.4f})"
            ):
                st.write(f"**Chunk ID:** {item['chunk_id']}")
                st.write(f"**Category:** {item['category']}")
                st.write(f"**URL:** {item['url']}")
                st.write("**Text preview:**")
                st.write(item["text"][:1200])


if __name__ == "__main__":
    main()
