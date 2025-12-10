import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

# ============================
# CONFIG
# ============================
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"   # use flan-t5-large if you have more RAM
TOP_K = 5


# ============================
# PDF & TEXT UTILITIES
# ============================
def extract_text_from_pdf_filelike(file) -> str:
    """Read text from a PDF uploaded via Streamlit (BytesIO-like object)."""
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


def build_chunks(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split large text into overlapping chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================
# LOAD MODELS (CACHED)
# ============================
@st.cache_resource(show_spinner="Loading models (embedding + Flan-T5)...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)
    return device, embedder, tokenizer, gen_model


# ============================
# INDEX BUILDING
# ============================
def build_index_from_files(uploaded_files):
    """Convert uploaded PDFs into chunked docs."""
    docs = []  # list of {"doc_id", "chunk_id", "text"}
    for f in uploaded_files:
        raw_text = extract_text_from_pdf_filelike(f)
        chunks = build_chunks(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, ch in enumerate(chunks):
            docs.append(
                {
                    "doc_id": f.name,
                    "chunk_id": idx,
                    "text": ch,
                }
            )
    return docs


def build_faiss_index(docs, embedder):
    """Build a FAISS index from chunk embeddings."""
    emb_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(emb_dim)

    vectors = []
    for d in docs:
        vec = embedder.encode(d["text"], convert_to_numpy=True, show_progress_bar=False)
        vectors.append(vec)

    vectors = np.vstack(vectors).astype("float32")
    index.add(vectors)
    return index


# ============================
# RETRIEVAL
# ============================
def retrieve_context(query: str, index, embedder, docs, top_k: int = TOP_K):
    """Retrieve top_k relevant chunks from the FAISS index."""
    q_vec = embedder.encode(query, convert_to_numpy=True).astype("float32")
    q_vec = np.expand_dims(q_vec, axis=0)
    distances, indices = index.search(q_vec, top_k)
    indices = indices[0]
    retrieved = [docs[i] for i in indices]
    return retrieved


# ============================
# QUESTION POST-PROCESSING
# ============================
def clean_and_extract_questions(raw_text: str, topic: str, num_questions: int):
    """
    Take raw model output and extract a clean list of questions.
    If the model doesn't give enough good lines, fill using templates.
    """
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    questions = []

    for ln in lines:
        text = ln

        # Remove numbering like "1. " or "2) "
        if text[0].isdigit():
            # find first '.' or ')'
            pos_dot = text.find(".")
            pos_paren = text.find(")")
            cut_pos = -1
            if pos_dot != -1 and pos_paren != -1:
                cut_pos = min(pos_dot, pos_paren)
            elif pos_dot != -1:
                cut_pos = pos_dot
            elif pos_paren != -1:
                cut_pos = pos_paren
            if cut_pos != -1 and cut_pos + 1 < len(text):
                text = text[cut_pos + 1 :].strip()

        # Remove leading bullet
        if text.startswith("- "):
            text = text[2:].strip()

        # Ignore very short junk like "On what??"
        if len(text.split()) < 4:
            continue

        # Ensure it ends with '?'
        if not text.endswith("?"):
            text = text.rstrip(". ") + "?"

        questions.append(text)

    # Fallback: use templates if not enough questions
    templates = [
        f"What do you mean by {topic}?",
        f"Explain {topic} in detail with suitable examples.",
        f"Why is {topic} important? Explain.",
        f"Describe {topic} in your own words.",
        f"List and explain the main features of {topic}.",
        f"How does {topic} affect our daily life? Explain.",
        f"Write a short note on {topic}.",
    ]

    # Add templates until we have at least num_questions
    for t in templates:
        if len(questions) >= num_questions:
            break
        if t not in questions:
            questions.append(t)

    # Return exactly num_questions
    return questions[:num_questions]


# ============================
# QUESTION GENERATION
# ============================
def generate_questions(
    topic: str,
    target_class: int,
    num_questions: int,
    index,
    docs,
    embedder,
    tokenizer,
    gen_model,
    device,
):
    """
    Generate questions by asking Flan-T5 for a numbered list,
    then cleaning and enforcing proper question format.
    """
    # 1) Retrieve context
    chunks = retrieve_context(topic, index, embedder, docs, top_k=TOP_K)
    context_text = "\n\n".join([c["text"] for c in chunks])

    # 2) Build prompt
    prompt = f"""
You are an experienced NCERT Class {target_class} teacher.

Using ONLY the textbook extract given in CONTEXT, write {num_questions} clear, exam-style questions
on the topic "{topic}".

Rules:
- Questions must be simple and meaningful for Class {target_class}.
- They should start with words like: What, Why, How, Explain, Describe, Define, List, etc.
- Each question must be complete and end with a question mark (?).
- Write them in this exact format:
1. Question 1?
2. Question 2?
3. Question 3?
Do not add any extra sentences before or after the list.

CONTEXT:
{context_text}
"""

    # 3) Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 4) Clean + enforce question format + fallback templates
    questions = clean_and_extract_questions(raw_text, topic, num_questions)

    # Nicely numbered block for display
    questions_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return questions_block, chunks


# ============================
# STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="NCERT Subjective Question Generator", layout="wide")
    st.title("📚 NCERT Subjective Question Generator (Classes 6–10)")

    st.markdown(
        """
Upload **NCERT PDFs (Science / Social Science / etc.) for Classes 6–10**  
and generate **exam-style questions** like:

- *What is a balanced diet?*  
- *How is photosynthesis useful to plants? Explain.*  
- *What do you mean by the Harappan civilisation?*
"""
    )

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs (you can select multiple files)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        target_class = st.selectbox("Select Class", [6, 7, 8, 9, 10], index=0)
    with col2:
        num_questions = st.slider("How many questions?", 1, 10, 3)

    topic = st.text_input(
        "Enter Topic (Example: Balanced diet, Motion, Acids Bases Salts, Ashoka, Mughal Empire)"
    )

    if not uploaded_files:
        st.info("👆 Please upload at least one NCERT PDF to begin.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Reading PDFs and building knowledge base..."):
        docs = build_index_from_files(uploaded_files)
        if not docs:
            st.error("No text could be extracted from the uploaded PDFs.")
            return
        index = build_faiss_index(docs, embedder)

    st.success(f"Indexed {len(docs)} text chunks from {len(uploaded_files)} PDF(s).")

    if topic and st.button("Generate Questions"):
        with st.spinner(f"Generating {num_questions} questions..."):
            questions_text, retrieved = generate_questions(
                topic=topic,
                target_class=target_class,
                num_questions=num_questions,
                index=index,
                docs=docs,
                embedder=embedder,
                tokenizer=tokenizer,
                gen_model=gen_model,
                device=device,
            )

        st.subheader("📄 Generated Questions")
        st.write(questions_text)

        with st.expander("Show textbook chunks used"):
            for r in retrieved:
                st.markdown(f"**{r['doc_id']} – chunk {r['chunk_id']}**")
                st.write(r["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()
