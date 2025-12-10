import re
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

# 🔥 IMPORTANT: This loads your fine-tuned model folder
GEN_MODEL_NAME = "flan_t5_ncert_subjective"

TOP_K = 4  # number of context chunks to retrieve


# ============================
# PDF EXTRACTION
# ============================

def extract_text_from_pdf_filelike(file):
    reader = PdfReader(file)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except:
            pages_text.append("")
    return "\n".join(pages_text)


def build_chunks(text, chunk_size=800, overlap=150):
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

@st.cache_resource(show_spinner="Loading embedding + fine-tuned model...")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    try:
        # 🔥 Try loading your fine-tuned model from local folder
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, local_files_only=True)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(
            GEN_MODEL_NAME,
            local_files_only=True
        ).to(device)
        print("Loaded fine-tuned model:", GEN_MODEL_NAME)

    except Exception as e:
        print("⚠️ Could not load fine-tuned model. Reason:", e)
        print("➡️ Falling back to base Flan-T5")

        BASE_MODEL = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    return device, embedder, tokenizer, gen_model


# ============================
# INDEXING + RETRIEVAL
# ============================

def build_index_from_files(uploaded_files):
    docs = []
    for f in uploaded_files:
        raw = extract_text_from_pdf_filelike(f)
        chunks = build_chunks(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            docs.append({"doc_id": f.name, "chunk_id": i, "text": ch})
    return docs


def build_faiss_index(docs, embedder):
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    vecs = []

    for d in docs:
        v = embedder.encode(d["text"], convert_to_numpy=True)
        vecs.append(v)

    vecs = np.vstack(vecs).astype("float32")
    index.add(vecs)
    return index


def retrieve_context(query, index, embedder, docs, top_k=4):
    q = embedder.encode(query, convert_to_numpy=True).astype("float32")
    q = np.expand_dims(q, 0)
    distances, idxs = index.search(q, top_k)
    return [docs[i] for i in idxs[0]]


# ============================
# QUESTION CLEANING
# ============================

def clean_and_extract_questions(raw_text, topic, num_questions):
    raw = " ".join(raw_text.split()).strip()

    # split numbered list: 1. ... 2. ...
    segments = [
        m.group(1).strip()
        for m in re.finditer(r"\d+\.\s*(.+?)(?=\d+\.|$)", raw)
    ]

    if not segments:
        segments = [raw]

    questions = []
    for seg in segments:
        if len(seg.split()) < 4:
            continue
        if not seg.endswith("?"):
            seg = seg.rstrip(". ") + "?"
        questions.append(seg)

    # fallback templates
    templates = [
        f"What do you mean by {topic}?",
        f"Explain {topic} with suitable examples.",
        f"Why is {topic} important? Explain.",
        f"Describe {topic} in your own words.",
        f"List and explain the main facts about {topic}.",
    ]

    seen = {q.lower() for q in questions}

    for t in templates:
        if len(questions) >= num_questions:
            break
        if t.lower() not in seen:
            questions.append(t)

    return questions[:num_questions]


# ============================
# MAIN QUESTION GENERATOR
# ============================

def generate_questions(topic, target_class, num_questions, index, docs, embedder, tokenizer, gen_model, device):
    retrieved = retrieve_context(topic, index, embedder, docs)
    context = "\n\n".join([c["text"] for c in retrieved])

    prompt = f"""
You are an experienced NCERT Class {target_class} teacher.
Using ONLY the textbook extract given in CONTEXT, write {num_questions} exam-style questions
on the topic "{topic}".

Rules:
- Questions must be simple, meaningful, and end with '?'
- Begin with: What, Why, How, Explain, Describe, Define, List
- Format:
1. Question 1?
2. Question 2?
3. Question 3?

CONTEXT:
{context}
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    questions = clean_and_extract_questions(raw, topic, num_questions)

    return questions, retrieved, raw


# ============================
# STREAMLIT UI
# ============================

def main():
    st.set_page_config(page_title="NCERT Subjective Question Generator", layout="wide")
    st.title("📚 NCERT Subjective Question Generator (Fine-tuned Model)")

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs (multiple allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    target_class = col1.selectbox("Select Class", [6,7,8,9,10])
    num_questions = col2.slider("How many questions?", 1, 10, 5)

    topic = st.text_input("Enter Topic (e.g., Balanced diet, Motion, Photosynthesis, Ashoka)")

    if not uploaded_files:
        st.info("Upload at least one NCERT PDF.")
        return

    device, embedder, tokenizer, gen_model = load_models()

    with st.spinner("Building knowledge base from PDFs..."):
        docs = build_index_from_files(uploaded_files)
        index = build_faiss_index(docs, embedder)

    st.success(f"Indexed {len(docs)} text chunks.")

    if topic and st.button("Generate Questions"):
        with st.spinner("Generating..."):
            questions, retrieved, raw = generate_questions(
                topic,
                target_class,
                num_questions,
                index,
                docs,
                embedder,
                tokenizer,
                gen_model,
                device
            )

        st.subheader("📘 Generated Questions")
        for i, q in enumerate(questions, start=1):
            st.write(f"**{i}. {q}**")

        with st.expander("🔍 Raw Model Output"):
            st.write(raw)

        with st.expander("📄 Context Chunks Used"):
            for c in retrieved:
                st.write(f"**{c['doc_id']} – chunk {c['chunk_id']}**")
                st.write(c["text"])
                st.markdown("---")


if __name__ == "__main__":
    main()
