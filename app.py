import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# -------------------------------
# CONFIG
# -------------------------------
BASE_MODEL = "google/flan-t5-base"
LORA_PATH = "./lora_ncert_finetuned"

device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("📘 NCERT Question Generator (Fine-tuned LoRA Model)")
st.write("Generate high-quality subjective questions for NCERT Classes 6–10.")

# -------------------------------
# LOAD MODEL + LORA ADAPTER
# -------------------------------
@st.cache_resource
def load_model():
    st.info("Loading base model... please wait 15–30 seconds.")

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    ).to(device)

    st.info("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.to(device)

    return tokenizer, model


tokenizer, model = load_model()

# -------------------------------
# GENERATION FUNCTION
# -------------------------------
def generate_questions(topic, num_questions):
    prompt = (
        f"You are an expert NCERT teacher.\n"
        f"Write {num_questions} detailed, high-quality subjective questions "
        f"from the chapter/topic: {topic}.\n"
        "Number the questions like:\n"
        "1.\n2.\n3.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.4,
        top_p=0.9,
        do_sample=True
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded


# -------------------------------
# STREAMLIT UI
# -------------------------------
topic = st.text_input("Enter chapter/topic:", placeholder="Nutrition in Plants")
num = st.number_input("How many questions?", min_value=1, max_value=20, value=5)

if st.button("Generate Questions"):
    if topic.strip() == "":
        st.error("Please enter a chapter or topic name.")
    else:
        st.write("⏳ Generating... please wait...")
        result = generate_questions(topic, num)
        st.subheader("Generated Questions")
        st.write(result)
