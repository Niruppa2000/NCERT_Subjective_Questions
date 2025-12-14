import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="NCERT Question Generator", layout="wide")
st.title("📘 NCERT Chapter-wise Question Generator")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # -------- Normalize column names --------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

uploaded_file = st.file_uploader("📂 Upload Questions CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    # ✅ Show detected columns (VERY IMPORTANT)
    st.subheader("🧾 Detected CSV Columns")
    st.code(list(df.columns))

    # -------- Expected columns after normalization --------
    required_columns = {
        "chapter",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer"
    }

    missing = required_columns - set(df.columns)

    if missing:
        st.error(f"❌ Missing columns in CSV: {missing}")
        st.stop()

    chapter_name = st.text_input("📖 Enter Chapter Name")
    num_questions = st.number_input("🔢 Number of Questions", 1, 50, 5)

    if st.button("🚀 Generate Questions"):
        filtered = df[df["chapter"].str.lower() == chapter_name.lower()]

        if filtered.empty:
            st.warning("⚠ No questions found for this chapter")
        else:
            sample = filtered.sample(
                min(num_questions, len(filtered)),
                random_state=random.randint(1, 9999)
            )

            for i, row in enumerate(sample.itertuples(), 1):
                st.markdown(f"### Q{i}. {row.question}")
                st.write(f"A. {row.option_a}")
                st.write(f"B. {row.option_b}")
                st.write(f"C. {row.option_c}")
                st.write(f"D. {row.option_d}")

                with st.expander("✅ Show Answer"):
                    st.write(row.answer)
