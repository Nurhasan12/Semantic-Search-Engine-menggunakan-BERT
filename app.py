# app.py

import streamlit as st
from model_script import load_data_and_embeddings, find_similar_texts

st.set_page_config(page_title="Text Search App", layout="wide")
st.title("🔍 Sistem Pencarian Teks Mirip dengan BERT")

st.markdown("Masukkan teks untuk menemukan dokumen yang paling mirip dari dataset.")

# Load dataset & embedding
@st.cache_resource
def load_all():
    return load_data_and_embeddings()

df_data, embeddings = load_all()

# Input user
user_input = st.text_area("Teks Pencarian", height=150)

if st.button("Cari Mirip") and user_input.strip():
    with st.spinner("Mencari teks mirip..."):
        result_df = find_similar_texts(user_input, df_data, embeddings, top_n=5)
        st.success("Hasil ditemukan!")
        for i, row in result_df.iterrows():
            st.markdown(f"**Skor: {row['score']:.4f} | Label: `{row['label']}`**")
            st.write(f"> {row['text_raw']}")
            st.markdown("---")
