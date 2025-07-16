# app.py
import streamlit as st
from model_script import load_data_and_embeddings, find_similar_texts

# --- Konfigurasi halaman ---
st.set_page_config(
    page_title="BERT Search",
    layout="centered",
    page_icon="🔍",
)

# --- CSS untuk gaya Google-like ---
st.markdown("""
    <style>
        body {
            background-color: #fff;
        }
        .search-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 100px;
        }
        .search-box {
            width: 100%;
            max-width: 600px;
        }
        .stTextInput > div > input {
            font-size: 20px;
            padding: 20px;
        }
        .result-box {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Logo Google-like ---
st.markdown("<h1 style='text-align: center; font-size: 48px; font-family: sans-serif;'>BERT Search</h1>", unsafe_allow_html=True)

# --- Load data ---
@st.cache_resource
def load_all():
    return load_data_and_embeddings()

df_data, embeddings = load_all()

# --- Input di tengah ---
with st.form("search_form"):
    user_input = st.text_input("", placeholder="Search...", key="search_input")
    search_button = st.form_submit_button("Search")

# --- Hasil pencarian ---
if search_button and user_input.strip():
    with st.spinner("Searching..."):
        result_df = find_similar_texts(user_input, df_data, embeddings, top_n=5)
        st.markdown("---")
        for i, row in result_df.iterrows():
            label_color = "green" if row['label'].lower() == "true" else "red"
            st.markdown(f"""
              <div class="result-box">
                <div style='font-size: 18px; color: blue;'>{row['text_raw']}</div>
                <div style='color: #555;'>
                  Label: <b style='color: {label_color};'>{row['label']}</b> | Score: {row['score']:.4f}
                </div>
              </div>
            """, unsafe_allow_html=True)
