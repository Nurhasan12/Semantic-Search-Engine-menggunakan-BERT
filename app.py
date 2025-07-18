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
        .stTextInput > div > input {
            font-size: 20px;
            padding: 20px;
        }
        .result-box {
            border-bottom: 1px solid #eee;
            padding: 20px 0;
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
    user_input = st.text_input("", placeholder="Search news article...", key="search_input")
    search_button = st.form_submit_button("Search")

# --- Hasil pencarian ---
if search_button and user_input.strip():
    with st.spinner("Searching..."):
        result_df = find_similar_texts(user_input, df_data, embeddings, top_n=5)
        st.markdown("---")
        for i, row in result_df.iterrows():
            label_color = "green" if row['label'].lower() == "true" else "red"
            preview_text = row['text']
            preview_text = preview_text.replace('\n', ' ').strip()
            if len(preview_text) > 200:
                preview_text = preview_text[:200].rsplit(' ', 1)[0] + "..."  # Potong jadi 2 baris (approx)

            st.markdown(f"""
                <div class="result-box">
                    <h4 style='margin-bottom:5px; color:#1a0dab;'>{row['title']}</h4>
                    <p style='margin:5px 0; color:#4d4d4d; font-size:15px; line-height:1.4em; max-height:2.8em; overflow:hidden;'>{preview_text}</p>
                    <div style='font-size:14px; color:#666;'>
                        Label: <b style='color: {label_color};'>{row['label']}</b> |
                        Score: <b>{row['score']:.4f}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

