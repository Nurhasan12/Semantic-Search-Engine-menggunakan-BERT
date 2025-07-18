# model_script.py

import pandas as pd
import numpy as np
import torch
import re
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Setup awal ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Load BERT ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# --- Fungsi Preprocessing ---
def clean_and_process_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --- Fungsi Embedding BERT ---
def get_bert_cls_embedding(text, tokenizer, model):
    if not text.strip():
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze()

# --- Load Dataset dan Embedding ---
def load_data_and_embeddings():
    df_fake = pd.read_csv('Fake.csv')
    df_true = pd.read_csv('True.csv')

    df_fake['label'] = 'fake'
    df_true['label'] = 'true'

    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Pastikan kolom title dan text ada
    if 'title' not in df.columns:
        df['title'] = '[Tidak Ada Judul]'
    else:
        df['title'] = df['title'].fillna('[Tidak Ada Judul]')

    if 'text' not in df.columns:
        df['text'] = '[Tidak Ada Isi Berita]'
    else:
        df['text'] = df['text'].fillna('[Tidak Ada Isi Berita]')

    # Gabungkan untuk embedding
    df['text_raw'] = df['title'] + ' ' + df['text']

    # Ganti kosong/nan
    df['text_raw'] = df['text_raw'].astype(str).replace(['nan', ''], '[Teks Kosong Tidak Ditemukan]')

    # Load embedding
    embeddings = pd.read_csv('bert_embeddings_all_sentences.csv').values
    return df.reset_index(drop=True), embeddings

# --- Fungsi Pencarian Teks Mirip ---
def find_similar_texts(user_query, df, embeddings, top_n=5):
    processed_query = clean_and_process_text(user_query)
    query_embedding = get_bert_cls_embedding(processed_query, tokenizer, model).reshape(1, -1)
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]

    # Ambil hasil: title, text, label, skor similarity
    results = df.iloc[top_indices][['title', 'text', 'label']].copy()
    results['score'] = scores[top_indices]
    return results.reset_index(drop=True)
