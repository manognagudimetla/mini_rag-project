import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("PDF RAG Question Answering")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Upload PDF
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    pages = [p.extract_text() for p in reader.pages]

    chunks = [p for p in pages if p.strip()]
    embeddings = model.encode(chunks)

    query = st.text_input("Ask a question")

    if query:
        q_emb = model.encode([query])
        scores = cosine_similarity(q_emb, embeddings)[0]
        best = np.argmax(scores)

        st.subheader("Answer")
        st.write(chunks[best])
