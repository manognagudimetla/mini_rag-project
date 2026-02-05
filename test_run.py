import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 RAG PIPELINE STARTED")

chunks = []
sources = []

def chunk_text(text, size=120):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

with open("sample.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    print("Pages:", len(reader.pages))

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            for chunk in chunk_text(text):
                chunks.append(chunk)
                sources.append(f"Page {page_no}")

print("Chunks created:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

questions = [
    "What is voltage regulation?",
    "Why is voltage regulation important?",
    "What factors affect voltage regulation?",
    "How is voltage regulation improved?",
    "What is the conclusion of the document?"
]

for q in questions:
    print("\n❓ QUESTION:", q)
    q_emb = model.encode([q])
    scores = cosine_similarity(q_emb, embeddings)[0]
    best = np.argmax(scores)

    print("✅ ANSWER:", chunks[best][:300])
    print("📄 SOURCE:", sources[best])

print("\n🎉 RAG PIPELINE COMPLETED")
