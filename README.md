# Mini-RAG Application 

This project implements a small Retrieval-Augmented Generation (RAG) pipeline
using a PDF file as the knowledge source.

The system reads a PDF, splits text into chunks, creates embeddings,
retrieves relevant chunks for a query, and reranks results to return
the best answer with citation.

------------------------------------------------------------

ARCHITECTURE

PDF → Text Extraction → Chunking → Embeddings → Retriever (Top-K)
→ Reranker → Answer + Citation

Pipeline steps:

1. Read PDF
2. Extract text page-wise
3. Split text into chunks
4. Generate embeddings
5. Retrieve top-K relevant chunks
6. Rerank candidates
7. Display best result

------------------------------------------------------------

CHUNKING STRATEGY

Chunking is performed using fixed-size word windows.

Parameters used:

Chunk size = 120 words
Overlap = 20 words
Strategy = sliding window chunking

Example:

Chunk 1 → words 0–120
Chunk 2 → words 100–220
Chunk 3 → words 200–320

This preserves semantic continuity across chunks.

------------------------------------------------------------

EMBEDDING MODEL

Model:
all-MiniLM-L6-v2

Vector dimension:
384

Used for:
- document chunk embeddings
- query embedding

------------------------------------------------------------

RETRIEVER

Retriever type:
Embedding similarity retriever

Similarity metric:
Cosine similarity

Top-K retrieved:
3

The retriever selects the most relevant chunks from the PDF.

------------------------------------------------------------

RERANKER

Reranker model:
cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker re-scores retrieved chunks using deeper semantic
comparison between the query and candidate text.

Final answer is selected from the highest scoring chunk.

------------------------------------------------------------

QUICKSTART

Step 1 — Install dependencies

pip install -r requirements.txt

Step 2 — Run Streamlit app

streamlit run app.py

Step 3 — Open browser

https://2pa3vpwadsm2s5mcwcex2f.streamlit.app/
Upload the sample PDF and enter a query.

------------------------------------------------------------

REPOSITORY STRUCTURE

mini-rag
│
├── app.py
├── sample.pdf
├── requirements.txt
└── README.md

------------------------------------------------------------

REMARKS

This implementation uses a PDF file as a small knowledge base to
demonstrate a minimal RAG pipeline.

For production systems, the retriever can be connected to a hosted
vector database such as Qdrant, Pinecone, or Weaviate.

Chunking parameters and Top-K retrieval values can be tuned based on
document size and latency requirements.
RESUME LINK :https://drive.google.com/file/d/1m3GbxdnhCBLXoVEEoBOB3LG8THDJ7LEc/view?usp=drivesdk
