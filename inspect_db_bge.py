# inspect_db_bge.py

import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ========== Config ==========
PERSIST_DIRECTORY = "qwen_vector_store"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ========== Load Embedding Model ==========
embedding = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ========== Load Vector Store ==========
print(f" Loading vector DB from: {PERSIST_DIRECTORY}")
vectordb = FAISS.load_local(
    PERSIST_DIRECTORY,
    embeddings=embedding,
    allow_dangerous_deserialization=True  # Safe since local and trusted
)

# ========== Inspect ==========
docs = vectordb.similarity_search("dummy", k=5)

print("\n Vector DB Metadata")
print("-" * 40)
print(f"Total documents in vector store: {vectordb.index.ntotal}")
print(f"Example chunk metadata and preview:")

for i, doc in enumerate(docs, 1):
    print(f"\n🔹 Document {i}")
    print(f"Title     : {doc.metadata.get('title', 'Untitled')}")
    print(f"URL       : {doc.metadata.get('url', 'N/A')}")
    print(f"Content   : {doc.page_content[:200]}...")

print("\n Vector DB inspection complete.\n")
