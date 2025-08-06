import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.schema.document import Document

# ========== Config ==========
PERSIST_DIRECTORY = "qwen_vector_store"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# ========== Load Embeddings & Vector Store ==========
print(" Loading vector DB from:", PERSIST_DIRECTORY)
embedding = HuggingFaceBgeEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectordb = FAISS.load_local(PERSIST_DIRECTORY, embeddings=embedding, allow_dangerous_deserialization=True)

# ========== Load Qwen3 Reranker ==========
print(" Loading Qwen3-Reranker model...")
tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)

# ========== Rerank Function ==========
def rerank(query, docs, top_n=10):
    scores = []
    for doc in docs:
        pair = tokenizer(
            query, doc.page_content,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            output = reranker(**pair)
            score = output.logits[0][1].item()
            scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_n]

# ========== CLI Loop ==========
print(" Ask your query (type 'exit' to quit):")
while True:
    query = input("\n Your Query: ").strip()
    if query.lower() == "exit":
        break

    print("\n Searching in vector DB...\n")
    retrieved = vectordb.similarity_search(query, k=20)
    top_scored = rerank(query, retrieved, top_n=5)

    print(f" Top 10 Results for '{query}':\n")
    for i, (score, doc) in enumerate(top_scored, 1):
        title = doc.metadata.get("title", "Untitled")
        url = doc.metadata.get("url", "No URL")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        print(f"{i}. Title: {title}")
        print(f"   Relevance Score: {score}")
        print(f"   Extracted Content: {snippet}...")
        print(f"   Document Link: {url}")
        print("-" * 60)

    print("\n" + "=" * 50)


# # hybrid_search.py

# import os
# import json
# import torch
# from rank_bm25 import BM25Okapi
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.schema.document import Document
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # === Config ===
# VECTOR_DB_PATH = "qwen_vector_store"
# JSON_PATH = "url_docs.json"
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# # === Load and prepare documents ===
# print("📥 Loading documents from JSON...")
# with open(JSON_PATH, 'r', encoding='utf-8') as f:
#     raw_docs = json.load(f)

# bm25_corpus = []
# langchain_docs = []

# for entry in raw_docs:
#     text = entry.get("response", "")
#     tokens = text.lower().split()
#     bm25_corpus.append(tokens)
#     langchain_docs.append(Document(page_content=text, metadata={
#         "title": entry.get("title", ""),
#         "url": entry.get("url", "")
#     }))

# print(f"✅ Loaded {len(langchain_docs)} documents for BM25 and reranking.")
# bm25 = BM25Okapi(bm25_corpus)

# # === Load Vector DB ===
# print("🔍 Loading FAISS vector DB...")
# embedding_model = HuggingFaceBgeEmbeddings(
#     model_name=MODEL_NAME,
#     model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#     encode_kwargs={"normalize_embeddings": True}
# )
# vectordb = FAISS.load_local(
#     VECTOR_DB_PATH,
#     embeddings=embedding_model,
#     allow_dangerous_deserialization=True
# )

# # === Load Qwen3 Reranker ===
# print("🚀 Loading Qwen3 reranker...")
# tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
# reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)

# # === Hybrid + Rerank Logic ===
# def hybrid_search(query, top_k=10, alpha=0.5):
#     query_tokens = query.lower().split()

#     # --- BM25 ---
#     bm25_scores = bm25.get_scores(query_tokens)

#     # --- Embedding ---
#     vector_results = vectordb.similarity_search_with_score(query, k=top_k * 3)

#     combined_scores = []
#     for doc, emb_score in vector_results:
#         doc_text = doc.page_content
#         doc_tokens = doc_text.lower().split()

#         # Match to BM25 score
#         bm25_index = next((i for i, d in enumerate(langchain_docs) if d.page_content == doc_text), -1)
#         bm25_score = bm25_scores[bm25_index] if bm25_index != -1 else 0

#         # Combined score (hybrid)
#         final_score = alpha * bm25_score + (1 - alpha) * emb_score

#         # Rerank with Qwen3
#         input_pair = tokenizer(query, doc_text, return_tensors="pt", truncation=True, padding=True)
#         with torch.no_grad():
#             output = reranker(**input_pair)
#             rerank_score = output.logits[0][1].item()

#         combined_scores.append((rerank_score, doc))

#     # Sort by reranked score
#     top_docs = sorted(combined_scores, key=lambda x: x[0], reverse=True)[:top_k]

#     # Display results
#     for i, (score, doc) in enumerate(top_docs, 1):
#         print(f"{i}. Title: {doc.metadata.get('title')}")
#         print(f"   Relevance Score: {round(score, 2)}%")
#         print(f"   Extracted Content: {doc.page_content[:300].strip()}...")
#         print(f"   Document Link: {doc.metadata.get('url')}\n{'-'*60}")

# # === Main CLI Loop ===
# if __name__ == "__main__":
#     while True:
#         query = input("\n🧠 Your Query (or type 'exit'): ").strip()
#         if query.lower() == "exit":
#             break
#         hybrid_search(query)
