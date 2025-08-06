# json_to_vector_bge.py

import os
import json
import torch
import shutil
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# ====== Config ======
INPUT_FILE = "url_docs.json"                #Change file name as needed
TEMP_FILE = "temp_input.json"               # Temporary file for processing
PERSIST_DIRECTORY = "qwen_vector_store"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ====== Load and Save JSON to temp ======
print(" Loading JSON...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f" Loaded {len(data)} entries")

with open(TEMP_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

# ====== Load into LangChain documents ======
loader = JSONLoader(
    file_path=TEMP_FILE,
    jq_schema=".[]",
    content_key="response",
    metadata_func=lambda record, metadata: {
        "title": record.get("title", "No Title"),
        "url": record.get("url", "No URL")
    }
)
docs = loader.load()
print(f" Loaded {len(docs)} documents")

# ====== Text Splitting ======
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f" Split into {len(chunks)} chunks")

# ====== Embedding Model ======
embedding = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ====== Vector Store ======
if os.path.exists(PERSIST_DIRECTORY):
    shutil.rmtree(PERSIST_DIRECTORY)

vectordb = FAISS.from_documents(chunks, embedding)
vectordb.save_local(PERSIST_DIRECTORY)
print(f" Vector store saved to: {PERSIST_DIRECTORY}")
