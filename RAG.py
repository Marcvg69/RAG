import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity


# Sample documents
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "The Great Wall of China is visible from space.",
    "The tallest mountain in the world is Mount Everest."
]

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Embed all documents
doc_embeddings = [embed(doc) for doc in documents]

def retrieve(query, doc_embeddings, documents, top_k=2):
    query_embedding = embed(query)
    similarities = [cosine_similarity(query_embedding, doc_emb)[0][0] for doc_emb in doc_embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices]

# Load generation model
generator = pipeline("text-generation", model="gpt2")

def rag(query, doc_embeddings, documents):
    retrieved_docs = retrieve(query, doc_embeddings, documents)
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"