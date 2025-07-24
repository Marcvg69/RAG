import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
cdfrom langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_ollama import OllamaLLM
from pdf_loader import PdfLoader
from vector_store import VectorStore
from chunk_text import Chunker

class RAG:

  def __init__(self, ):
    self.instructor_prompt = """Instruction: You're an expert problem solver you answer questions from context given below. You strictly adhere to the context and never move away from it. You're honest and if you do not find the answer to the question in the context you politely say "I Don't know!"
    So help me answer the user question mentioned below with the help of the context provided
    User Question: {user_query}
    Answer Context: {answer_context}
    """
    self.prompt = PromptTemplate.from_template(self.instructor_prompt)
    self.llm = OllamaLLM(model="llama3.2:3b") #OpenAI()
    self.vectorStore = VectorStore()
    self.pdfloader = PdfLoader()
    self.chunker = Chunker()
    pass

  def run(self, filePath, query):
    docs = self.pdfloader.read_file(filePath)
    list_of_docs = self.chunker.chunk_docs(docs)
    self.vectorStore.add_docs(list_of_docs)
    results = self.vectorStore.search_docs(query)
    answer_context = "\n\n"
    for res in results:
      answer_context = answer_context + "\n\n" + res.page_content
    chain = self.prompt | self.llm
    response = chain.invoke(
        {
            "user_query": query,
            "answer_context": answer_context,
        }
    )
    return response
  pass

if __name__ == "__main__":
  rag = RAG()
  filePath="investment.pdf"
  query="How to invest?"
  response = rag.run(filePath, query)
  print(response)
  pass

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