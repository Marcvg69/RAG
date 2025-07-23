from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load PDF
pdf_path = "mydoc.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()

# 2. Create vectorstore (embedding the documents)
embedding = OllamaEmbeddings(model="nomic-embed-text")  # You must have this model pulled
vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory="./chroma_db")

# 3. Define the retriever
retriever = vectordb.as_retriever()

# 4. Load local LLM from Ollama
llm = Ollama(model="llama3")  # Change to "mistral" or another if you prefer

# 5. Build RetrievalQA pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Run a chat loop
print("Ask me anything about the PDF. Type 'exit' to quit.")
while True:
    question = input("\nYou: ")
    if question.lower() in ["exit", "quit"]:
        break

    result = qa_chain({"query": question})
    print(f"\n📘 Answer:\n{result['result']}")

