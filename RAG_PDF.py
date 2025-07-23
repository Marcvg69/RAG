# Install required packages
# pip install langchain openai faiss-cpu tiktoken unstructured

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Step 1: Load and split PDF
loader = PyPDFLoader("example.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Step 2: Embed and store vectors
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)

# Step 3: Create RAG pipeline
retriever = vectorstore.as_retriever()
llm = ChatOpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 4: Ask a question!
query = "What is the main conclusion of the document?"
result = qa_chain.run(query)
print(result)

