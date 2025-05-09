import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Paths to the PDFs
pdf_files = [
    "HomeEase_FAQ.pdf",
    "product_specs.pdf",
    "setup_guide.pdf"
]

# Step 1: Load and combine documents
all_docs = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    all_docs.extend(loader.load())

# Step 2: Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(all_docs)

# Step 3: Generate embeddings using a local model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in Chroma DB
persist_directory = "chroma_db"
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)
vectordb.persist()

print(f"✅ Ingestion complete. {len(chunks)} chunks indexed.")
