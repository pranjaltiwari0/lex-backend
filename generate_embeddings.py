import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # v2-compatible Document

# 📁 Path to your PDF files
pdf_folder = r"C:\Users\pranj\Desktop\Lex AI\backend\data"

# 📦 Load and split all PDFs
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
        docs = loader.load()
        
        # Convert to Pydantic v2-compatible Document if needed
        clean_docs = [Document(page_content=doc.page_content, metadata=dict(doc.metadata)) for doc in docs]
        chunks = splitter.split_documents(clean_docs)
        all_chunks.extend(chunks)

print(f"✅ Loaded and split {len(all_chunks)} chunks.")

# 🧠 Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧾 Store in FAISS index
vectorstore = FAISS.from_documents(all_chunks, embeddings)

# 💾 Save FAISS index (compatible with Pydantic v2)
save_path = r"C:\Users\pranj\Desktop\Lex AI\backend\embeddings"
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)

print("✅ Embeddings generated and saved to FAISS index!")
