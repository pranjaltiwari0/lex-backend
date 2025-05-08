import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 📁 Path to your PDF files
pdf_folder = r"C:\Users\pranj\Desktop\Lex AI\backend\data"

# 📦 Load all PDFs
all_chunks = []
for file_name in os.listdir(pdf_folder):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file_name))
        docs = loader.load()

        # 📚 Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

print(f"✅ Loaded and split {len(all_chunks)} chunks.")

# 🧠 Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧾 Store in FAISS index
vectorstore = FAISS.from_documents(all_chunks, embeddings)

# 💾 Save FAISS index
save_path = r"C:\Users\pranj\Desktop\Lex AI\backend\embeddings"
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)

print("✅ Embeddings generated and saved to FAISS index!")
