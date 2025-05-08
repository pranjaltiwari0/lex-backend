import PyPDF2
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="your_pinecone_env")
index = pinecone.Index("lex-ai-legal-db")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50  
    )
    return splitter.split_text(text)

# Convert chunks to embeddings
def convert_to_embeddings(text_chunks):
    embedding_model = OpenAIEmbeddings()  # Using OpenAI's embedding model
    return [embedding_model.embed_query(chunk) for chunk in text_chunks]

# Store embeddings in Pinecone
def store_embeddings(text_chunks, embeddings):
    for i, (chunk, vector) in enumerate(zip(text_chunks, embeddings)):
        index.upsert(vectors=[(f"doc_chunk_{i}", vector, {"text": chunk})])

#Retrieve relevant chunks for a user query
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = OpenAIEmbeddings().embed_query(query)
    results = index.query(queries=[query_embedding], top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results["matches"]]

# Main Execution Flow
if __name__ == "__main__":
    pdf_path = "legal_document.pdf"  
    text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(text)
    embeddings = convert_to_embeddings(text_chunks)
    store_embeddings(text_chunks, embeddings)


    
    print("Relevant Legal Texts:")
    for chunk in retrieved_chunks:
        print(f"- {chunk}\n")
