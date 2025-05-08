from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic.v1 import BaseModel  # Pydantic v2 compatibility
import os
from dotenv import load_dotenv
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Together

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input/output models
class Query(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    response: str

# Load embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(BASE_DIR, "embeddings")

vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Load Together LLM
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.3,
    max_tokens=512,
    top_p=0.9,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# Define route
@app.post("/query", response_model=Answer)
async def process_query(query: Query):
    try:
        response = qa_chain.run(query.question)
        return {"question": query.question, "response": response}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
