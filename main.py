from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# Load the embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load FAISS index and chunks
index = faiss.read_index("faiss_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# FastAPI setup
app = FastAPI()

# Enable CORS so frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "RAG Chatbot is running."}

@app.post("/chat")
def chat(request: QueryRequest):
    query = request.query
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k=1)

    best_chunk = chunks[I[0][0]]
    return {
        "section": best_chunk["section"],
        "chunk_id": best_chunk["chunk_id"],
        "response": best_chunk["text"]
    }
