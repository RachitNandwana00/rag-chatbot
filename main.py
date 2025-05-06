from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

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

# Pydantic model for JSON API input
class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Chatbot</h2>
            <form action="/chat-form" method="post">
                <input type="text" name="query" placeholder="Ask your question here" required>
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """

# 🟦 Form endpoint for browser/manual use
@app.post("/chat-form")
def chat_form(query: str = Form(...)):
    return run_query(query)

# 🟨 JSON API endpoint for frontend/app use
@app.post("/chat")
def chat_api(request: QueryRequest):
    return run_query(request.query)

# 🔁 Shared logic
def run_query(query: str):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k=1)

    best_chunk = chunks[I[0][0]]
    return {
        "section": best_chunk["section"],
        "chunk_id": best_chunk["chunk_id"],
        "response": best_chunk["text"]
    }
