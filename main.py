from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from datetime import datetime

# Load the embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load FAISS index and chunks
index = faiss.read_index("faiss_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load generation model (flan-t5-small)
rephraser = pipeline("text2text-generation", model="google/flan-t5-small")

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

# Time-based greeting helper
def get_time_based_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! How can I help you today?"
    elif hour < 17:
        return "Good afternoon! How can I help you today?"
    else:
        return "Good evening! How can I help you today?"

# Greeting keywords
GREETINGS = ["hi", "hello", "hey", "heyyyy", "good morning", "good afternoon", "good evening"]

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

# 🔁 Shared query logic
def run_query(query: str):
    # Handle greeting
    if query.lower().strip() in GREETINGS:
        return {"response": get_time_based_greeting()}

    # Encode and normalize query embedding
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # FAISS search
    D, I = index.search(np.array(query_embedding).astype("float32"), k=1)
    best_chunk = chunks[I[0][0]]
    chunk_text = best_chunk["text"]

    # Strict prompt to reduce hallucination
    prompt = f"""
You are a helpful assistant. Answer the question only based on the information in the context below.
If the answer is not found in the context, say "I don't know."

Context:
{chunk_text}

Question: {query}
Answer:
    """

    # Generate response using rephraser
    generated = rephraser(prompt, max_length=100, do_sample=False)[0]['generated_text']

    return {
        "section": best_chunk["section"],
        "chunk_id": best_chunk["chunk_id"],
        "response": generated
    }
