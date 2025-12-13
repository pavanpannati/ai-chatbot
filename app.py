from fastapi import FastAPI
from pydantic import BaseModel
from rag_graph import rag_app

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    result = rag_app.invoke({"question": q.question})

    return {
        "answer": result["answer"],
        "context_chunks": result["context"],
        "confidence_score": result["confidence"]
    }
