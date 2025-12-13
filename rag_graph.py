import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langgraph.graph import StateGraph


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


index = faiss.read_index("vector.index")
chunks = pickle.load(open("chunks.pkl", "rb"))


llm = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)


class RAGState(dict):
    question: str
    context: list
    answer: str
    confidence: float


def retrieve(state: RAGState):
    query_embedding = embed_model.encode([state["question"]])
    distances, indices = index.search(np.array(query_embedding), 5)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    confidence = float(1 / (1 + distances[0][0]))

    return {
        "context": retrieved_chunks,
        "confidence": confidence
    }

def generate(state: RAGState):
    context_text = "\n\n".join(state["context"])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not found, say "Not available in the document".

Context:
{context_text}

Question:
{state["question"]}
"""

    response = llm(prompt)[0]["generated_text"]

    return {"answer": response}


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")

rag_app = graph.compile()
