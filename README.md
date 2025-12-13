User Question
     ↓
Text Embedding (Sentence Transformers)
     ↓
FAISS Vector Search
     ↓
Relevant Chunks Retrieved
     ↓
LLM Answer Generation
     ↓
Final Answer + Context + Confidence


├── ingest.py             # PDF ingestion & vector creation
├── rag_graph.py          # LangGraph RAG pipeline
├── vector.index          # FAISS vector index
├── chunks.pkl            # Stored text chunks
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation


Installation
pip install -r requirements.txt

py ingest.py

This will create:
vector.index
chunks.pkl

py rag_graph.py

uvicorn app:main --reload
