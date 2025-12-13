import PyPDF2
import requests
from io import BytesIO
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

PDF_PATH = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

def read_pdf(url):
  response = requests.get(url)
  response.raise_for_status()  
  
  
  pdf_file = BytesIO(response.content)
  reader = PyPDF2.PdfReader(pdf_file)

  text = ""
  for page in reader.pages:
      page_text = page.extract_text()
      if page_text:
          text += page_text + "\n"

  return text

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_text = read_pdf(PDF_PATH)
chunks = chunk_text(pdf_text)

embeddings = model.encode(chunks)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "vector.index")
pickle.dump(chunks, open("chunks.pkl", "wb"))

print("PDF ingested successfully")
