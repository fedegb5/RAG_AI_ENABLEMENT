import PyPDF2
import numpy as np
import pandas as pd
import faiss
from embedding import EmbeddingModel
from dotenv import load_dotenv

load_dotenv()

class FaissDB:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = EmbeddingModel(model_name)

        # self.documents = self.load_marvel_csv()
        self.documents = self.load_pdf_chunks("futbol.pdf", 150)
        self.embeddings = [self.embedding_model.get_embedding(doc) for doc in self.documents]
        self.embeddings = np.array(self.embeddings).astype('float32')

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def add_documents(self, docs):
        new_embeddings = [self.embedding_model.get_embedding(doc) for doc in docs]
        new_embeddings = np.array(new_embeddings).astype('float32')
        self.index.add(new_embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding, top_k=10, threshold=1.2):
        query_embedding = np.array(query_embedding).astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if threshold is None or dist <= threshold:
                #print(f"Documento: {self.documents[idx]} | Distancia: {dist}")
                results.append(self.documents[idx])
        return results

    def load_fifa_csv(self):
        csv_path = "Fifa_world_cup_matches.csv"
        df = pd.read_csv(csv_path)
        docs = []
        for _, row in df.iterrows():
            doc = (
                f"{row['team1']} ({row['number of goals team1']}) vs {row['team2']} ({row['number of goals team2']}) on {row['date']} ({row['category']})"
            )
            docs.append(doc)
        return docs

    def load_marvel_csv(self):
        csv_path = "Marvel_Characters.csv"
        df = pd.read_csv(csv_path)
        docs = []
        for _, row in df.iterrows():
            doc = f"{row['charname']} - Types: {row['types']} - Powers: {row['superpowers']} - Universe: {row['universes']}"
            docs.append(doc)
        return docs
    
    def load_pdf_chunks(self, pdf_path, chunk_size=500):
        reader = PyPDF2.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return chunks