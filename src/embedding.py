
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
	def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
		self.model = SentenceTransformer(model_name)

	def get_embedding(self, text: str):
		return self.model.encode(text)
