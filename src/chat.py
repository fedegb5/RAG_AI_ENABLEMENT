class Chatbot:
    def __init__(self, openai_api, embedding_model, faiss_db):
        self.openai_api = openai_api
        self.embedding_model = embedding_model
        self.faiss_db = faiss_db

    def ask_question(self, question):
        question_embedding = self.embedding_model.get_embedding(question)

        relevant_docs = self.faiss_db.search(question_embedding)
        print(f"\nRelevant documents: {relevant_docs}\n")

        context = self._format_context(relevant_docs)

        response = self.openai_api.generate_response(question, context)
        return response

    def _format_context(self, documents):
        return "\n".join(documents)