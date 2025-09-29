import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class OpenAIAPI:
    def __init__(self, apikey=None):
        if apikey is None:
            apikey = os.getenv("API_KEY")
        self.apikey = apikey


    def generate_response(self, question, context):
        client = OpenAI(api_key=self.apikey)
        prompt = f"Eres capaz de responder la siguiente pregunta: {question}\n Dado el siguiente contexto?\nContext:\n{context}\n En caso de no tener la información necesaria en el contexto responde 'No lo sé'."
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=1.4,
        )
        return response.choices[0].message.content