from chat import Chatbot
from faiss_db import FaissDB
from openai_api import OpenAIAPI
from embedding import EmbeddingModel  

def main():
    # Initialize the FAISS database
    faiss_db = FaissDB()

    # Initialize the embedding model
    embedding_model = EmbeddingModel()

    # Initialize the OpenAI API
    openai_api = OpenAIAPI()
    
    # Initialize the chatbot
    chatbot = Chatbot(openai_api, embedding_model, faiss_db)

    print("Bienvenido! Escribe 'exit' para salir.")

    while True:
        user_input = input("\nPregunta: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.ask_question(user_input)
        print(f"\n\nChatbot: {response}")

if __name__ == "__main__":
    main()