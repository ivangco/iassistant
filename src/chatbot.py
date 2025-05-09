from typing import Dict, List
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_processing import DocumentProcessor
from auth import AuthManager
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

class EnterpriseChatbot:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.auth_manager = AuthManager()
        self.document_processor = DocumentProcessor()
        self.vectorstore = self.document_processor.process_documents(docs_dir)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Validar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se encontró la API key de OpenAI en el archivo .env")
        
        try:
            # Probar la API key
            client = OpenAI(api_key=api_key)
            client.models.list()
        except openai.AuthenticationError:
            raise ValueError("API key de OpenAI inválida")
        except openai.RateLimitError:
            raise ValueError("Se ha excedido la cuota de la API de OpenAI. Por favor, verifica tu plan y facturación.")
        except Exception as e:
            raise ValueError(f"Error al validar la API key: {str(e)}")
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
    
    def process_query(self, username: str, query: str) -> str:
        """Procesa una consulta del usuario y devuelve una respuesta."""
        try:
            # Verificar autenticación
            if not self.auth_manager.authenticate(username, ""):
                return "Error: Usuario no autenticado"
            
            # Verificar si la consulta está vacía
            if not query.strip():
                return "Por favor, ingresa una consulta válida."
            
            # Procesar la consulta
            result = self.qa_chain({"question": query})
            
            # Verificar si se encontró información relevante
            if not result.get("answer"):
                return "Lo siento, no pude encontrar información relevante para tu consulta."
            
            # Formatear la respuesta con las fuentes
            sources = result.get("source_documents", [])
            response = result["answer"]
            
            if sources:
                response += "\n\nFuentes:"
                for i, doc in enumerate(sources, 1):
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        response += f"\n{i}. {doc.metadata['source']}"
            
            return response
            
        except openai.RateLimitError:
            return "Error: Se ha excedido la cuota de la API de OpenAI. Por favor, intenta más tarde."
        except openai.AuthenticationError:
            return "Error: Problema de autenticación con la API de OpenAI. Por favor, verifica la configuración."
        except Exception as e:
            print(f"Error al procesar la consulta: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta nuevamente."
    
    def reset_conversation(self):
        """Reinicia la conversación."""
        self.memory.clear() 