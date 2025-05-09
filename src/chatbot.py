from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from data_processing import DocumentProcessor
from auth import AuthManager
import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
import logging
import requests
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Obtener URL del servicio del modelo
        self.model_service_url = os.getenv("MODEL_SERVICE_URL", "http://localhost:8000")
        
        # Validar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ No se encontró la API key de OpenAI en el archivo .env")
        
        try:
            # Probar la API key
            client = OpenAI(api_key=api_key)
            client.models.list()
            self.api_available = True
            logger.info("✅ API de OpenAI disponible y validada")
        except openai.AuthenticationError:
            raise ValueError("❌ API key de OpenAI inválida")
        except openai.RateLimitError:
            logger.warning("⚠️ Se ha excedido la cuota de OpenAI. Usando modelo alternativo.")
            self.api_available = False
        except Exception as e:
            logger.warning(f"⚠️ Error al validar la API key: {str(e)}. Usando modelo alternativo.")
            self.api_available = False
        
        # Inicializar modelos
        self.llm = self._get_llm()
        
        try:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            logger.info("✅ Cadena de conversación inicializada correctamente")
        except Exception as e:
            logger.error(f"❌ Error al inicializar la cadena de conversación: {str(e)}")
            raise
    
    def _get_llm(self):
        """Obtiene el modelo de lenguaje apropiado según disponibilidad."""
        if self.api_available:
            try:
                logger.info("🔄 Inicializando GPT-3.5-turbo")
                return ChatOpenAI(
                    temperature=0.7,
                    model_name="gpt-3.5-turbo",
                    verbose=True
                )
            except Exception as e:
                logger.warning(f"⚠️ Error al inicializar GPT-3.5: {str(e)}. Usando modelo alternativo.")
                self.api_available = False
        
        # Modelo alternativo
        logger.info("🔄 Inicializando modelo alternativo")
        return ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo-16k",
            request_timeout=30,
            verbose=True
        )
    
    def _generate_local_response(self, query: str, context: str) -> str:
        """Genera una respuesta usando el servicio del modelo local."""
        try:
            # Preparar la petición
            payload = {
                "text": query,
                "context": context
            }
            
            # Hacer la petición al servicio
            response = requests.post(
                f"{self.model_service_url}/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                logger.error(f"❌ Error en el servicio del modelo: {response.text}")
                return "Lo siento, hubo un error al procesar tu consulta con el modelo local."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al conectar con el servicio del modelo: {str(e)}")
            return "Lo siento, no se pudo conectar con el servicio del modelo local."
        except Exception as e:
            logger.error(f"❌ Error al generar respuesta local: {str(e)}")
            return "Lo siento, hubo un error al procesar tu consulta con el modelo local."
    
    def process_query(self, username: str, query: str) -> str:
        """Procesa una consulta del usuario y devuelve una respuesta."""
        try:
            # Verificar autenticación
            if not self.auth_manager.authenticate(username, ""):
                return "❌ Error: Usuario no autenticado"
            
            # Verificar si la consulta está vacía
            if not query.strip():
                return "⚠️ Por favor, ingresa una consulta válida."
            
            # Mostrar información sobre el modelo actual
            model_info = "🤖 Usando GPT-3.5-turbo" if self.api_available else "🤖 Usando modelo alternativo"
            logger.info(model_info)
            
            # Intentar procesar con el modelo actual
            try:
                logger.info(f"📝 Procesando consulta: {query[:50]}...")
                result = self.qa_chain({"question": query})
            except Exception as e:
                if "insufficient_quota" in str(e).lower():
                    logger.warning("⚠️ Cambiando a modelo alternativo debido a cuota insuficiente...")
                    self.api_available = False
                    self.llm = self._get_llm()
                    self.qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.vectorstore.as_retriever(),
                        memory=self.memory,
                        return_source_documents=True,
                        verbose=True
                    )
                    try:
                        result = self.qa_chain({"question": query})
                    except Exception as e2:
                        logger.warning("⚠️ Error con modelo alternativo, usando modelo local...")
                        # Obtener contexto relevante
                        docs = self.vectorstore.similarity_search(query, k=3)
                        context = "\n".join([doc.page_content for doc in docs])
                        response = self._generate_local_response(query, context)
                        return f"🤖 Usando modelo local (BLOOM-560m)\n\n{response}"
                else:
                    raise e
            
            # Verificar si se encontró información relevante
            if not result.get("answer"):
                return "❌ Lo siento, no pude encontrar información relevante para tu consulta."
            
            # Formatear la respuesta con las fuentes
            sources = result.get("source_documents", [])
            response = f"{model_info}\n\n{result['answer']}"
            
            if sources:
                response += "\n\n📚 Fuentes:"
                for i, doc in enumerate(sources, 1):
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        response += f"\n{i}. {doc.metadata['source']}"
            
            return response
            
        except openai.RateLimitError:
            logger.error("❌ Error de cuota en OpenAI")
            return "❌ Error: Se ha excedido la cuota de la API de OpenAI. Por favor, intenta más tarde."
        except openai.AuthenticationError:
            logger.error("❌ Error de autenticación en OpenAI")
            return "❌ Error: Problema de autenticación con la API de OpenAI. Por favor, verifica la configuración."
        except Exception as e:
            logger.error(f"❌ Error al procesar la consulta: {str(e)}")
            return "❌ Lo siento, ha ocurrido un error al procesar tu consulta. Por favor, intenta nuevamente."
    
    def reset_conversation(self):
        """Reinicia la conversación."""
        try:
            self.memory.clear()
            logger.info("🔄 Conversación reiniciada")
        except Exception as e:
            logger.error(f"❌ Error al reiniciar la conversación: {str(e)}")
            raise 