from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict, List
from data_processing import DocumentProcessor
from auth import AuthManager

class EnterpriseChatbot:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.processor = DocumentProcessor()
        self.auth_manager = AuthManager()
        self.vectorstore = self.processor.process_documents(docs_dir)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = ChatOpenAI(temperature=0.7)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
    
    def process_query(self, username: str, query: str) -> str:
        """Procesa una consulta del usuario y retorna una respuesta."""
        # Verificar acceso a documentos
        if not self.auth_manager.can_access_document(username, "*"):
            return "Lo siento, no tienes permiso para acceder a esta información."
        
        try:
            response = self.qa_chain({"question": query})
            return response["answer"]
        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"
    
    def reset_conversation(self):
        """Reinicia la conversación."""
        self.memory.clear() 