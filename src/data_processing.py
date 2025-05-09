from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Carga un documento basado en su extensión."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path}")
        
        return loader.load()
    
    def process_documents(self, docs_dir: str) -> FAISS:
        """Procesa todos los documentos en el directorio y crea el índice FAISS."""
        documents = []
        
        for filename in os.listdir(docs_dir):
            if filename.endswith(('.pdf', '.txt')):
                file_path = os.path.join(docs_dir, filename)
                documents.extend(self.load_document(file_path))
        
        # Dividir documentos en chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Crear índice FAISS
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        return vectorstore 