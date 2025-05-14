try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError as e:
    import logging
    logging.error(f"❌ Error al importar PyPDFLoader: {str(e)}")
    logging.error("Intentando instalar dependencias faltantes...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf>=4.0.1", "pdfminer.six>=20221105"])
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import List, Dict
from dotenv import load_dotenv
import openai
from openai import OpenAI
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Validar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ No se encontró la API key de OpenAI en el archivo .env")
        
        try:
            # Probar la API key
            client = OpenAI(api_key=api_key)
            client.models.list()
            self.api_available = True
            self.embeddings = OpenAIEmbeddings()
            logger.info("✅ API de OpenAI disponible para embeddings")
        except openai.AuthenticationError:
            raise ValueError("❌ API key de OpenAI inválida")
        except openai.RateLimitError:
            logger.warning("⚠️ Se ha excedido la cuota de OpenAI. Usando embeddings locales.")
            self.api_available = False
            self._initialize_local_embeddings()
        except Exception as e:
            logger.warning(f"⚠️ Error al validar la API key: {str(e)}. Usando embeddings locales.")
            self.api_available = False
            self._initialize_local_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def _initialize_local_embeddings(self):
        """Inicializa los embeddings locales."""
        try:
            logger.info("🔄 Inicializando embeddings locales...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ Embeddings locales inicializados correctamente")
        except Exception as e:
            logger.error(f"❌ Error al inicializar embeddings locales: {str(e)}")
            raise
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Carga un documento basado en su extensión."""
        try:
            if file_path.endswith('.pdf'):
                logger.info(f"📄 Cargando documento PDF: {os.path.basename(file_path)}")
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                logger.info(f"📝 Cargando documento de texto: {os.path.basename(file_path)}")
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"❌ Formato de archivo no soportado: {file_path}")
            
            docs = loader.load()
            logger.info(f"✅ Documento cargado: {len(docs)} páginas/secciones")
            return docs
        except Exception as e:
            logger.error(f"❌ Error al cargar el documento {file_path}: {str(e)}")
            return []
    
    def process_documents(self, docs_dir: str) -> FAISS:
        """Procesa todos los documentos en el directorio y crea el índice FAISS."""
        documents = []
        
        # Verificar si el directorio existe y tiene archivos
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
            logger.info(f"📁 Directorio {docs_dir} creado.")
        
        # Crear archivo de ejemplo si el directorio está vacío
        if not any(f.endswith(('.pdf', '.txt')) for f in os.listdir(docs_dir)):
            example_path = os.path.join(docs_dir, "preguntas_frecuentes.txt")
            with open(example_path, 'w', encoding='utf-8') as f:
                f.write("""PREGUNTAS FRECUENTES

1. ¿Cómo solicito vacaciones?
Para solicitar vacaciones, debes enviar un correo a recursos.humanos@empresa.com con al menos 2 semanas de anticipación.

2. ¿Cuál es el horario de trabajo?
El horario de trabajo es de lunes a viernes de 9:00 AM a 6:00 PM, con una hora de almuerzo.

3. ¿Cómo reporto un problema técnico?
Los problemas técnicos deben reportarse a través del portal de IT en intranet.empresa.com/soporte

4. ¿Cuál es la política de trabajo remoto?
El trabajo remoto está permitido hasta 2 días por semana, previa aprobación del supervisor.

5. ¿Cómo accedo a los beneficios de la empresa?
Los beneficios pueden ser consultados en el portal de empleados en intranet.empresa.com/beneficios""")
            logger.info(f"📝 Archivo de ejemplo creado en {example_path}")
        
        # Procesar documentos
        total_files = len([f for f in os.listdir(docs_dir) if f.endswith(('.pdf', '.txt'))])
        processed_files = 0
        
        for filename in os.listdir(docs_dir):
            if filename.endswith(('.pdf', '.txt')):
                file_path = os.path.join(docs_dir, filename)
                try:
                    docs = self.load_document(file_path)
                    if docs:
                        documents.extend(docs)
                        processed_files += 1
                        logger.info(f"✅ Documento procesado exitosamente: {filename} ({processed_files}/{total_files})")
                    else:
                        logger.warning(f"⚠️ No se pudo procesar el documento: {filename}")
                except Exception as e:
                    logger.error(f"❌ Error al procesar {filename}: {str(e)}")
        
        if not documents:
            raise ValueError("❌ No se encontraron documentos válidos para procesar")
        
        try:
            # Dividir documentos en chunks
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"📊 Documentos divididos en {len(texts)} chunks")
            
            # Crear índice FAISS
            embedding_type = "OpenAI" if self.api_available else "Local (HuggingFace)"
            logger.info(f"🔄 Creando índice FAISS con embeddings de {embedding_type}...")
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            logger.info("✅ Índice FAISS creado exitosamente")
            
            return vectorstore
        except Exception as e:
            if "insufficient_quota" in str(e).lower() and self.api_available:
                logger.warning("⚠️ Cambiando a embeddings locales debido a cuota insuficiente...")
                self.api_available = False
                self._initialize_local_embeddings()
                return FAISS.from_documents(texts, self.embeddings)
            logger.error(f"❌ Error al crear el índice de documentos: {str(e)}")
            raise 