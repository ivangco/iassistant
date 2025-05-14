from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import os
from contextlib import asynccontextmanager
import sys

# Configurar logging más detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Modelo y tokenizador globales
model = None
tokenizer = None

class Query(BaseModel):
    text: str
    context: str

class Response(BaseModel):
    response: str
    model_name: str = "GPT-2"
    model_config = ConfigDict(protected_namespaces=())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador del ciclo de vida de la aplicación."""
    global model, tokenizer
    try:
        logger.info("🔄 Iniciando inicialización del modelo...")
        logger.debug("Verificando disponibilidad de CUDA...")
        logger.debug(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
        
        # Usar GPT-2 en lugar de BLOOM (modelo más pequeño)
        model_name = "gpt2"
        logger.info(f"🔄 Cargando tokenizador para {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                cache_dir="/root/.cache/huggingface"
            )
            logger.info("✅ Tokenizador cargado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error al cargar el tokenizador: {str(e)}")
            raise
        
        logger.info(f"🔄 Cargando modelo {model_name}...")
        try:
            # Configuraciones para reducir el uso de memoria
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False,
                cache_dir="/root/.cache/huggingface",
                device_map="auto"  # Distribuir automáticamente en CPU/GPU
            )
            logger.info("✅ Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {str(e)}")
            raise
        
        logger.info("✅ Modelo local inicializado correctamente")
        yield
    except Exception as e:
        logger.error(f"❌ Error durante la inicialización: {str(e)}")
        logger.exception("Detalles del error:")
        raise
    finally:
        logger.info("🔄 Limpiando recursos...")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("✅ Limpieza completada")

app = FastAPI(title="Modelo Local API", lifespan=lifespan)

@app.post("/generate", response_model=Response)
async def generate_response(query: Query):
    """Genera una respuesta usando el modelo local."""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Modelo no inicializado")
        
        # Preparar el prompt
        prompt = f"Contexto: {query.context}\n\nPregunta: {query.text}\n\nRespuesta:"
        
        # Generar respuesta con configuraciones optimizadas
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Generar con parámetros más conservadores
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Respuesta:")[-1].strip()
        
        return Response(
            response=response,
            model_name="GPT-2"
        )
        
    except Exception as e:
        logger.error(f"❌ Error al generar respuesta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    return {"status": "healthy", "model": "GPT-2"} 