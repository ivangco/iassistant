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
    model_name: str = "BLOOM-560m"
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
        
        # Usar BLOOM-560m (modelo multilingüe más pequeño)
        model_name = "bigscience/bloom-560m"
        logger.info(f"🔄 Cargando tokenizador para {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                cache_dir="/root/.cache/huggingface"
            )
            # Configurar el token de padding
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Tokenizador cargado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error al cargar el tokenizador: {str(e)}")
            raise
        
        logger.info(f"🔄 Cargando modelo {model_name}...")
        try:
            # Configuraciones para reducir el uso de memoria
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Volver a float32 para compatibilidad
                low_cpu_mem_usage=True,
                local_files_only=False,
                cache_dir="/root/.cache/huggingface",
                device_map="cpu",  # Forzar uso de CPU
                max_memory={0: "4GB"},  # Limitar uso de memoria
                offload_folder="offload"  # Carpeta para offloading
            )
            # Poner el modelo en modo evaluación
            model.eval()
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
        # Solo limpiar recursos cuando la aplicación se detenga completamente
        logger.info("🔄 Limpiando recursos...")
        if model is not None:
            model.cpu()  # Mover a CPU antes de liberar
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
        logger.info("🔄 Iniciando generación de respuesta...")
        logger.debug(f"Query recibida - Texto: {query.text}, Contexto: {query.context}")
        
        if model is None or tokenizer is None:
            logger.error("❌ Modelo o tokenizador no inicializado")
            raise HTTPException(status_code=503, detail="Modelo no inicializado")
        
        # Asegurarse de que el modelo está en modo evaluación
        logger.debug("Configurando modelo en modo evaluación...")
        model.eval()
        
        # Preparar el prompt en español
        prompt = f"Contexto: {query.context}\n\nPregunta: {query.text}\n\nRespuesta en español:"
        logger.debug(f"Prompt preparado: {prompt}")
        
        try:
            # Generar respuesta con configuraciones optimizadas
            logger.debug("Tokenizando input...")
            with torch.no_grad():  # Deshabilitar gradientes para inferencia
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,  # Reducido para usar menos memoria
                    truncation=True,
                    padding=True
                )
                logger.debug(f"Input tokenizado. Shape: {inputs['input_ids'].shape}")
                
                # Generar con parámetros más conservadores
                logger.debug("Iniciando generación...")
                try:
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=100,  # Reducido para usar menos memoria
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.2,
                        top_p=0.9,
                        top_k=50,
                        early_stopping=True,
                        use_cache=True  # Habilitar cache para mejor rendimiento
                    )
                    logger.debug(f"Generación completada. Shape de output: {outputs.shape}")
                    
                    logger.debug("Decodificando respuesta...")
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split("Respuesta en español:")[-1].strip()
                    logger.debug(f"Respuesta generada: {response}")
                    
                    # Limpiar memoria explícitamente
                    del outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    logger.info("✅ Respuesta generada exitosamente")
                    return Response(
                        response=response,
                        model_name="BLOOM-560m"
                    )
                except RuntimeError as e:
                    logger.error(f"❌ Error durante la generación: {str(e)}")
                    # Intentar liberar memoria y reintentar una vez
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    raise HTTPException(status_code=503, detail="Error durante la generación. Por favor, intente nuevamente.")
            
        except RuntimeError as e:
            logger.error(f"❌ Error de runtime durante la generación: {str(e)}")
            if "CUDA out of memory" in str(e):
                logger.error("Error de memoria CUDA. Intentando liberar memoria...")
                torch.cuda.empty_cache()
                raise HTTPException(status_code=503, detail="Error de memoria GPU. Por favor, intente nuevamente.")
            raise HTTPException(status_code=500, detail=f"Error durante la generación: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Error inesperado: {str(e)}")
        logger.exception("Detalles del error:")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.debug("Finalizando generación de respuesta")

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio."""
    try:
        logger.debug("Verificando estado del servicio...")
        if model is None or tokenizer is None:
            logger.error("Modelo o tokenizador no inicializado")
            raise HTTPException(status_code=503, detail="Modelo no inicializado")
        
        # Verificar memoria GPU si está disponible
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            logger.debug(f"Memoria GPU - Allocada: {memory_allocated:.2f}MB, Reservada: {memory_reserved:.2f}MB")
        
        logger.info("✅ Servicio saludable")
        return {
            "status": "healthy",
            "model": "BLOOM-560m",
            "gpu_memory": {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else None,
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2) if torch.cuda.is_available() else None
            }
        }
    except Exception as e:
        logger.error(f"❌ Error en health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 