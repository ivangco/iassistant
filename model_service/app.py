from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Modelo Local API")

# Modelo y tokenizador globales
model = None
tokenizer = None

class Query(BaseModel):
    text: str
    context: str

class Response(BaseModel):
    response: str
    model_name: str

@app.on_event("startup")
async def startup_event():
    """Inicializa el modelo al arrancar el servicio."""
    global model, tokenizer
    try:
        logger.info("🔄 Inicializando modelo local (BLOOM-560m)...")
        model_name = "bigscience/bloom-560m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Mover el modelo a GPU si está disponible
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("✅ Modelo cargado en GPU")
        else:
            logger.info("ℹ️ GPU no disponible, usando CPU")
            
        logger.info("✅ Modelo local inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error al inicializar el modelo: {str(e)}")
        raise

@app.post("/generate", response_model=Response)
async def generate_response(query: Query):
    """Genera una respuesta usando el modelo local."""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Modelo no inicializado")
        
        # Preparar el prompt
        prompt = f"Contexto: {query.context}\n\nPregunta: {query.text}\n\nRespuesta:"
        
        # Generar respuesta
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = model.generate(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Respuesta:")[-1].strip()
        
        return Response(
            response=response,
            model_name="BLOOM-560m"
        )
        
    except Exception as e:
        logger.error(f"❌ Error al generar respuesta: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado")
    return {"status": "healthy", "model": "BLOOM-560m"} 