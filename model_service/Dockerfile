FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar la caché de Docker
COPY model_requirements.txt .
RUN pip install --no-cache-dir -r model_requirements.txt

# Crear directorio para caché de Hugging Face
RUN mkdir -p /root/.cache/huggingface

# Copiar el código de la aplicación
COPY app.py .

# Exponer el puerto
EXPOSE 8000

# Usar uvicorn directamente con --host 0.0.0.0 para permitir conexiones externas
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 