FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY src/ ./src/
COPY docs/ ./docs/

# Exponer el puerto
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "src/app.py"] 