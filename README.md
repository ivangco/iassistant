# 🤖 Asistente Empresarial con IA Híbrida

Un asistente empresarial inteligente que combina modelos de IA de OpenAI con un modelo local, diseñado para responder preguntas sobre documentación empresarial.

## 🌟 Características

- **Sistema Híbrido de IA**:
  - GPT-3.5-turbo como modelo principal
  - GPT-3.5-turbo-16k como respaldo
  - BLOOM-560m como modelo local (cuando los anteriores no están disponibles)

- **Procesamiento de Documentos**:
  - Soporte para PDF y archivos de texto
  - Embeddings para búsqueda semántica
  - Extracción de contexto relevante

- **Arquitectura Distribuida**:
  - Servicio principal con interfaz web
  - Servicio separado para el modelo local
  - Comunicación vía API REST

- **Interfaz Amigable**:
  - Chat interactivo con Gradio
  - Historial de conversaciones
  - Fuentes de información
  - Indicador del modelo en uso

## 🏗️ Arquitectura

El proyecto está dividido en dos servicios principales:

### 1. Servicio Principal (app)
- Interfaz web con Gradio
- Procesamiento de documentos
- Integración con OpenAI
- Comunicación con el modelo local

### 2. Servicio del Modelo Local (model)
- API REST con FastAPI
- Modelo BLOOM-560m
- Soporte para GPU
- Caché persistente

## 🚀 Requisitos

- Docker y Docker Compose
- NVIDIA GPU (opcional, para mejor rendimiento del modelo local)
- API key de OpenAI

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd asistenteia
```

2. Crear archivo `.env`:
```bash
OPENAI_API_KEY=tu_api_key_aqui
```

3. Construir y ejecutar los servicios:
```bash
docker-compose up --build
```

## 🔧 Configuración

### Variables de Entorno
- `OPENAI_API_KEY`: API key de OpenAI
- `MODEL_SERVICE_URL`: URL del servicio del modelo local (por defecto: http://model:8000)

### Puertos
- `7860`: Interfaz web (Gradio)
- `8000`: API del modelo local

## 💻 Uso

1. Acceder a la interfaz web:
   - Local: http://localhost:7860
   - Remoto: URL proporcionada por Gradio

2. Ingresar nombre de usuario (por defecto: "usuario")

3. Hacer preguntas sobre la documentación

4. El sistema automáticamente:
   - Seleccionará el mejor modelo disponible
   - Buscará información relevante
   - Proporcionará respuestas con fuentes

## 🔄 Flujo de Trabajo

1. **Procesamiento de Documentos**:
   - Los documentos se cargan en el directorio `docs/`
   - Se procesan y se crean embeddings
   - Se almacenan en un índice FAISS

2. **Procesamiento de Consultas**:
   - El usuario hace una pregunta
   - El sistema busca información relevante
   - Se selecciona el modelo apropiado
   - Se genera y devuelve la respuesta

3. **Sistema de Fallback**:
   - Intenta usar GPT-3.5-turbo
   - Si falla, usa GPT-3.5-turbo-16k
   - Si ambos fallan, usa BLOOM-560m local

## 📁 Estructura del Proyecto

```
.
├── src/
│   ├── app.py              # Interfaz web con Gradio
│   ├── chatbot.py          # Lógica principal del chatbot
│   └── data_processing.py  # Procesamiento de documentos
├── model_service/
│   └── app.py             # API del modelo local
├── docs/                  # Documentos a procesar
├── Dockerfile.app         # Dockerfile para el servicio principal
├── Dockerfile.model       # Dockerfile para el servicio del modelo
├── docker-compose.yml     # Configuración de Docker Compose
├── requirements.txt       # Dependencias del servicio principal
└── model_requirements.txt # Dependencias del servicio del modelo
```

## 🔍 Monitoreo y Logging

- Logs detallados de cada servicio
- Indicadores de estado del modelo
- Mensajes de error descriptivos
- Emojis para mejor visualización

## 🔒 Seguridad

- Validación de API keys
- Autenticación de usuarios
- Manejo seguro de documentos
- Comunicación segura entre servicios

## 🤝 Contribuir

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- OpenAI por GPT-3.5
- Hugging Face por BLOOM-560m
- Gradio por la interfaz web
- FastAPI por el framework de API 