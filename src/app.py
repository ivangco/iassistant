import gradio as gr
from chatbot import EnterpriseChatbot
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear directorio docs si no existe
docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)
    logger.info(f"📁 Directorio {docs_dir} creado.")

try:
    chatbot = EnterpriseChatbot(docs_dir)
    logger.info("✅ Chatbot inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error al inicializar el chatbot: {str(e)}")
    raise

def respond(message, chat_history, username):
    """Procesa la consulta del usuario y devuelve una respuesta."""
    try:
        if not message.strip():
            return chat_history, "⚠️ Por favor, ingresa una consulta válida."
        
        logger.info(f"📝 Procesando consulta de {username}: {message[:50]}...")
        response = chatbot.process_query(username, message)
        chat_history.append((message, response))
        return chat_history, ""
    except Exception as e:
        logger.error(f"❌ Error al procesar la consulta: {str(e)}")
        error_msg = f"❌ Error: {str(e)}"
        chat_history.append((message, error_msg))
        return chat_history, ""

def reset_chat():
    """Reinicia la conversación."""
    try:
        chatbot.reset_conversation()
        logger.info("🔄 Conversación reiniciada")
        return [], ""
    except Exception as e:
        logger.error(f"❌ Error al reiniciar la conversación: {str(e)}")
        return [], f"❌ Error al reiniciar: {str(e)}"

# Crear la interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Asistente Empresarial")
    
    with gr.Row():
        chatbot_interface = gr.Chatbot(height=600)
        with gr.Column():
            username = gr.Textbox(label="Usuario", value="usuario")
            reset = gr.Button("🔄 Reiniciar")
    
    msg = gr.Textbox(placeholder="Escribe tu pregunta aquí...")
    submit = gr.Button("Enviar")
    
    submit.click(
        respond,
        [msg, chatbot_interface, username],
        [chatbot_interface, msg]
    )
    
    msg.submit(
        respond,
        [msg, chatbot_interface, username],
        [chatbot_interface, msg]
    )
    
    reset.click(reset_chat, None, [chatbot_interface, msg])

# Iniciar la aplicación
if __name__ == "__main__":
    try:
        logging.info("✅ Chatbot inicializado correctamente")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            allowed_paths=["docs"]
        )
    except Exception as e:
        logging.error(f"❌ Error al iniciar la aplicación: {str(e)}")
        raise 