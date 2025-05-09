import gradio as gr
from chatbot import EnterpriseChatbot
import os
from dotenv import load_dotenv
import sys

# Cargar variables de entorno
load_dotenv()

# Verificar si existe el directorio docs
docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)
    print(f"Directorio {docs_dir} creado.")

try:
    # Inicializar el chatbot
    chatbot = EnterpriseChatbot(docs_dir)
except ValueError as e:
    print(f"Error de inicialización: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado: {str(e)}")
    sys.exit(1)

def respond(message, chat_history, username):
    """Procesa una consulta y devuelve la respuesta."""
    if not username.strip():
        return "Por favor, ingresa un nombre de usuario.", chat_history
    
    try:
        response = chatbot.process_query(username, message)
        chat_history.append((message, response))
        return "", chat_history
    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        print(error_msg)
        chat_history.append((message, "Lo siento, ha ocurrido un error. Por favor, intenta nuevamente."))
        return "", chat_history

def reset_chat():
    """Reinicia la conversación."""
    try:
        chatbot.reset_conversation()
        return [], ""
    except Exception as e:
        print(f"Error al reiniciar la conversación: {str(e)}")
        return [], "Error al reiniciar la conversación"

# Crear la interfaz de Gradio
with gr.Blocks(title="Asistente Empresarial IA") as demo:
    gr.Markdown("# 🤖 Asistente Empresarial IA")
    gr.Markdown("""
    Este asistente puede responder preguntas sobre documentos internos de la empresa.
    Por favor, ingresa tu nombre de usuario y tu pregunta.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot_interface = gr.Chatbot(height=600)
            with gr.Row():
                message = gr.Textbox(
                    label="Tu pregunta",
                    placeholder="Escribe tu pregunta aquí...",
                    lines=2
                )
                submit = gr.Button("Enviar")
        
        with gr.Column(scale=1):
            username = gr.Textbox(
                label="Usuario",
                placeholder="Ingresa tu nombre de usuario"
            )
            reset = gr.Button("Reiniciar conversación")
    
    # Configurar eventos
    submit.click(
        respond,
        inputs=[message, chatbot_interface, username],
        outputs=[message, chatbot_interface]
    )
    
    message.submit(
        respond,
        inputs=[message, chatbot_interface, username],
        outputs=[message, chatbot_interface]
    )
    
    reset.click(
        reset_chat,
        outputs=[chatbot_interface, message]
    )

# Iniciar la aplicación
if __name__ == "__main__":
    try:
        demo.launch(share=True)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {str(e)}")
        sys.exit(1) 