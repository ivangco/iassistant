import gradio as gr
import os
from chatbot import EnterpriseChatbot
from auth import AuthManager

# Inicializar componentes
auth_manager = AuthManager()
chatbot = EnterpriseChatbot("docs")

def login(username, password):
    user = auth_manager.authenticate(username, password)
    if user:
        return gr.update(visible=True), gr.update(visible=False), f"Bienvenido, {username}!"
    return gr.update(visible=False), gr.update(visible=True), "Credenciales inválidas"

def chat(message, history, username):
    response = chatbot.process_query(username, message)
    history.append((message, response))
    return "", history

def reset_chat():
    chatbot.reset_conversation()
    return []

# Crear la interfaz de Gradio
with gr.Blocks(title="Asistente Empresarial IA") as demo:
    gr.Markdown("# 🤖 Asistente Empresarial IA")
    
    with gr.Row():
        with gr.Column():
            username = gr.Textbox(label="Usuario")
            password = gr.Textbox(label="Contraseña", type="password")
            login_btn = gr.Button("Iniciar Sesión")
            status = gr.Textbox(label="Estado")
    
    with gr.Row(visible=False) as chat_interface:
        chatbot_interface = gr.ChatInterface(
            fn=chat,
            additional_inputs=[username],
            title="Chat con el Asistente",
            description="Haz preguntas sobre los documentos de la empresa",
            examples=[
                "¿Cuáles son los procedimientos estándar?",
                "¿Dónde puedo encontrar información sobre políticas?",
                "¿Cuáles son las preguntas más frecuentes?"
            ],
            retry_btn=None,
            undo_btn=None,
            clear_btn="Nueva Conversación"
        )
    
    login_btn.click(
        fn=login,
        inputs=[username, password],
        outputs=[chat_interface, gr.Group(), status]
    )

if __name__ == "__main__":
    demo.launch() 