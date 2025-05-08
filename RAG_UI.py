import gradio as gr
import json
import os
from RAGSearchV4 import (
    initialize_model,
    process_query,
    query_ollama
)

CHAT_HISTORY_FILE = "chat_history.json"


"""
Ollama History Format:
# ❌ List of Dictionaries (Ollama Format)
[
    {"role": "user", "content": "Hello"},
    {"role": "system", "content": "Hi there!"}
]
"""

"""
Gradio History Format:
[
    ("Hello", "Hi there!"),
    ("How are you?", "I'm good, thanks!")
]
"""



def load_chat_history():
    """Load chat history as a list of tuples from a JSON file if it exists."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Ensure it's a list of tuples
                if isinstance(data, list):
                    # Convert any list of lists → list of tuples
                    return [tuple(x) for x in data if isinstance(x, list) or isinstance(x, tuple)]
            except json.JSONDecodeError:
                pass
    return []            


def save_chat_history(history):
    """Save chat history (list of tuples) to a JSON file."""
    # Convert each tuple to a list before dumping to JSON (JSON can't handle tuples directly)
    json_ready = [list(pair) for pair in history]
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)

# (B) Conversation in Scene Two
def continue_conversation(user_message, history):
    """
    IMPORTANT: `gr.ChatInterface` wants a single string or a dictionary with
    a "response" field. It does NOT want a list of tuples as a return.
    """
    history = load_chat_history()
    bot_reply = query_ollama(user_message, history)

    # Store the new turn locally in your list-of-tuples format
    history.append((user_message, bot_reply))
    save_chat_history(history)

    # Return only the bot's reply to avoid the 'tuple' object error
    return bot_reply

def run_app():
    with gr.Blocks() as demo:
        gr.Markdown("# ASLS RAG v4")

        # -----------------------
        # Scene One: Inital Prompt
        # -----------------------
        with gr.Column(visible=True) as scene_one:
            search_query = gr.Textbox(label="Embedding Query")
            instructions = gr.Textbox(label="Instruction Query")
            num_functions = gr.Number(
                label="Number of Functions to Return",
                value=5,
                precision=0
            )

            submit_btn = gr.Button("Submit")

        # -----------------------
        # Loading Screen
        # -----------------------
        with gr.Column(visible=False) as scene_loading:
            gr.Markdown("### ⏳ Processing...")

        # -----------------------
        # Results Scene
        # -----------------------
        with gr.Column(visible=False) as results_screen:

            conversation_mode_btn = gr.Button("Continue to Conversation Mode")
            back_to_search_btn = gr.Button("Back to RAG")  # 🔹 New Button

            # Some output placeholders
            with gr.Row():
                with gr.Column(scale=2):
                    answer_output = gr.Markdown()
                with gr.Column(scale=2):
                    functions_output = gr.Markdown()
 

        #-------------------------
        # Conversation Scene
        #-------------------------
        with gr.Column(visible=False) as conversation_scene:
            back_to_search_btn_2 = gr.Button("Back to RAG")  

            chat_history = gr.State(list(load_chat_history() or []))
            
            chatbot = gr.Chatbot(height=750)
            
            #error returning here??
            gr.ChatInterface(
                fn=continue_conversation,
                title="Conversation Mode",
                chatbot=chatbot,
                textbox=gr.Textbox(placeholder="Ask a question", container=False, scale=7),
                theme="ocean"
            )


            clear_memory_btn = gr.Button("Clear Memory and Return to RAG", elem_id="clear-memory-btn")

        # Global Buttons
        #view_memory_btn = gr.Button("View Memory")
        #clear_btn = gr.Button("Clear History")

        # ------- EVENT HANDLERS -------

        # (A) Scene One: Code search
        def handle_code_search(embeddedQ, instructionQ, numberOfFunctions, history):
            # 1) Show loading, hide everything else
            yield (
                gr.update(visible=False),  # scene_one
                gr.update(visible=True),   # scene_loading
                gr.update(visible=False),  # results screen
                gr.update(value=""),       # answer_output
                gr.update(value=""),       # functions_output
                gr.update(value=[]),       # chat_history
            )

            # 2) Process query
            answer, listing = process_query(embeddedQ, instructionQ, numberOfFunctions, False, history)
                        
            history.append((f"RAG Embedding Query: {embeddedQ}", listing))
            history.append((f"RAG Instruction: {instructionQ}", answer))

            save_chat_history(history)

            yield (
                gr.update(visible=False),   # scene_one
                gr.update(visible=False),   # scene_loading
                gr.update(visible=True),    # results screen 
                gr.update(value=answer),    # answer_output
                gr.update(value=listing),   # functions_output
                gr.update(value=history)  # chat_history updated
            )

        

        submit_btn.click(
            fn=handle_code_search,
            inputs=[search_query, instructions, num_functions, chat_history],
            outputs=[scene_one, scene_loading, results_screen, answer_output, functions_output, chat_history],
            queue=True
        )

        def handle_conversation_mode_switch(history):
             # Ensure history is always a list
            if isinstance(history, set):
                history = list(history)
            return (
                gr.update(visible=False),    # scene_one
                gr.update(visible=False),    # scene_loading
                gr.update(visible=False),    # results screen 
                gr.update(visible=True),     # results screen 
                gr.update(value=history) # update histroy
            )

        conversation_mode_btn.click(
            fn=handle_conversation_mode_switch,
            inputs=[chat_history],
            outputs=[scene_one, scene_loading, results_screen, conversation_scene, chatbot],
            queue = True
        )

        def reset_to_search():
            return (
                gr.update(visible=True),   # scene_one (Show)
                gr.update(visible=False),  # scene_loading (Hide)
                gr.update(visible=False),  # results_screen (Hide)
                gr.update(visible=False),  # conversation_scene (Hide)
                gr.update(value=""),       # answer_output (Clear)
                gr.update(value=""),       # functions_output (Clear)
                gr.update(value=[])        # chat_history (Reset)
            )

        back_to_search_btn.click(
        fn=reset_to_search,
        inputs=[],
        outputs=[scene_one, scene_loading, results_screen, conversation_scene, answer_output, functions_output, chat_history],
        queue=True
        )

        back_to_search_btn_2.click(
            fn=reset_to_search,
            inputs=[],
            outputs=[scene_one, scene_loading, results_screen, conversation_scene, answer_output, functions_output, chat_history],
            queue=True
        )

        def clear_memory():
            """Clear chat history and reset the UI."""
            # Empty chat history
            empty_history = []
            
            # Save empty history to file
            save_chat_history(empty_history)

            return (
                gr.update(value=empty_history),  # chat_history reset
                gr.update(value=""),  # answer_output cleared
                gr.update(value=""),  # functions_output cleared
                gr.update(visible=True),   # Show scene_one
                gr.update(visible=False),  # Hide loading screen
                gr.update(visible=False),  # Hide results_screen
                gr.update(visible=False),  # Hide conversation_scene
            )

        clear_memory_btn.click(
        fn=clear_memory,
        inputs=[],
        outputs=[chat_history, answer_output, functions_output, scene_one, scene_loading, results_screen, conversation_scene],
        queue=True
        )



        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    initialize_model()
    run_app()
