import gradio as gr
import threading
import queue
from run_agent import AIAgent

def chat_interface(message, history, provider, base_url, model_name, api_key):
    """
    Main chat logic integrating the AIAgent with Gradio.
    Since AIAgent.run_conversation blocks and doesn't natively yield to Gradio,
    we spawn a thread to run it and use a queue to pass streaming updates
    (text tokens and tool logs) back to the UI.
    """
    if not model_name or not provider:
        yield "⚠️ Please connect/load a model API in the settings panel before starting a chat."
        return

    # We will use this queue to receive tokens/events from the agent
    q = queue.Queue()

    # We construct a stream_delta_callback that puts text chunks into the queue
    def stream_callback(token):
        if token is None:
            # None usually signals end of response, but run_conversation itself returning
            # is our definitive end signal. We can ignore None or use it as a flush.
            pass
        else:
            q.put({"type": "text", "content": token})

    # We construct a tool_progress_callback to show tool executions as logs
    def tool_callback(event, *args):
        # Format the event and args into a readable log message
        if event == "tool.started":
            name = args[0] if len(args) > 0 else "tool"
            preview = args[1] if len(args) > 1 else ""
            log_msg = f"🔧 **Running {name}**\n```json\n{preview}\n```\n"
            q.put({"type": "log", "content": log_msg})
        elif event == "_thinking":
            q.put({"type": "log", "content": f"🤔 *Thinking*: {args[0] if len(args)>0 else ''}\n"})
        else:
            q.put({"type": "log", "content": f"⚙️ {event}: {args}\n"})

    def run_agent_thread():
        try:
            # Initialize the agent.
            # We configure quiet_mode=True to reduce terminal spam, as we are piping to the UI.
            # Using defaults for model/provider, but these can be extended with Gradio inputs later.
            agent_kwargs = {
                "quiet_mode": True,
                "stream_delta_callback": stream_callback,
                "tool_progress_callback": tool_callback,
                "max_iterations": 10, # Cap for webui
                "provider": provider,
                "model": model_name
            }
            if base_url:
                agent_kwargs["base_url"] = base_url
            if api_key:
                agent_kwargs["api_key"] = api_key

            agent = AIAgent(**agent_kwargs)

            # Reconstruct conversation history in the format AIAgent expects.
            # Gradio history is a list of dicts: {"role": "user"/"assistant", "content": "..."}
            formatted_history = []
            for msg in history:
                # Gradio new format uses Message dicts directly
                if isinstance(msg, dict):
                    formatted_history.append({"role": msg["role"], "content": msg["content"]})
                else:
                    # Legacy fallback
                    user_msg, asst_msg = msg
                    formatted_history.append({"role": "user", "content": user_msg})
                    formatted_history.append({"role": "assistant", "content": asst_msg})

            # Run the conversation blockingly
            result = agent.run_conversation(message, conversation_history=formatted_history)

            # Put a done signal when finished
            q.put({"type": "done", "result": result})
        except Exception as e:
            q.put({"type": "error", "content": str(e)})

    # Start the thread
    thread = threading.Thread(target=run_agent_thread)
    thread.start()

    response_text = ""

    # Process the queue
    while True:
        try:
            # Block until an item is available
            item = q.get(block=True, timeout=120.0)

            if item["type"] == "text":
                response_text += item["content"]
                yield response_text
            elif item["type"] == "log":
                # We can append logs to the response text so they are visible inline,
                # or just stream them out. Appending them before the text stream works.
                response_text += item["content"]
                yield response_text
            elif item["type"] == "error":
                response_text += f"\n\n❌ **Error**: {item['content']}"
                yield response_text
                break
            elif item["type"] == "done":
                # Check if we should override with the final response,
                # but since we streamed, we are mostly good.
                result = item["result"]
                # In case streaming wasn't fully capturing everything or missed the final touch
                if result.get("final_response") and not response_text.strip():
                    response_text = result["final_response"]
                    yield response_text
                break

        except queue.Empty:
            # If the queue is empty for too long, maybe the thread hung or crashed silently
            yield response_text + "\n\n⚠️ *Timed out waiting for response.*"
            break

    thread.join(timeout=1.0)


# Define the Gradio App
with gr.Blocks(title="Hermes Agent") as app:
    gr.Markdown(
        """
        # ☤ Hermes Agent Web UI
        Interact with your self-improving AI agent locally.
        """
    )

    with gr.Accordion("⚙️ Model API Configuration", open=True):
        gr.Markdown("Configure the API provider and model you want to use. You must load a model API before starting a chat.")
        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=["ollama", "openai", "openrouter", "anthropic", "custom"],
                value="ollama",
                label="Provider",
                interactive=True
            )
            model_input = gr.Textbox(
                label="Model Name",
                placeholder="e.g. hermes3",
                value="hermes3",
                interactive=True
            )
        with gr.Row():
            base_url_input = gr.Textbox(
                label="Base URL (Optional)",
                placeholder="e.g. http://localhost:11434/v1",
                value="http://localhost:11434/v1",
                interactive=True
            )
            api_key_input = gr.Textbox(
                label="API Key (Optional)",
                placeholder="Required for OpenAI/OpenRouter/Anthropic",
                type="password",
                interactive=True
            )

        connect_btn = gr.Button("Connect & Save", variant="primary")
        status_text = gr.Markdown("*Status: Disconnected*")

    def connect_model(provider, model_name, base_url, api_key):
        if not provider or not model_name:
            return gr.update(value="⚠️ *Error: Provider and Model Name are required.*", visible=True), gr.update(interactive=False)
        return gr.update(value=f"✅ *Connected to {provider} ({model_name})*", visible=True), gr.update(interactive=True)


    chat_box = gr.ChatInterface(
        fn=chat_interface,
        additional_inputs=[provider_dropdown, base_url_input, model_input, api_key_input],
        examples=[
            ["Hello, who are you?", "ollama", "http://localhost:11434/v1", "hermes3", ""],
            ["What directory am I currently in?", "ollama", "http://localhost:11434/v1", "hermes3", ""],
            ["Can you write a simple python script to calculate fibonacci numbers and save it?", "ollama", "http://localhost:11434/v1", "hermes3", ""]
        ]
    )

    # We gray out the submit button initially or handle it via a connection check.
    # We will use the connect_btn to establish that the user checked their settings.
    # Note: `gr.ChatInterface` doesn't natively support full disabling of the chat box via a boolean update easily,
    # but it will immediately yield an error message if the provider/model are empty.

    connect_btn.click(
        fn=connect_model,
        inputs=[provider_dropdown, model_input, base_url_input, api_key_input],
        outputs=[status_text]
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Default())
