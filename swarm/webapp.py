"""
Web Interface for AI Assistant Swarm.

This module provides a web-based interface using Gradio
for interacting with the AI Assistant Swarm through a browser.
"""

import os
import sys
import logging
import asyncio
import argparse
import threading
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Generator

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.base import Base

from .interfaces.common import UserInterface, Role, Conversation, Message
from .swarm_manager import SwarmManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("swarm_webapp.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("webapp")

# Custom Gradio theme for AI Assistant Swarm
class SwarmTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.indigo,
            secondary_hue=colors.blue,
            neutral_hue=colors.gray,
            font=(fonts.GoogleFont("Inter"), fonts.GoogleFont("IBM Plex Mono")),
            font_mono=(fonts.GoogleFont("IBM Plex Mono")),
        )
        self.radius_size = sizes.radius_md
        
        # Card styling
        self.card_background_fill = "white"
        self.card_background_fill_dark = "*neutral_800"
        self.card_border_width = "1px"
        self.card_border_color = "*neutral_100"
        self.card_border_color_dark = "*neutral_700"
        self.card_radius = sizes.radius_lg
        self.card_shadow = "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)"
        
        # Chat styling
        self.chatbot_code_background_color = "*neutral_100"
        self.chatbot_code_background_color_dark = "*neutral_800"

# Helper to run async functions in Gradio sync context
def run_async(func):
    """Decorator to run an async function in a synchronous context."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(func(*args, **kwargs))
        loop.close()
        return result
    return wrapper

class WebInterface(UserInterface):
    """
    Web Interface for the AI Assistant Swarm using Gradio.
    
    This class provides a web-based interface for interacting
    with the AI Assistant Swarm, with a chatbot UI, agent selection,
    and file upload functionality.
    """
    
    def __init__(
        self,
        swarm_manager: SwarmManager,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
    ):
        """
        Initialize the web interface for AI Assistant Swarm.
        
        Args:
            swarm_manager: Manager for the assistant swarm
            server_name: Host to run the server on
            server_port: Port to run the server on
            share: Whether to create a shareable link
        """
        super().__init__(swarm_manager)
        self.server_name = server_name
        self.server_port = server_port
        self.share = share
        self.app = None
        self.theme = SwarmTheme()
        
    async def start(self) -> None:
        """Start the web interface server."""
        # Check if models are available
        models_status = await self.swarm_manager.check_models()
        missing_models = [model for model, status in models_status.items() if not status]
        
        # Log model status
        if missing_models:
            logger.warning(f"Missing models: {', '.join(missing_models)}")
            logger.warning("The application will still start but some functionality may be limited.")
        else:
            logger.info("All required models are available.")
        
        # Create Gradio interface
        self._build_interface()
        
        # Start the server
        logger.info(f"Starting web interface on {self.server_name}:{self.server_port}")
        self.app.launch(
            server_name=self.server_name,
            server_port=self.server_port,
            share=self.share,
            prevent_thread_lock=True
        )
        
    async def shutdown(self) -> None:
        """Shut down the web interface server."""
        if self.app:
            logger.info("Shutting down web interface")
            try:
                # Close the Gradio interface
                self.app.close()
                # Shutdown the swarm manager
                await self.swarm_manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down: {str(e)}")
    
    def _build_interface(self) -> None:
        """Build the Gradio interface."""
        with gr.Blocks(theme=self.theme, title="AI Assistant Swarm") as app:
            # Header
            with gr.Row():
                gr.Markdown(
                    """
                    # AI Assistant Swarm
                    ### A locally hosted multi-agent AI assistant system
                    
                    This system provides a cooperative network of specialized AI agents to help with:
                    - **Python Programming** - Coding help, debugging, and explanations for beginners
                    - **Medical Inquiries** - Health-related questions with beginner-friendly explanations
                    - **General Tasks** - Everyday questions and assistance
                    
                    All processing happens locally on your machine, with zero cloud dependencies.
                    """
                )
            
            # Main layout
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1):
                    # Agent selection
                    agent_dropdown = gr.Dropdown(
                        choices=["Auto (Orchestrator)", "Medical Specialist", "Python Developer", "General Assistant"],
                        value="Auto (Orchestrator)",
                        label="Select Agent",
                        info="Choose which agent to talk to, or let the system decide automatically"
                    )
                    
                    # Upload section
                    with gr.Group():
                        gr.Markdown("### Upload Documents")
                        file_upload = gr.File(
                            label="Upload a file to the assistant's memory",
                            file_types=["pdf", "txt", "py", "md", "docx"]
                        )
                        upload_button = gr.Button("Upload")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    
                    # Memory search
                    with gr.Group():
                        gr.Markdown("### Memory Search")
                        memory_query = gr.Textbox(label="Search Memory")
                        search_button = gr.Button("Search")
                        memory_results = gr.Dataframe(
                            headers=["Type", "Content", "Source", "Date"],
                            label="Search Results",
                            wrap=True,
                            max_rows=5
                        )
                        
                    # Model status
                    with gr.Group():
                        gr.Markdown("### System Status")
                        model_status_btn = gr.Button("Check Model Status")
                        model_status = gr.Dataframe(
                            headers=["Model", "Status"],
                            label="Ollama Models Status",
                            row_count=3
                        )
                    
                    # New conversation button
                    new_conversation_btn = gr.Button("Start New Conversation")
                
                # Main chat area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="AI Assistant Swarm",
                        show_label=True,
                        bubble=True,
                        avatar_images=("avatar-user.png", "avatar-ai.png"),
                        height=600
                    )
                    
                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask a question or provide a task...",
                            show_label=False, 
                            container=False,
                            scale=9
                        )
                        submit_btn = gr.Button("Send", scale=1)
                        
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                    
                    with gr.Accordion("Advanced Options", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature (creativity)",
                            info="Higher values make output more creative, lower values make it more deterministic"
                        )
            
            # Event handlers
            
            # Chat submission
            def user_message_submitted(user_message, history, agent_selection, temperature):
                """Handle user message submission."""
                if not user_message:
                    return "", history
                
                # Add user message to history
                history = history + [(user_message, None)]
                return "", history
            
            def bot_response(history, agent_selection, temperature):
                """Generate bot response based on user input."""
                if not history or not history[-1][0]:
                    return history
                
                user_message = history[-1][0]
                
                # Map agent selection to agent type
                agent_type_map = {
                    "Medical Specialist": "medical",
                    "Python Developer": "python",
                    "General Assistant": "fallback",
                    "Auto (Orchestrator)": None
                }
                agent_type = agent_type_map.get(agent_selection)
                
                # Configure temperature
                self.swarm_manager.set_temperature(temperature)
                
                # Get response
                response_text, agent_name = run_async(self.swarm_manager.process_query)(
                    user_message, agent_type
                )
                
                # Update history with response
                history[-1] = (user_message, f"**{agent_name}**: {response_text}" if agent_name else response_text)
                
                return history
            
            # File upload handler
            def handle_file_upload(file):
                """Handle file upload to memory system."""
                if not file:
                    return "No file selected."
                
                try:
                    document_id = run_async(self.swarm_manager.add_file_to_memory)(file.name)
                    return f"Successfully uploaded: {os.path.basename(file.name)}\nDocument ID: {document_id}"
                except Exception as e:
                    logger.error(f"Error uploading file: {str(e)}")
                    return f"Error uploading file: {str(e)}"
            
            # Memory search handler
            def handle_memory_search(query):
                """Search the memory for matching documents."""
                if not query:
                    return [["", "Please enter a search query.", "", ""]]
                
                try:
                    documents = run_async(self.swarm_manager.search_memory)(query)
                    
                    if not documents:
                        return [["", "No matching documents found.", "", ""]]
                    
                    results = []
                    for doc in documents:
                        doc_type = doc.metadata.get('doc_type', 'Unknown')
                        content = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                        source = doc.metadata.get('source', 'Unknown')
                        date = doc.metadata.get('added_at', 'Unknown')
                        results.append([doc_type, content, source, date])
                    
                    return results
                except Exception as e:
                    logger.error(f"Error searching memory: {str(e)}")
                    return [["Error", str(e), "", ""]]
            
            # Model status handler
            def check_model_status():
                """Check the status of required Ollama models."""
                try:
                    models_status = run_async(self.swarm_manager.check_models)()
                    return [[model, "Available" if status else "Not Found"] for model, status in models_status.items()]
                except Exception as e:
                    logger.error(f"Error checking models: {str(e)}")
                    return [["Error", str(e)]]
            
            # Clear chat handler
            def clear_chat():
                """Clear the chat history."""
                # Create a new conversation in the interface
                self.new_conversation()
                return []
            
            # Connect events
            user_input.submit(
                user_message_submitted,
                [user_input, chatbot, agent_dropdown, temperature_slider],
                [user_input, chatbot],
                queue=False
            ).then(
                bot_response,
                [chatbot, agent_dropdown, temperature_slider],
                [chatbot]
            )
            
            submit_btn.click(
                user_message_submitted,
                [user_input, chatbot, agent_dropdown, temperature_slider],
                [user_input, chatbot],
                queue=False
            ).then(
                bot_response,
                [chatbot, agent_dropdown, temperature_slider],
                [chatbot]
            )
            
            upload_button.click(
                handle_file_upload,
                [file_upload],
                [upload_status]
            )
            
            search_button.click(
                handle_memory_search,
                [memory_query],
                [memory_results]
            )
            
            model_status_btn.click(
                check_model_status,
                [],
                [model_status]
            )
            
            clear_btn.click(
                clear_chat,
                [],
                [chatbot]
            )
            
            new_conversation_btn.click(
                clear_chat,
                [],
                [chatbot]
            )
            
            # Save app
            self.app = app

async def main():
    """Main entry point for the web interface application."""
    parser = argparse.ArgumentParser(description="AI Assistant Swarm Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set logging level")
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize the swarm manager
    swarm_manager = SwarmManager()
    
    # Initialize and start the web interface
    web_interface = WebInterface(
        swarm_manager=swarm_manager,
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
    
    try:
        await web_interface.start()
        
        # Keep the main thread alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        # Shut down the interface
        await web_interface.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1) 