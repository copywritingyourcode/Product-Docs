"""
Command Line Interface for AI Assistant Swarm.

This module provides a simple CLI for interacting with the AI Assistant Swarm,
allowing users to have conversations with the agents through a terminal.
"""

import os
import sys
import logging
import asyncio
import signal
import argparse
from typing import List, Dict, Any, Optional, Tuple, NoReturn
import readline  # For command history

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner

from .interfaces.common import UserInterface, Role, Conversation, Message
from .swarm_manager import SwarmManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("swarm_cli.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cli")

# Rich console for pretty terminal output
console = Console()

class CommandLineInterface(UserInterface):
    """
    Command Line Interface for the AI Assistant Swarm.
    
    This class provides a terminal-based interface for interacting
    with the AI Assistant Swarm, with support for conversation
    management, agent selection, and basic commands.
    """
    
    HELP_TEXT = """
    AI Assistant Swarm CLI - Available Commands:
    
    /help                - Show this help message
    /exit                - Exit the application
    /clear               - Clear the screen
    /new                 - Start a new conversation
    /list                - List available conversations
    /load <id>           - Load a specific conversation
    /agent <type>        - Switch to a specific agent
        Types: auto (default), medical, python, general
    /upload <file_path>  - Upload a file to the memory system
    /memory <query>      - Search the memory for documents matching a query
    /models              - Check status of Ollama models
    
    For any other input, just type your message to chat with the assistant.
    """
    
    def __init__(
        self, 
        swarm_manager: SwarmManager,
        show_agent_name: bool = True,
        streaming: bool = True
    ):
        """
        Initialize the CLI for AI Assistant Swarm.
        
        Args:
            swarm_manager: Manager for the assistant swarm
            show_agent_name: Whether to show which agent responded
            streaming: Whether to stream responses as they are generated
        """
        super().__init__(swarm_manager)
        self.show_agent_name = show_agent_name
        self.streaming = streaming
        self.current_agent_type = "auto"  # auto, medical, python, general
        self.running = False
        self.spinner = None
        
    async def start(self) -> NoReturn:
        """Start the CLI interface and process user input until exit."""
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # Welcome message
        console.clear()
        console.print(Panel.fit(
            "[bold blue]AI Assistant Swarm[/bold blue] - [italic]A locally hosted multi-agent AI assistant[/italic]",
            border_style="blue"
        ))
        console.print("Type [bold green]/help[/bold green] for available commands or start chatting!")
        console.print()
        
        # Check if Ollama models are available
        await self._check_models()
        
        # Main input loop
        while self.running:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if not user_input.strip():
                    continue
                    
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                else:
                    await self._handle_message(user_input)
            except KeyboardInterrupt:
                self._handle_interrupt(None, None)
            except Exception as e:
                logger.error(f"Error in CLI: {str(e)}")
                console.print(f"[bold red]Error:[/bold red] {str(e)}", style="red")
                
    async def shutdown(self) -> None:
        """Shut down the CLI interface gracefully."""
        self.running = False
        await self.swarm_manager.shutdown()
        console.print("\n[bold yellow]Shutting down...[/bold yellow]")
    
    def _handle_interrupt(self, sig, frame) -> None:
        """Handle interrupt signal (Ctrl+C)."""
        console.print("\n[bold yellow]Interrupted![/bold yellow]")
        asyncio.create_task(self.shutdown())
        
    async def _handle_command(self, command: str) -> None:
        """
        Handle a command (starting with /).
        
        Args:
            command: The command string
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == '/help':
            console.print(Markdown(self.HELP_TEXT))
            
        elif cmd == '/exit':
            await self.shutdown()
            sys.exit(0)
            
        elif cmd == '/clear':
            console.clear()
            
        elif cmd == '/new':
            self.new_conversation()
            console.print("[bold blue]Started a new conversation[/bold blue]")
            
        elif cmd == '/list':
            await self._list_conversations()
            
        elif cmd == '/load':
            if not args:
                console.print("[bold red]Error:[/bold red] Please specify a conversation ID", style="red")
            else:
                loaded = self.load_conversation(args)
                if loaded:
                    console.print(f"[bold blue]Loaded conversation: {args}[/bold blue]")
                    await self._show_conversation()
                else:
                    console.print(f"[bold red]Error:[/bold red] Conversation not found: {args}", style="red")
                    
        elif cmd == '/agent':
            await self._change_agent(args)
            
        elif cmd == '/upload':
            if not args:
                console.print("[bold red]Error:[/bold red] Please specify a file path", style="red")
            else:
                await self._upload_file(args)
                
        elif cmd == '/memory':
            if not args:
                console.print("[bold red]Error:[/bold red] Please specify a query", style="red")
            else:
                await self._search_memory(args)
                
        elif cmd == '/models':
            await self._check_models()
            
        else:
            console.print(f"[bold red]Unknown command:[/bold red] {cmd}", style="red")
            console.print("Type [bold green]/help[/bold green] for available commands.")
    
    async def _handle_message(self, message: str) -> None:
        """
        Handle a user message (not a command).
        
        Args:
            message: The user's message
        """
        # Start spinner
        with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
            # Process message through the swarm
            if self.streaming:
                response_text = ""
                agent_name = None
                
                # Stream the response
                async for chunk, is_done, metadata in self.swarm_manager.process_query_stream(
                    message, 
                    agent_type=self.current_agent_type if self.current_agent_type != "auto" else None
                ):
                    response_text += chunk
                    if is_done and metadata:
                        agent_name = metadata.get("agent_name")
                        
                # Add to conversation history
                self.current_conversation.add_message(Role.USER, message)
                self.current_conversation.add_message(Role.ASSISTANT, response_text, agent_name)
                
                # Print the full response with agent name if available
                if agent_name and self.show_agent_name:
                    console.print(f"[bold purple]{agent_name}[/bold purple]")
                else:
                    console.print("[bold purple]Assistant[/bold purple]")
                
                console.print(Markdown(response_text))
            else:
                # Non-streaming mode
                response, agent_name = await self.swarm_manager.process_query(
                    message,
                    agent_type=self.current_agent_type if self.current_agent_type != "auto" else None
                )
                
                # Add to conversation history
                self.current_conversation.add_message(Role.USER, message)
                self.current_conversation.add_message(Role.ASSISTANT, response, agent_name)
                
                # Print the response with agent name if available
                if agent_name and self.show_agent_name:
                    console.print(f"[bold purple]{agent_name}[/bold purple]")
                else:
                    console.print("[bold purple]Assistant[/bold purple]")
                
                console.print(Markdown(response))
    
    async def _list_conversations(self) -> None:
        """List available conversations."""
        conversations = self.list_conversations()
        
        if not conversations:
            console.print("[italic]No conversations available[/italic]")
            return
            
        table = Table(title="Available Conversations")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Messages", style="magenta")
        table.add_column("Last Updated", style="yellow")
        table.add_column("Preview", style="blue")
        
        for conv in conversations:
            table.add_row(
                conv["id"],
                conv["title"],
                str(conv["message_count"]),
                datetime.fromtimestamp(conv["last_updated"]).strftime("%Y-%m-%d %H:%M:%S"),
                conv["preview"]
            )
            
        console.print(table)
    
    async def _change_agent(self, agent_type: str) -> None:
        """
        Change the current agent.
        
        Args:
            agent_type: The type of agent to switch to
        """
        agent_type = agent_type.lower()
        valid_types = ["auto", "medical", "python", "general"]
        
        if agent_type not in valid_types:
            console.print(f"[bold red]Error:[/bold red] Unknown agent type: {agent_type}", style="red")
            console.print(f"Valid types: {', '.join(valid_types)}")
            return
            
        self.current_agent_type = agent_type
        console.print(f"[bold blue]Switched to agent: {agent_type}[/bold blue]")
        
        if agent_type == "medical":
            console.print("[italic]You are now talking to the Medical Specialist agent[/italic]")
        elif agent_type == "python":
            console.print("[italic]You are now talking to the Senior Python Developer agent[/italic]")
        elif agent_type == "general":
            console.print("[italic]You are now talking to the General Assistant agent[/italic]")
        else:  # auto
            console.print("[italic]The system will automatically select the best agent for your query[/italic]")
    
    async def _upload_file(self, file_path: str) -> None:
        """
        Upload a file to the memory system.
        
        Args:
            file_path: Path to the file to upload
        """
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File not found: {file_path}", style="red")
            return
            
        try:
            with console.status(f"[bold blue]Uploading {os.path.basename(file_path)}...[/bold blue]", spinner="dots"):
                document_id = await self.swarm_manager.add_file_to_memory(file_path)
                
            console.print(f"[bold green]Successfully uploaded:[/bold green] {os.path.basename(file_path)}")
            console.print(f"Document ID: {document_id}")
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            console.print(f"[bold red]Error uploading file:[/bold red] {str(e)}", style="red")
    
    async def _search_memory(self, query: str) -> None:
        """
        Search the memory system for documents matching a query.
        
        Args:
            query: The search query
        """
        try:
            with console.status(f"[bold blue]Searching memory...[/bold blue]", spinner="dots"):
                documents = await self.swarm_manager.search_memory(query)
                
            if not documents:
                console.print("[italic]No matching documents found[/italic]")
                return
                
            console.print(f"[bold blue]Found {len(documents)} matching documents:[/bold blue]")
            
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get('source', 'Unknown')
                date = doc.metadata.get('added_at', 'Unknown')
                doc_type = doc.metadata.get('doc_type', 'Unknown')
                
                panel = Panel(
                    Text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""), 
                         style="white", 
                         no_wrap=False),
                    title=f"[bold]{i}. {doc_type}[/bold]",
                    subtitle=f"Source: {source} | Added: {date}",
                    border_style="blue"
                )
                console.print(panel)
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            console.print(f"[bold red]Error searching memory:[/bold red] {str(e)}", style="red")
    
    async def _show_conversation(self) -> None:
        """Display the current conversation."""
        if not self.current_conversation.messages:
            console.print("[italic]No messages in this conversation[/italic]")
            return
            
        for message in self.current_conversation.messages:
            if message.role == Role.USER:
                console.print(f"[bold green]You[/bold green]")
                console.print(message.content)
            elif message.role == Role.ASSISTANT:
                if message.agent_name and self.show_agent_name:
                    console.print(f"[bold purple]{message.agent_name}[/bold purple]")
                else:
                    console.print("[bold purple]Assistant[/bold purple]")
                console.print(Markdown(message.content))
            console.print()
    
    async def _check_models(self) -> None:
        """Check if required Ollama models are available."""
        try:
            with console.status("[bold blue]Checking Ollama models...[/bold blue]", spinner="dots"):
                models_status = await self.swarm_manager.check_models()
                
            table = Table(title="Ollama Models Status")
            table.add_column("Model", style="cyan")
            table.add_column("Status", style="green")
            
            for model, status in models_status.items():
                icon = "✅" if status else "❌"
                status_text = "[green]Available[/green]" if status else "[red]Not Found[/red]"
                table.add_row(model, f"{icon} {status_text}")
                
            console.print(table)
            
            missing = [model for model, status in models_status.items() if not status]
            if missing:
                console.print("\n[bold yellow]Some models are missing![/bold yellow]")
                console.print("You can install them with the following commands:")
                for model in missing:
                    console.print(f"  [bold]ollama pull {model}[/bold]")
        except Exception as e:
            logger.error(f"Error checking models: {str(e)}")
            console.print(f"[bold red]Error checking Ollama models:[/bold red] {str(e)}", style="red")
            console.print("[italic]Is Ollama running?[/italic]")

async def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="AI Assistant Swarm CLI")
    parser.add_argument("--no-streaming", action="store_true", help="Disable response streaming")
    parser.add_argument("--hide-agent", action="store_true", help="Hide agent names in responses")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set logging level")
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize the swarm manager
    swarm_manager = SwarmManager()
    
    # Initialize and start the CLI
    cli = CommandLineInterface(
        swarm_manager=swarm_manager,
        show_agent_name=not args.hide_agent,
        streaming=not args.no_streaming
    )
    
    try:
        await cli.start()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. Exiting...[/bold yellow]")
        sys.exit(0) 