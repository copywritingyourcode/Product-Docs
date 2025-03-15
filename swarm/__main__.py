"""
Main entry point for the AI Assistant Swarm package.

This module provides a convenient way to start the system
from the command line with different interface options.
"""

import os
import sys
import logging
import asyncio
import argparse
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("main")

async def main():
    """Main entry point for the AI Assistant Swarm."""
    parser = argparse.ArgumentParser(description="AI Assistant Swarm")
    
    # Interface options
    parser.add_argument("--interface", "-i", choices=["cli", "web"], default="cli",
                        help="Interface to use (cli or web)")
    
    # Web interface options
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the web server on")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the web server on")
    parser.add_argument("--share", action="store_true",
                        help="Create a shareable link for the web interface")
    
    # CLI interface options
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable response streaming in CLI")
    parser.add_argument("--hide-agent", action="store_true",
                        help="Hide agent names in CLI responses")
    
    # General options
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory for storing data")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Import here to avoid circular imports
    from .swarm_manager import SwarmManager
    
    # Initialize the swarm manager
    swarm_manager = SwarmManager(
        data_dir=args.data_dir
    )
    
    try:
        # Check if models are available
        models_status = await swarm_manager.check_models()
        missing_models = [model for model, status in models_status.items() if not status]
        
        if missing_models:
            logger.warning(f"Missing models: {', '.join(missing_models)}")
            logger.warning("The system will still start but some functionality may be limited.")
            logger.warning("You can install the missing models with:")
            for model in missing_models:
                logger.warning(f"  ollama pull {model}")
        else:
            logger.info("All required models are available.")
        
        # Start the requested interface
        if args.interface == "cli":
            from .cli import CommandLineInterface
            
            # Initialize and start the CLI
            cli = CommandLineInterface(
                swarm_manager=swarm_manager,
                show_agent_name=not args.hide_agent,
                streaming=not args.no_streaming
            )
            
            await cli.start()
        else:  # web interface
            from .webapp import WebInterface
            
            # Initialize and start the web interface
            web_interface = WebInterface(
                swarm_manager=swarm_manager,
                server_name=args.host,
                server_port=args.port,
                share=args.share
            )
            
            await web_interface.start()
            
            # Keep the main thread alive
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        # Shut down the swarm manager
        await swarm_manager.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1) 