#!/usr/bin/env python3
"""
Second Opinion MCP Server
A modular MCP server for getting AI opinions from multiple providers
"""

import asyncio
import logging
import sys

# Import our modular components
from client_manager import ClientManager
from conversation_manager import ConversationManager
from ai_providers import AIProviders
from mcp_server import MCPServer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecondOpinionServer:
    """Main server orchestrator - coordinates all components"""
    
    def __init__(self):
        """Initialize the server with all components"""
        logger.info("Initializing Second Opinion MCP Server...")
        
        # Initialize components in dependency order
        logger.info("Setting up client manager...")
        self.client_manager = ClientManager()
        
        logger.info("Setting up conversation manager...")
        self.conversation_manager = ConversationManager()
        
        logger.info("Setting up AI providers...")
        self.ai_providers = AIProviders(
            self.client_manager, 
            self.conversation_manager
        )
        
        logger.info("Setting up MCP server...")
        self.mcp_server = MCPServer(
            self.client_manager,
            self.conversation_manager,
            self.ai_providers
        )
        
        # Get the MCP app instance
        self.app = self.mcp_server.get_app()
        
        logger.info("Second Opinion MCP Server initialized successfully!")
        
        # Log available services
        self._log_available_services()
    
    def _log_available_services(self):
        """Log which AI services are available"""
        available = self.client_manager.get_available_clients()
        available_services = [name for name, status in available.items() if status]
        
        if available_services:
            logger.info(f"Available AI services: {', '.join(available_services)}")
        else:
            logger.warning("No AI services configured! Please set up API keys.")
        
        # Log personality feature
        personalities = self.conversation_manager.get_available_personalities()
        logger.info(f"Available personalities: {', '.join(personalities)}")
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Second Opinion MCP Server...")
        try:
            # Import and run MCP server
            from mcp.server.stdio import stdio_server
            
            async with stdio_server() as (read_stream, write_stream):
                await self.app.run(
                    read_stream,
                    write_stream,
                    self.app.create_initialization_options()
                )
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            raise


async def main():
    """Main entry point"""
    try:
        # Create and run the server
        server = SecondOpinionServer()
        await server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


def run_server():
    """Synchronous wrapper for running the server"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.error(f"Server failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()