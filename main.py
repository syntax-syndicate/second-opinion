#!/usr/bin/env python3
"""
Second Opinion MCP Server
Allows AI models to get second opinions from other AI models (OpenAI, Gemini, Grok, Claude)
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence
import logging

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# AI API imports
import openai

# Gemini API imports - handle both possible installations gracefully
GEMINI_AVAILABLE = False
USE_NEW_SDK = False

try:
    # Try new SDK first
    import google.genai as genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
    USE_NEW_SDK = True
    print("Using new google-genai SDK", file=sys.stderr)
except ImportError:
    try:
        # Fallback to old SDK
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        USE_NEW_SDK = False
        print("Using legacy google-generativeai SDK", file=sys.stderr)
    except ImportError:
        GEMINI_AVAILABLE = False
        print("Gemini API not available. Install with: pip install google-generativeai", file=sys.stderr)

# Anthropic Claude imports
CLAUDE_AVAILABLE = False
try:
    import anthropic
    CLAUDE_AVAILABLE = True
    print("Anthropic Claude SDK available", file=sys.stderr)
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Claude API not available. Install with: pip install anthropic", file=sys.stderr)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecondOpinionServer:
    def __init__(self):
        self.app = Server("second-opinion")
        
        # Initialize API clients
        self.openai_client = None
        self.gemini_client = None
        self.grok_client = None
        self.claude_client = None
        
        self._setup_clients()
        self._setup_handlers()
    
    def _setup_clients(self):
        """Initialize API clients with environment variables"""
        # OpenAI setup
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OPENAI_API_KEY not found - OpenAI features disabled")
        
        # Gemini setup
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key and GEMINI_AVAILABLE:
            if USE_NEW_SDK:
                self.gemini_client = genai.Client(api_key=gemini_api_key)
            else:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai
            logger.info("Gemini client initialized")
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini API not available - install google-generativeai")
            else:
                logger.warning("GEMINI_API_KEY not found - Gemini features disabled")
        
        # Grok setup (xAI API - compatible with OpenAI SDK)
        grok_api_key = os.getenv("GROK_API_KEY")
        if grok_api_key:
            self.grok_client = openai.OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1"
            )
            logger.info("Grok client initialized")
        else:
            logger.warning("GROK_API_KEY not found - Grok features disabled")
        
        # Claude setup
        claude_api_key = os.getenv("CLAUDE_API_KEY")
        if claude_api_key and CLAUDE_AVAILABLE:
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
            logger.info("Claude client initialized")
        else:
            if not CLAUDE_AVAILABLE:
                logger.warning("Claude API not available - install anthropic")
            else:
                logger.warning("CLAUDE_API_KEY not found - Claude features disabled")
    
    def _setup_handlers(self):
        """Set up MCP handlers"""
        
        @self.app.list_tools()
        async def handle_list_tools() -> List[Tool]:
            tools = []
            
            if self.openai_client:
                tools.extend([
                    Tool(
                        name="get_openai_opinion",
                        description="Get a second opinion from an OpenAI model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "OpenAI model to use",
                                    "enum": [
                                        "o4-mini",
                                        "gpt-4.1",
                                        "gpt-4o"
                                    ],
                                    "default": "gpt-4.1"
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Temperature for response randomness (0.0-2.0)",
                                    "minimum": 0.0,
                                    "maximum": 2.0,
                                    "default": 0.7
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Maximum tokens in response",
                                    "default": 1000
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Optional system prompt to guide the response",
                                    "default": "You are providing a thoughtful second opinion. Be concise but thorough."
                                }
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="compare_openai_models",
                        description="Get opinions from multiple OpenAI models for comparison",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to compare across models"
                                },
                                "models": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "o4-mini",
                                            "gpt-4.1",
                                            "gpt-4o"
                                        ]
                                    },
                                    "description": "List of OpenAI models to compare",
                                    "default": ["gpt-4.1", "gpt-4o"]
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7
                                }
                            },
                            "required": ["prompt"]
                        }
                    )
                ])
            
            if self.gemini_client:
                tools.extend([
                    Tool(
                        name="get_gemini_opinion",
                        description="Get a second opinion from a Gemini model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Gemini model to use",
                                    "enum": [
                                        "gemini-2.0-flash-001",
                                        "gemini-2.5-flash-preview-05-20"
                                    ],
                                    "default": "gemini-2.0-flash-001"
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Temperature for response randomness (0.0-2.0)",
                                    "minimum": 0.0,
                                    "maximum": 2.0,
                                    "default": 0.7
                                },
                                "max_output_tokens": {
                                    "type": "integer",
                                    "description": "Maximum tokens in response",
                                    "default": 1000
                                }
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="compare_gemini_models",
                        description="Get opinions from multiple Gemini models for comparison",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to compare across models"
                                },
                                "models": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "gemini-2.0-flash-001",
                                            "gemini-2.5-pro-experimental"
                                        ]
                                    },
                                    "description": "List of Gemini models to compare",
                                    "default": ["gemini-2.0-flash-001", "gemini-2.5-pro-experimental"]
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7
                                }
                            },
                            "required": ["prompt"]
                        }
                    )
                ])
            
            if self.openai_client and self.gemini_client:
                tools.append(
                    Tool(
                        name="cross_platform_comparison",
                        description="Get opinions from both OpenAI and Gemini models for cross-platform comparison",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to compare across platforms"
                                },
                                "openai_model": {
                                    "type": "string",
                                    "enum": [
                                        "o4-mini",
                                        "gpt-4.1",
                                        "gpt-4o"
                                    ],
                                    "default": "gpt-4.1"
                                },
                                "gemini_model": {
                                    "type": "string",
                                    "enum": [
                                        "gemini-2.0-flash-001",
                                        "gemini-2.5-pro-experimental"
                                    ],
                                    "default": "gemini-2.0-flash-001"
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7
                                }
                            },
                            "required": ["prompt"]
                        }
                    )
                )
            
            return tools
        
        @self.app.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
            try:
                if name == "get_openai_opinion":
                    return await self._get_openai_opinion(**arguments)
                elif name == "get_gemini_opinion":
                    return await self._get_gemini_opinion(**arguments)
                elif name == "compare_openai_models":
                    return await self._compare_openai_models(**arguments)
                elif name == "compare_gemini_models":
                    return await self._compare_gemini_models(**arguments)
                elif name == "cross_platform_comparison":
                    return await self._cross_platform_comparison(**arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _get_openai_opinion(
        self, 
        prompt: str, 
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = "You are providing a thoughtful second opinion. Be concise but thorough."
    ) -> Sequence[TextContent]:
        if not self.openai_client:
            return [TextContent(type="text", text="OpenAI client not configured. Please set OPENAI_API_KEY environment variable.")]
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = f"**OpenAI {model} Opinion:**\n\n{response.choices[0].message.content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"OpenAI API Error: {str(e)}")]
    
    async def _get_gemini_opinion(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.7,
        max_output_tokens: int = 1000
    ) -> Sequence[TextContent]:
        if not self.gemini_client:
            return [TextContent(type="text", text="Gemini client not configured. Please set GEMINI_API_KEY environment variable.")]
        
        try:
            if USE_NEW_SDK:
                config = genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
                
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                result_text = response.text
            else:
                # Using old SDK
                model_obj = self.gemini_client.GenerativeModel(model)
                response = model_obj.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens
                    }
                )
                result_text = response.text
            
            result = f"**Gemini {model} Opinion:**\n\n{result_text}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Gemini API Error: {str(e)}")]
    
    async def _compare_openai_models(
        self,
        prompt: str,
        models: List[str] = None,
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        if not self.openai_client:
            return [TextContent(type="text", text="OpenAI client not configured.")]
        
        if models is None:
            models = ["gpt-4.1", "gpt-4o"]
        
        results = []
        results.append("## OpenAI Model Comparison\n")
        
        for model in models:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Provide a thoughtful analysis. Be concise but thorough."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=800
                )
                
                results.append(f"### {model}\n{response.choices[0].message.content}\n")
                
            except Exception as e:
                results.append(f"### {model}\n❌ Error: {str(e)}\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _compare_gemini_models(
        self,
        prompt: str,
        models: List[str] = None,
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        if not self.gemini_client:
            return [TextContent(type="text", text="Gemini client not configured.")]
        
        if models is None:
            models = ["gemini-2.0-flash-001", "gemini-2.5-pro-experimental"]
        
        results = []
        results.append("## Gemini Model Comparison\n")
        
        for model in models:
            try:
                if USE_NEW_SDK:
                    config = genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=800
                    )
                    
                    response = self.gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config
                    )
                    result_text = response.text
                else:
                    # Using old SDK
                    model_obj = self.gemini_client.GenerativeModel(model)
                    response = model_obj.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": 800
                        }
                    )
                    result_text = response.text
                
                results.append(f"### {model}\n{result_text}\n")
                
            except Exception as e:
                results.append(f"### {model}\n❌ Error: {str(e)}\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _cross_platform_comparison(
        self,
        prompt: str,
        openai_model: str = "gpt-4.1",
        gemini_model: str = "gemini-2.0-flash-001",
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        results = []
        results.append("## Cross-Platform AI Comparison\n")
        
        # Get OpenAI opinion
        if self.openai_client:
            try:
                openai_response = self.openai_client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": "Provide a thoughtful analysis. Be concise but thorough."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=800
                )
                results.append(f"### OpenAI ({openai_model})\n{openai_response.choices[0].message.content}\n")
            except Exception as e:
                results.append(f"### OpenAI ({openai_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### OpenAI\n❌ Not configured\n")
        
        # Get Gemini opinion
        if self.gemini_client:
            try:
                if USE_NEW_SDK:
                    config = genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=800
                    )
                    
                    gemini_response = self.gemini_client.models.generate_content(
                        model=gemini_model,
                        contents=prompt,
                        config=config
                    )
                    result_text = gemini_response.text
                else:
                    # Using old SDK
                    model_obj = self.gemini_client.GenerativeModel(gemini_model)
                    gemini_response = model_obj.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": 800
                        }
                    )
                    result_text = gemini_response.text
                
                results.append(f"### Gemini ({gemini_model})\n{result_text}\n")
            except Exception as e:
                results.append(f"### Gemini ({gemini_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### Gemini\n❌ Not configured\n")
        
        return [TextContent(type="text", text="\n".join(results))]

def main():
    """Main entry point"""
    # Check for required environment variables
    required_vars = []
    if not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    if not os.getenv("GEMINI_API_KEY"):
        required_vars.append("GEMINI_API_KEY")
    
    if required_vars:
        print("⚠️  Warning: Missing environment variables:", file=sys.stderr)
        for var in required_vars:
            print(f"   - {var}", file=sys.stderr)
        print("\nSome features will be disabled. Set these variables to enable full functionality.", file=sys.stderr)
    
    # Create and run server
    server = SecondOpinionServer()
    
    # Use the correct MCP server startup
    import mcp.server.stdio
    
    async def run_server():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name="second-opinion",
                server_version="1.0.0",
                capabilities={}
            )
            await server.app.run(read_stream, write_stream, init_options)
    
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
