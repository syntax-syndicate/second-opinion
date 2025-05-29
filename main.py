#!/usr/bin/env python3
"""
Second Opinion MCP Server
Allows AI models to get second opinions from other AI models (OpenAI, Gemini, Grok, Claude)
Features conversation history and collaborative prompting
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence
import logging
from collections import defaultdict

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
        
        # Conversation history storage
        # Format: {platform_model: [conversation_history]}
        self.conversation_histories = defaultdict(list)
        
        # Collaborative system prompt
        self.collaborative_system_prompt = """You are an AI assistant working in a collaborative environment with Claude (an Anthropic AI) and other AI models to help users. Claude is coordinating this collaboration and has sent you this message as part of a multi-AI consultation. 

Your role is to provide your unique perspective and expertise to help answer the user's question. You're part of a team of AI assistants, each bringing different strengths and viewpoints. Be thoughtful, helpful, and concise in your response, as your input will be combined with responses from other AI models to give the user a comprehensive answer.

Remember that you're working together with Claude and other AIs to provide the best possible assistance to the user."""
        
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
    
    def _get_conversation_key(self, platform: str, model: str) -> str:
        """Generate a key for conversation history storage"""
        return f"{platform}_{model}"
    
    def _add_to_conversation_history(self, key: str, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_histories[key].append({"role": role, "content": content})
        
        # Keep only last 10 exchanges (20 messages) to manage memory
        if len(self.conversation_histories[key]) > 20:
            self.conversation_histories[key] = self.conversation_histories[key][-20:]
    
    def _get_openai_messages(self, key: str, prompt: str, system_prompt: str = None) -> List[Dict]:
        """Build OpenAI messages array with conversation history"""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": self.collaborative_system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_histories[key])
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _get_gemini_history_and_prompt(self, key: str, prompt: str):
        """Build Gemini conversation history and current prompt"""
        history = []
        
        # Convert conversation history to Gemini format
        for msg in self.conversation_histories[key]:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [{"text": msg["content"]}]})
        
        return history, prompt
    
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
                                    "default": ""
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
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
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
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
                                            "gemini-2.5-flash-preview-05-20"
                                        ]
                                    },
                                    "description": "List of Gemini models to compare",
                                    "default": ["gemini-2.0-flash-001", "gemini-2.5-flash-preview-05-20"]
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
            
            if self.grok_client:
                tools.extend([
                    Tool(
                        name="get_grok_opinion",
                        description="Get a second opinion from a Grok model (xAI)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Grok model to use",
                                    "enum": [
                                        "grok-3",
                                        "grok-2",
                                        "grok-beta"
                                    ],
                                    "default": "grok-3"
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
                                    "default": ""
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
                                }
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="compare_grok_models",
                        description="Get opinions from multiple Grok models for comparison",
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
                                            "grok-3",
                                            "grok-2",
                                            "grok-beta"
                                        ]
                                    },
                                    "description": "List of Grok models to compare",
                                    "default": ["grok-3", "grok-2"]
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
            
            if self.claude_client:
                tools.extend([
                    Tool(
                        name="get_claude_opinion",
                        description="Get a second opinion from a Claude model (Anthropic)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Claude model to use",
                                    "enum": [
                                        "claude-4-opus-20250522",
                                        "claude-4-sonnet-20250522",
                                        "claude-3-7-sonnet-20250224",
                                        "claude-3-5-sonnet-20241022"
                                    ],
                                    "default": "claude-4-sonnet-20250522"
                                },
                                "temperature": {
                                    "type": "number",
                                    "description": "Temperature for response randomness (0.0-1.0)",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
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
                                    "default": ""
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
                                }
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="compare_claude_models",
                        description="Get opinions from multiple Claude models for comparison",
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
                                            "claude-4-opus-20250522",
                                            "claude-4-sonnet-20250522",
                                            "claude-3-7-sonnet-20250224",
                                            "claude-3-5-sonnet-20241022"
                                        ]
                                    },
                                    "description": "List of Claude models to compare",
                                    "default": ["claude-4-opus-20250522", "claude-4-sonnet-20250522"]
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
            
            # Cross-platform comparison tools
            available_providers = []
            if self.openai_client: available_providers.append("OpenAI")
            if self.gemini_client: available_providers.append("Gemini")
            if self.grok_client: available_providers.append("Grok")
            if self.claude_client: available_providers.append("Claude")
            
            if len(available_providers) >= 2:
                tools.append(
                    Tool(
                        name="cross_platform_comparison",
                        description=f"Get opinions from multiple AI platforms for cross-platform comparison. Available: {', '.join(available_providers)}",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to compare across platforms"
                                },
                                "openai_model": {
                                    "type": "string",
                                    "enum": ["o4-mini", "gpt-4.1", "gpt-4o"],
                                    "default": "gpt-4.1"
                                },
                                "gemini_model": {
                                    "type": "string",
                                    "enum": ["gemini-2.0-flash-001", "gemini-2.5-flash-preview-05-20"],
                                    "default": "gemini-2.0-flash-001"
                                },
                                "grok_model": {
                                    "type": "string",
                                    "enum": ["grok-3", "grok-2", "grok-beta"],
                                    "default": "grok-3"
                                },
                                "claude_model": {
                                    "type": "string",
                                    "enum": ["claude-4-opus-20250522", "claude-4-sonnet-20250522", "claude-3-7-sonnet-20250224"],
                                    "default": "claude-4-sonnet-20250522"
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
            
            # Add conversation management tools
            tools.append(
                Tool(
                    name="list_conversation_histories",
                    description="List all active conversation histories with AI models",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            )
            
            tools.append(
                Tool(
                    name="clear_conversation_history",
                    description="Clear conversation history for a specific AI model or all models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "platform": {
                                "type": "string",
                                "description": "Platform to clear (openai, gemini, grok, claude, or 'all')",
                                "enum": ["openai", "gemini", "grok", "claude", "all"]
                            },
                            "model": {
                                "type": "string",
                                "description": "Specific model to clear (optional, clears all models for platform if not specified)",
                                "default": ""
                            }
                        },
                        "required": ["platform"]
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
                elif name == "get_grok_opinion":
                    return await self._get_grok_opinion(**arguments)
                elif name == "get_claude_opinion":
                    return await self._get_claude_opinion(**arguments)
                elif name == "compare_openai_models":
                    return await self._compare_openai_models(**arguments)
                elif name == "compare_gemini_models":
                    return await self._compare_gemini_models(**arguments)
                elif name == "compare_grok_models":
                    return await self._compare_grok_models(**arguments)
                elif name == "compare_claude_models":
                    return await self._compare_claude_models(**arguments)
                elif name == "cross_platform_comparison":
                    return await self._cross_platform_comparison(**arguments)
                elif name == "list_conversation_histories":
                    return await self._list_conversation_histories()
                elif name == "clear_conversation_history":
                    return await self._clear_conversation_history(**arguments)
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
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.openai_client:
            return [TextContent(type="text", text="OpenAI client not configured. Please set OPENAI_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("openai", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            # Use max_completion_tokens for o4-mini and other o-series models
            token_param = "max_completion_tokens" if model.startswith("o") else "max_tokens"
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                token_param: max_tokens
            }
            
            response = self.openai_client.chat.completions.create(**kwargs)
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**OpenAI {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"OpenAI API Error: {str(e)}")]
    
    async def _get_gemini_opinion(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.7,
        max_output_tokens: int = 1000,
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.gemini_client:
            return [TextContent(type="text", text="Gemini client not configured. Please set GEMINI_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("gemini", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            if USE_NEW_SDK:
                # Build conversation history for new SDK
                history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                
                config = genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    system_instruction=self.collaborative_system_prompt
                )
                
                # Create chat with history
                chat = self.gemini_client.chats.create(
                    model=model,
                    config=config,
                    history=history
                )
                
                response = chat.send_message(current_prompt)
                result_text = response.text
            else:
                # Using old SDK
                model_obj = self.gemini_client.GenerativeModel(
                    model,
                    system_instruction=self.collaborative_system_prompt
                )
                
                # Build conversation history for old SDK
                history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                
                if history:
                    chat = model_obj.start_chat(history=history)
                    response = chat.send_message(
                        current_prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_output_tokens
                        }
                    )
                else:
                    response = model_obj.generate_content(
                        current_prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_output_tokens
                        }
                    )
                result_text = response.text
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", result_text)
            
            result = f"**Gemini {model} Opinion:**\n\n{result_text}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Gemini API Error: {str(e)}")]
    
    async def _get_grok_opinion(
        self,
        prompt: str,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.grok_client:
            return [TextContent(type="text", text="Grok client not configured. Please set GROK_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("grok", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.grok_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Grok {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Grok API Error: {str(e)}")]
    
    async def _get_claude_opinion(
        self,
        prompt: str,
        model: str = "claude-4-sonnet-20250522",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.claude_client:
            return [TextContent(type="text", text="Claude client not configured. Please set CLAUDE_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("claude", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history (Claude format)
            messages = []
            for msg in self.conversation_histories[conversation_key]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Use custom system prompt or default collaborative prompt
            final_system_prompt = system_prompt if system_prompt else self.collaborative_system_prompt
            
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=final_system_prompt,
                messages=messages
            )
            
            response_content = response.content[0].text
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Claude {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Claude API Error: {str(e)}")]
    
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
                conversation_key = self._get_conversation_key("openai", model)
                messages = self._get_openai_messages(conversation_key, prompt)
                
                # Use max_completion_tokens for o-series models
                token_param = "max_completion_tokens" if model.startswith("o") else "max_tokens"
                
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    token_param: 800
                }
                
                response = self.openai_client.chat.completions.create(**kwargs)
                response_content = response.choices[0].message.content
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### {model}\n{response_content}\n")
                
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
            models = ["gemini-2.0-flash-001", "gemini-2.5-flash-preview-05-20"]
        
        results = []
        results.append("## Gemini Model Comparison\n")
        
        for model in models:
            try:
                conversation_key = self._get_conversation_key("gemini", model)
                
                if USE_NEW_SDK:
                    history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                    
                    config = genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=800,
                        system_instruction=self.collaborative_system_prompt
                    )
                    
                    if history:
                        chat = self.gemini_client.chats.create(
                            model=model,
                            config=config,
                            history=history
                        )
                        response = chat.send_message(current_prompt)
                    else:
                        response = self.gemini_client.models.generate_content(
                            model=model,
                            contents=current_prompt,
                            config=config
                        )
                    result_text = response.text
                else:
                    # Using old SDK
                    model_obj = self.gemini_client.GenerativeModel(
                        model,
                        system_instruction=self.collaborative_system_prompt
                    )
                    
                    history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                    
                    if history:
                        chat = model_obj.start_chat(history=history)
                        response = chat.send_message(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 800
                            }
                        )
                    else:
                        response = model_obj.generate_content(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 800
                            }
                        )
                    result_text = response.text
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", result_text)
                
                results.append(f"### {model}\n{result_text}\n")
                
            except Exception as e:
                results.append(f"### {model}\n❌ Error: {str(e)}\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _compare_grok_models(
        self,
        prompt: str,
        models: List[str] = None,
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        if not self.grok_client:
            return [TextContent(type="text", text="Grok client not configured.")]
        
        if models is None:
            models = ["grok-3", "grok-2"]
        
        results = []
        results.append("## Grok Model Comparison\n")
        
        for model in models:
            try:
                conversation_key = self._get_conversation_key("grok", model)
                messages = self._get_openai_messages(conversation_key, prompt)
                
                response = self.grok_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800
                )
                
                response_content = response.choices[0].message.content
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### {model}\n{response_content}\n")
                
            except Exception as e:
                results.append(f"### {model}\n❌ Error: {str(e)}\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _compare_claude_models(
        self,
        prompt: str,
        models: List[str] = None,
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        if not self.claude_client:
            return [TextContent(type="text", text="Claude client not configured.")]
        
        if models is None:
            models = ["claude-4-opus-20250522", "claude-4-sonnet-20250522"]
        
        results = []
        results.append("## Claude Model Comparison\n")
        
        for model in models:
            try:
                conversation_key = self._get_conversation_key("claude", model)
                
                # Build messages with conversation history
                messages = []
                for msg in self.conversation_histories[conversation_key]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                
                response = self.claude_client.messages.create(
                    model=model,
                    max_tokens=800,
                    temperature=temperature,
                    system=self.collaborative_system_prompt,
                    messages=messages
                )
                
                response_content = response.content[0].text
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### {model}\n{response_content}\n")
                
            except Exception as e:
                results.append(f"### {model}\n❌ Error: {str(e)}\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _cross_platform_comparison(
        self,
        prompt: str,
        openai_model: str = "gpt-4.1",
        gemini_model: str = "gemini-2.0-flash-001",
        grok_model: str = "grok-3",
        claude_model: str = "claude-4-sonnet-20250522",
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        results = []
        results.append("## Cross-Platform AI Comparison\n")
        
        # Get OpenAI opinion
        if self.openai_client:
            try:
                conversation_key = self._get_conversation_key("openai", openai_model)
                messages = self._get_openai_messages(conversation_key, prompt)
                
                token_param = "max_completion_tokens" if openai_model.startswith("o") else "max_tokens"
                kwargs = {
                    "model": openai_model,
                    "messages": messages,
                    "temperature": temperature,
                    token_param: 800
                }
                
                openai_response = self.openai_client.chat.completions.create(**kwargs)
                response_content = openai_response.choices[0].message.content
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### OpenAI ({openai_model})\n{response_content}\n")
            except Exception as e:
                results.append(f"### OpenAI ({openai_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### OpenAI\n❌ Not configured\n")
        
        # Get Gemini opinion
        if self.gemini_client:
            try:
                conversation_key = self._get_conversation_key("gemini", gemini_model)
                
                if USE_NEW_SDK:
                    history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                    
                    config = genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=800,
                        system_instruction=self.collaborative_system_prompt
                    )
                    
                    if history:
                        chat = self.gemini_client.chats.create(
                            model=gemini_model,
                            config=config,
                            history=history
                        )
                        response = chat.send_message(current_prompt)
                    else:
                        response = self.gemini_client.models.generate_content(
                            model=gemini_model,
                            contents=current_prompt,
                            config=config
                        )
                    result_text = response.text
                else:
                    # Using old SDK
                    model_obj = self.gemini_client.GenerativeModel(
                        gemini_model,
                        system_instruction=self.collaborative_system_prompt
                    )
                    
                    history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
                    
                    if history:
                        chat = model_obj.start_chat(history=history)
                        response = chat.send_message(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 800
                            }
                        )
                    else:
                        response = model_obj.generate_content(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 800
                            }
                        )
                    result_text = response.text
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", result_text)
                
                results.append(f"### Gemini ({gemini_model})\n{result_text}\n")
            except Exception as e:
                results.append(f"### Gemini ({gemini_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### Gemini\n❌ Not configured\n")
        
        # Get Grok opinion
        if self.grok_client:
            try:
                conversation_key = self._get_conversation_key("grok", grok_model)
                messages = self._get_openai_messages(conversation_key, prompt)
                
                grok_response = self.grok_client.chat.completions.create(
                    model=grok_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800
                )
                
                response_content = grok_response.choices[0].message.content
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### Grok ({grok_model})\n{response_content}\n")
            except Exception as e:
                results.append(f"### Grok ({grok_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### Grok\n❌ Not configured\n")
        
        # Get Claude opinion
        if self.claude_client:
            try:
                conversation_key = self._get_conversation_key("claude", claude_model)
                
                # Build messages with conversation history
                messages = []
                for msg in self.conversation_histories[conversation_key]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                
                claude_response = self.claude_client.messages.create(
                    model=claude_model,
                    max_tokens=800,
                    temperature=temperature,
                    system=self.collaborative_system_prompt,
                    messages=messages
                )
                
                response_content = claude_response.content[0].text
                
                # Add to conversation history
                self._add_to_conversation_history(conversation_key, "user", prompt)
                self._add_to_conversation_history(conversation_key, "assistant", response_content)
                
                results.append(f"### Claude ({claude_model})\n{response_content}\n")
            except Exception as e:
                results.append(f"### Claude ({claude_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### Claude\n❌ Not configured\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
    async def _list_conversation_histories(self) -> Sequence[TextContent]:
        """List all active conversation histories"""
        if not self.conversation_histories:
            return [TextContent(type="text", text="No active conversation histories.")]
        
        result = "## Active Conversation Histories\n\n"
        
        for key, history in self.conversation_histories.items():
            if history:  # Only show keys with actual conversation data
                platform, model = key.split("_", 1)
                message_count = len(history)
                result += f"**{platform.title()} - {model}**: {message_count} messages\n"
        
        if result == "## Active Conversation Histories\n\n":
            result += "No conversation histories with messages found."
        
        return [TextContent(type="text", text=result)]
    
    async def _clear_conversation_history(self, platform: str, model: str = "") -> Sequence[TextContent]:
        """Clear conversation history for specified platform/model"""
        if platform == "all":
            self.conversation_histories.clear()
            return [TextContent(type="text", text="✅ Cleared all conversation histories.")]
        
        if model:
            # Clear specific model
            key = self._get_conversation_key(platform, model)
            if key in self.conversation_histories:
                del self.conversation_histories[key]
                return [TextContent(type="text", text=f"✅ Cleared conversation history for {platform} {model}.")]
            else:
                return [TextContent(type="text", text=f"No conversation history found for {platform} {model}.")]
        else:
            # Clear all models for platform
            keys_to_remove = [key for key in self.conversation_histories.keys() if key.startswith(f"{platform}_")]
            for key in keys_to_remove:
                del self.conversation_histories[key]
            
            if keys_to_remove:
                return [TextContent(type="text", text=f"✅ Cleared all conversation histories for {platform} ({len(keys_to_remove)} models).")]
            else:
                return [TextContent(type="text", text=f"No conversation histories found for {platform}.")]

def main():
    """Main entry point"""
    # Check for required environment variables
    required_vars = []
    if not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY")
    if not os.getenv("GEMINI_API_KEY"):
        required_vars.append("GEMINI_API_KEY")
    if not os.getenv("GROK_API_KEY"):
        required_vars.append("GROK_API_KEY")
    if not os.getenv("CLAUDE_API_KEY"):
        required_vars.append("CLAUDE_API_KEY")
    
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
