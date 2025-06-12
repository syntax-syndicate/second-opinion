#!/usr/bin/env python3
"""
Second Opinion MCP Server
Allows AI models to get second opinions from other AI models (OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, OpenRouter)
Features conversation history, collaborative prompting, and group discussions
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
import requests

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
        self.huggingface_api_key = None
        self.deepseek_client = None
        self.openrouter_client = None
        self.mistral_client = None
        self.together_client = None
        self.cohere_client = None
        self.groq_client_fast = None  # New fast Groq client
        self.perplexity_client = None
        
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
        
        # HuggingFace setup
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if huggingface_api_key:
            self.huggingface_api_key = huggingface_api_key
            logger.info("HuggingFace client initialized")
        else:
            logger.warning("HUGGINGFACE_API_KEY not found - HuggingFace features disabled")
        
        # DeepSeek setup (uses OpenAI SDK with different base URL)
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            self.deepseek_client = openai.OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            logger.info("DeepSeek client initialized")
        else:
            logger.warning("DEEPSEEK_API_KEY not found - DeepSeek features disabled")
        
        # OpenRouter setup (uses OpenAI SDK with OpenRouter base URL)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            self.openrouter_client = openai.OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized")
        else:
            logger.warning("OPENROUTER_API_KEY not found - OpenRouter features disabled")
        
        # Mistral AI setup (uses OpenAI SDK with Mistral base URL)
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key:
            self.mistral_client = openai.OpenAI(
                api_key=mistral_api_key,
                base_url="https://api.mistral.ai/v1"
            )
            logger.info("Mistral AI client initialized")
        else:
            logger.warning("MISTRAL_API_KEY not found - Mistral AI features disabled")
        
        # Together AI setup (uses OpenAI SDK with Together base URL)
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if together_api_key:
            self.together_client = openai.OpenAI(
                api_key=together_api_key,
                base_url="https://api.together.xyz/v1"
            )
            logger.info("Together AI client initialized")
        else:
            logger.warning("TOGETHER_API_KEY not found - Together AI features disabled")
        
        # Cohere setup (uses native Cohere SDK)
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            try:
                import cohere
                self.cohere_client = cohere.Client(api_key=cohere_api_key)
                logger.info("Cohere client initialized")
            except ImportError:
                logger.warning("Cohere package not available. Install with: pip install cohere")
                self.cohere_client = None
        else:
            logger.warning("COHERE_API_KEY not found - Cohere features disabled")
        
        # Enhanced Groq setup (using their fast inference API)
        groq_fast_api_key = os.getenv("GROQ_FAST_API_KEY") or os.getenv("GROQ_API_KEY")
        if groq_fast_api_key:
            self.groq_client_fast = openai.OpenAI(
                api_key=groq_fast_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            logger.info("Groq Fast client initialized")
        else:
            logger.warning("GROQ_FAST_API_KEY not found - Enhanced Groq features disabled")
        
        # Perplexity AI setup (uses OpenAI SDK with Perplexity base URL)
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_api_key:
            self.perplexity_client = openai.OpenAI(
                api_key=perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            logger.info("Perplexity AI client initialized")
        else:
            logger.warning("PERPLEXITY_API_KEY not found - Perplexity AI features disabled")
    
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
                                    "default": 4000
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
                                    "default": 4000
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
                                        "grok-3-thinking",
                                        "grok-3-mini",
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
                                    "default": 4000
                                },
                                "reasoning_effort": {
                                    "type": "string",
                                    "description": "Reasoning effort for thinking models (low/high)",
                                    "enum": ["low", "high"],
                                    "default": "low"
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
                                            "grok-3-thinking",
                                            "grok-3-mini",
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
                                        "claude-4-opus-20250514",
                                        "claude-4-sonnet-20250514",
                                        "claude-3-7-sonnet-20250224",
                                        "claude-3-5-sonnet-20241022"
                                    ],
                                    "default": "claude-4-sonnet-20250514"
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
                                    "default": 4000
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
                                            "claude-4-opus-20250514",
                                            "claude-4-sonnet-20250514",
                                            "claude-3-7-sonnet-20250224",
                                            "claude-3-5-sonnet-20241022"
                                        ]
                                    },
                                    "description": "List of Claude models to compare",
                                    "default": ["claude-4-opus-20250514", "claude-4-sonnet-20250514"]
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
            
            # HuggingFace tools
            if self.huggingface_api_key:
                tools.append(
                    Tool(
                        name="get_huggingface_opinion",
                        description="Get a second opinion from any HuggingFace model via Inference API",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "HuggingFace model to use (e.g., 'microsoft/DialoGPT-large', 'meta-llama/Llama-3.3-70B-Instruct')",
                                    "default": "meta-llama/Llama-3.3-70B-Instruct"
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
                                    "default": 4000
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
                                }
                            },
                            "required": ["prompt", "model"]
                        }
                    )
                )
            
            # DeepSeek tools
            if self.deepseek_client:
                tools.extend([
                    Tool(
                        name="get_deepseek_opinion",
                        description="Get a second opinion from a DeepSeek model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "DeepSeek model to use",
                                    "enum": [
                                        "deepseek-chat",
                                        "deepseek-reasoner"
                                    ],
                                    "default": "deepseek-chat"
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
                                    "default": 4000
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
                    )
                ])
            
            # OpenRouter tools
            if self.openrouter_client:
                tools.extend([
                    Tool(
                        name="get_openrouter_opinion",
                        description="Get a second opinion from any OpenRouter model (300+ models available)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "OpenRouter model to use (e.g., 'anthropic/claude-3-5-sonnet', 'openai/gpt-4', 'meta-llama/llama-3.1-405b-instruct')",
                                    "default": "anthropic/claude-3-5-sonnet"
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
                                    "default": 4000
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
                            "required": ["prompt", "model"]
                        }
                    ),
                    Tool(
                        name="list_openrouter_models",
                        description="List all available OpenRouter models (300+ models from various providers)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "filter_by": {
                                    "type": "string",
                                    "description": "Filter models by provider (e.g., 'openai', 'anthropic', 'meta-llama', 'google')",
                                    "default": ""
                                }
                            },
                            "required": []
                        }
                    )
                ])
            
            # Mistral AI tools
            if self.mistral_client:
                tools.append(
                    Tool(
                        name="get_mistral_opinion",
                        description="Get a second opinion from a Mistral AI model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Mistral model to use",
                                    "enum": [
                                        "mistral-large-latest",
                                        "mistral-small-latest",
                                        "mistral-medium-latest",
                                        "codestral-latest"
                                    ],
                                    "default": "mistral-small-latest"
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
                                    "default": 4000
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
                    )
                )
            
            # Together AI tools
            if self.together_client:
                tools.append(
                    Tool(
                        name="get_together_opinion",
                        description="Get a second opinion from Together AI (200+ open-source models)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Together AI model to use",
                                    "enum": [
                                        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                                        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                                        "meta-llama/Llama-3.1-405B-Instruct-Turbo",
                                        "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                        "microsoft/WizardLM-2-8x22B",
                                        "Qwen/Qwen2.5-72B-Instruct-Turbo"
                                    ],
                                    "default": "meta-llama/Llama-3.1-8B-Instruct-Turbo"
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
                                    "default": 4000
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
                    )
                )
            
            # Cohere tools
            if self.cohere_client:
                tools.append(
                    Tool(
                        name="get_cohere_opinion",
                        description="Get a second opinion from a Cohere model",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Cohere model to use",
                                    "enum": [
                                        "command-r-plus",
                                        "command-r",
                                        "command"
                                    ],
                                    "default": "command-r-plus"
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
                                    "default": 4000
                                },
                                "reset_conversation": {
                                    "type": "boolean",
                                    "description": "Reset conversation history for this model",
                                    "default": False
                                }
                            },
                            "required": ["prompt"]
                        }
                    )
                )
            
            # Groq Fast tools
            if self.groq_client_fast:
                tools.append(
                    Tool(
                        name="get_groq_fast_opinion",
                        description="Get a second opinion from Groq's ultra-fast inference API",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Groq fast model to use",
                                    "enum": [
                                        "llama-3.1-70b-versatile",
                                        "llama-3.1-8b-instant",
                                        "mixtral-8x7b-32768",
                                        "gemma2-9b-it"
                                    ],
                                    "default": "llama-3.1-70b-versatile"
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
                                    "default": 4000
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
                    )
                )
            
            # Perplexity AI tools
            if self.perplexity_client:
                tools.append(
                    Tool(
                        name="get_perplexity_opinion",
                        description="Get a second opinion from Perplexity AI (with web search capabilities)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The question or prompt to get an opinion on"
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Perplexity model to use",
                                    "enum": [
                                        "llama-3.1-sonar-large-128k-online",
                                        "llama-3.1-sonar-small-128k-online",
                                        "llama-3.1-sonar-large-128k-chat",
                                        "llama-3.1-sonar-small-128k-chat"
                                    ],
                                    "default": "llama-3.1-sonar-large-128k-online"
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
                                    "default": 4000
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
                    )
                )
            
            # Cross-platform comparison tools
            available_providers = []
            if self.openai_client: available_providers.append("OpenAI")
            if self.gemini_client: available_providers.append("Gemini")
            if self.grok_client: available_providers.append("Grok")
            if self.claude_client: available_providers.append("Claude")
            if self.huggingface_api_key: available_providers.append("HuggingFace")
            if self.deepseek_client: available_providers.append("DeepSeek")
            if self.openrouter_client: available_providers.append("OpenRouter")
            if self.mistral_client: available_providers.append("Mistral")
            if self.together_client: available_providers.append("Together")
            if self.cohere_client: available_providers.append("Cohere")
            if self.groq_client_fast: available_providers.append("GroqFast")
            if self.perplexity_client: available_providers.append("Perplexity")
            
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
                                    "enum": ["grok-3", "grok-3-thinking", "grok-2", "grok-beta"],
                                    "default": "grok-3"
                                },
                                "claude_model": {
                                    "type": "string",
                                    "enum": ["claude-4-opus-20250514", "claude-4-sonnet-20250514", "claude-3-7-sonnet-20250224"],
                                    "default": "claude-4-sonnet-20250514"
                                },
                                "huggingface_model": {
                                    "type": "string",
                                    "default": "meta-llama/Llama-3.3-70B-Instruct"
                                },
                                "deepseek_model": {
                                    "type": "string",
                                    "enum": ["deepseek-chat", "deepseek-reasoner"],
                                    "default": "deepseek-chat"
                                },
                                "openrouter_model": {
                                    "type": "string",
                                    "description": "OpenRouter model to use (e.g., 'anthropic/claude-3-5-sonnet', 'openai/gpt-4')",
                                    "default": "anthropic/claude-3-5-sonnet"
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
            
            # Group discussion feature
            if len(available_providers) >= 2:
                tools.append(
                    Tool(
                        name="group_discussion",
                        description=f"Start a group discussion between multiple AI models where each can see what others said. Available: {', '.join(available_providers)}",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "The topic or question for group discussion"
                                },
                                "participants": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "platform": {
                                                "type": "string",
                                                "enum": ["openai", "gemini", "grok", "claude", "huggingface", "deepseek", "openrouter"]
                                            },
                                            "model": {
                                                "type": "string"
                                            }
                                        },
                                        "required": ["platform", "model"]
                                    },
                                    "description": "List of AI models to participate in discussion",
                                    "default": [
                                        {"platform": "openai", "model": "gpt-4.1"},
                                        {"platform": "claude", "model": "claude-4-sonnet-20250514"},
                                        {"platform": "gemini", "model": "gemini-2.0-flash-001"}
                                    ]
                                },
                                "rounds": {
                                    "type": "integer",
                                    "description": "Number of discussion rounds",
                                    "minimum": 1,
                                    "maximum": 5,
                                    "default": 2
                                },
                                "temperature": {
                                    "type": "number",
                                    "default": 0.7
                                }
                            },
                            "required": ["topic"]
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
                                "description": "Platform to clear (openai, gemini, grok, claude, huggingface, deepseek, openrouter, mistral, together, cohere, groq_fast, perplexity, or 'all')",
                                "enum": ["openai", "gemini", "grok", "claude", "huggingface", "deepseek", "openrouter", "mistral", "together", "cohere", "groq_fast", "perplexity", "all"]
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
                elif name == "get_huggingface_opinion":
                    return await self._get_huggingface_opinion(**arguments)
                elif name == "get_deepseek_opinion":
                    return await self._get_deepseek_opinion(**arguments)
                elif name == "get_openrouter_opinion":
                    return await self._get_openrouter_opinion(**arguments)
                elif name == "list_openrouter_models":
                    return await self._list_openrouter_models(**arguments)
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
                elif name == "group_discussion":
                    return await self._group_discussion(**arguments)
                elif name == "list_conversation_histories":
                    return await self._list_conversation_histories()
                elif name == "clear_conversation_history":
                    return await self._clear_conversation_history(**arguments)
                elif name == "get_mistral_opinion":
                    return await self._get_mistral_opinion(**arguments)
                elif name == "get_together_opinion":
                    return await self._get_together_opinion(**arguments)
                elif name == "get_cohere_opinion":
                    return await self._get_cohere_opinion(**arguments)
                elif name == "get_groq_fast_opinion":
                    return await self._get_groq_fast_opinion(**arguments)
                elif name == "get_perplexity_opinion":
                    return await self._get_perplexity_opinion(**arguments)
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
        max_tokens: int = 4000,
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
        max_output_tokens: int = 4000,
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.gemini_client:
            return [TextContent(type="text", text="Gemini client not configured. Please set GEMINI_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("gemini", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            result_text = None
            max_retries = 3
            
            for retry_attempt in range(max_retries):
                try:
                    if USE_NEW_SDK:
                        result_text = await self._call_gemini_new_sdk(
                            conversation_key, prompt, model, temperature, max_output_tokens
                        )
                    else:
                        result_text = await self._call_gemini_old_sdk(
                            conversation_key, prompt, model, temperature, max_output_tokens
                        )
                    
                    # Validate response
                    if result_text and len(result_text.strip()) > 0:
                        break  # Success!
                    else:
                        logger.warning(f"Gemini returned empty response on attempt {retry_attempt + 1}")
                        if retry_attempt < max_retries - 1:
                            await asyncio.sleep(2 ** retry_attempt)  # Exponential backoff
                        
                except Exception as retry_error:
                    logger.warning(f"Gemini attempt {retry_attempt + 1} failed: {str(retry_error)}")
                    if retry_attempt < max_retries - 1:
                        await asyncio.sleep(2 ** retry_attempt)  # Exponential backoff
                    else:
                        raise retry_error
            
            # Final validation
            if not result_text or len(result_text.strip()) == 0:
                # Try with simplified prompt for conversation issues
                if len(self.conversation_histories[conversation_key]) > 0:
                    logger.info(f"Trying Gemini {model} with reset conversation due to empty response")
                    self.conversation_histories[conversation_key] = []  # Reset conversation
                    
                    if USE_NEW_SDK:
                        result_text = await self._call_gemini_new_sdk(
                            conversation_key, prompt, model, temperature, max_output_tokens
                        )
                    else:
                        result_text = await self._call_gemini_old_sdk(
                            conversation_key, prompt, model, temperature, max_output_tokens
                        )
                
                if not result_text or len(result_text.strip()) == 0:
                    error_msg = f"Gemini {model} returned empty response after {max_retries} attempts.\n\n"
                    error_msg += "**Possible issues:**\n"
                    error_msg += "• Long conversation history may be confusing the model\n"
                    error_msg += "• The prompt may trigger content safety filters\n"
                    error_msg += "• Model may be experiencing temporary issues\n\n"
                    error_msg += "**Try:**\n"
                    error_msg += "• Reset conversation with `reset_conversation: true`\n"
                    error_msg += "• Rephrase your prompt\n"
                    error_msg += "• Try a different Gemini model\n"
                    error_msg += "• Use a shorter, more direct question"
                    return [TextContent(type="text", text=error_msg)]
            
            # Clean up response
            result_text = result_text.strip()
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", result_text)
            
            result = f"**Gemini {model} Opinion:**\n\n{result_text}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return [TextContent(type="text", text=f"Gemini API Error: {str(e)}")]
    
    async def _call_gemini_new_sdk(self, conversation_key: str, prompt: str, model: str, temperature: float, max_output_tokens: int) -> str:
        """Call Gemini using the new SDK with improved error handling"""
        history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
        
        # Enhanced generation config
        config = genai_types.GenerateContentConfig(
            temperature=min(max(temperature, 0.0), 2.0),  # Clamp temperature
            max_output_tokens=max_output_tokens,
            system_instruction=self.collaborative_system_prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        )
        
        try:
            if history:
                # Use chat with history
                chat = self.gemini_client.chats.create(
                    model=model,
                    config=config,
                    history=history
                )
                response = chat.send_message(current_prompt)
            else:
                # Generate content without history
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=[{"parts": [{"text": current_prompt}]}],
                    config=config
                )
            
            # Extract text with validation
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        return candidate.content.parts[0].text
            
            return ""
            
        except Exception as e:
            logger.error(f"Gemini new SDK error: {str(e)}")
            raise e
    
    async def _call_gemini_old_sdk(self, conversation_key: str, prompt: str, model: str, temperature: float, max_output_tokens: int) -> str:
        """Call Gemini using the old SDK with improved error handling"""
        model_obj = self.gemini_client.GenerativeModel(
            model,
            system_instruction=self.collaborative_system_prompt,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        )
        
        history, current_prompt = self._get_gemini_history_and_prompt(conversation_key, prompt)
        
        generation_config = {
            "temperature": min(max(temperature, 0.0), 2.0),  # Clamp temperature
            "max_output_tokens": max_output_tokens,
            "top_p": 0.95,
            "top_k": 40
        }
        
        try:
            if history:
                # Use chat with history
                chat = model_obj.start_chat(history=history)
                response = chat.send_message(
                    current_prompt,
                    generation_config=generation_config
                )
            else:
                # Generate content without history
                response = model_obj.generate_content(
                    current_prompt,
                    generation_config=generation_config
                )
            
            # Extract text with validation
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        return candidate.content.parts[0].text
            
            return ""
            
        except Exception as e:
            logger.error(f"Gemini old SDK error: {str(e)}")
            raise e
    
    async def _get_grok_opinion(
        self,
        prompt: str,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        reasoning_effort: str = "low",
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
            
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add reasoning_effort for thinking models (grok-3-mini and grok-3-thinking)
            if model in ["grok-3-mini", "grok-3-thinking"]:
                request_params["reasoning_effort"] = reasoning_effort
            
            response = self.grok_client.chat.completions.create(**request_params)
            
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
        model: str = "claude-4-sonnet-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4000,
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
    
    async def _get_huggingface_opinion(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.huggingface_api_key:
            return [TextContent(type="text", text="HuggingFace client not configured. Please set HUGGINGFACE_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("huggingface", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # HuggingFace Inference API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {self.huggingface_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "second-opinion-mcp/2.0"
            }
            
            # Build conversation context for instruction models
            conversation_context = ""
            chat_history = []
            
            # For chat format models, build proper chat history
            if "instruct" in model.lower() or "chat" in model.lower():
                for msg in self.conversation_histories[conversation_key]:
                    chat_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                chat_history.append({"role": "user", "content": prompt})
            else:
                # For other models, build text context
                for msg in self.conversation_histories[conversation_key]:
                    if msg["role"] == "user":
                        conversation_context += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        conversation_context += f"Assistant: {msg['content']}\n"
            
            # Determine the best prompt format based on model type
            if "instruct" in model.lower() or "chat" in model.lower():
                # Try chat format first for instruction models
                if "llama" in model.lower() and chat_history:
                    # Llama format
                    full_prompt = ""
                    for msg in chat_history:
                        if msg["role"] == "user":
                            full_prompt += f"### Human: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            full_prompt += f"### Assistant: {msg['content']}\n"
                    if not full_prompt.endswith("### Assistant: "):
                        full_prompt += "### Assistant: "
                elif "mistral" in model.lower() and chat_history:
                    # Mistral format
                    full_prompt = ""
                    for msg in chat_history:
                        if msg["role"] == "user":
                            full_prompt += f"[INST] {msg['content']} [/INST]"
                        elif msg["role"] == "assistant":
                            full_prompt += f" {msg['content']}</s>"
                else:
                    # Generic instruction format
                    if conversation_context:
                        full_prompt = f"{conversation_context}User: {prompt}\nAssistant:"
                    else:
                        full_prompt = f"User: {prompt}\nAssistant:"
            else:
                # For base models, use simple continuation
                if conversation_context:
                    full_prompt = f"{conversation_context}{prompt}"
                else:
                    full_prompt = prompt
            
            # Enhanced payload formats with better error handling
            payloads_to_try = [
                # Format 1: Full featured format with better parameters
                {
                    "inputs": full_prompt,
                    "parameters": {
                        "temperature": min(max(temperature, 0.1), 1.0),  # Clamp temperature
                        "max_new_tokens": min(max_tokens, 2048),
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": 0.95,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                        "stop_sequences": ["User:", "Human:", "### Human:", "[INST]", "\n\n\n"],
                        "pad_token_id": 50256,  # Common padding token
                        "eos_token_id": 50256   # Common end-of-sequence token
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                },
                # Format 2: Simplified format with basic parameters
                {
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": min(max_tokens, 1024),
                        "temperature": min(max(temperature, 0.1), 1.0),
                        "return_full_text": False,
                        "do_sample": True
                    },
                    "options": {
                        "wait_for_model": True
                    }
                },
                # Format 3: Minimal format for compatibility
                {
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": min(max_tokens, 512),
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True
                    }
                },
                # Format 4: Ultra-minimal for problematic models
                {
                    "inputs": full_prompt
                }
            ]
            
            response_content = None
            last_error = None
            retry_count = 0
            max_retries = 3
            
            # Enhanced retry logic with exponential backoff
            for payload_idx, payload in enumerate(payloads_to_try):
                for retry in range(max_retries):
                    try:
                        # Progressive timeout increase
                        timeout = 30 + (retry * 15)
                        
                        response = requests.post(
                            api_url, 
                            headers=headers, 
                            json=payload, 
                            timeout=timeout
                        )
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            
                            # Enhanced response parsing
                            response_content = self._extract_huggingface_response(result_data, full_prompt)
                            
                            if response_content and len(response_content.strip()) > 0:
                                break  # Success! Exit both loops
                                
                        elif response.status_code == 503:
                            # Model loading - wait and retry
                            wait_time = min(15 + (retry * 10), 60)  # Progressive wait
                            logger.info(f"Model {model} loading, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            last_error = f"Model {model} is loading. Retrying..."
                            continue
                            
                        elif response.status_code == 429:
                            # Rate limited - wait longer
                            wait_time = min(30 + (retry * 20), 120)
                            logger.info(f"Rate limited for {model}, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            last_error = "Rate limited. Retrying..."
                            continue
                            
                        elif response.status_code == 400:
                            # Bad request - try next payload format
                            last_error = f"Bad request (format {payload_idx+1}): {response.text}"
                            break  # Try next payload format
                            
                        else:
                            last_error = f"HTTP {response.status_code}: {response.text}"
                            
                    except requests.exceptions.Timeout:
                        last_error = f"Timeout after {timeout}s for model {model}"
                        logger.warning(f"Request timeout on attempt {retry+1} for {model}")
                        continue
                        
                    except requests.exceptions.ConnectionError:
                        last_error = f"Connection error for model {model}"
                        await asyncio.sleep(5)  # Brief wait for connection issues
                        continue
                        
                    except Exception as e:
                        last_error = f"Request error: {str(e)}"
                        continue
                
                # If we got a valid response, break out of payload loop
                if response_content and len(response_content.strip()) > 0:
                    break
            
            # Final response validation and error handling
            if not response_content or len(response_content.strip()) == 0:
                return await self._handle_huggingface_error(model, last_error)
            
            # Clean and validate response
            response_content = self._clean_huggingface_response(response_content, full_prompt)
            
            if len(response_content.strip()) < 3:  # Too short response
                return [TextContent(type="text", text=f"HuggingFace {model} returned an unexpectedly short response. The model may not be suitable for this task. Try a different model or rephrase your prompt.")]
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**HuggingFace {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            logger.error(f"HuggingFace API error for {model}: {str(e)}")
            return [TextContent(type="text", text=f"HuggingFace API Error: {str(e)}")]
    
    def _extract_huggingface_response(self, result_data, full_prompt: str) -> str:
        """Extract and clean response content from HuggingFace API response"""
        response_content = None
        
        # Handle different response formats
        if isinstance(result_data, list) and len(result_data) > 0:
            item = result_data[0]
            if isinstance(item, dict):
                response_content = item.get("generated_text") or item.get("text") or str(item)
            else:
                response_content = str(item)
        elif isinstance(result_data, dict):
            response_content = result_data.get("generated_text") or result_data.get("text") or str(result_data)
        else:
            response_content = str(result_data)
        
        return response_content if response_content else ""
    
    def _clean_huggingface_response(self, response_content: str, full_prompt: str) -> str:
        """Clean and format HuggingFace response content"""
        if not response_content:
            return ""
        
        # Remove the input prompt if it's included in the response
        if full_prompt in response_content:
            response_content = response_content.replace(full_prompt, "").strip()
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            "Assistant:", "AI:", "Bot:", "Response:", "Answer:", 
            "### Assistant:", "### Response:", "[/INST]", "</s>"
        ]
        
        for prefix in prefixes_to_remove:
            if response_content.startswith(prefix):
                response_content = response_content[len(prefix):].strip()
        
        # Remove trailing artifacts
        suffixes_to_remove = ["</s>", "<|endoftext|>", "<|end|>", "###"]
        for suffix in suffixes_to_remove:
            if response_content.endswith(suffix):
                response_content = response_content[:-len(suffix)].strip()
        
        # Clean up excessive whitespace and newlines
        lines = response_content.split('\n')
        cleaned_lines = []
        consecutive_empty = 0
        
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty <= 1:  # Allow max 1 consecutive empty line
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    async def _handle_huggingface_error(self, model: str, last_error: str) -> Sequence[TextContent]:
        """Handle HuggingFace errors with helpful suggestions"""
        # Updated model suggestions with more recent and reliable models
        working_models = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "microsoft/DialoGPT-large",
            "HuggingFaceH4/zephyr-7b-beta",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "google/flan-t5-large",
            "Qwen/Qwen2.5-7B-Instruct",
            "microsoft/phi-3-medium-4k-instruct"
        ]
        
        # Check if it's a model-not-found error
        if "404" in str(last_error) or "not found" in str(last_error).lower():
            error_msg = f"❌ **Model '{model}' not found on HuggingFace.**\n\n"
            error_msg += "**🔍 Troubleshooting:**\n"
            error_msg += "• Check the model name spelling and format (e.g., 'username/model-name')\n"
            error_msg += "• Verify the model exists at huggingface.co/models\n"
            error_msg += "• Ensure the model supports text generation\n\n"
        elif "loading" in str(last_error).lower():
            error_msg = f"⏳ **Model '{model}' is currently loading.**\n\n"
            error_msg += "**💡 What this means:**\n"
            error_msg += "• HuggingFace models 'sleep' when unused and need time to wake up\n"
            error_msg += "• This usually takes 30-60 seconds for first request\n"
            error_msg += "• Try again in a minute!\n\n"
        else:
            error_msg = f"❌ **HuggingFace Error with '{model}':** {last_error}\n\n"
        
        error_msg += "**🚀 Try these reliable models instead:**\n"
        for suggestion in working_models:
            error_msg += f"• `{suggestion}`\n"
        
        error_msg += "\n**💡 Pro Tips:**\n"
        error_msg += "• Smaller models (7B) load faster than larger ones (70B)\n"
        error_msg += "• 'Instruct' models work better for conversations\n"
        error_msg += "• Popular models are usually faster to load\n"
        error_msg += "• If a model fails, wait 1-2 minutes before retrying"
        
        return [TextContent(type="text", text=error_msg)]
    
    async def _get_mistral_opinion(
        self,
        prompt: str,
        model: str = "mistral-small-latest",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.mistral_client:
            return [TextContent(type="text", text="Mistral AI client not configured. Please set MISTRAL_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("mistral", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.mistral_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Mistral {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Mistral API Error: {str(e)}")]
    
    async def _get_together_opinion(
        self,
        prompt: str,
        model: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.together_client:
            return [TextContent(type="text", text="Together AI client not configured. Please set TOGETHER_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("together", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.together_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Together AI {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Together AI Error: {str(e)}")]
    
    async def _get_cohere_opinion(
        self,
        prompt: str,
        model: str = "command-r-plus",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.cohere_client:
            return [TextContent(type="text", text="Cohere client not configured. Please set COHERE_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("cohere", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build chat history for Cohere format
            chat_history = []
            for msg in self.conversation_histories[conversation_key]:
                if msg["role"] == "user":
                    chat_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg["content"]})
            
            # Make the API call
            response = self.cohere_client.chat(
                model=model,
                message=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=chat_history,
                preamble=self.collaborative_system_prompt
            )
            
            response_content = response.text
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Cohere {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Cohere API Error: {str(e)}")]
    
    async def _get_groq_fast_opinion(
        self,
        prompt: str,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.groq_client_fast:
            return [TextContent(type="text", text="Groq Fast client not configured. Please set GROQ_FAST_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("groq_fast", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.groq_client_fast.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Groq Fast {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Groq Fast API Error: {str(e)}")]
    
    async def _get_perplexity_opinion(
        self,
        prompt: str,
        model: str = "llama-3.1-sonar-large-128k-online",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.perplexity_client:
            return [TextContent(type="text", text="Perplexity AI client not configured. Please set PERPLEXITY_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("perplexity", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.perplexity_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**Perplexity {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Perplexity API Error: {str(e)}")]
    
    async def _get_deepseek_opinion(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.deepseek_client:
            return [TextContent(type="text", text="DeepSeek client not configured. Please set DEEPSEEK_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("deepseek", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**DeepSeek {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"DeepSeek API Error: {str(e)}")]
    
    async def _get_openrouter_opinion(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: str = "",
        reset_conversation: bool = False
    ) -> Sequence[TextContent]:
        if not self.openrouter_client:
            return [TextContent(type="text", text="OpenRouter client not configured. Please set OPENROUTER_API_KEY environment variable.")]
        
        try:
            conversation_key = self._get_conversation_key("openrouter", model)
            
            # Reset conversation if requested
            if reset_conversation:
                self.conversation_histories[conversation_key] = []
            
            # Build messages with conversation history
            messages = self._get_openai_messages(conversation_key, prompt, system_prompt)
            
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_key, "user", prompt)
            self._add_to_conversation_history(conversation_key, "assistant", response_content)
            
            result = f"**OpenRouter {model} Opinion:**\n\n{response_content}"
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"OpenRouter API Error: {str(e)}")]
    
    async def _list_openrouter_models(
        self,
        filter_by: str = ""
    ) -> Sequence[TextContent]:
        if not self.openrouter_client:
            return [TextContent(type="text", text="OpenRouter client not configured. Please set OPENROUTER_API_KEY environment variable.")]
        
        try:
            # Use the models endpoint to get available models
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {self.openrouter_client.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code != 200:
                return [TextContent(type="text", text=f"Failed to fetch OpenRouter models: HTTP {response.status_code}")]
            
            models_data = response.json()
            models = models_data.get("data", [])
            
            if not models:
                return [TextContent(type="text", text="No models found in OpenRouter response.")]
            
            # Filter models if requested
            if filter_by:
                filtered_models = []
                for model in models:
                    model_id = model.get("id", "")
                    if filter_by.lower() in model_id.lower():
                        filtered_models.append(model)
                models = filtered_models
            
            # Format the response
            result = f"## OpenRouter Models ({len(models)} available)\n\n"
            
            if filter_by:
                result += f"**Filter:** {filter_by}\n\n"
            
            # Group models by provider
            providers = {}
            for model in models:
                model_id = model.get("id", "")
                if "/" in model_id:
                    provider = model_id.split("/")[0]
                    if provider not in providers:
                        providers[provider] = []
                    providers[provider].append(model)
                else:
                    if "other" not in providers:
                        providers["other"] = []
                    providers["other"].append(model)
            
            # Display models by provider
            for provider, provider_models in sorted(providers.items()):
                result += f"### {provider.title()}\n"
                for model in provider_models[:10]:  # Limit to first 10 per provider to avoid huge responses
                    model_id = model.get("id", "Unknown")
                    model_name = model.get("name", model_id)
                    pricing = model.get("pricing", {})
                    
                    # Format pricing info if available
                    pricing_info = ""
                    if pricing:
                        prompt_price = pricing.get("prompt", "")
                        completion_price = pricing.get("completion", "")
                        if prompt_price and completion_price:
                            pricing_info = f" (${prompt_price}/1M tokens input, ${completion_price}/1M tokens output)"
                    
                    result += f"- **{model_id}** - {model_name}{pricing_info}\n"
                
                if len(provider_models) > 10:
                    result += f"- ... and {len(provider_models) - 10} more {provider} models\n"
                result += "\n"
            
            # Add usage tip
            result += "**Usage:** Use the exact model ID (e.g., 'anthropic/claude-3-5-sonnet') with the get_openrouter_opinion tool.\n"
            result += "**Popular models:** anthropic/claude-3-5-sonnet, openai/gpt-4, meta-llama/llama-3.1-405b-instruct, google/gemini-pro-1.5\n"
            
            return [TextContent(type="text", text=result)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error fetching OpenRouter models: {str(e)}")]
    
    async def _group_discussion(
        self,
        topic: str,
        participants: List[Dict[str, str]] = None,
        rounds: int = 2,
        temperature: float = 0.7
    ) -> Sequence[TextContent]:
        """Start a group discussion between multiple AI models"""
        
        if participants is None:
            participants = [
                {"platform": "openai", "model": "gpt-4.1"},
                {"platform": "claude", "model": "claude-4-sonnet-20250514"},
                {"platform": "gemini", "model": "gemini-2.0-flash-001"}
            ]
        
        # Validate participants and filter to only available ones
        valid_participants = []
        for participant in participants:
            platform = participant["platform"]
            model = participant["model"]
            
            # Force HuggingFace to use Llama 3.3 70B Instruct for group discussions
            if platform == "huggingface":
                model = "meta-llama/Llama-3.3-70B-Instruct"
                participant = {"platform": platform, "model": model}
            
            if platform == "openai" and self.openai_client:
                valid_participants.append(participant)
            elif platform == "gemini" and self.gemini_client:
                valid_participants.append(participant)
            elif platform == "grok" and self.grok_client:
                valid_participants.append(participant)
            elif platform == "claude" and self.claude_client:
                valid_participants.append(participant)
            elif platform == "huggingface" and self.huggingface_api_key:
                valid_participants.append(participant)
            elif platform == "deepseek" and self.deepseek_client:
                valid_participants.append(participant)
            elif platform == "openrouter" and self.openrouter_client:
                valid_participants.append(participant)
        
        if len(valid_participants) < 2:
            return [TextContent(type="text", text="Need at least 2 valid participants for group discussion.")]
        
        # Track discussion history
        discussion_history = []
        
        results = []
        results.append(f"## 🎭 AI Group Discussion: {topic}\n")
        participant_list = ', '.join([f"{p['platform']}/{p['model']}" for p in valid_participants])
        results.append(f"**Participants:** {participant_list}\n")
        
        for round_num in range(rounds):
            results.append(f"\n### Round {round_num + 1}\n")
            
            for i, participant in enumerate(valid_participants):
                platform = participant["platform"]
                model = participant["model"]
                
                # Build context for this participant
                context_prompt = f"Topic for discussion: {topic}\n\n"
                
                if discussion_history:
                    context_prompt += "Previous discussion:\n"
                    for entry in discussion_history:
                        context_prompt += f"- **{entry['participant']}**: {entry['response']}\n"
                    context_prompt += "\n"
                
                if round_num == 0:
                    context_prompt += f"You are participating in round {round_num + 1} of {rounds}. Please share your thoughts on this topic (keep it concise, around 2-3 sentences)."
                else:
                    context_prompt += f"This is round {round_num + 1} of {rounds}. Please respond to the previous discussion and add your perspective (keep it concise, around 2-3 sentences)."
                
                # Get response from this participant
                try:
                    if platform == "openai":
                        response = await self._get_openai_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    elif platform == "gemini":
                        response = await self._get_gemini_opinion(
                            context_prompt, model=model, temperature=temperature, max_output_tokens=500
                        )
                    elif platform == "grok":
                        response = await self._get_grok_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    elif platform == "claude":
                        response = await self._get_claude_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    elif platform == "huggingface":
                        response = await self._get_huggingface_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    elif platform == "deepseek":
                        response = await self._get_deepseek_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    elif platform == "openrouter":
                        response = await self._get_openrouter_opinion(
                            context_prompt, model=model, temperature=temperature, max_tokens=500
                        )
                    
                    if response and len(response) > 0:
                        participant_name = f"{platform.title()}-{model}"
                        response_text = response[0].text
                        
                        # Clean up the response (remove the header we added)
                        if f"**{platform.title()}" in response_text:
                            response_text = response_text.split("Opinion:**\n\n", 1)[-1]
                        
                        # Add to discussion history
                        discussion_history.append({
                            "participant": participant_name,
                            "response": response_text,
                            "round": round_num + 1
                        })
                        
                        results.append(f"**{participant_name}:** {response_text}\n")
                    else:
                        results.append(f"**{platform.title()}-{model}:** ❌ No response\n")
                        
                except Exception as e:
                    results.append(f"**{platform.title()}-{model}:** ❌ Error: {str(e)}\n")
        
        results.append(f"\n---\n*Discussion completed with {len(valid_participants)} participants over {rounds} rounds*")
        
        return [TextContent(type="text", text="\n".join(results))]
    
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
                    token_param: 2000
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
                        max_output_tokens=2000,
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
                                "max_output_tokens": 2000
                            }
                        )
                    else:
                        response = model_obj.generate_content(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 2000
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
                
                request_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
                
                # Add reasoning_effort for thinking models
                if model in ["grok-3-mini", "grok-3-thinking"]:
                    request_params["reasoning_effort"] = "low"
                
                response = self.grok_client.chat.completions.create(**request_params)
                
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
            models = ["claude-4-opus-20250514", "claude-4-sonnet-20250514"]
        
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
                    max_tokens=2000,
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
        claude_model: str = "claude-4-sonnet-20250514",
        huggingface_model: str = "meta-llama/Llama-3.3-70B-Instruct",
        deepseek_model: str = "deepseek-chat",
        openrouter_model: str = "anthropic/claude-3-5-sonnet",
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
                    token_param: 2000
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
                        max_output_tokens=2000,
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
                                "max_output_tokens": 2000
                            }
                        )
                    else:
                        response = model_obj.generate_content(
                            current_prompt,
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": 2000
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
                
                request_params = {
                    "model": grok_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                }
                
                # Add reasoning_effort for thinking models
                if grok_model in ["grok-3-mini", "grok-3-thinking"]:
                    request_params["reasoning_effort"] = "low"
                
                grok_response = self.grok_client.chat.completions.create(**request_params)
                
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
                    max_tokens=2000,
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
        
        # Get HuggingFace opinion
        if self.huggingface_api_key:
            try:
                hf_response = await self._get_huggingface_opinion(
                    prompt, huggingface_model, temperature, 2000
                )
                response_content = hf_response[0].text
                
                # Clean up the response
                if f"**HuggingFace" in response_content:
                    response_content = response_content.split("Opinion:**\n\n", 1)[-1]
                
                results.append(f"### HuggingFace ({huggingface_model})\n{response_content}\n")
            except Exception as e:
                results.append(f"### HuggingFace ({huggingface_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### HuggingFace\n❌ Not configured\n")
        
        # Get DeepSeek opinion
        if self.deepseek_client:
            try:
                ds_response = await self._get_deepseek_opinion(
                    prompt, deepseek_model, temperature, 2000
                )
                response_content = ds_response[0].text
                
                # Clean up the response
                if f"**DeepSeek" in response_content:
                    response_content = response_content.split("Opinion:**\n\n", 1)[-1]
                
                results.append(f"### DeepSeek ({deepseek_model})\n{response_content}\n")
            except Exception as e:
                results.append(f"### DeepSeek ({deepseek_model})\n❌ Error: {str(e)}\n")
        else:
            results.append("### DeepSeek\n❌ Not configured\n")
        
        return [TextContent(type="text", text="\n".join(results))]
    
        # Get OpenRouter opinion
        if self.openrouter_client:
            try:
                or_response = await self._get_openrouter_opinion(
                    prompt, openrouter_model, temperature, 2000
                )
                response_content = or_response[0].text
                
                # Clean up the response
                if f"**OpenRouter" in response_content:
                    response_content = response_content.split("Opinion:**\\n\\n", 1)[-1]
                
                results.append(f"### OpenRouter ({openrouter_model})\\n{response_content}\\n")
            except Exception as e:
                results.append(f"### OpenRouter ({openrouter_model})\\n❌ Error: {str(e)}\\n")
        else:
            results.append("### OpenRouter\\n❌ Not configured\\n")
        
        return [TextContent(type="text", text="\\n".join(results))]
    
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
    if not os.getenv("HUGGINGFACE_API_KEY"):
        required_vars.append("HUGGINGFACE_API_KEY")
    if not os.getenv("DEEPSEEK_API_KEY"):
        required_vars.append("DEEPSEEK_API_KEY")
    if not os.getenv("OPENROUTER_API_KEY"):
        required_vars.append("OPENROUTER_API_KEY")
    if not os.getenv("MISTRAL_API_KEY"):
        required_vars.append("MISTRAL_API_KEY")
    if not os.getenv("TOGETHER_API_KEY"):
        required_vars.append("TOGETHER_API_KEY")
    if not os.getenv("COHERE_API_KEY"):
        required_vars.append("COHERE_API_KEY")
    if not os.getenv("GROQ_FAST_API_KEY") and not os.getenv("GROQ_API_KEY"):
        required_vars.append("GROQ_FAST_API_KEY")
    if not os.getenv("PERPLEXITY_API_KEY"):
        required_vars.append("PERPLEXITY_API_KEY")
    
    if required_vars:
        print("⚠️  Warning: Missing environment variables:", file=sys.stderr)
        for var in required_vars:
            print(f"   - {var}", file=sys.stderr)
        print("\nSome features will be disabled. Set these variables to enable full functionality.", file=sys.stderr)
    
    # Create server instance
    server_instance = SecondOpinionServer()
    
    # Run the MCP server
    async def run_server():
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.app.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="second-opinion",
                    server_version="3.0.0",
                    capabilities=server_instance.app.get_capabilities()
                )
            )
    
    # Run the server
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
