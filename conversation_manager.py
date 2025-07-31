#!/usr/bin/env python3
"""
Conversation Manager for Second Opinion MCP Server
Handles conversation history, context management, and message formatting
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and context across different AI platforms"""
    
    def __init__(self):
        """Initialize the conversation manager"""
        # Conversation history storage
        # Format: {platform_model: [conversation_history]}
        self.conversation_histories = defaultdict(list)
        
        # Collaborative system prompt used across platforms
        self.collaborative_system_prompt = """You are an AI assistant working in a collaborative environment with Claude (an Anthropic AI) and other AI models to help users. Claude is coordinating this collaboration and has sent you this message as part of a multi-AI consultation. 

Your role is to provide your unique perspective and expertise to help answer the user's question. You're part of a team of AI assistants, each bringing different strengths and viewpoints. Be thoughtful, helpful, and concise in your response, as your input will be combined with responses from other AI models to give the user a comprehensive answer.

Remember that you're working together with Claude and other AIs to provide the best possible assistance to the user."""
        
        # Personality system prompts for the new feature
        self.personality_prompts = {
            "honest": """You are a brutally honest AI assistant. Tell the truth even when it's uncomfortable. Don't sugarcoat anything - be direct, frank, and straightforward. If something is a bad idea, say so clearly. If someone is wrong, correct them bluntly but constructively. Your honesty helps people make better decisions.""",
            
            "friend": """You are a supportive, caring friend. Be warm, affectionate, and emotionally supportive. Use friendly terms naturally, show genuine interest in the user's life and feelings. Offer comfort during difficult times and celebrate their successes. Be encouraging and positive when appropriate, but always maintain respect and boundaries.""",
            
            "coach": """You are a motivational life coach. Be energetic, encouraging, and focused on helping people achieve their goals. Push them to be their best selves, offer practical advice, and hold them accountable. Use motivational language and help them break down big challenges into manageable steps. Celebrate progress and keep them focused on success.""",
            
            "wise": """You are an ancient wise sage with deep understanding of life, philosophy, and human nature. Speak with wisdom gained through centuries of observation. Offer profound insights, philosophical perspectives, and timeless advice. Use metaphors and parables when appropriate. Help people see the bigger picture and understand deeper truths.""",
            
            "creative": """You are an incredibly creative and artistic AI. Think outside the box, offer innovative solutions, and approach problems from unique angles. Be imaginative, experimental, and unafraid to suggest unconventional ideas. Inspire creativity in others and help them see possibilities they might have missed. Make everything more interesting and original."""
        }
        
        logger.info("Conversation manager initialized with personality system prompts")
    
    def get_conversation_key(self, platform: str, model: str) -> str:
        """Generate a key for conversation history storage"""
        return f"{platform}_{model}"
    
    def add_to_conversation_history(self, key: str, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_histories[key].append({"role": role, "content": content})
        
        # Keep only last 10 exchanges (20 messages) to manage memory
        if len(self.conversation_histories[key]) > 20:
            self.conversation_histories[key] = self.conversation_histories[key][-20:]
        
        logger.debug(f"Added {role} message to conversation {key}, total messages: {len(self.conversation_histories[key])}")
    
    def get_openai_messages(self, key: str, prompt: str, system_prompt: str = None, personality: str = None) -> List[Dict]:
        """Build OpenAI messages array with conversation history"""
        messages = []
        
        # Determine system prompt
        if personality and personality in self.personality_prompts:
            final_system_prompt = self.personality_prompts[personality]
        elif system_prompt:
            final_system_prompt = system_prompt
        else:
            final_system_prompt = self.collaborative_system_prompt
        
        # Add system prompt
        messages.append({"role": "system", "content": final_system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_histories[key])
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def get_gemini_history_and_prompt(self, key: str, prompt: str, system_prompt: str = None, personality: str = None) -> tuple:
        """Build Gemini chat history and current prompt"""
        history = []
        
        # Convert conversation history to Gemini format
        for msg in self.conversation_histories[key]:
            if msg["role"] == "user":
                history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                history.append({"role": "model", "parts": [msg["content"]]})
        
        # Determine system prompt
        if personality and personality in self.personality_prompts:
            final_system_prompt = self.personality_prompts[personality]
        elif system_prompt:
            final_system_prompt = system_prompt
        else:
            final_system_prompt = self.collaborative_system_prompt
        
        # Combine system prompt with user prompt
        full_prompt = f"{final_system_prompt}\n\nUser: {prompt}"
        
        return history, full_prompt
    
    def get_claude_messages(self, key: str, prompt: str, system_prompt: str = None, personality: str = None) -> tuple:
        """Build Claude messages and system prompt"""
        messages = []
        
        # Add conversation history (Claude uses same format as OpenAI)
        messages.extend(self.conversation_histories[key])
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Determine system prompt
        if personality and personality in self.personality_prompts:
            final_system_prompt = self.personality_prompts[personality]
        elif system_prompt:
            final_system_prompt = system_prompt
        else:
            final_system_prompt = self.collaborative_system_prompt
        
        return messages, final_system_prompt
    
    def clear_conversation_history(self, platform: str = None, model: str = None) -> Dict[str, Any]:
        """Clear conversation history for specific platform/model or all"""
        if platform == "all":
            # Clear all conversations
            cleared_count = len(self.conversation_histories)
            self.conversation_histories.clear()
            logger.info(f"Cleared all conversation histories ({cleared_count} total)")
            return {"status": "success", "message": f"Cleared all conversation histories ({cleared_count} total)"}
        
        elif platform and model:
            # Clear specific conversation
            key = self.get_conversation_key(platform, model)
            if key in self.conversation_histories:
                del self.conversation_histories[key]
                logger.info(f"Cleared conversation history for {platform}/{model}")
                return {"status": "success", "message": f"Cleared conversation history for {platform}/{model}"}
            else:
                return {"status": "error", "message": f"No conversation history found for {platform}/{model}"}
        
        elif platform:
            # Clear all conversations for a platform
            keys_to_clear = [k for k in self.conversation_histories.keys() if k.startswith(f"{platform}_")]
            for key in keys_to_clear:
                del self.conversation_histories[key]
            logger.info(f"Cleared {len(keys_to_clear)} conversation histories for platform {platform}")
            return {"status": "success", "message": f"Cleared {len(keys_to_clear)} conversation histories for platform {platform}"}
        
        else:
            return {"status": "error", "message": "Must specify platform or 'all'"}
    
    def list_conversation_histories(self) -> Dict[str, Any]:
        """List all active conversation histories"""
        histories = {}
        
        for key, messages in self.conversation_histories.items():
            if messages:  # Only include non-empty histories
                platform_model = key.split("_", 1)
                if len(platform_model) == 2:
                    platform, model = platform_model
                    histories[key] = {
                        "platform": platform,
                        "model": model,
                        "message_count": len(messages),
                        "last_message": messages[-1]["content"][:100] + "..." if len(messages[-1]["content"]) > 100 else messages[-1]["content"]
                    }
        
        logger.info(f"Listed {len(histories)} active conversation histories")
        return {
            "total_conversations": len(histories),
            "conversations": histories
        }
    
    def get_conversation_context(self, key: str) -> List[Dict]:
        """Get the full conversation context for a given key"""
        return self.conversation_histories.get(key, []).copy()
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personality options"""
        return list(self.personality_prompts.keys())
    
    def get_personality_description(self, personality: str) -> str:
        """Get description of a specific personality"""
        descriptions = {
            "honest": "Brutally honest and direct - tells the truth even when uncomfortable",
            "friend": "Supportive friend - warm, caring, and emotionally supportive",
            "coach": "Motivational life coach - energetic, encouraging, goal-focused",
            "wise": "Ancient wise sage - philosophical, deep insights, timeless wisdom",
            "creative": "Highly creative and artistic - innovative, imaginative, unique perspectives"
        }
        return descriptions.get(personality, "Unknown personality")
    
    def reset_conversation(self, key: str):
        """Reset a specific conversation to empty state"""
        if key in self.conversation_histories:
            self.conversation_histories[key] = []
            logger.info(f"Reset conversation history for {key}")
        else:
            logger.warning(f"Attempted to reset non-existent conversation {key}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation usage"""
        total_conversations = len(self.conversation_histories)
        total_messages = sum(len(msgs) for msgs in self.conversation_histories.values())
        active_conversations = len([k for k, v in self.conversation_histories.items() if v])
        
        platform_stats = defaultdict(int)
        for key in self.conversation_histories.keys():
            platform = key.split("_")[0]
            platform_stats[platform] += 1
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages,
            "platform_breakdown": dict(platform_stats),
            "average_messages_per_conversation": total_messages / max(active_conversations, 1)
        }