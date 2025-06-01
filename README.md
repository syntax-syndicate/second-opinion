# Second Opinion MCP

Get instant second opinions from multiple AI models (OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek & OpenRouter) directly within Claude conversations.

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:

* **Coding problems** - Compare approaches across models
* **Creative writing** - Get diverse style feedback  
* **Problem solving** - Validate logic and reasoning
* **Cross-model analysis** - See how different AIs tackle the same task
* **Group discussions** - Host AI debates with multiple models (now with 2+ models!)
* **Custom model access** - Use any HuggingFace model via Inference API
* **300+ OpenRouter models** - Access any model through OpenRouter's unified API

## ✨ New Features

### 🌐 OpenRouter Integration (NEW!)
Access 300+ AI models through OpenRouter's unified API including:
- `anthropic/claude-3-5-sonnet` - Latest Claude 3.5 Sonnet
- `openai/gpt-4` - GPT-4 via OpenRouter
- `meta-llama/llama-3.1-405b-instruct` - Massive 405B Llama model
- `google/gemini-pro-1.5` - Gemini Pro 1.5
- *Any model available on OpenRouter*

### 🎭 Improved Group Discussions
- **Reduced requirements**: Now works with just 2+ models (previously required 3+)
- **OpenRouter support**: Include any OpenRouter model in discussions
- **HuggingFace standardization**: Automatically uses Llama 3.3 70B for group discussions
- **Better participant filtering**: Only uses available models automatically

### 🤖 HuggingFace Integration
Access any of the 800,000+ models on HuggingFace Hub via their Inference API. Use cutting-edge open source models like:
- `meta-llama/Llama-3.3-70B-Instruct`
- `microsoft/DialoGPT-large`
- `Qwen/Qwen2.5-72B-Instruct`

### 🧠 DeepSeek Models
Get opinions from DeepSeek's powerful reasoning models:
- `deepseek-chat` (DeepSeek-V3) - Fast and efficient
- `deepseek-reasoner` (DeepSeek-R1) - Advanced reasoning

### 🤔 Grok 3 Thinking
Access xAI's latest thinking models with enhanced reasoning:
- `grok-3` - Latest flagship model
- `grok-3-thinking` - Step-by-step reasoning model
- `grok-3-mini` - Lightweight thinking model with `reasoning_effort` control

### 🎭 Group Discussions
Start multi-AI discussions where models can see and respond to each other's input:
```
> "Start a group discussion about the future of AI with GPT-4.1, Claude-4, and Gemini"
```

### 🔧 Enhanced Performance
- **Longer responses**: Increased max_tokens (4000 default) to prevent cut-off responses
- **Better error handling**: More robust API interactions
- **Conversation persistence**: Enhanced memory management

## 📋 Requirements

* Python 3.8+
* Claude Desktop
* API keys for any combination of: OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, OpenRouter

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ProCreations-Official/second-opinion.git
   cd second-opinion
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Get API Keys**

   * **OpenAI**: Get your key at [platform.openai.com](https://platform.openai.com/api-keys)
   * **Gemini**: Get your key at [aistudio.google.com](https://aistudio.google.com/app/apikey)
   * **Grok**: Get your key at [x.ai](https://x.ai/api)
   * **Claude**: Get your key at [anthropic.com](https://console.anthropic.com/settings/keys)
   * **HuggingFace**: Get your key at [huggingface.co](https://huggingface.co/settings/tokens)
   * **DeepSeek**: Get your key at [deepseek.com](https://api-docs.deepseek.com/)
   * **OpenRouter**: Get your key at [openrouter.ai](https://openrouter.ai/keys)

4. **Configure Claude Desktop**

   Add this to your Claude Desktop MCP configuration:

   ```json
   {
     "mcpServers": {
       "second-opinion": {
         "command": "python3",
         "args": ["/path/to/your/main.py"],
         "env": {
           "OPENAI_API_KEY": "your_openai_key_here",
           "GEMINI_API_KEY": "your_gemini_key_here",
           "GROK_API_KEY": "your_grok_key_here",
           "CLAUDE_API_KEY": "your_claude_key_here",
           "HUGGINGFACE_API_KEY": "your_huggingface_key_here",
           "DEEPSEEK_API_KEY": "your_deepseek_key_here",
           "OPENROUTER_API_KEY": "your_openrouter_key_here"
         }
       }
     }
   }
   ```

   **Note**: You only need to add API keys for the services you want to use. Missing keys will simply disable those specific features.

5. **Restart Claude Desktop**

## 🎯 Available Models

**OpenAI**
* `o4-mini` - Fast reasoning model
* `gpt-4.1` - Latest flagship model  
* `gpt-4o` - Multimodal powerhouse

**Gemini**
* `gemini-2.0-flash-001` - Fast and efficient
* `gemini-2.5-flash-preview-05-20` - Advanced reasoning

**Grok**
* `grok-3` - Latest flagship model
* `grok-3-thinking` - Step-by-step reasoning
* `grok-3-mini` - Lightweight thinking model
* `grok-2` - Robust and reliable
* `grok-beta` - Experimental features

**Claude**
* `claude-4-opus-20250514` - Most advanced Claude model
* `claude-4-sonnet-20250514` - Versatile model for general tasks
* `claude-3-7-sonnet-20250224` - Stable and reliable
* `claude-3-5-sonnet-20241022` - Efficient, lighter model

**HuggingFace** (800,000+ models available)
* `meta-llama/Llama-3.3-70B-Instruct` - Meta's latest Llama model
* `microsoft/DialoGPT-large` - Conversational AI
* `microsoft/phi-4` - Microsoft's efficient reasoning model
* `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` - DeepSeek reasoning model 
* `google/gemma-3-27b-it` - Google's Gemma instruction-tuned model
* `Qwen/Qwen3-235B-A22B` - Alibaba's massive 235B parameter model
* `mistralai/Mistral-7B-Instruct-v0.3` - Mistral's instruction model
* *Any model on HuggingFace Hub that supports text generation*

**DeepSeek**
* `deepseek-chat` - DeepSeek-V3 for general tasks
* `deepseek-reasoner` - DeepSeek-R1 for advanced reasoning

**OpenRouter** (300+ models available)
* `anthropic/claude-3-5-sonnet` - Latest Claude 3.5 Sonnet
* `openai/gpt-4` - GPT-4 via OpenRouter  
* `meta-llama/llama-3.1-405b-instruct` - Massive 405B Llama model
* `google/gemini-pro-1.5` - Gemini Pro 1.5
* `mistralai/mistral-large-2407` - Mistral Large
* `qwen/qwen-2.5-72b-instruct` - Qwen 2.5 72B
* *Browse all 300+ models with the `list_openrouter_models` tool*

## 💡 Usage Examples

Once configured, ask Claude things like:

> "Get a second opinion from GPT-4.1 on this coding approach"

> "What would Grok-3-thinking think about this solution?"

> "Compare how Claude-4-opus and gemini-2.0-flash would solve this problem"

> "Get an opinion from meta-llama/Llama-3.3-70B-Instruct on HuggingFace"

> "What does DeepSeek-reasoner think about this math problem?"

> "Get an opinion from anthropic/claude-3-5-sonnet on OpenRouter"

> "List all available OpenRouter models from Anthropic"

> "Start a group discussion about AI ethics with GPT-4.1, Claude-4, and meta-llama/llama-3.1-405b-instruct on OpenRouter"

> "Start a group discussion about AI ethics with GPT-4.1, Claude-4, Gemini, and DeepSeek"

> "Cross-platform comparison of this algorithm across all available models"

## 🔧 Available Tools

### Single Model Opinions
* **`get_openai_opinion`** - Get opinion from any OpenAI model
* **`get_gemini_opinion`** - Get opinion from any Gemini model  
* **`get_grok_opinion`** - Get opinion from any Grok model (includes thinking models)
* **`get_claude_opinion`** - Get opinion from any Claude model
* **`get_huggingface_opinion`** - Get opinion from any HuggingFace model
* **`get_deepseek_opinion`** - Get opinion from DeepSeek models
* **`get_openrouter_opinion`** - Get opinion from any OpenRouter model (300+ available)
* **`list_openrouter_models`** - Browse all available OpenRouter models

### Model Comparisons  
* **`compare_openai_models`** - Compare multiple OpenAI models
* **`compare_gemini_models`** - Compare multiple Gemini models
* **`compare_grok_models`** - Compare multiple Grok models
* **`compare_claude_models`** - Compare multiple Claude models

### Cross-Platform Features
* **`cross_platform_comparison`** - Compare across OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek & OpenRouter
* **`group_discussion`** - Multi-round discussions between AI models with shared context (now supports 2+ models)

### Conversation Management
* **`list_conversation_histories`** - See active conversation threads
* **`clear_conversation_history`** - Reset conversation memory for specific models

## 🧠 Advanced Features

### Grok 3 Thinking Models
For deeper reasoning, use thinking models:
```
> "Get a Grok-3-thinking opinion on this complex math problem with high reasoning effort"
```

The `reasoning_effort` parameter controls thinking depth:
- `low` - Faster responses with basic reasoning
- `high` - Deeper analysis with step-by-step thinking

### Group Discussions
Create AI debates and collaborative problem-solving:
```
> "Start a group discussion about renewable energy solutions with 3 rounds between GPT-4.1, Claude-4, Gemini, and DeepSeek"
```

Each AI can see previous responses and build on the discussion.

### HuggingFace Model Access
Access cutting-edge open source models:
```
> "Get an opinion from microsoft/DialoGPT-large about chatbot design patterns"
```

Perfect for testing specialized models or comparing open source vs proprietary AI.

## 🔒 Security

Your API keys stay private on your machine. The MCP server only sends model responses to Claude, never your credentials.

## 🛟 Troubleshooting

**Import errors**: Ensure you've installed all dependencies with `pip install -r requirements.txt`

**API errors**: Check that your API keys are correct and active

**Server not connecting**: Verify the file path in your MCP configuration

**Cut-off responses**: The new version uses 4000 max_tokens by default to prevent truncation

**HuggingFace timeouts**: Some models may take time to load. Try again after a few moments.

**Model not available**: Check if the HuggingFace model supports text generation or chat completion

## 🚀 What's New in v2.0

- ✅ **HuggingFace Integration**: Access 800,000+ open source models
- ✅ **DeepSeek Support**: V3 and R1 reasoning models  
- ✅ **Grok 3 Thinking**: Advanced reasoning with controllable effort
- ✅ **Group Discussions**: Multi-AI collaborative conversations
- ✅ **Enhanced Responses**: 4x longer responses (4000 tokens default)
- ✅ **Better Error Handling**: More robust API interactions
- ✅ **Expanded Model Support**: 6 AI platforms, 800,000+ models

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

---

**Built for developers who want access to the entire AI ecosystem at their fingertips** 🧠✨
