# Second Opinion MCP

Get instant second opinions from multiple AI models (OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, Mistral, Together AI, Cohere, Groq Fast & Perplexity) directly within Claude conversations.

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:

* **Coding problems** - Compare approaches across models
* **Creative writing** - Get diverse style feedback  
* **Problem solving** - Validate logic and reasoning
* **Cross-model analysis** - See how different AIs tackle the same task
* **Group discussions** - Host AI debates with multiple models
* **Custom model access** - Use any HuggingFace model via Inference API

## ✨ Version 3.0 Features & Improvements

### 🚀 Major Bug Fixes
- **Fixed HuggingFace Models**: Completely rebuilt HuggingFace integration with advanced retry logic, better model format detection, and comprehensive error handling
- **Fixed Gemini Blank Responses**: Enhanced Gemini conversation handling to prevent empty responses in long chats with smart fallback and retry mechanisms
- **Improved Error Handling**: Better error messages with helpful suggestions for troubleshooting

### 🆕 New AI Platform Support
- **🤖 Mistral AI**: Direct access to Mistral's latest models including mistral-large-latest and codestral-latest
- **🔗 Together AI**: Access to 200+ open-source models with fast inference
- **🧠 Cohere**: Enterprise-grade language models with Command R+ and Command R
- **⚡ Groq Fast**: Ultra-fast inference API for lightning-quick responses
- **🔍 Perplexity AI**: Web-connected AI with real-time search capabilities

### 🤖 HuggingFace Integration (Enhanced)
Access any of the 800,000+ models on HuggingFace Hub via their Inference API with improved reliability:
- `meta-llama/Llama-3.1-8B-Instruct` - Fast and reliable
- `meta-llama/Llama-3.1-70B-Instruct` - Powerful reasoning
- `mistralai/Mistral-7B-Instruct-v0.3` - Efficient French-developed model
- `Qwen/Qwen2.5-7B-Instruct` - Alibaba's latest model

### 🧠 DeepSeek Models
Get opinions from DeepSeek's powerful reasoning models:
- `deepseek-chat` (DeepSeek-V3) - Fast and efficient
- `deepseek-reasoner` (DeepSeek-R1) - Advanced reasoning

### 🤔 Grok 3 Thinking
Access xAI's latest thinking models with enhanced reasoning:
- `grok-3` - Latest flagship model
- `grok-3-thinking` - Step-by-step reasoning model
- `grok-3-mini` - Lightweight thinking model with `reasoning_effort` control

### 🎭 Group Discussions (Enhanced)
Start multi-AI discussions where models can see and respond to each other's input:
```
> "Start a group discussion about the future of AI with GPT-4.1, Claude-4, Mistral, and Perplexity"
```

### 🔧 Enhanced Performance
- **Longer responses**: Increased max_tokens (4000 default) to prevent cut-off responses
- **Better error handling**: More robust API interactions with exponential backoff
- **Conversation persistence**: Enhanced memory management with better context handling
- **Smart retry logic**: Automatic retries with progressive delays for better reliability

## 📋 Requirements

* Python 3.8+
* Claude Desktop
* API keys for any combination of: OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, Mistral, Together AI, Cohere, Groq Fast, Perplexity

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
   * **Mistral**: Get your key at [console.mistral.ai](https://console.mistral.ai/)
   * **Together AI**: Get your key at [api.together.xyz](https://api.together.xyz/settings/api-keys)
   * **Cohere**: Get your key at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys)
   * **Groq**: Get your key at [console.groq.com](https://console.groq.com/keys)
   * **Perplexity**: Get your key at [perplexity.ai](https://www.perplexity.ai/settings/api)

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
           "MISTRAL_API_KEY": "your_mistral_key_here",
           "TOGETHER_API_KEY": "your_together_key_here",
           "COHERE_API_KEY": "your_cohere_key_here",
           "GROQ_FAST_API_KEY": "your_groq_key_here",
           "PERPLEXITY_API_KEY": "your_perplexity_key_here"
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
* `claude-4-opus-20250522` - Most advanced Claude model
* `claude-4-sonnet-20250522` - Versatile model for general tasks
* `claude-3-7-sonnet-20250224` - Stable and reliable
* `claude-3-5-sonnet-20241022` - Efficient, lighter model

**HuggingFace** (800,000+ models available - Enhanced with better reliability)
* `meta-llama/Llama-3.1-8B-Instruct` - Fast and reliable Meta model
* `meta-llama/Llama-3.1-70B-Instruct` - Powerful reasoning model
* `mistralai/Mistral-7B-Instruct-v0.3` - Efficient French-developed model
* `Qwen/Qwen2.5-7B-Instruct` - Alibaba's latest model
* *Any model on HuggingFace Hub that supports text generation*

**DeepSeek**
* `deepseek-chat` - DeepSeek-V3 for general tasks
* `deepseek-reasoner` - DeepSeek-R1 for advanced reasoning

**Mistral AI** (NEW)
* `mistral-large-latest` - Most powerful Mistral model
* `mistral-small-latest` - Fast and cost-effective
* `mistral-medium-latest` - Balanced performance
* `codestral-latest` - Specialized for code generation

**Together AI** (NEW - 200+ open-source models)
* `meta-llama/Llama-3.1-8B-Instruct-Turbo` - Fast Llama turbo
* `meta-llama/Llama-3.1-70B-Instruct-Turbo` - Powerful Llama turbo
* `meta-llama/Llama-3.1-405B-Instruct-Turbo` - Largest Llama model
* `mistralai/Mixtral-8x7B-Instruct-v0.1` - Mixture of experts
* `Qwen/Qwen2.5-72B-Instruct-Turbo` - Alibaba's fast model

**Cohere** (NEW - Enterprise-grade)
* `command-r-plus` - Most capable Cohere model
* `command-r` - Balanced performance model
* `command` - Standard command model

**Groq Fast** (NEW - Ultra-fast inference)
* `llama-3.1-70b-versatile` - Fast 70B Llama
* `llama-3.1-8b-instant` - Lightning-fast 8B model
* `mixtral-8x7b-32768` - Fast Mixtral variant
* `gemma2-9b-it` - Google's Gemma model

**Perplexity AI** (NEW - Web-connected)
* `llama-3.1-sonar-large-128k-online` - Web search + large context
* `llama-3.1-sonar-small-128k-online` - Web search + fast responses
* `llama-3.1-sonar-large-128k-chat` - Pure chat without web
* `llama-3.1-sonar-small-128k-chat` - Fast chat model

## 💡 Usage Examples

Once configured, ask Claude things like:

> "Get a second opinion from GPT-4.1 on this coding approach"

> "What would Grok-3-thinking think about this solution?"

> "Compare how Claude-4-opus and gemini-2.0-flash would solve this problem"

> "Get an opinion from meta-llama/Llama-3.1-70B-Instruct on HuggingFace"

> "What does DeepSeek-reasoner think about this math problem?"

> "Ask Mistral-large-latest to review my code architecture"

> "Get a fast response from Groq's llama-3.1-8b-instant model"

> "Use Perplexity's web search to research the latest AI developments"

> "What does Cohere's command-r-plus think about this business strategy?"

> "Get Together AI's Llama-405B opinion on this complex problem"

> "Start a group discussion about AI ethics with GPT-4.1, Claude-4, Mistral, and Perplexity"

> "Cross-platform comparison of this algorithm across all 11 available platforms"

## 🔧 Available Tools

### Single Model Opinions
* **`get_openai_opinion`** - Get opinion from any OpenAI model
* **`get_gemini_opinion`** - Get opinion from any Gemini model (enhanced with better conversation handling)
* **`get_grok_opinion`** - Get opinion from any Grok model (includes thinking models)
* **`get_claude_opinion`** - Get opinion from any Claude model
* **`get_huggingface_opinion`** - Get opinion from any HuggingFace model (enhanced with better reliability)
* **`get_deepseek_opinion`** - Get opinion from DeepSeek models
* **`get_mistral_opinion`** - Get opinion from Mistral AI models (NEW)
* **`get_together_opinion`** - Get opinion from Together AI's 200+ models (NEW)
* **`get_cohere_opinion`** - Get opinion from Cohere enterprise models (NEW)
* **`get_groq_fast_opinion`** - Get ultra-fast responses from Groq (NEW)
* **`get_perplexity_opinion`** - Get web-connected AI responses (NEW)

### Model Comparisons  
* **`compare_openai_models`** - Compare multiple OpenAI models
* **`compare_gemini_models`** - Compare multiple Gemini models
* **`compare_grok_models`** - Compare multiple Grok models
* **`compare_claude_models`** - Compare multiple Claude models

### Cross-Platform Features
* **`cross_platform_comparison`** - Compare across all 11 AI platforms: OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, Mistral, Together AI, Cohere, Groq Fast & Perplexity
* **`group_discussion`** - Multi-round discussions between AI models with shared context (supports all platforms)

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

## 🚀 What's New in v3.0

- ✅ **Major Bug Fixes**: Fixed HuggingFace empty responses and Gemini blank chat issues
- ✅ **5 New AI Platforms**: Mistral AI, Together AI, Cohere, Groq Fast, Perplexity AI
- ✅ **Enhanced HuggingFace**: Completely rebuilt with advanced retry logic and better error handling
- ✅ **Improved Gemini**: Smart conversation handling prevents blank responses in long chats
- ✅ **11 Total Platforms**: Now supporting 11 different AI platforms and 800,000+ models
- ✅ **Ultra-Fast Inference**: Groq Fast for lightning-quick responses
- ✅ **Web-Connected AI**: Perplexity AI with real-time search capabilities
- ✅ **Enterprise Models**: Cohere's command models for business use cases
- ✅ **Better Error Messages**: Helpful troubleshooting suggestions and model recommendations

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

---

**Built for developers who want access to the entire AI ecosystem at their fingertips** 🧠✨
