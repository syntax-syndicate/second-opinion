# Second Opinion MCP

Get instant second opinions from multiple AI models including local, cloud, and enterprise services directly within Claude conversations.

**Supported Platforms (25+ Services):**
- **Local AI**: Ollama, LM Studio 
- **Cloud Services**: OpenAI, Gemini, Grok, Claude, Azure OpenAI, AWS Bedrock, Vertex AI
- **Specialized Services**: Mistral, Together AI, Cohere, Groq, Perplexity, Writer.com, HuggingFace
- **Enterprise Services**: AI21, Stability AI, Fireworks AI, Anyscale, OpenRouter, DeepSeek
- **Emerging Platforms**: Moonshot AI, 01.AI (Yi models), Baichuan AI, Replicate

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:

* **Coding problems** - Compare approaches across models
* **Creative writing** - Get diverse style feedback  
* **Problem solving** - Validate logic and reasoning
* **Cross-model analysis** - See how different AIs tackle the same task
* **Group discussions** - Host AI debates with multiple models
* **Custom model access** - Use any HuggingFace model via Inference API

## ✨ Version 4.0 Features & Improvements

### 🚀 Major New Platform Integrations
- **🎭 Replicate**: Access to open-source models including Llama 2, CodeLlama, Mistral, and more
- **🌟 AI21 Labs**: Jamba 1.5 models with advanced reasoning capabilities
- **🎨 Stability AI**: StableLM models including code-specialized variants
- **🔥 Fireworks AI**: Ultra-fast inference for popular open-source models
- **🚀 Anyscale**: Ray-powered LLM serving with enterprise-grade reliability

### 🆕 Enhanced Existing Platform Support
- **🤖 Mistral AI**: Direct access to Mistral's latest models including mistral-large-latest and codestral-latest
- **🔗 Together AI**: Access to 200+ open-source models with fast inference
- **🧠 Cohere**: Enterprise-grade language models with Command R+ and Command R
- **⚡ Groq Fast**: Ultra-fast inference API for lightning-quick responses
- **🔍 Perplexity AI**: Web-connected AI with real-time search capabilities

### 🔧 Previous Bug Fixes (v3.0)
- **Fixed HuggingFace Models**: Completely rebuilt HuggingFace integration with advanced retry logic, better model format detection, and comprehensive error handling
- **Fixed Gemini Blank Responses**: Enhanced Gemini conversation handling to prevent empty responses in long chats with smart fallback and retry mechanisms
- **Improved Error Handling**: Better error messages with helpful suggestions for troubleshooting

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
* API keys for any combination of the 25+ supported services:
  - **Required for local**: Ollama server or LM Studio running locally
  - **Cloud APIs**: OpenAI, Gemini, Grok, Claude, Azure OpenAI, AWS credentials, GCP credentials
  - **Specialized APIs**: Mistral, Together AI, Cohere, Groq, Perplexity, Writer.com, HuggingFace
  - **Enterprise APIs**: AI21, Stability AI, Fireworks AI, Anyscale, OpenRouter, DeepSeek
  - **Emerging APIs**: Moonshot AI, 01.AI, Baichuan AI, Replicate

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

3. **Get API Keys (Optional - choose which services you want to use)**

   **Core AI Services:**
   * **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
   * **Gemini**: [aistudio.google.com](https://aistudio.google.com/app/apikey)
   * **Grok**: [x.ai](https://x.ai/api)
   * **Claude**: [anthropic.com](https://console.anthropic.com/settings/keys)
   * **HuggingFace**: [huggingface.co](https://huggingface.co/settings/tokens)
   
   **Cloud Platforms:**
   * **Azure OpenAI**: Azure portal + deployment URL
   * **AWS Bedrock**: AWS access/secret keys
   * **Google Vertex AI**: GCP project ID + authentication
   
   **Specialized Services:**
   * **DeepSeek**: [deepseek.com](https://api-docs.deepseek.com/)
   * **Mistral**: [console.mistral.ai](https://console.mistral.ai/)
   * **Together AI**: [api.together.xyz](https://api.together.xyz/settings/api-keys)
   * **Cohere**: [dashboard.cohere.com](https://dashboard.cohere.com/api-keys)
   * **Groq**: [console.groq.com](https://console.groq.com/keys)
   * **Perplexity**: [perplexity.ai](https://www.perplexity.ai/settings/api)
   * **Writer.com**: [writer.com](https://writer.com)
   
   **Enterprise Services:**
   * **AI21**: [studio.ai21.com](https://studio.ai21.com)
   * **Stability AI**: [platform.stability.ai](https://platform.stability.ai)
   * **Fireworks AI**: [fireworks.ai](https://fireworks.ai)
   * **Anyscale**: [anyscale.com](https://anyscale.com)
   * **OpenRouter**: [openrouter.ai](https://openrouter.ai)
   * **Replicate**: [replicate.com](https://replicate.com)
   
   **Emerging Platforms:**
   * **Moonshot AI**: [moonshot.cn](https://moonshot.cn)
   * **01.AI**: [lingyiwanwu.com](https://lingyiwanwu.com)
   * **Baichuan AI**: [baichuan-ai.com](https://baichuan-ai.com)
   
   **Local Services (No API keys required):**
   * **Ollama**: Install and run `ollama serve`
   * **LM Studio**: Start local server

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
           "PERPLEXITY_API_KEY": "your_perplexity_key_here",
           "AZURE_OPENAI_API_KEY": "your_azure_key_here",
           "AZURE_OPENAI_ENDPOINT": "your_azure_endpoint_here",
           "AWS_ACCESS_KEY_ID": "your_aws_access_key",
           "AWS_SECRET_ACCESS_KEY": "your_aws_secret_key",
           "AWS_REGION": "us-east-1",
           "VERTEX_PROJECT_ID": "your_gcp_project_id",
           "VERTEX_LOCATION": "us-central1",
           "WRITER_API_KEY": "your_writer_key_here",
           "AI21_API_KEY": "your_ai21_key_here",
           "STABILITY_API_KEY": "your_stability_key_here",
           "FIREWORKS_API_KEY": "your_fireworks_key_here",
           "ANYSCALE_API_KEY": "your_anyscale_key_here",
           "OPENROUTER_API_KEY": "your_openrouter_key_here",
           "REPLICATE_API_TOKEN": "your_replicate_token_here",
           "MOONSHOT_API_KEY": "your_moonshot_key_here",
           "YI_API_KEY": "your_yi_key_here",
           "BAICHUAN_API_KEY": "your_baichuan_key_here",
           "OLLAMA_BASE_URL": "http://localhost:11434/v1",
           "LMSTUDIO_BASE_URL": "http://localhost:1234/v1"
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
* `gpt-4o-mini` - Lightweight version of GPT-4o
* `gpt-3.5-turbo` - Fast and cost-effective model

**Gemini**
* `gemini-2.5-flash-lite-preview-06-17` - Lightweight and fast
* `gemini-2.5-flash` - Advanced reasoning and efficiency

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

**Perplexity AI** (Web-connected)
* `llama-3.1-sonar-large-128k-online` - Web search + large context
* `llama-3.1-sonar-small-128k-online` - Web search + fast responses
* `llama-3.1-sonar-large-128k-chat` - Pure chat without web
* `llama-3.1-sonar-small-128k-chat` - Fast chat model

**Replicate** (NEW - Open-source model hosting)
* `meta/llama-2-70b-chat` - Large Llama 2 chat model
* `meta/llama-2-13b-chat` - Medium Llama 2 chat model  
* `meta/llama-2-7b-chat` - Small Llama 2 chat model
* `meta/codellama-34b-instruct` - Code-specialized Llama model
* `stability-ai/stable-code-instruct-3b` - Stability's code model
* `mistralai/mistral-7b-instruct-v0.2` - Mistral instruction model
* `microsoft/wizardcoder-34b` - Microsoft's coding model

**AI21 Labs** (NEW - Advanced reasoning)
* `jamba-1.5-large` - Large Jamba model with state-space capabilities
* `jamba-1.5-mini` - Compact Jamba model
* `j2-ultra` - Jurassic-2 Ultra model
* `j2-mid` - Jurassic-2 Mid model

**Stability AI** (NEW - StableLM family)
* `stablelm-2-zephyr-1_6b` - Efficient 1.6B parameter model
* `stable-code-instruct-3b` - Code-specialized 3B model
* `japanese-stablelm-instruct-beta-70b` - Japanese language model
* `stablelm-zephyr-3b` - Balanced 3B parameter model

**Fireworks AI** (NEW - Ultra-fast inference)
* `accounts/fireworks/models/llama-v3p1-70b-instruct` - Fast Llama 3.1 70B
* `accounts/fireworks/models/llama-v3p1-8b-instruct` - Fast Llama 3.1 8B
* `accounts/fireworks/models/mixtral-8x7b-instruct` - Fast Mixtral model
* `accounts/fireworks/models/qwen2p5-72b-instruct` - Fast Qwen 2.5 model
* `accounts/fireworks/models/deepseek-coder-v2-lite-instruct` - Code-specialized model

**Anyscale** (NEW - Ray-powered serving)
* `meta-llama/Llama-2-70b-chat-hf` - Enterprise Llama 2 70B
* `meta-llama/Llama-2-13b-chat-hf` - Enterprise Llama 2 13B
* `meta-llama/Llama-2-7b-chat-hf` - Enterprise Llama 2 7B
* `codellama/CodeLlama-34b-Instruct-hf` - Enterprise CodeLlama
* `mistralai/Mistral-7B-Instruct-v0.1` - Enterprise Mistral model

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

> "Cross-platform comparison of this algorithm across all 16 available platforms"

> "Get a Replicate opinion from meta/llama-2-70b-chat on this open-source approach"

> "What does AI21's Jamba-1.5-large think about this reasoning problem?"

> "Ask Stability AI's StableLM about this code optimization"

> "Get a super-fast response from Fireworks AI's Llama model"

> "Use Anyscale's enterprise-grade Llama serving for this complex task"

## 🔧 Available Tools

### Core AI Services
* **`get_openai_opinion`** - Get opinion from any OpenAI model
* **`get_gemini_opinion`** - Get opinion from any Gemini model (enhanced conversation handling)
* **`get_grok_opinion`** - Get opinion from any Grok model (includes thinking models)
* **`get_claude_opinion`** - Get opinion from any Claude model
* **`get_huggingface_opinion`** - Get opinion from any of 800,000+ HuggingFace models

### Local AI Services
* **`get_ollama_opinion`** - Get opinion from local Ollama models (NEW)
* **`get_lmstudio_opinion`** - Get opinion from LM Studio local models (NEW)

### Cloud Enterprise Services
* **`get_azure_openai_opinion`** - Get opinion from Azure OpenAI (NEW)
* **`get_aws_bedrock_opinion`** - Get opinion from AWS Bedrock models (NEW)
* **`get_vertex_ai_opinion`** - Get opinion from Google Vertex AI (NEW)

### Specialized Services
* **`get_deepseek_opinion`** - Get opinion from DeepSeek models
* **`get_mistral_opinion`** - Get opinion from Mistral AI models
* **`get_together_opinion`** - Get opinion from Together AI's 200+ models
* **`get_cohere_opinion`** - Get opinion from Cohere enterprise models
* **`get_groq_fast_opinion`** - Get ultra-fast responses from Groq
* **`get_perplexity_opinion`** - Get web-connected AI responses
* **`get_writer_opinion`** - Get opinion from Writer.com AI (NEW)

### Enterprise & Emerging Services
* **`get_ai21_opinion`** - Get opinion from AI21 Labs models
* **`get_stability_opinion`** - Get opinion from Stability AI models
* **`get_fireworks_opinion`** - Get opinion from Fireworks AI models
* **`get_anyscale_opinion`** - Get opinion from Anyscale models
* **`get_openrouter_opinion`** - Get opinion from OpenRouter models
* **`get_replicate_opinion`** - Get opinion from Replicate models
* **`get_moonshot_opinion`** - Get opinion from Moonshot AI (NEW)
* **`get_yi_opinion`** - Get opinion from 01.AI Yi models (NEW)

### Model Comparisons  
* **`compare_openai_models`** - Compare multiple OpenAI models
* **`compare_gemini_models`** - Compare multiple Gemini models
* **`compare_grok_models`** - Compare multiple Grok models
* **`compare_claude_models`** - Compare multiple Claude models

### Cross-Platform Features
* **`cross_platform_comparison`** - Compare across all 25+ AI platforms and services
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

## 🚀 What's New in v4.0

### 🌟 Major Expansion: 25+ AI Services
- ✅ **Local AI Support**: Ollama and LM Studio for private, local AI inference
- ✅ **Enterprise Cloud**: Azure OpenAI, AWS Bedrock, Google Vertex AI
- ✅ **Specialized Services**: Writer.com for content creation
- ✅ **Emerging Platforms**: Moonshot AI, 01.AI Yi models, Baichuan AI
- ✅ **Complete Coverage**: Now supporting 25+ different AI platforms and services

### 🏢 Enterprise-Ready Features
- ✅ **Multi-Cloud Support**: Works with all major cloud providers
- ✅ **Local Deployment**: Run AI models privately with Ollama/LM Studio
- ✅ **Enterprise APIs**: Full support for business-grade AI services
- ✅ **Flexible Configuration**: Mix and match any combination of services

### 🔧 Technical Improvements
- ✅ **Enhanced Architecture**: Modular design for easy service addition
- ✅ **Better Error Handling**: Comprehensive error messages and fallbacks
- ✅ **Robust Configuration**: Environment-based setup with graceful degradation
- ✅ **Cross-Platform Tools**: Updated comparison and discussion features

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

---

**Built for developers who want access to the entire AI ecosystem at their fingertips** 🧠✨

*Now with 25+ AI services including local, cloud, and enterprise platforms - the most comprehensive AI second opinion service available*
