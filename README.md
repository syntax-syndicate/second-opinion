<div align="center">

# 🤖 Second Opinion MCP

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Get+instant+second+opinions;From+16+AI+platforms;800%2C000%2B+models+available;Directly+within+Claude+conversations" alt="Typing SVG" />

[![License](https://img.shields.io/badge/License-CC--BY--ND--4.0-blue?style=for-the-badge&logo=creative-commons&logoColor=white)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-purple?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K)](https://modelcontextprotocol.io)

**🎯 Get instant second opinions from 16 AI platforms and 800,000+ models**

*OpenAI • Gemini • Grok • Claude • HuggingFace • DeepSeek • Mistral • Together AI • Cohere • Groq • Perplexity • Replicate • AI21 Labs • Stability AI • Fireworks AI • Anyscale*

---

</div>

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:

* **Coding problems** - Compare approaches across models
* **Creative writing** - Get diverse style feedback  
* **Problem solving** - Validate logic and reasoning
* **Cross-model analysis** - See how different AIs tackle the same task
* **Group discussions** - Host AI debates with multiple models
* **Custom model access** - Use any HuggingFace model via Inference API

## ✨ Version 5.0 Features & Improvements

### 🎭 **NEW: AI Personality System**
- **5 Distinct Personalities**: `honest`, `gf`, `coach`, `wise`, `creative`
- **Intelligent Model Matching**: Each personality uses models best suited for their character
- **Always Available**: Works with any configured AI provider

### 🧠 **NEW: Intelligent Model Selection**
- **Quality-Based Ranking**: 34+ models ranked by capability (Grok-4 → Gemini Pro → GPT-4.1)
- **Smart Defaults**: Automatically selects the best available model
- **Personality Optimization**: Different models for different personality types

### 🏗️ **NEW: Modular Architecture**
- **5 Clean Files**: Replaced 51k+ token monolith with maintainable modules
- **Professional Structure**: `client_manager.py`, `ai_providers.py`, `conversation_manager.py`, `mcp_server.py`, `main.py`
- **JSON Configuration**: Easy model priority updates via `model_priority.json`

### 🚀 Major Platform Integrations
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

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![Claude Desktop](https://img.shields.io/badge/Claude-Desktop-purple?style=flat-square&logo=anthropic&logoColor=white)
![API Keys](https://img.shields.io/badge/API-Keys-Optional-green?style=flat-square&logo=key&logoColor=white)

</div>

* **Python 3.8+** - Programming language runtime
* **Claude Desktop or Claude Code** - Choose your preferred MCP integration
* **API Keys** - For any combination of the 16 supported AI platforms

## 📦 Installation Guide

### 🚀 Quick Start

1. **📥 Clone the repository**

   ```bash
   git clone https://github.com/ProCreations-Official/second-opinion.git
   cd second-opinion
   ```

2. **⚙️ Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **🔑 Get API Keys**

   <details>
   <summary>🔗 <strong>Click to expand API key links</strong></summary>

   | Platform | Link | Required |
   |----------|------|----------|
   | **OpenAI** | [platform.openai.com](https://platform.openai.com/api-keys) | ⭐ Popular |
   | **Gemini** | [aistudio.google.com](https://aistudio.google.com/app/apikey) | ⭐ Popular |
   | **Grok** | [x.ai](https://x.ai/api) | 🔥 Thinking Models |
   | **Claude** | [anthropic.com](https://console.anthropic.com/settings/keys) | 🧠 Advanced |
   | **HuggingFace** | [huggingface.co](https://huggingface.co/settings/tokens) | 🤗 800k+ Models |
   | **DeepSeek** | [deepseek.com](https://api-docs.deepseek.com/) | 🔬 Reasoning |
   | **Mistral** | [console.mistral.ai](https://console.mistral.ai/) | 🇫🇷 European |
   | **Together AI** | [api.together.xyz](https://api.together.xyz/settings/api-keys) | 🔗 200+ Models |
   | **Cohere** | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) | 🏢 Enterprise |
   | **Groq** | [console.groq.com](https://console.groq.com/keys) | ⚡ Ultra-Fast |
   | **Perplexity** | [perplexity.ai](https://www.perplexity.ai/settings/api) | 🔍 Web Search |
   | **Replicate** | [replicate.com](https://replicate.com/account/api-tokens) | 🎭 Open Source |
   | **AI21 Labs** | [studio.ai21.com](https://studio.ai21.com/account/api-key) | 🧬 Jamba Models |
   | **Stability AI** | [platform.stability.ai](https://platform.stability.ai/account/keys) | 🎨 StableLM |
   | **Fireworks AI** | [fireworks.ai](https://fireworks.ai/account/api-keys) | 🔥 Fast Inference |
   | **Anyscale** | [console.anyscale.com](https://console.anyscale.com/credentials) | 🚀 Ray Serving |

   </details>

4. **🔧 Choose Your Integration Method**

   Select the method that matches your Claude setup:

<details>
<summary>🖥️ <strong>Option A: Claude Desktop Installation</strong></summary>

### For Claude Desktop Users

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
        "REPLICATE_API_TOKEN": "your_replicate_key_here",
        "AI21_API_KEY": "your_ai21_key_here",
        "STABILITY_API_KEY": "your_stability_key_here",
        "FIREWORKS_API_KEY": "your_fireworks_key_here",
        "ANYSCALE_API_KEY": "your_anyscale_key_here"
      }
    }
  }
}
```

> **💡 Note**: You only need to add API keys for the services you want to use. Missing keys will simply disable those specific features.

**🔄 Restart Claude Desktop** after configuration.

</details>

<details>
<summary>🛠️ <strong>Option B: Claude Code Installation</strong></summary>

### 🎯 For Claude Code CLI Users

<div align="center">

![Claude Code](https://img.shields.io/badge/Claude-Code-purple?style=flat-square&logo=anthropic&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-Server-blue?style=flat-square&logo=server&logoColor=white)

</div>

#### 📦 Prerequisites

First, ensure Claude Code CLI is installed globally:

```bash
npm install -g @anthropic-ai/claude-code
```

#### 🚀 Installation Methods

<details>
<summary>🎯 <strong>Method 1: Direct CLI Configuration (Recommended)</strong></summary>

Use the `claude mcp add` command to add the Second Opinion server:

```bash
# Navigate to your second-opinion directory
cd /path/to/your/second-opinion

# Add the MCP server with environment variables (use -e for each API key)
claude mcp add second-opinion -s user \
  -e OPENAI_API_KEY=your_openai_key_here \
  -e GEMINI_API_KEY=your_gemini_key_here \
  -e GROK_API_KEY=your_grok_key_here \
  -e CLAUDE_API_KEY=your_claude_key_here \
  -e HUGGINGFACE_API_KEY=your_huggingface_key_here \
  -e DEEPSEEK_API_KEY=your_deepseek_key_here \
  -e MISTRAL_API_KEY=your_mistral_key_here \
  -e TOGETHER_API_KEY=your_together_key_here \
  -e COHERE_API_KEY=your_cohere_key_here \
  -e GROQ_FAST_API_KEY=your_groq_key_here \
  -e PERPLEXITY_API_KEY=your_perplexity_key_here \
  -e REPLICATE_API_TOKEN=your_replicate_key_here \
  -e AI21_API_KEY=your_ai21_key_here \
  -e STABILITY_API_KEY=your_stability_key_here \
  -e FIREWORKS_API_KEY=your_fireworks_key_here \
  -e ANYSCALE_API_KEY=your_anyscale_key_here \
  -- /path/to/your/second-opinion/run.sh
```

> **💡 Quick Setup**: You only need to include `-e` flags for the API keys you have. For example, if you only have OpenAI and Gemini keys:

```bash
claude mcp add second-opinion -s user \
  -e OPENAI_API_KEY=your_openai_key_here \
  -e GEMINI_API_KEY=your_gemini_key_here \
  -- /path/to/your/second-opinion/run.sh
```

</details>

<details>
<summary>⚙️ <strong>Method 2: Manual Configuration</strong></summary>

Alternatively, you can manually add the server to your `.claude.json` file:

```json
{
  "mcpServers": {
    "second-opinion": {
      "type": "stdio",
      "command": "/path/to/your/second-opinion/run.sh",
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
        "REPLICATE_API_TOKEN": "your_replicate_key_here",
        "AI21_API_KEY": "your_ai21_key_here",
        "STABILITY_API_KEY": "your_stability_key_here",
        "FIREWORKS_API_KEY": "your_fireworks_key_here",
        "ANYSCALE_API_KEY": "your_anyscale_key_here"
      }
    }
  }
}
```

</details>

#### 🔧 Why Use run.sh?

<div align="center">

| Feature | Benefit |
|---------|---------|
| 📦 **Dependency Management** | Automatically installs/updates requirements |
| 🛡️ **Error Handling** | Checks for python3 availability and required files |
| 🔄 **Cross-platform** | Works better than direct Python execution |
| ⚡ **Reliability** | Ensures consistent execution regardless of system |

</div>

#### ✅ Verification

Check that your MCP server is properly installed:

```bash
claude mcp list
```

You should see `second-opinion` in the list of available MCP servers.

> **🔑 Environment Variables**: You only need to add API keys for the services you want to use. Missing keys will simply disable those specific AI platforms. The server will work with any combination of available API keys.

</details>

<details>
<summary>🎯 <strong>Available Models (800,000+ Models Across 16 Platforms)</strong></summary>

<div align="center">

![Total Models](https://img.shields.io/badge/Total_Models-800%2C000%2B-brightgreen?style=flat-square)
![Platforms](https://img.shields.io/badge/Platforms-16-blue?style=flat-square)
![Updated](https://img.shields.io/badge/Updated-v4.0-orange?style=flat-square)

</div>

### 🚀 **Premium AI Platforms**

<details>
<summary>🧠 <strong>OpenAI Models</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `o4-mini` | Fast reasoning model | ⚡ Quick reasoning |
| `gpt-4.1` | Latest flagship model | 🎯 General tasks |
| `gpt-4o` | Multimodal powerhouse | 🖼️ Vision + text |
| `gpt-4o-mini` | Lightweight GPT-4o | 💰 Cost-effective |
| `gpt-3.5-turbo` | Fast and efficient | 🏃 Speed |

</details>

<details>
<summary>💎 <strong>Google Gemini Models</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `gemini-2.5-flash-lite-preview-06-17` | Lightweight and fast | ⚡ Quick responses |
| `gemini-2.5-flash` | Advanced reasoning and efficiency | 🧮 Complex analysis |
| `gemini-2.5-pro` | Most capable Gemini model | 🧠 Advanced tasks |

</details>

<details>
<summary>🔥 <strong>Grok Models (xAI)</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `grok-3` | Latest flagship model | 🎯 General excellence |
| `grok-3-thinking` | Step-by-step reasoning | 🤔 Deep thinking |
| `grok-3-mini` | Lightweight thinking model | 💡 Quick insights |
| `grok-2` | Robust and reliable | 🛡️ Stability |
| `grok-beta` | Experimental features | 🧪 Cutting edge |

</details>

<details>
<summary>🎭 <strong>Anthropic Claude Models</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `claude-4-opus-20250522` | Most advanced Claude | 🧠 Complex reasoning |
| `claude-4-sonnet-20250522` | Versatile general tasks | ⚖️ Balanced performance |
| `claude-3-7-sonnet-20250224` | Stable and reliable | 🛡️ Production use |
| `claude-3-5-sonnet-20241022` | Efficient, lighter model | 💨 Fast responses |

</details>

<details>
<summary>🤗 <strong>HuggingFace Hub (800,000+ Models)</strong></summary>

**Featured Models:**
| Model | Description | Best For |
|-------|-------------|----------|
| `meta-llama/Llama-3.1-8B-Instruct` | Fast Meta model | ⚡ Speed |
| `meta-llama/Llama-3.1-70B-Instruct` | Powerful reasoning | 🧠 Complex tasks |
| `mistralai/Mistral-7B-Instruct-v0.3` | French-developed | 🇫🇷 European AI |
| `Qwen/Qwen2.5-7B-Instruct` | Alibaba's latest | 🏢 Enterprise |

> **🌟 Special**: Access to *any model on HuggingFace Hub that supports text generation*

</details>

### 🔬 **Specialized AI Platforms**

<details>
<summary>🧬 <strong>DeepSeek Models (Advanced Reasoning)</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `deepseek-chat` | DeepSeek-V3 general tasks | 💬 Conversations |
| `deepseek-reasoner` | DeepSeek-R1 advanced reasoning | 🧠 Complex logic |

</details>

<details>
<summary>🇫🇷 <strong>Mistral AI Models</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `mistral-large-latest` | Most powerful Mistral | 🎯 Top performance |
| `mistral-small-latest` | Fast and cost-effective | 💰 Budget-friendly |
| `mistral-medium-latest` | Balanced performance | ⚖️ General use |
| `codestral-latest` | Code generation specialist | 💻 Programming |

</details>

<details>
<summary>🔗 <strong>Together AI (200+ Open-Source Models)</strong></summary>

| Model | Description | Best For |
|-------|-------------|----------|
| `meta-llama/Llama-3.1-8B-Instruct-Turbo` | Fast Llama turbo | ⚡ Speed |
| `meta-llama/Llama-3.1-70B-Instruct-Turbo` | Powerful Llama turbo | 🚀 Performance |
| `meta-llama/Llama-3.1-405B-Instruct-Turbo` | Largest Llama model | 🦣 Massive scale |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Mixture of experts | 🎭 Specialized tasks |
| `Qwen/Qwen2.5-72B-Instruct-Turbo` | Alibaba's fast model | 🏢 Enterprise |

</details>

<details>
<summary>🏢 <strong>Enterprise & Fast Inference</strong></summary>

**Cohere (Enterprise-grade)**
| Model | Description | Best For |
|-------|-------------|----------|
| `command-r-plus` | Most capable Cohere | 🎯 Enterprise |
| `command-r` | Balanced performance | ⚖️ General business |
| `command` | Standard command model | 💼 Basic tasks |

**Groq (Ultra-fast inference)**
| Model | Description | Best For |
|-------|-------------|----------|
| `llama-3.1-70b-versatile` | Fast 70B Llama | ⚡ Quick power |
| `llama-3.1-8b-instant` | Lightning-fast 8B | 🏃 Instant responses |
| `mixtral-8x7b-32768` | Fast Mixtral variant | 🎭 Quick specialization |
| `gemma2-9b-it` | Google's Gemma model | 🔍 Search-optimized |

</details>

<details>
<summary>🔍 <strong>Web-Connected & Open Source</strong></summary>

**Perplexity AI (Web-connected)**
| Model | Description | Best For |
|-------|-------------|----------|
| `llama-3.1-sonar-large-128k-online` | Web search + large context | 🌐 Research |
| `llama-3.1-sonar-small-128k-online` | Web search + fast responses | 🔍 Quick search |
| `llama-3.1-sonar-large-128k-chat` | Pure chat without web | 💬 Conversations |
| `llama-3.1-sonar-small-128k-chat` | Fast chat model | ⚡ Quick chat |

**Replicate (Open-source hosting)**
| Model | Description | Best For |
|-------|-------------|----------|
| `meta/llama-2-70b-chat` | Large Llama 2 chat | 🦣 Powerful chat |
| `meta/llama-2-13b-chat` | Medium Llama 2 chat | ⚖️ Balanced |
| `meta/codellama-34b-instruct` | Code-specialized Llama | 💻 Programming |
| `microsoft/wizardcoder-34b` | Microsoft's coding model | 🧙 Code magic |

</details>

<details>
<summary>🎨 <strong>Specialized Model Families</strong></summary>

**AI21 Labs (Advanced reasoning)**
| Model | Description | Best For |
|-------|-------------|----------|
| `jamba-1.5-large` | State-space capabilities | 🧬 Complex reasoning |
| `jamba-1.5-mini` | Compact Jamba model | 💎 Efficient reasoning |
| `j2-ultra` | Jurassic-2 Ultra model | 🦕 Powerful |
| `j2-mid` | Jurassic-2 Mid model | ⚖️ Balanced |

**Stability AI (StableLM family)**
| Model | Description | Best For |
|-------|-------------|----------|
| `stablelm-2-zephyr-1_6b` | Efficient 1.6B parameter | ⚡ Lightweight |
| `stable-code-instruct-3b` | Code-specialized 3B | 💻 Programming |
| `japanese-stablelm-instruct-beta-70b` | Japanese language | 🇯🇵 Japanese tasks |
| `stablelm-zephyr-3b` | Balanced 3B parameter | ⚖️ General use |

**Fireworks AI (Ultra-fast inference)**
| Model | Description | Best For |
|-------|-------------|----------|
| `accounts/fireworks/models/llama-v3p1-70b-instruct` | Fast Llama 3.1 70B | 🔥 Speed + power |
| `accounts/fireworks/models/llama-v3p1-8b-instruct` | Fast Llama 3.1 8B | ⚡ Quick responses |
| `accounts/fireworks/models/mixtral-8x7b-instruct` | Fast Mixtral model | 🎭 Fast specialization |
| `accounts/fireworks/models/deepseek-coder-v2-lite-instruct` | Code-specialized | 💻 Fast coding |

**Anyscale (Ray-powered serving)**
| Model | Description | Best For |
|-------|-------------|----------|
| `meta-llama/Llama-2-70b-chat-hf` | Enterprise Llama 2 70B | 🏢 Enterprise chat |
| `meta-llama/Llama-2-13b-chat-hf` | Enterprise Llama 2 13B | 💼 Business tasks |
| `codellama/CodeLlama-34b-Instruct-hf` | Enterprise CodeLlama | 💻 Enterprise coding |
| `mistralai/Mistral-7B-Instruct-v0.1` | Enterprise Mistral | 🇫🇷 Enterprise EU |

</details>

</details>

<div align="center">

---

```
████████████████████████████████████████████████████████████████████████████████
```

# 💡 Usage Examples

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=2000&pause=500&color=F75C7E&center=true&vCenter=true&width=800&lines=Ask+Claude+for+second+opinions...;Get+diverse+AI+perspectives...;Compare+across+multiple+models...;Access+800%2C000%2B+AI+models!" alt="Usage Examples" />

---

</div>

<details>
<summary>🎯 <strong>Quick Examples - Get Started Now!</strong></summary>

### 🎭 **NEW: Personality Examples**

```
😤 "Give me an honest opinion about this code" (brutally frank feedback)

💕 "I need some encouragement with this project" (supportive girlfriend mode)

🏆 "Help me stay motivated to finish this task" (motivational coach)

🧙 "What's the deeper meaning behind this design pattern?" (ancient wisdom)

🎨 "Think of a creative solution to this problem" (innovative thinking)

🤖 "Just give me the best available opinion" (automatic smart selection)
```

### 🔥 **Popular Requests**

```
💬 "Get a second opinion from GPT-4.1 on this coding approach"

🤔 "What would Grok-4 think about this solution?" (NEW: Latest model)

⚖️ "Compare how Claude-4-opus and gemini-2.5-flash would solve this problem"

🤗 "Get an opinion from meta-llama/Llama-3.1-70B-Instruct on HuggingFace"

🧠 "What does DeepSeek-reasoner think about this math problem?"

🇫🇷 "Ask Mistral-large-latest to review my code architecture"

⚡ "Get a fast response from Groq's llama-3.1-8b-instant model"

🌐 "Use Perplexity's web search to research the latest AI developments"

🏢 "What does Cohere's command-r-plus think about this business strategy?"

🔗 "Get Together AI's Llama-405B opinion on this complex problem"
```

### 🎭 **Advanced Features**

```
🗣️ "Start a group discussion about AI ethics with GPT-4.1, Claude-4, Mistral, and Perplexity"

📊 "Cross-platform comparison of this algorithm across all 16 available platforms"

🎭 "Get a Replicate opinion from meta/llama-2-70b-chat on this open-source approach"

🧬 "What does AI21's Jamba-1.5-large think about this reasoning problem?"

🎨 "Ask Stability AI's StableLM about this code optimization"

🔥 "Get a super-fast response from Fireworks AI's Llama model"

🚀 "Use Anyscale's enterprise-grade Llama serving for this complex task"
```

</details>

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
* **`get_perplexity_opinion`** - Get web-connected AI responses
* **`get_replicate_opinion`** - Get opinion from Replicate's open-source models (NEW)
* **`get_ai21_opinion`** - Get opinion from AI21 Labs' Jamba models (NEW)
* **`get_stability_opinion`** - Get opinion from Stability AI's StableLM models (NEW)
* **`get_fireworks_opinion`** - Get ultra-fast responses from Fireworks AI (NEW)
* **`get_anyscale_opinion`** - Get enterprise-grade responses from Anyscale (NEW)

### Model Comparisons  
* **`compare_openai_models`** - Compare multiple OpenAI models
* **`compare_gemini_models`** - Compare multiple Gemini models
* **`compare_grok_models`** - Compare multiple Grok models
* **`compare_claude_models`** - Compare multiple Claude models

### 🎭 NEW: Personality & Smart Default Tools
* **`get_personality_opinion`** - Get AI responses with specific personality (honest, gf, coach, wise, creative)
* **`get_default_opinion`** - Automatically uses the best available model (Grok-4 → Gemini Pro → GPT-4.1)
* **`list_personalities`** - See all available AI personalities and their descriptions

### Cross-Platform Features
* **`cross_platform_comparison`** - Compare across all 16 AI platforms: OpenAI, Gemini, Grok, Claude, HuggingFace, DeepSeek, Mistral, Together AI, Cohere, Groq Fast, Perplexity, Replicate, AI21 Labs, Stability AI, Fireworks AI & Anyscale
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

## 🚀 What's New in v5.0

- ✅ **🎭 AI Personality System**: 5 distinct personalities (honest, gf, coach, wise, creative) with optimized model selection
- ✅ **🧠 Intelligent Model Selection**: Quality-ranked models with Grok-4 as top priority, smart defaults
- ✅ **🏗️ Modular Architecture**: Refactored from 51k+ token monolith into 5 maintainable modules
- ✅ **📊 JSON Configuration**: Easy model priority updates via configuration files
- ✅ **🤖 Always-Available Tools**: Personality and default tools work with any provider setup
- ✅ **⚡ Enhanced Performance**: Optimized model selection and better error handling
- ✅ **🔄 Future-Proof**: Easy to add new models and update priorities

### Previous Updates (v4.0)
- ✅ **5 New Major AI Platforms**: Replicate, AI21 Labs, Stability AI, Fireworks AI, Anyscale
- ✅ **16 Total Platforms**: Now supporting 16 different AI platforms and 800,000+ models
- ✅ **Advanced Reasoning**: AI21 Labs' Jamba models with state-space architecture
- ✅ **Ultra-Fast Inference**: Fireworks AI for blazing-fast open model serving

### Previous Improvements (v3.0)
- ✅ **Major Bug Fixes**: Fixed HuggingFace empty responses and Gemini blank chat issues
- ✅ **Enhanced HuggingFace**: Completely rebuilt with advanced retry logic and better error handling
- ✅ **Improved Gemini**: Smart conversation handling prevents blank responses in long chats
- ✅ **Web-Connected AI**: Perplexity AI with real-time search capabilities
- ✅ **Enterprise Models**: Cohere's command models for business use cases

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

<div align="center">

---

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

### 🧠✨ **Built for developers who want access to the entire AI ecosystem at their fingertips**

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=4000&pause=1000&color=36BCF7&center=true&vCenter=true&width=900&lines=16+AI+platforms+%E2%80%A2+800%2C000%2B+models;The+most+comprehensive+AI+second+opinion+service;Get+diverse+perspectives+%E2%80%A2+Make+better+decisions;Open+source+%E2%80%A2+Community+driven" alt="Footer" />

![GitHub stars](https://img.shields.io/github/stars/ProCreations-Official/second-opinion?style=social)
![GitHub forks](https://img.shields.io/github/forks/ProCreations-Official/second-opinion?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ProCreations-Official/second-opinion?style=social)

**⭐ Star us on GitHub • 🍴 Fork the project • 💖 Contribute to the future of AI**

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

</div>
