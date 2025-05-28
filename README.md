# Second Opinion MCP

Get instant second opinions from multiple AI models (OpenAI, Gemini, Grok, & Claude) directly within Claude conversations.

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:

* **Coding problems** - Compare approaches across models
* **Creative writing** - Get diverse style feedback
* **Problem solving** - Validate logic and reasoning
* **Cross-model analysis** - See how different AIs tackle the same task

## 📋 Requirements

* Python 3.8+
* Claude Desktop
* API keys for OpenAI, Gemini, Grok, and/or Claude

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
   * **Grok**: Get your key at [x.ai](https://x.ai/team/api-keys)
   * **Claude**: Get your key at [anthropic.com](https://console.anthropic.com/settings/keys))

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
           "CLAUDE_API_KEY": "your_claude_key_here"
         }
       }
     }
   }
   ```

Note: Any models not explicitly added in the MCP configuration will automatically adjust to ensure Claude exclusively uses the models you’ve added your API key to. For instance, if you’ve only added Gemini (since it’s free), Claude will be appropriately be restricted to using Gemini for obtaining another opinion and will not attempt to use other models, resulting in failure.


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

* `grok-3` - Latest Grok model
* `grok-2` - Robust and reliable
* `grok-beta` - Experimental features

**Claude**

* `claude-4-opus-20250522` - Most advanced Claude model
* `claude-4-sonnet-20250522` - Versatile model for general tasks
* `claude-3-7-sonnet-20250224` - Stable and reliable
* `claude-3-5-sonnet-20241022` - Efficient, lighter model

## 💡 Usage Examples

Once configured, ask Claude things like:

> "Get a second opinion from GPT-4.1 on this coding approach"

> "What would Grok-3 think about this solution?"

> "Compare how Claude-4-opus and gemini-2.0-flash would solve this problem"

> "Cross-platform comparison of this algorithm"

## 🔧 Available Tools

* **`get_openai_opinion`** - Get opinion from any OpenAI model
* **`get_gemini_opinion`** - Get opinion from any Gemini model
* **`get_grok_opinion`** - Get opinion from any Grok model
* **`get_claude_opinion`** - Get opinion from any Claude model
* **`compare_openai_models`** - Compare multiple OpenAI models
* **`compare_gemini_models`** - Compare multiple Gemini models
* **`compare_grok_models`** - Compare multiple Grok models
* **`compare_claude_models`** - Compare multiple Claude models
* **`cross_platform_comparison`** - Compare OpenAI, Gemini, Grok, and Claude

## 🔒 Security

Your API keys stay private on your machine. The MCP server only sends model responses to Claude, never your credentials.

## 🛟 Troubleshooting

**Import errors**: Ensure you've installed all dependencies
**API errors**: Check that your API keys are correct and active
**Server not connecting**: Verify the file path in your MCP configuration

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

---

**Built for developers who want multiple AI perspectives at their fingertips** 🧠✨
