# Second Opinion MCP

Get instant second opinions from multiple AI models (OpenAI & Gemini) directly within Claude conversations.

## 🚀 What it does

This MCP server allows Claude to consult other AI models for different perspectives on:
- **Coding problems** - Compare approaches across models
- **Creative writing** - Get diverse style feedback  
- **Problem solving** - Validate logic and reasoning
- **Cross-model analysis** - See how different AIs tackle the same task

## 📋 Requirements

- Python 3.8+
- Claude Desktop
- API keys for OpenAI and/or Gemini

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
   - **OpenAI**: Get your key at [platform.openai.com](https://platform.openai.com/api-keys)
   - **Gemini**: Get your key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

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
           "GEMINI_API_KEY": "your_gemini_key_here"
         }
       }
     }
   }
   ```

5. **Restart Claude Desktop**

## 🎯 Available Models

**OpenAI**
- `o4-mini` - Fast reasoning model
- `gpt-4.1` - Latest flagship model  
- `gpt-4o` - Multimodal powerhouse

**Gemini**
- `gemini-2.0-flash-001` - Fast and efficient
- `gemini-2.5-pro-experimental` - Advanced reasoning

## 💡 Usage Examples

Once configured, ask Claude things like:

> "Get a second opinion from GPT-4.1 on this coding approach"

> "What would Gemini think about this solution?"

> "Compare how o4-mini and gemini-2.0-flash would solve this problem"

> "Cross-platform comparison of this algorithm"

## 🔧 Available Tools

- **`get_openai_opinion`** - Get opinion from any OpenAI model
- **`get_gemini_opinion`** - Get opinion from any Gemini model  
- **`compare_openai_models`** - Compare multiple OpenAI models
- **`compare_gemini_models`** - Compare multiple Gemini models
- **`cross_platform_comparison`** - Compare OpenAI vs Gemini

## 🔒 Security

Your API keys stay private on your machine. The MCP server only sends model responses to Claude, never your credentials.

## 🛟 Troubleshooting

**Import errors**: Make sure you've installed all requirements
**API errors**: Verify your API keys are correct and have billing enabled
**Server not connecting**: Check the file path in your MCP config

## 🤝 Contributing

Issues and pull requests welcome! This is an open-source project for the AI community.

---

**Built for developers who want multiple AI perspectives at their fingertips** 🧠✨
