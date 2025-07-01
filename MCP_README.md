# MCP Integration Guide

## Model Context Protocol (MCP) Support

This notebook analyzer now supports the Model Context Protocol (MCP), allowing AI assistants like Claude, ChatGPT, and others to directly analyze Jupyter notebooks (.ipynb) and marimo notebooks (.py) for GPU requirements.

## Quick Setup

1. **Start the service:**
   ```bash
   docker compose up -d
   ```

2. **Copy the MCP configuration:**
   ```bash
   cp mcp_config.json ~/.config/your-ai-assistant/mcp_servers.json
   ```

3. **Test the connection:**
   ```bash
   curl -X POST http://localhost:5001/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
   ```

## Available Tools

### 1. `analyze_notebook`
Performs complete GPU requirements analysis on Jupyter and marimo notebooks.

**Parameters:**
- `url` (string, required): URL to the notebook (GitHub, GitLab, direct .ipynb or .py)
- `include_reasoning` (boolean, optional): Include detailed analysis reasoning (default: true)
- `include_compliance` (boolean, optional): Include NVIDIA compliance assessment (default: true)

**Returns:**
- GPU requirements (minimum and optimal configurations)
- VRAM estimates and runtime predictions
- SXM requirements and ARM compatibility
- NVIDIA compliance scoring
- Detailed reasoning (if requested)

### 2. `get_gpu_recommendations`
Provides GPU recommendations for specific workload types.

**Parameters:**
- `workload_type` (string, required): One of `inference`, `training`, `fine-tuning`
- `model_size` (string, optional): One of `small`, `medium`, `large`, `xlarge` (default: medium)
- `batch_size` (integer, optional): Expected batch size (default: 1)

**Returns:**
- Recommended GPU type and quantity
- VRAM requirements per GPU
- Performance considerations

## Example Usage

### Direct MCP Calls

```bash
# Analyze a notebook
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "analyze_notebook",
      "arguments": {
        "url": "https://github.com/brevdev/launchables/blob/main/llama3_finetune_inference.ipynb",
        "include_compliance": true
      }
    },
    "id": 1
  }'

# Get recommendations
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", 
    "method": "tools/call",
    "params": {
      "name": "get_gpu_recommendations",
      "arguments": {
        "workload_type": "training",
        "model_size": "large",
        "batch_size": 4
      }
    },
    "id": 2
  }'
```

### AI Assistant Integration

Once connected via MCP, you can simply ask your AI assistant:

- *"What GPU requirements does this notebook have?"* (provide URL)
- *"Analyze this training notebook for NVIDIA compliance"*
- *"What GPUs do I need for fine-tuning a large model with batch size 8?"*
- *"Compare the GPU requirements between these two notebooks"*

The AI will automatically use the appropriate tools to provide detailed analysis.

## Configuration for Popular AI Assistants

### Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "notebook-analyzer": {
      "command": "curl",
      "args": ["-X", "POST", "http://localhost:5001/mcp", "-H", "Content-Type: application/json"],
      "transport": "http"
    }
  }
}
```

### Cursor/VSCode with Claude
Add to your workspace settings or global MCP configuration.

### Custom Integrations
Use the HTTP transport at `http://localhost:5001/mcp` with JSON-RPC 2.0 protocol.

## Benefits of MCP Integration

1. **Seamless Analysis**: AI assistants can analyze notebooks directly in conversation
2. **Comparative Analysis**: Easy comparison of GPU requirements across multiple notebooks  
3. **Automated Recommendations**: Get instant GPU sizing for different workload scenarios
4. **Context-Aware**: AI can provide reasoning based on your specific use case
5. **Batch Processing**: Analyze multiple notebooks efficiently through AI orchestration

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure the service is running on port 5001
2. **Tool Not Found**: Check that you're using the correct tool names
3. **Invalid URL**: Ensure notebook URLs are publicly accessible
4. **Timeout**: Large notebooks may take longer to analyze

### Debug Commands

```bash
# Check service health
curl http://localhost:5001/health

# Test MCP initialization  
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}'

# View container logs
docker compose logs -f
```

## Advanced Usage

### Batch Analysis Script
```python
import requests
import json

def analyze_notebooks(urls):
    results = []
    for url in urls:
        response = requests.post('http://localhost:5001/mcp', 
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "analyze_notebook", 
                    "arguments": {"url": url}
                },
                "id": len(results) + 1
            })
        results.append(response.json())
    return results

# Analyze multiple notebooks (both Jupyter and marimo)
notebooks = [
    "https://github.com/user/repo1/blob/main/notebook1.ipynb",
    "https://github.com/user/repo2/blob/main/marimo-app.py",
    "https://github.com/user/repo3/blob/main/analysis.ipynb"
]
results = analyze_notebooks(notebooks)
```

This MCP integration makes your notebook analyzer incredibly powerful when combined with AI assistants, enabling natural language queries about GPU requirements and automated analysis workflows. 