# Notebook Analyzer - Web Interface

A containerized web application for analyzing Jupyter notebooks (.ipynb) and marimo notebooks (.py) to determine optimal NVIDIA GPU requirements, VRAM needs, and compliance with best practices.

## 🚀 Features

### Web Interface
- **Modern, Responsive UI**: Clean Bootstrap-based interface optimized for desktop and mobile
- **Dual Input Methods**: Analyze notebooks via URL or file upload
- **Real-time Analysis**: Fast processing with loading indicators and progress feedback
- **Comprehensive Results**: GPU recommendations, compliance scoring, and detailed reasoning
- **API Access**: RESTful API and MCP for programmatic integration and AI assistant connectivity
- **MCP Integration**: Model Context Protocol support enables AI assistants to analyze notebooks directly

### Analysis Capabilities
- **GPU Requirements Analysis**: Minimum & optimal GPU recommendations with VRAM estimation
- **Runtime Predictions**: Estimated execution times for different GPU configurations
- **Multi-GPU Detection**: Identifies distributed training and model parallelism requirements
- **SXM Form Factor Analysis**: Determines when SXM GPUs with NVLink are beneficial
- **ARM/Grace Compatibility**: Evaluates notebook compatibility with ARM-based systems
- **NVIDIA Compliance Scoring**: 0-100 score based on official notebook guidelines
- **LLM Enhancement**: Optional context-aware analysis for improved accuracy

## 📋 Requirements

### For Local Development
- Docker and Docker Compose
- Python 3.8+ (if running without Docker)

### For Production Deployment
- Docker runtime environment
- Optional: Load balancer/reverse proxy (nginx, traefik, etc.)

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

1. **Clone and build**:
   ```bash
   git clone <repository-url>
   cd notebook-analyzer-web
   docker-compose up --build
   ```

2. **Access the application**:
   Open your browser to [http://localhost:5001](http://localhost:5001)

### Using Docker directly

```bash
# Build the image
docker build -t notebook-analyzer-web .

# Run the container
docker run -p 5000:5000 notebook-analyzer-web
```

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python app.py
```

## 🌐 Usage

### Web Interface

1. **URL Analysis**: Enter any public GitHub, GitLab, or direct notebook URL (.ipynb or .py)
2. **File Upload**: Drag and drop or select local .ipynb (Jupyter) or .py (marimo) files (up to 16MB)
3. **View Results**: Get comprehensive GPU recommendations and compliance analysis

### API Access

#### REST API

**Analyze from URL**
```bash
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/user/repo/blob/main/notebook.ipynb"}'
```

**Analyze uploaded file**
```bash
# Jupyter notebook
curl -X POST http://localhost:5001/api/analyze \
  -F "file=@your-notebook.ipynb"

# marimo notebook
curl -X POST http://localhost:5001/api/analyze \
  -F "file=@your-marimo-app.py"
```

#### MCP (Model Context Protocol) Integration

The service now supports MCP for AI assistant integration! This allows AI assistants like Claude to directly analyze notebooks through standardized tool calls.

**Available MCP Tools:**
- `analyze_notebook` - Complete notebook analysis with GPU requirements and compliance
- `get_gpu_recommendations` - Get recommendations for specific workload types

**Example MCP Tool Call:**
```bash
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "analyze_notebook",
      "arguments": {
        "url": "https://github.com/user/repo/blob/main/notebook.ipynb",
        "include_reasoning": true,
        "include_compliance": true
      }
    },
    "id": 1
  }'
```

**Connect to AI Assistants:**
Copy `mcp_config.json` to your AI assistant's MCP configuration to enable notebook analysis capabilities.

**MCP Tools Available:**

1. **`analyze_notebook`** - Complete notebook analysis
   - **Parameters:**
     - `url` (required): URL to Jupyter or marimo notebook
     - `include_reasoning` (optional): Include detailed reasoning
     - `include_compliance` (optional): Include NVIDIA compliance assessment
   - **Returns:** GPU requirements, compliance score, runtime estimates

2. **`get_gpu_recommendations`** - Workload-specific recommendations  
   - **Parameters:**
     - `workload_type` (required): `inference`, `training`, or `fine-tuning`
     - `model_size` (optional): `small`, `medium`, `large`, or `xlarge`
     - `batch_size` (optional): Expected batch size
   - **Returns:** Recommended GPU configuration

**Example with Claude/Cursor:**
Once connected via MCP, you can ask: *"Analyze this notebook for GPU requirements: https://github.com/user/repo/blob/main/training.ipynb"* and the AI will automatically use the analysis tool.

**Testing MCP directly:**
```bash
# List available tools
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Analyze a notebook  
curl -X POST http://localhost:5001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call", 
    "params": {
      "name": "analyze_notebook",
      "arguments": {"url": "https://github.com/user/repo/blob/main/notebook.ipynb"}
    },
    "id": 2
  }'
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HOST` | Server bind address | `0.0.0.0` | No |
| `PORT` | Server port | `5000` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `SECRET_KEY` | Flask secret key | Auto-generated | No |
| `OPENAI_BASE_URL` | LLM API endpoint | None | No |
| `OPENAI_API_KEY` | LLM API key | None | No |
| `OPENAI_MODEL` | LLM model name | None | No |
| `GITHUB_TOKEN` | GitHub access token | None | No |
| `GITLAB_TOKEN` | GitLab access token | None | No |

### Quick Configuration Options

#### Option 1: Using .env file (Recommended)
```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
nano .env

# Start the application
docker compose up -d
```

#### Option 2: Shell environment variables
```bash
# LLM Enhancement
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1"

# Private Repository Access
export GITHUB_TOKEN="ghp_your_personal_access_token"
export GITLAB_TOKEN="glpat_your_personal_access_token"

# Start with current environment
docker compose up -d
```

#### Option 3: Inline with docker compose
```bash
OPENAI_API_KEY=your-key docker compose up -d
```

### LLM Provider Examples

#### NVIDIA NIM
```bash
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_API_KEY=your-nvidia-api-key
OPENAI_MODEL=nvidia/llama-3.1-nemotron-ultra-253b-v1
```

#### OpenAI
```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4
```

#### Local Ollama
```bash
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=dummy
OPENAI_MODEL=llama3:8b
```

## 🔒 Security Considerations

### Container Security
- Runs as non-root user (`appuser`)
- Minimal base image (Python 3.11 slim)
- No unnecessary packages or services
- Proper file permissions and isolation

### Application Security
- File upload restrictions (.ipynb and .py files only, 16MB limit)
- Secure filename handling
- Input validation and sanitization
- Temporary file cleanup

### Production Deployment
Consider these additional security measures:
- Use HTTPS with proper SSL certificates
- Implement rate limiting
- Add authentication/authorization if needed
- Use a reverse proxy (nginx, traefik)
- Regular security updates

## 📊 Sample Output

The web interface provides:
- **GPU Requirements**: Minimum and optimal configurations with VRAM and runtime estimates
- **Compliance Assessment**: NVIDIA notebook best practices scoring (0-100)
- **Detailed Analysis**: Reasoning, insights, and recommendations
- **Compatibility Info**: SXM requirements and ARM/Grace compatibility

## 🚧 Development

### Project Structure
```
notebook-analyzer-web/
├── app.py                    # Flask web application with MCP support
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container build instructions
├── docker-compose.yml       # Multi-service deployment
├── mcp_config.json          # MCP configuration for AI assistants
├── MCP_README.md            # MCP integration guide
├── .env.example             # Environment configuration template
├── templates/               # HTML templates
│   ├── base.html           # Base template with common layout
│   ├── index.html          # Main analysis form
│   ├── results.html        # Analysis results display
│   └── results_stream.html # Streaming analysis display
└── notebook-analyzer/       # Original analyzer code
    ├── notebook-analyzer.py # Core analysis engine
    ├── requirements.txt    # Original dependencies
    └── README.md          # Original documentation
```

### Adding Features
1. **New Analysis Features**: Extend the `GPUAnalyzer` class in `notebook-analyzer.py`
2. **UI Improvements**: Modify templates in the `templates/` directory
3. **API Endpoints**: Add new routes to `app.py`

### Testing
```bash
# Run the application locally
python app.py

# Test API endpoints
curl http://localhost:5001/health
curl -X POST http://localhost:5001/api/analyze -H "Content-Type: application/json" -d '{"url": "test-url"}'
```

## 🐳 Docker Deployment

### Production Docker Compose

```yaml
version: '3.8'
services:
  notebook-analyzer:
    image: notebook-analyzer-web:latest
    ports:
      - "5000:5000"
    environment:
      - HOST=0.0.0.0
      - PORT=5000
      - DEBUG=false
      - SECRET_KEY=your-production-secret-key
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - notebook-analyzer
    restart: unless-stopped
```

### Health Monitoring
The application includes health check endpoints:
- **Health Check**: `GET /health` - Returns service status
- **Docker Health**: Automated container health monitoring

## 📄 License

This project is licensed under the Apache License 2.0 - see the original [LICENSE](notebook-analyzer/LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🆘 Troubleshooting

### Common Issues

**Container fails to start**:
- Check Docker daemon is running
- Verify port 5000 is not in use
- Check Docker logs: `docker-compose logs`

**Analysis fails**:
- Verify notebook URL is accessible
- Check file format (.ipynb or .py required)
- Review application logs for errors

**Private repo access fails**:
- Ensure tokens have proper permissions
- Verify token format and validity
- Check environment variable configuration

### Getting Help
- Check the application logs: `docker-compose logs notebook-analyzer-web`
- Review the original analyzer documentation: [notebook-analyzer/README.md](notebook-analyzer/README.md)
- Submit issues through the project repository 