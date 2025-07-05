# Notebook Analyzer

A comprehensive tool for analyzing Jupyter notebooks (.ipynb) and marimo notebooks (.py) to determine NVIDIA GPU requirements, runtime estimates, and compliance with NVIDIA notebook best practices.

**Available as both CLI tool and web application with MCP (Model Context Protocol) support for AI assistant integration.**

## 🚀 Features

### GPU Requirements Analysis
- **3-Tier GPU Recommendations**: Get comprehensive hardware suggestions across three tiers:
  - **Minimum**: Entry-level viable option (lowest cost that works)
  - **Recommended**: Balanced price/performance (best value for most users)
  - **Optimal**: High performance option (best performance regardless of cost)
- **VRAM Estimation**: Accurate memory requirements based on workload analysis
- **Runtime Predictions**: Estimated execution times for different GPU configurations
- **Multi-GPU Detection**: Identifies distributed training and model parallelism requirements
- **SXM Form Factor Analysis**: Determines when SXM GPUs with NVLink are beneficial

### Advanced Compatibility Assessment
- **ARM/Grace Compatibility**: Evaluates notebook compatibility with ARM-based systems
- **Workload Complexity Analysis**: Categorizes notebooks from simple inference to extreme training workloads
- **Memory Optimization Detection**: Identifies techniques like LoRA, quantization, gradient checkpointing

### NVIDIA Notebook Compliance
- **Comprehensive Best Practices Integration**: Loads official NVIDIA guidelines from `analyzer/nvidia_best_practices.md`
- **Structure & Layout Assessment**: Evaluates title format, introduction completeness, navigation, conclusions
- **Content Quality Analysis**: Checks documentation ratio, code explanations, educational value, professional writing
- **Technical Standards**: Reviews requirements management, environment variables, reproducibility, file complexity
- **Enhanced Compliance Scoring**: 0-100 score based on NVIDIA's official notebook guidelines with detailed criteria
- **Guidelines-Based Evaluation**: Both static analysis and LLM evaluation use comprehensive NVIDIA standards

### LLM Enhancement (Optional)
- **Context-Aware Analysis**: Deep understanding of notebook intent and workflow
- **Enhanced Accuracy**: Combines static analysis with LLM reasoning for better recommendations
- **Compliance Evaluation**: Advanced assessment of content quality and best practices

### 🆕 Enhanced Input Support
- **Dual Format Support**: Analyze both Jupyter (.ipynb) and marimo (.py) notebooks seamlessly
- **Local File Analysis**: Analyze notebooks directly from your file system
- **Private Repository Access**: Built-in GitHub authentication for private repos
- **Automatic URL Handling**: Smart parsing of URLs with query parameters (no manual quoting needed)
- **Flexible Input**: Supports GitHub URLs, raw URLs, and local file paths for both notebook formats
- **JSON Output**: Machine-readable output for automation and integration

## 📋 Requirements

- Python 3.8+
- Internet connection (for fetching remote notebooks and optional LLM analysis)
- OpenAI-compatible API access (optional, for enhanced analysis)
- GitHub Personal Access Token (optional, for private repositories)
- Docker (optional, for web interface deployment)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nvidia-kejones/notebook-analyzer.git
   ```

2. **Install dependencies**:
   ```bash
   cd notebook-analyzer
   # highly recommend using a virtual env
   pip install -r requirements.txt
   ```

3. **Optional: Configure Environment Variables**:
   ```bash
   # For LLM Enhancement
   export OPENAI_BASE_URL="https://integrate.api.nvidia.com"  # NOTE: Do NOT include /v1 - it's added automatically
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1"  # optional, defaults to gpt-4
   
   # For Private GitHub Repositories
   export GITHUB_TOKEN="ghp_your_personal_access_token"
   
   # For Private GitLab Repositories
   export GITLAB_TOKEN="glpat_your_personal_access_token"
   ```

## 📖 Usage

### Basic Analysis
```bash
# Public GitHub Jupyter notebook
python notebook-analyzer.py https://github.com/user/repo/blob/main/notebook.ipynb

# Public GitHub marimo notebook
python notebook-analyzer.py https://github.com/user/repo/blob/main/analysis.py

# Local notebook files
python notebook-analyzer.py ./my-notebook.ipynb
python notebook-analyzer.py ./my-marimo-app.py
python notebook-analyzer.py /path/to/notebook.ipynb
```

The analyzer provides **3-tier GPU recommendations** for every workload:

**Example Output:**
```
📊 MINIMUM Configuration:
   GPU Type: RTX 4060
   Quantity: 1
   VRAM: 8 GB
   Estimated Runtime: 1-3 hours

🎯 RECOMMENDED Configuration:
   GPU Type: RTX 4080
   Quantity: 1
   VRAM: 16 GB
   Estimated Runtime: 30-60 minutes

🚀 OPTIMAL CONFIGURATION:
   GPU Type: L40S
   Quantity: 1
   VRAM: 48 GB
   Estimated Runtime: 5-15 minutes
```

### Private Repository Access
```bash
# Set GitHub token for private repos
export GITHUB_TOKEN=ghp_your_personal_access_token
python notebook-analyzer.py https://github.com/private-org/private-repo/blob/main/notebook.ipynb

# Set GitLab token for private repos
export GITLAB_TOKEN=glpat_your_personal_access_token
python notebook-analyzer.py https://gitlab.com/private-group/private-project/-/blob/main/notebook.ipynb
```

### GitLab Repository Analysis
```bash
# Public GitLab repository
python notebook-analyzer.py https://gitlab.com/user/repo/-/blob/main/notebook.ipynb

# Self-hosted GitLab instance (Jupyter)
export GITLAB_TOKEN=your_token_here
python notebook-analyzer.py https://gitlab.company.com/team/project/-/blob/develop/analysis.ipynb

# Self-hosted GitLab instance (marimo)
export GITLAB_TOKEN=your_token_here
python notebook-analyzer.py https://gitlab.company.com/team/project/-/blob/develop/marimo-app.py
```

### Raw URLs with Authentication Tokens
```bash
# No manual quoting needed - tool handles it automatically!
python notebook-analyzer.py https://raw.githubusercontent.com/org/repo/file.ipynb?token=GHSAT0AAAAAADDFMT5FA...
```

### Verbose Analysis with Detailed Reasoning
```bash
python notebook-analyzer.py -v https://github.com/user/repo/blob/main/notebook.ipynb
```

### With LLM Enhancement
```bash
export OPENAI_BASE_URL="https://integrate.api.nvidia.com"
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="nvidia/llama-3.1-nemotron-ultra-253b-v1"
python notebook-analyzer.py -v https://github.com/brevdev/launchables/blob/main/llama3_finetune_inference.ipynb
```

### Using Local LLM (e.g., Ollama)
```bash
export OPENAI_BASE_URL="http://localhost:11434"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="llama3:8b"
python notebook-analyzer.py -v ./local-notebook.ipynb
```

## 🔧 Troubleshooting

### JSON Output for Automation
```bash
# Pure JSON output (no status messages)
python notebook-analyzer.py --json https://github.com/user/repo/blob/main/notebook.ipynb

# Pretty-printed JSON with verbose flag
python notebook-analyzer.py --json --verbose ./notebook.ipynb

# Pipeline integration with jq - extract all 3 tiers
python notebook-analyzer.py --json notebook.ipynb | jq '.min_gpu_type, .recommended_gpu_type, .optimal_gpu_type'

# Extract VRAM requirements for each tier
python notebook-analyzer.py --json notebook.ipynb | jq '.min_vram_gb, .recommended_vram_gb, .optimal_vram_gb'

# Save results to file
python notebook-analyzer.py --json notebook.ipynb > analysis_results.json
```

**JSON Structure:**
```json
{
  "min_gpu_type": "RTX 4060",
  "min_quantity": 1,
  "min_vram_gb": 8,
  "min_runtime_estimate": "1-3 hours",
  "recommended_gpu_type": "RTX 4080", 
  "recommended_quantity": 1,
  "recommended_vram_gb": 16,
  "recommended_runtime_estimate": "30-60 minutes",
  "optimal_gpu_type": "L40S",
  "optimal_quantity": 1,
  "optimal_vram_gb": 48,
  "optimal_runtime_estimate": "5-15 minutes",
  "confidence": 0.85,
  "nvidia_compliance_score": 75
}
```

## 🌐 Web Interface & API

### Web Application
The analyzer includes a modern web interface with real-time streaming analysis and MCP support:

```bash
# Start web interface (requires Flask dependencies)
python app.py

# Or use Docker for easy deployment
docker-compose up --build
```

**Features:**
- Modern responsive UI with Bootstrap
- Dual input methods (URL or file upload)  
- Real-time streaming analysis with progress updates
- Comprehensive results display
- RESTful API endpoints
- **MCP (Model Context Protocol) integration** for AI assistants

### Docker Deployment

#### Using Docker Compose (Recommended)

1. **Clone and build**:
   ```bash
   git clone https://github.com/nvidia-kejones/notebook-analyzer.git
   cd notebook-analyzer
   docker-compose up --build
   ```

2. **Access the application**:
   Open your browser to [http://localhost:8080](http://localhost:8080)

#### Using Docker directly

```bash
# Build the image
docker build -t notebook-analyzer-web .

# Run the container
docker run -p 5000:5000 notebook-analyzer-web
```

### API Usage

#### REST API

**Analyze from URL**
```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/user/repo/blob/main/notebook.ipynb"}'
```

**Analyze uploaded file**
```bash
# Jupyter notebook
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@your-notebook.ipynb"

# marimo notebook
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@your-marimo-app.py"
```

### MCP (Model Context Protocol) Integration

The service supports MCP for AI assistant integration! This allows AI assistants like Claude to directly analyze notebooks through standardized tool calls.

#### Quick MCP Setup

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
   curl -X POST http://localhost:8080/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
   ```

#### Available MCP Tools

1. **`analyze_notebook`** - Complete notebook analysis
   - **Parameters:**
     - `url` (required): URL to Jupyter or marimo notebook
     - `include_reasoning` (optional): Include detailed reasoning
     - `include_compliance` (optional): Include NVIDIA compliance assessment
   - **Returns:** 3-tier GPU requirements, compliance score, runtime estimates

2. **`get_gpu_recommendations`** - Workload-specific recommendations  
   - **Parameters:**
     - `workload_type` (required): `inference`, `training`, or `fine-tuning`
     - `model_size` (optional): `small`, `medium`, `large`, or `xlarge`
     - `batch_size` (optional): Expected batch size
   - **Returns:** Recommended GPU configuration

#### MCP Usage Examples

```bash
# Analyze a notebook
curl -X POST http://localhost:8080/mcp \
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
curl -X POST http://localhost:8080/mcp \
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

#### AI Assistant Integration

Once connected via MCP, you can simply ask your AI assistant:

- *"What GPU requirements does this notebook have?"* (provide URL)
- *"Analyze this training notebook for NVIDIA compliance"*
- *"What GPUs do I need for fine-tuning a large model with batch size 8?"*
- *"Compare the GPU requirements between these two notebooks"*

The AI will automatically use the appropriate tools to provide detailed analysis.

### Vercel Deployment

The Flask application can be deployed to Vercel, but with important limitations due to the serverless nature of Vercel's platform.

#### ⚠️ Important Limitations

- **Execution Time**: Hobby Plan (10s max), Pro Plan (60s max)
- **Memory Usage**: Large notebooks may hit memory limits
- **File Processing**: Uses `/tmp` directory (max 500MB)

#### Deployment Steps

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Configure Environment Variables**:
   ```bash
   vercel env add OPENAI_BASE_URL
   vercel env add OPENAI_API_KEY
   vercel env add OPENAI_MODEL
   vercel env add GITHUB_TOKEN
   vercel env add GITLAB_TOKEN
   ```

3. **Deploy**:
   ```bash
   vercel --prod
   ```

4. **Test the Deployment**:
   ```bash
   curl https://your-app.vercel.app/health
   ```

#### When NOT to Use Vercel

Consider alternatives if you need:
- Long-running analysis (>60 seconds)
- Large file processing (>16MB notebooks)
- Persistent storage between requests
- Complex multi-step workflows

#### Recommended Alternatives

- **Railway** - Better for long-running processes
- **Render** - Good for background processing  
- **Google Cloud Run** - Flexible container deployment
- **Docker** - Full control with unlimited resources

## 📊 Sample Output

### Human-Readable Format (Default)

#### GPU Workload Example
```
✅ LLM enhancement enabled using nvidia/llama-3.1-nemotron-ultra-253b-v1
✅ GitHub authentication enabled
📁 Loading local notebook: ./fine-tune-analysis.ipynb
✅ Successfully loaded local notebook
🤖 Enhancing analysis with LLM...
✅ LLM analysis complete (confidence: 87%)
📋 Evaluating NVIDIA compliance...
✅ Compliance evaluation complete (score: 78/100)

======================================================================
📋 ENHANCED GPU REQUIREMENTS ANALYSIS
🎯 WITH NVIDIA BEST PRACTICES COMPLIANCE
======================================================================

📊 MINIMUM Configuration:
   GPU Type: L4
   Quantity: 1
   VRAM: 24 GB
   Estimated Runtime: 4.2 hours

🎯 RECOMMENDED Configuration:
   GPU Type: RTX 4090
   Quantity: 1
   VRAM: 24 GB
   Estimated Runtime: 2.1 hours

🚀 OPTIMAL CONFIGURATION:
   GPU Type: A100 SXM 80G
   Quantity: 1
   VRAM: 80 GB
   Estimated Runtime: 1.1 hours

📋 NVIDIA NOTEBOOK COMPLIANCE: 78/100
🟡 Good - Generally follows NVIDIA best practices

💡 ADDITIONAL INFO:
   SXM Form Factor Required: No
   ARM/Grace Compatibility: Likely Compatible
   Analysis Confidence: 87%
   LLM Enhanced: Yes
   NVIDIA Best Practices: ✅ Loaded

📚 Structure & Layout Assessment:
     Title: ✅ Good title format
     Introduction: ⚠️ Introduction present but could be enhanced
     Navigation: ✅ Good use of headers for navigation
     Conclusion: ✅ Has summary/conclusion

🎯 Content Quality Recommendations:
     • Consider adding more links to relevant documentation
     • Some code cells lack explanatory text

🔧 Technical Standards Recommendations:
     • Pin package versions (e.g., torch==2.1.0)
     • Set seeds for reproducibility

🤖 LLM Analysis Insights:
     • LLM estimated 28GB vs static analysis 16GB
     • Memory optimizations detected: LoRA, gradient_checkpointing
     • LLM identified workload complexity: complex
======================================================================
```

#### CPU-Only Workload Example
```
✅ GitHub authentication enabled
📁 Loading local notebook: ./data-analysis.ipynb
✅ Successfully loaded local notebook
🤖 Enhancing analysis with LLM...
✅ LLM analysis complete (confidence: 92%)
📋 Evaluating NVIDIA compliance...
✅ Compliance evaluation complete (score: 65/100)

======================================================================
📋 ENHANCED GPU REQUIREMENTS ANALYSIS
🎯 WITH NVIDIA BEST PRACTICES COMPLIANCE
======================================================================

💻 CPU-OPTIMIZED WORKLOAD:
   This notebook is designed for CPU execution and does not require GPU acceleration.
   Estimated Runtime: 5-15 minutes
   GPU Required: No

📋 NVIDIA NOTEBOOK COMPLIANCE: 65/100
🟠 Fair - Some improvements needed for NVIDIA standards

💡 ADDITIONAL INFO:
   SXM Form Factor Required: No
   ARM/Grace Compatibility: Likely Compatible
   Analysis Confidence: 92%
   LLM Enhanced: Yes
   NVIDIA Best Practices: ✅ Loaded

📚 Structure & Layout Assessment:
     Title: ⚠️ Consider NVIDIA-style title format
     Introduction: ✅ Good introduction
     Navigation: ✅ Good use of headers for navigation
     Conclusion: ⚠️ Could benefit from stronger conclusion

🎯 Content Quality Recommendations:
     • Add more explanatory text for code cells
     • Include links to relevant documentation

🔧 Technical Standards Recommendations:
     • Pin package versions (e.g., pandas==1.5.3)
     • Add requirements.txt file

🤖 LLM Analysis Insights:
     • Detected pandas, numpy, matplotlib usage (CPU-optimized)
     • No GPU-accelerated operations found
     • Workload complexity: simple data analysis
======================================================================
```

### JSON Format (--json flag)

#### GPU Workload JSON Example
```json
{
  "min_gpu_type": "L4",
  "min_quantity": 1,
  "min_vram_gb": 24,
  "min_runtime_estimate": "4.2 hours",
  "recommended_gpu_type": "RTX 4090",
  "recommended_quantity": 1,
  "recommended_vram_gb": 24,
  "recommended_runtime_estimate": "2.1 hours",
  "recommended_viable": true,
  "recommended_limitation": null,
  "optimal_gpu_type": "A100 SXM 80G",
  "optimal_quantity": 1,
  "optimal_vram_gb": 80,
  "optimal_runtime_estimate": "1.1 hours",
  "sxm_required": false,
  "sxm_reasoning": [],
  "arm_compatibility": "Likely Compatible",
  "arm_reasoning": ["Uses 3 ARM-compatible frameworks"],
  "confidence": 0.87,
  "reasoning": ["LLM: LLM estimated 28GB vs static analysis 16GB"],
  "llm_enhanced": true,
  "llm_reasoning": ["Memory optimizations detected: LoRA, gradient_checkpointing"],
  "nvidia_compliance_score": 78.0,
  "structure_assessment": {
    "title": "✅ Good title format",
    "introduction": "⚠️ Introduction present but could be enhanced",
    "navigation": "✅ Good use of headers for navigation",
    "conclusion": "✅ Has summary/conclusion"
  },
  "content_quality_issues": [
    "Consider adding more links to relevant documentation",
    "Some code cells lack explanatory text"
  ],
  "technical_recommendations": [
    "Pin package versions (e.g., torch==2.1.0)",
    "Set seeds for reproducibility"
  ],
  "workload_detected": true,
  "analysis_metadata": {
    "analyzed_url_or_path": "./fine-tune-analysis.ipynb",
    "timestamp": "2024-12-19T10:30:45.123456",
    "version": "3.1.0",
    "enhanced_features": "NVIDIA Best Practices Integration"
  }
}
```

#### CPU-Only Workload JSON Example
```json
{
  "min_gpu_type": "CPU-only",
  "min_quantity": 0,
  "min_vram_gb": 0,
  "min_runtime_estimate": "CPU execution",
  "recommended_gpu_type": null,
  "recommended_quantity": null,
  "recommended_vram_gb": null,
  "recommended_runtime_estimate": null,
  "recommended_viable": false,
  "recommended_limitation": "CPU-optimized workload - GPU not needed",
  "optimal_gpu_type": "CPU-only",
  "optimal_quantity": 0,
  "optimal_vram_gb": 0,
  "optimal_runtime_estimate": "CPU execution",
  "sxm_required": false,
  "sxm_reasoning": [],
  "arm_compatibility": "Likely Compatible",
  "arm_reasoning": ["Uses CPU-optimized libraries"],
  "confidence": 0.92,
  "reasoning": ["No GPU workload detected", "CPU-optimized libraries: pandas, numpy, matplotlib"],
  "llm_enhanced": true,
  "llm_reasoning": ["Detected pandas, numpy, matplotlib usage (CPU-optimized)", "No GPU-accelerated operations found"],
  "nvidia_compliance_score": 65.0,
  "structure_assessment": {
    "title": "⚠️ Consider NVIDIA-style title format",
    "introduction": "✅ Good introduction",
    "navigation": "✅ Good use of headers for navigation",
    "conclusion": "⚠️ Could benefit from stronger conclusion"
  },
  "content_quality_issues": [
    "Add more explanatory text for code cells",
    "Include links to relevant documentation"
  ],
  "technical_recommendations": [
    "Pin package versions (e.g., pandas==1.5.3)",
    "Add requirements.txt file"
  ],
  "workload_detected": false,
  "analysis_metadata": {
    "analyzed_url_or_path": "./data-analysis.ipynb",
    "timestamp": "2024-12-19T10:30:45.123456",
    "version": "3.1.0",
    "enhanced_features": "NVIDIA Best Practices Integration"
  }
}
```

## 🎯 Supported GPU Models

### Consumer GPUs
- RTX 50, 40, and 30 Series (various VRAM configurations)

### Data Center GPUs
- **B200**: SXM (192GB) - Dual-GPU design
- **H200**: SXM (141GB), NVL (141GB)
- **H100**: SXM (80GB), PCIe (80GB), NVL (94GB)
- **A100**: SXM 80G/40G, PCIe 80G/40G
- **L40S** (48GB), **L40** (48GB), **L4** (24GB)

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_BASE_URL` | OpenAI API endpoint | No* | None |
| `OPENAI_API_KEY` | API authentication key | No* | None |
| `OPENAI_MODEL` | Model name to use | No | gpt-4 |
| `GITHUB_TOKEN` | GitHub Personal Access Token | No** | None |
| `GITLAB_TOKEN` | GitLab Personal Access Token | No*** | None |
| `HOST` | Web server bind address | No | 0.0.0.0 |
| `PORT` | Web server port | No | 5000 |
| `DEBUG` | Enable debug mode | No | false |
| `SECRET_KEY` | Flask secret key | No | Auto-generated |

*Required for LLM enhancement  
**Required for private GitHub repositories  
***Required for private GitLab repositories

### Command Line Arguments

```bash
python notebook-analyzer.py [-h] [-v] [-j] [URL_OR_PATH ...]

positional arguments:
  URL_OR_PATH          URL to notebook, local file path, or multiple URL fragments

optional arguments:
  -h, --help          show help message and exit
  -v, --verbose       verbose output with detailed reasoning
  -j, --json          output results in JSON format (pure JSON, no status messages)
```

### 🔧 Automatic URL Handling Examples

The tool automatically handles complex URLs that might be split by the shell:

```bash
# These all work without manual quoting:
python notebook-analyzer.py https://raw.githubusercontent.com/repo/file.ipynb?token=abc123&ref=main
python notebook-analyzer.py https://github.com/org/repo/blob/feature/branch-name/notebook.ipynb
python notebook-analyzer.py ./notebooks/analysis.ipynb

# JSON output examples:
python notebook-analyzer.py --json https://github.com/user/repo/blob/main/notebook.ipynb
python notebook-analyzer.py --json --verbose ./local-notebook.ipynb
```

## 🧠 Analysis Methodology

### 3-Tier GPU Recommendation System
Our core analysis provides three distinct hardware recommendations:

- **🟢 Minimum**: Entry-level viable option (lowest cost that works)
- **🟡 Recommended**: Balanced price/performance (best value for most users)  
- **🔴 Optimal**: High performance option (best performance regardless of cost)

**CPU-First Analysis**: Automatically detects CPU-only workloads (basic Python, simple data analysis) and correctly recommends no GPU when appropriate.

### Multi-Phase Analysis Pipeline

#### Phase 1: Static Analysis
- **Framework Detection**: Identifies GPU-accelerated libraries (PyTorch, TensorFlow, RAPIDS, etc.)
- **Model Pattern Recognition**: Detects specific models (BERT, GPT, LLaMA, ResNet, etc.) and estimates requirements
- **Workload Classification**: Distinguishes between inference, training, fine-tuning, and GPU computing
- **Multi-GPU Detection**: Identifies distributed training patterns and SXM requirements
- **VRAM Estimation**: Calculates memory requirements based on models, batch sizes, and optimizations
- **ARM/Grace Compatibility**: Evaluates compatibility with ARM-based systems

#### Phase 2: LLM Enhancement (Optional)
When OpenAI API is configured, adds intelligent analysis:
- **Contextual Understanding**: Analyzes code intent and workflow complexity
- **Memory Optimization Detection**: Identifies LoRA, quantization, gradient checkpointing
- **Tutorial vs Production Classification**: Distinguishes demos from production workloads
- **Runtime Estimation**: Provides realistic time estimates based on hardware performance
- **Confidence Calibration**: Adjusts confidence based on evidence quality

#### Phase 2.5: Self-Review (Development Mode)
Advanced consistency checking:
- **Accuracy Validation**: Reviews recommendations for logical consistency
- **3-Tier Compliance**: Ensures proper minimum → recommended → optimal progression
- **Workload Alignment**: Validates GPU recommendations match detected complexity
- **Error Correction**: Automatically fixes common analysis inconsistencies

#### Phase 3: NVIDIA Compliance Evaluation
Based on comprehensive NVIDIA notebook guidelines:
- **Structure (25%)**: Title format, introduction quality, navigation headers, conclusion
- **Content (25%)**: Documentation ratio, code explanations, educational value, external links
- **Technical (25%)**: Requirements management, reproducibility, GPU optimizations, file organization
- **Brand (25%)**: NVIDIA messaging, brand consistency, developer focus, maintenance quality

### Advanced Features

#### Smart Workload Detection
- **CPU-Only Recognition**: Identifies basic Python operations requiring no GPU
- **GPU Benefit Levels**: Classifies workloads as none/beneficial/recommended/required
- **Scale Assessment**: Distinguishes tutorial datasets from production workloads
- **Framework Analysis**: Detects specific ML/AI frameworks and their GPU requirements

#### Performance-Aware Recommendations
- **Relative Performance Factors**: Uses real-world GPU performance data
- **Runtime Estimation**: Provides time estimates across different hardware tiers
- **Multi-GPU Scaling**: Accounts for distributed training performance benefits
- **Form Factor Analysis**: Determines when SXM GPUs with NVLink are beneficial

#### Security & Reliability
- **Content Sanitization**: Secure processing of notebook content
- **Fallback Analysis**: Graceful degradation when LLM services unavailable
- **Rate Limiting**: Respects API limits with intelligent retry mechanisms
- **Error Handling**: Robust processing of malformed or incomplete notebooks

## 🔗 Supported Input Sources

### Local Files
```bash
python notebook-analyzer.py ./notebook.ipynb
python notebook-analyzer.py /Users/username/projects/analysis.ipynb
python notebook-analyzer.py ~/Documents/research/model-training.ipynb
```

### Public GitHub Repositories
```bash
python notebook-analyzer.py https://github.com/user/repo/blob/main/notebook.ipynb
```

### Public GitLab Repositories
```bash
python notebook-analyzer.py https://gitlab.com/user/repo/-/blob/main/notebook.ipynb
```

### Private GitHub Repositories
```bash
export GITHUB_TOKEN=ghp_your_personal_access_token
python notebook-analyzer.py https://github.com/private-org/repo/blob/branch/notebook.ipynb
```

### Private GitLab Repositories
```bash
export GITLAB_TOKEN=glpat_your_personal_access_token
python notebook-analyzer.py https://gitlab.com/private-group/repo/-/blob/branch/notebook.ipynb
```

### Self-Hosted GitLab Instances
```bash
export GITLAB_TOKEN=your_enterprise_token
python notebook-analyzer.py https://gitlab.company.com/team/project/-/blob/main/notebook.ipynb
```

### Raw URLs with Authentication
```bash
# GitHub raw URLs with tokens
python notebook-analyzer.py https://raw.githubusercontent.com/org/repo/file.ipynb?token=GHSAT0AAA...

# GitLab raw URLs  
python notebook-analyzer.py https://gitlab.com/user/repo/-/raw/main/notebook.ipynb
```

### GitHub URL Conversion
The tool automatically converts GitHub blob URLs to raw content:
```
https://github.com/user/repo/blob/main/notebook.ipynb
↓ (automatically converted to)
https://raw.githubusercontent.com/user/repo/main/notebook.ipynb
```

### GitLab URL Conversion
The tool automatically converts GitLab blob URLs to raw content:
```
https://gitlab.com/user/repo/-/blob/main/notebook.ipynb
↓ (automatically converted to)
https://gitlab.com/user/repo/-/raw/main/notebook.ipynb
```

## 🔐 Git Platform Authentication Setup

### GitHub Authentication

For analyzing private GitHub repositories, create a GitHub Personal Access Token:

1. **Go to GitHub Settings** → Developer settings → Personal access tokens → Tokens (classic)
2. **Generate new token** with `repo` scope
3. **Set environment variable**:
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```
4. **Verify access**:
   ```bash
   python notebook-analyzer.py https://github.com/your-private-org/private-repo/blob/main/notebook.ipynb
   ```

### GitLab Authentication

For analyzing private GitLab repositories (GitLab.com or self-hosted):

1. **Go to GitLab** → User Settings → Access Tokens (or Project Settings → Access Tokens)
2. **Create token** with `read_repository` scope
3. **Set environment variable**:
   ```bash
   export GITLAB_TOKEN=glpat_your_token_here
   ```
4. **Verify access**:
   ```bash
   python notebook-analyzer.py https://gitlab.com/your-private-group/private-project/-/blob/main/notebook.ipynb
   ```

### Self-Hosted GitLab

For enterprise GitLab instances:

1. **Create token** on your GitLab instance with appropriate permissions
2. **Set environment variable** with your enterprise token:
   ```bash
   export GITLAB_TOKEN=your_enterprise_token
   ```
3. **Analyze notebooks** from your self-hosted instance:
   ```bash
   python notebook-analyzer.py https://gitlab.company.com/team/project/-/blob/develop/analysis.ipynb
   ```

## 🎓 Use Cases

### For Data Scientists & ML Engineers
- **Hardware Planning**: Determine optimal GPU configuration for projects with 3-tier recommendations
- **Cost Optimization**: Balance performance vs budget with minimum/recommended/optimal configurations
- **Runtime Planning**: Get realistic execution time estimates for different GPU configurations
- **Platform Selection**: Choose between consumer and enterprise GPU offerings based on workload
- **Local Development**: Analyze notebooks before committing to repositories
- **Automation Integration**: Use JSON output for CI/CD pipelines and infrastructure provisioning
- **CPU-First Analysis**: Automatically detect CPU-only workloads to avoid unnecessary GPU costs

### For NVIDIA Teams & Partners
- **Content Quality Assurance**: Ensure notebooks meet NVIDIA's comprehensive standards (100-point scoring)
- **Launchable Validation**: Verify notebooks before publication with compliance scoring
- **Developer Experience**: Improve notebook quality with structure, content, and technical recommendations
- **Brand Consistency**: Maintain unified voice across NVIDIA content with best practices integration
- **Private Repository Analysis**: Analyze internal notebooks securely with GitHub/GitLab token support
- **Tutorial vs Production Classification**: Distinguish between demo content and production workloads

### For DevOps & Infrastructure Teams
- **Resource Allocation**: Plan GPU cluster requirements with accurate VRAM estimates
- **Performance Monitoring**: Validate actual vs predicted performance with runtime estimates
- **Platform Compatibility**: Assess ARM/Grace system compatibility automatically
- **Batch Analysis**: Process multiple notebooks for infrastructure planning and budgeting
- **Automated Workflows**: Integrate JSON output into monitoring, provisioning, and cost management systems
- **Security Compliance**: Use security sandbox for safe analysis of untrusted notebooks
- **Multi-Cloud Planning**: Get recommendations across consumer and enterprise GPU tiers

### For AI/ML Platform Providers
- **Resource Optimization**: Automatically right-size GPU instances based on workload analysis
- **Cost Management**: Provide users with minimum viable vs optimal configurations
- **Quality Gates**: Implement notebook quality checks in deployment pipelines
- **User Experience**: Guide users to appropriate hardware choices with confidence scoring
- **Compliance Monitoring**: Track notebook quality across teams and projects

## 🔄 Integration Examples

### CI/CD Pipeline Integration
```bash
#!/bin/bash
# Check notebook GPU compliance in CI/CD

RESULT=$(python notebook-analyzer.py --json "$NOTEBOOK_PATH")
COMPLIANCE_SCORE=$(echo "$RESULT" | jq '.nvidia_compliance_score')

if (( $(echo "$COMPLIANCE_SCORE < 70" | bc -l) )); then
  echo "❌ Notebook compliance score too low: $COMPLIANCE_SCORE"
  exit 1
fi

# Check if workload actually needs GPU
WORKLOAD_DETECTED=$(echo "$RESULT" | jq '.workload_detected')
if [ "$WORKLOAD_DETECTED" = "false" ]; then
  echo "💡 CPU-only workload detected - no GPU required"
  exit 0
fi

MIN_VRAM=$(echo "$RESULT" | jq '.min_vram_gb')
if (( MIN_VRAM > 32 )); then
  echo "⚠️ High VRAM requirement detected: ${MIN_VRAM}GB"
fi

echo "✅ Notebook analysis passed"
```

### Batch Processing for Infrastructure Planning
```bash
# Analyze multiple notebooks and aggregate results
for notebook in notebooks/*.ipynb notebooks/*.py; do
  python notebook-analyzer.py --json "$notebook" >> batch_results.jsonl
done

# Extract insights with jq
jq -s 'map(select(.nvidia_compliance_score < 80))' batch_results.jsonl > low_compliance.json
jq -s 'map(select(.workload_detected == true)) | map(.min_vram_gb) | add / length' batch_results.jsonl  # Average VRAM for GPU workloads
jq -s 'map(select(.workload_detected == false)) | length' batch_results.jsonl  # Count CPU-only workloads
```

### Monitoring Integration
```bash
# Send results to monitoring system
RESULT=$(python notebook-analyzer.py --json notebook.ipynb)
curl -X POST https://monitoring.company.com/metrics \
  -H "Content-Type: application/json" \
  -d "$RESULT"
```

### Infrastructure Provisioning with 3-Tier Support
```python
import subprocess
import json

def analyze_and_provision(notebook_path, tier="recommended"):
    result = subprocess.run(
        ["python", "notebook-analyzer.py", "--json", notebook_path],
        capture_output=True, text=True
    )
    
    analysis = json.loads(result.stdout)
    
    # Check if GPU is actually needed
    if not analysis.get("workload_detected", False):
        return provision_cpu_instance(analysis)
    
    # Auto-configure cloud instance based on tier
    if tier == "minimum":
        gpu_type = analysis["min_gpu_type"]
        vram_gb = analysis["min_vram_gb"]
        runtime = analysis["min_runtime_estimate"]
    elif tier == "optimal":
        gpu_type = analysis["optimal_gpu_type"]
        vram_gb = analysis["optimal_vram_gb"]
        runtime = analysis["optimal_runtime_estimate"]
    else:  # recommended
        gpu_type = analysis["recommended_gpu_type"] or analysis["min_gpu_type"]
        vram_gb = analysis["recommended_vram_gb"] or analysis["min_vram_gb"]
        runtime = analysis["recommended_runtime_estimate"] or analysis["min_runtime_estimate"]
    
    instance_config = {
        "gpu_type": gpu_type,
        "vram_requirement": vram_gb,
        "estimated_runtime": runtime,
        "confidence": analysis["confidence"],
        "compliance_score": analysis["nvidia_compliance_score"]
    }
    
    return provision_cloud_instance(instance_config)

def provision_cpu_instance(analysis):
    """Provision CPU-only instance for non-GPU workloads"""
    return {
        "instance_type": "cpu_optimized",
        "estimated_runtime": analysis["min_runtime_estimate"],
        "cost_savings": "Significant vs GPU deployment"
    }
```

### MCP (Model Context Protocol) Integration
```python
# Available as MCP server for AI assistants
# Endpoint: POST /mcp
# Supports Claude, ChatGPT, and other AI assistants

{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "analyze_notebook",
    "arguments": {
      "notebook_url": "https://github.com/user/repo/blob/main/notebook.ipynb"
    }
  }
}
```

## 🚨 Limitations

### Analysis Limitations
- **Estimation Accuracy**: Runtime estimates are approximations based on patterns and GPU performance data
- **Dynamic Content**: Cannot analyze notebooks with runtime-dependent behavior or dynamic imports
- **Language Support**: Primarily designed for Python notebooks (Jupyter .ipynb and marimo .py)
- **Complex Workflows**: May not capture intricate distributed training setups or custom CUDA kernels
- **Security Scope**: Security sandbox focuses on preventing code execution, not comprehensive security audit

### API and Authentication
- **API Dependencies**: LLM features require internet access and API availability
- **Token Expiration**: GitHub/GitLab tokens may expire and need renewal
- **Rate Limiting**: GitHub/GitLab APIs have rate limits that may affect batch processing
- **Platform Differences**: GitLab and GitHub have different URL formats and authentication methods

### Environment Constraints
- **Resource Limits**: Web interface has memory and time limits for large notebooks
- **Concurrent Analysis**: Limited concurrent analysis capacity in web deployment
- **File Size Limits**: Large notebooks (>10MB) may not be supported in web interface

## 🛠️ Troubleshooting

### Common Issues

**❌ "Not found (404)" for GitHub/GitLab URLs**
- Repository may be private (set `GITHUB_TOKEN` or `GITLAB_TOKEN`)
- File may not exist at the specified path
- Branch name may be incorrect in URL
- For GitLab: ensure URL uses `/-/blob/` format, not `/blob/`

**❌ "Forbidden (403)" errors**
- Authentication required for private repository
- GitHub/GitLab token may be expired or have insufficient permissions
- Rate limiting (wait and retry, or use different token)
- For GitLab: ensure token has `read_repository` scope

**❌ Shell quoting issues with URLs**
- Tool automatically handles most URL reconstruction cases
- For complex URLs with tokens, the tool will reconstruct them automatically
- Use verbose mode (`-v`) to see URL reconstruction process

**❌ Local file not found**
- Check file path spelling and existence
- Ensure file has `.ipynb` extension for Jupyter or `.py` for marimo notebooks
- Verify file permissions are readable

**❌ "CPU-only" results when expecting GPU recommendations**
- This is often correct - the tool detects truly CPU-only workloads
- Basic Python operations, simple data analysis don't need GPU acceleration
- Use verbose mode (`-v`) to see detailed reasoning

**❌ Low compliance scores**
- Check notebook structure: needs proper title, introduction, headers, conclusion
- Ensure adequate documentation-to-code ratio
- Add requirements.txt or pin package versions
- Include links to relevant documentation

**❌ LLM enhancement failures**
- Verify `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` environment variables
- Check API endpoint availability and model name
- Tool will fallback to static analysis if LLM fails

### Advanced Troubleshooting

**🔧 Web Interface Issues**
- Check browser developer tools for JavaScript errors
- Verify network connectivity for streaming analysis
- Large notebooks may timeout - try CLI version instead

**🔧 MCP Integration Issues**
- Ensure MCP endpoint is accessible at `/mcp`
- Check JSON-RPC 2.0 message format
- Verify AI assistant has proper MCP client implementation

**🔧 Security Sandbox Issues**
- Sandbox may reject notebooks with certain patterns
- Check verbose output for specific security concerns
- Some legitimate notebooks may trigger false positives

## 📈 Contributing

This tool is designed to be extensible. Areas for contribution:

### Core Analysis
- Additional GPU model support (new NVIDIA releases)
- Enhanced pattern recognition for emerging ML frameworks
- Improved runtime estimation models based on real benchmarks
- Better compliance rule definitions and scoring

### Platform Support
- Support for additional notebook formats (Observable, Databricks, etc.)
- Enhanced GitHub/GitLab authentication methods
- Support for additional Git platforms (Bitbucket, Azure DevOps)
- Integration with cloud ML platforms (SageMaker, Vertex AI)

### Security & Reliability
- Enhanced security sandbox capabilities
- Better handling of large notebooks and complex dependencies
- Improved error recovery and fallback mechanisms
- Performance optimizations for batch processing

### User Experience
- Interactive web interface improvements
- Better visualization of analysis results
- Enhanced progress reporting and streaming
- Mobile-responsive design improvements

## 📄 License

Apache 2.0 - For external use, please ensure compliance with relevant licensing terms.

## 🆘 Support

For issues with the tool:
1. **Check verbose output** (`-v`) for detailed analysis and reasoning
2. **Verify environment variables** are set correctly (including `GITLAB_TOKEN` for GitLab repos)
3. **Test with a simple public notebook** first to verify basic functionality
4. **Check GitHub/GitLab token permissions** for private repositories
5. **Review the troubleshooting section** above for common issues
6. **Check compliance score** if results seem unexpected - low scores indicate notebook quality issues

For NVIDIA-specific notebook guidelines, refer to the NVIDIA notebook standards documentation and the loaded best practices (`analyzer/nvidia_best_practices.md`).

## 🧪 Testing

The project includes comprehensive test scripts in the `tests/` directory:

```bash
# Run all tests
cd tests && ./test.sh

# Run quick tests only (faster, core functionality)
cd tests && ./test.sh --quick

# Test against different URL
cd tests && ./test.sh --url http://localhost:5000

# Run accuracy tests (validates analysis quality)
cd tests && ./test_accuracy.sh

# Test streaming functionality (web interface)
cd tests && ./test_streaming.sh

# Verify security headers (web deployment)
cd tests && ./verify_security_headers.sh
```

See `tests/README.md` for detailed testing documentation and test result interpretation.

