# Notebook Analyzer

A comprehensive tool for analyzing Jupyter notebooks to determine NVIDIA GPU requirements, runtime estimates, and compliance with NVIDIA notebook best practices.

## 🚀 Features

### GPU Requirements Analysis
- **Minimum & Optimal GPU Recommendations**: Get both cost-effective and performance-optimized hardware suggestions
- **VRAM Estimation**: Accurate memory requirements based on workload analysis
- **Runtime Predictions**: Estimated execution times for different GPU configurations
- **Multi-GPU Detection**: Identifies distributed training and model parallelism requirements
- **SXM Form Factor Analysis**: Determines when SXM GPUs with NVLink are beneficial

### Advanced Compatibility Assessment
- **ARM/Grace Compatibility**: Evaluates notebook compatibility with ARM-based systems
- **Workload Complexity Analysis**: Categorizes notebooks from simple inference to extreme training workloads
- **Memory Optimization Detection**: Identifies techniques like LoRA, quantization, gradient checkpointing

### NVIDIA Notebook Compliance
- **Structure & Layout Assessment**: Evaluates title format, introduction completeness, navigation
- **Content Quality Analysis**: Checks documentation ratio, code explanations, educational value
- **Technical Standards**: Reviews requirements management, environment variables, reproducibility
- **Compliance Scoring**: 0-100 score based on NVIDIA's official notebook guidelines

### LLM Enhancement (Optional)
- **Context-Aware Analysis**: Deep understanding of notebook intent and workflow
- **Enhanced Accuracy**: Combines static analysis with LLM reasoning for better recommendations
- **Compliance Evaluation**: Advanced assessment of content quality and best practices

### 🆕 Enhanced Input Support
- **Local File Analysis**: Analyze notebooks directly from your file system
- **Private Repository Access**: Built-in GitHub authentication for private repos
- **Automatic URL Handling**: Smart parsing of URLs with query parameters (no manual quoting needed)
- **Flexible Input**: Supports GitHub URLs, raw URLs, and local file paths
- **JSON Output**: Machine-readable output for automation and integration

## 📋 Requirements

- Python 3.8+
- Internet connection (for fetching remote notebooks and optional LLM analysis)
- OpenAI-compatible API access (optional, for enhanced analysis)
- GitHub Personal Access Token (optional, for private repositories)

## 🛠️ Installation

1. **Clone or download the script**:
   ```bash
   git clone https://github.com/nvidia-kejones/notebook-analyzer.git
   # or copy the script to your local machine
   ```

2. **Install dependencies**:
   ```bash
   cd notebook-analyzer
   pip install -r requirements.txt
   ```

3. **Optional: Configure Environment Variables**:
   ```bash
   # For LLM Enhancement
   export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1/"
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
# Public GitHub notebook
python notebook-analyzer.py https://github.com/user/repo/blob/main/notebook.ipynb

# Local notebook file
python notebook-analyzer.py ./my-notebook.ipynb
python notebook-analyzer.py /path/to/notebook.ipynb
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

# Self-hosted GitLab instance
export GITLAB_TOKEN=your_token_here
python notebook-analyzer.py https://gitlab.company.com/team/project/-/blob/develop/analysis.ipynb
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

### JSON Output for Automation
```bash
# Pure JSON output (no status messages)
python notebook-analyzer.py --json https://github.com/user/repo/blob/main/notebook.ipynb

# Pretty-printed JSON with verbose flag
python notebook-analyzer.py --json --verbose ./notebook.ipynb

# Pipeline integration with jq
python notebook-analyzer.py --json notebook.ipynb | jq '.min_vram_gb'

# Save results to file
python notebook-analyzer.py --json notebook.ipynb > analysis_results.json
```

## 📊 Sample Output

### Human-Readable Format (Default)

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
GPU REQUIREMENTS ANALYSIS
======================================================================

📊 MINIMUM REQUIREMENTS:
   GPU Type: L4
   Quantity: 1
   VRAM: 24 GB
   Estimated Runtime: 4.2 hours

🚀 OPTIMAL CONFIGURATION:
   GPU Type: A100 SXM 80G
   Quantity: 1
   VRAM: 80 GB
   Estimated Runtime: 1.1 hours

📋 NVIDIA NOTEBOOK COMPLIANCE: 78/100
🟡 Overall Quality Score

💡 ADDITIONAL INFO:
   SXM Form Factor Required: No
   ARM/Grace Compatibility: Likely Compatible
   Analysis Confidence: 87%
   LLM Enhanced: Yes

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

### JSON Format (--json flag)

```json
{
  "min_gpu_type": "L4",
  "min_quantity": 1,
  "min_vram_gb": 24,
  "optimal_gpu_type": "A100 SXM 80G",
  "optimal_quantity": 1,
  "optimal_vram_gb": 80,
  "min_runtime_estimate": "4.2 hours",
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
  "analysis_metadata": {
    "analyzed_url_or_path": "./fine-tune-analysis.ipynb",
    "timestamp": "2024-12-19T10:30:45.123456",
    "version": "3.0.0"
  }
}
```

## 🎯 Supported GPU Models

### Consumer GPUs
- RTX 50 Series: 5090 (32GB), 5080 (16GB)
- RTX 40 Series: 4090 (24GB), 4080 (16GB)
- RTX 30 Series: 3090 (24GB), 3080 (10GB)

### Professional GPUs
- RTX 6000 Pro Server/Workstation (48GB)
- L40S (48GB), L40 (48GB), L4 (24GB)

### Data Center GPUs
- **B200**: SXM (192GB), PCIe (192GB)
- **H200**: SXM (141GB), NVL (141GB)
- **H100**: SXM (80GB), PCIe (80GB), NVL (188GB)
- **A100**: SXM 80G/40G, PCIe 80G/40G
- **V100** (32GB), **T4** (16GB)

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_BASE_URL` | OpenAI API endpoint | No* | None |
| `OPENAI_API_KEY` | API authentication key | No* | None |
| `OPENAI_MODEL` | Model name to use | No | gpt-4 |
| `GITHUB_TOKEN` | GitHub Personal Access Token | No** | None |
| `GITLAB_TOKEN` | GitLab Personal Access Token | No*** | None |

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

### Static Analysis
- **Library Detection**: Identifies GPU frameworks (PyTorch, TensorFlow, etc.)
- **Model Pattern Recognition**: Detects specific models and their requirements
- **Batch Size Analysis**: Extracts and analyzes batch size implications
- **Training Detection**: Identifies training vs inference workloads
- **Multi-GPU Patterns**: Detects distributed computing requirements

### LLM Enhancement
- **Contextual Understanding**: Analyzes code intent and workflow context
- **Memory Optimization Detection**: Identifies advanced techniques
- **Workload Classification**: Better complexity assessment
- **Quality Evaluation**: Comprehensive notebook best practices review

### NVIDIA Compliance Evaluation
Based on official NVIDIA notebook guidelines:
- **Structure (25%)**: Title format, introduction, navigation, conclusion
- **Content (25%)**: Documentation ratio, explanations, educational value
- **Technical (25%)**: Requirements, environment variables, reproducibility
- **Brand (25%)**: NVIDIA messaging, professional presentation

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
- **Hardware Planning**: Determine optimal GPU configuration for projects
- **Cost Optimization**: Balance performance vs budget with min/optimal recommendations
- **Runtime Planning**: Estimate project completion times
- **Platform Selection**: Choose between different cloud GPU offerings
- **Local Development**: Analyze notebooks before committing to repositories
- **Automation Integration**: Use JSON output for CI/CD pipelines and tooling

### For NVIDIA Teams
- **Content Quality Assurance**: Ensure notebooks meet company standards
- **Launchable Validation**: Verify notebooks before publication
- **Developer Experience**: Improve notebook quality for better user experience
- **Brand Consistency**: Maintain unified voice across NVIDIA content
- **Private Repository Analysis**: Analyze internal notebooks securely

### For DevOps & Infrastructure
- **Resource Allocation**: Plan GPU cluster requirements
- **Performance Monitoring**: Validate actual vs predicted performance
- **Platform Compatibility**: Assess ARM/Grace system compatibility
- **Batch Analysis**: Process multiple notebooks for infrastructure planning
- **Automated Workflows**: Integrate JSON output into monitoring and provisioning systems
- **Cost Management**: Aggregate VRAM requirements across projects for budgeting

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

MIN_VRAM=$(echo "$RESULT" | jq '.min_vram_gb')
if (( MIN_VRAM > 32 )); then
  echo "⚠️ High VRAM requirement detected: ${MIN_VRAM}GB"
fi

echo "✅ Notebook analysis passed"
```

### Batch Processing
```bash
# Analyze multiple notebooks and aggregate results
for notebook in notebooks/*.ipynb; do
  python notebook-analyzer.py --json "$notebook" >> batch_results.jsonl
done

# Extract insights with jq
jq -s 'map(select(.nvidia_compliance_score < 80))' batch_results.jsonl > low_compliance.json
jq -s 'map(.min_vram_gb) | add / length' batch_results.jsonl  # Average VRAM requirement
```

### Monitoring Integration
```bash
# Send results to monitoring system
RESULT=$(python notebook-analyzer.py --json notebook.ipynb)
curl -X POST https://monitoring.company.com/metrics \
  -H "Content-Type: application/json" \
  -d "$RESULT"
```

### Infrastructure Provisioning
```python
import subprocess
import json

def analyze_and_provision(notebook_path):
    result = subprocess.run(
        ["python", "notebook-analyzer.py", "--json", notebook_path],
        capture_output=True, text=True
    )
    
    analysis = json.loads(result.stdout)
    
    # Auto-configure cloud instance
    gpu_type = analysis["optimal_gpu_type"]
    vram_gb = analysis["optimal_vram_gb"]
    
    instance_config = {
        "gpu_type": gpu_type,
        "vram_requirement": vram_gb,
        "estimated_runtime": analysis["optimal_runtime_estimate"]
    }
    
    return provision_cloud_instance(instance_config)
```

- **Estimation Accuracy**: Runtime estimates are approximations based on patterns
- **Dynamic Content**: Cannot analyze notebooks with runtime-dependent behavior
- **API Dependencies**: LLM features require internet access and API availability
- **Language Support**: Primarily designed for Python notebooks
- **Complex Workflows**: May not capture intricate distributed training setups
- **Token Expiration**: GitHub/GitLab tokens may expire and need renewal
- **Platform Differences**: GitLab and GitHub have different URL formats and authentication methods

## 🚨 Limitations

### Common Issues

**❌ "Not found (404)" for GitHub/GitLab URLs**
- Repository may be private (set `GITHUB_TOKEN` or `GITLAB_TOKEN`)
- File may not exist at the specified path
- Branch name may be incorrect
- For GitLab: ensure URL uses `/-/blob/` format

**❌ "Forbidden (403)" errors**
- Authentication required for private repository
- GitHub/GitLab token may be expired or have insufficient permissions
- Rate limiting (wait and retry)
- For GitLab: ensure token has `read_repository` scope

**❌ Shell quoting issues with URLs**
- Tool automatically handles most cases
- For complex URLs, the tool will reconstruct them automatically
- Use verbose mode (`-v`) to see URL reconstruction

**❌ Local file not found**
- Check file path spelling and existence
- Ensure file has `.ipynb` extension
- Verify file permissions

**❌ GitLab URL format issues**
- Ensure GitLab URLs use `/-/blob/` not `/blob/`
- Self-hosted GitLab instances should work with proper authentication

## 🛠️ Troubleshooting

This tool is designed to be extensible. Areas for contribution:
- Additional GPU model support
- Enhanced pattern recognition for new frameworks
- Improved runtime estimation models
- Better compliance rule definitions
- Support for additional notebook formats
- Enhanced GitHub authentication methods
- GitLab API integration improvements
- Support for additional Git platforms (Bitbucket, Azure DevOps)

## 📄 License

Apache 2.0 - For external use, please ensure compliance with relevant licensing terms.

## 🆘 Support

For issues with the tool:
1. **Check verbose output** (`-v`) for detailed analysis
2. **Verify environment variables** are set correctly (including `GITLAB_TOKEN` for GitLab repos)
3. **Test with a simple public notebook** first
4. **Check GitHub/GitLab token permissions** for private repositories
5. **Review the troubleshooting section** above

For NVIDIA-specific notebook guidelines, refer to the NVIDIA notebook standards documentation.

