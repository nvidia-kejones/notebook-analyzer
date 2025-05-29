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
- **Compliance Scoring**: 0-100 score based on NVIDIA's notebook guidelines

### LLM Enhancement (Optional)
- **Context-Aware Analysis**: Deep understanding of notebook intent and workflow
- **Enhanced Accuracy**: Combines static analysis with LLM reasoning for better recommendations
- **Compliance Evaluation**: Advanced assessment of content quality and best practices

## 📋 Requirements

- Python 3.8+
- Internet connection (for fetching notebooks and optional LLM analysis)
- OpenAI-compatible API access (optional, for enhanced analysis)

## 🛠️ Installation

1. **Clone or download the script**:
   ```bash
   wget https://raw.githubusercontent.com/your-repo/notebook-analyzer.py
   # or copy the script to your local machine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Configure LLM Enhancement**:
   ```bash
   export OPENAI_BASE_URL="https://api.openai.com"
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="gpt-4"  # optional, defaults to gpt-4
   ```

## 📖 Usage

### Basic Analysis
```bash
python notebook-analyzer.py https://github.com/user/repo/blob/main/notebook.ipynb
```

### Verbose Analysis with Detailed Reasoning
```bash
python notebook-analyzer.py -v https://github.com/user/repo/blob/main/notebook.ipynb
```

### With LLM Enhancement
```bash
export OPENAI_BASE_URL="https://api.openai.com"
export OPENAI_API_KEY="your-api-key"
python notebook-analyzer.py -v https://github.com/user/repo/blob/main/notebook.ipynb
```

### Using Local LLM (e.g., Ollama)
```bash
export OPENAI_BASE_URL="http://localhost:11434"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="llama3:8b"
python notebook-analyzer.py -v https://github.com/user/repo/blob/main/notebook.ipynb
```

## 📊 Sample Output

```
✅ LLM enhancement enabled using gpt-4
Fetching notebook from: https://github.com/example/llama-finetune.ipynb
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

## 🎯 Supported GPU Models

### Consumer GPUs
- RTX 50 Series: 5090 (32GB), 5080 (16GB)
- RTX 40 Series: 4090 (24GB), 4080 (16GB)
- RTX 30 Series: 3090 (24GB), 3080 (10GB)

### Professional GPUs
- RTX 6000 Pro Server/Workstation (48GB)
- L40S (48GB), L40 (48GB), L4 (24GB)

### Data Center GPUs
- **H200**: SXM (141GB), NVL (141GB)
- **B200**: SXM (192GB), PCIe (192GB)
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

*Required for LLM enhancement

### Command Line Arguments

```bash
python notebook-analyzer.py [-h] [-v] URL

positional arguments:
  URL                   URL to the Jupyter notebook

optional arguments:
  -h, --help           show help message and exit
  -v, --verbose        verbose output with detailed reasoning
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

## 🔗 Supported Notebook Sources

- **GitHub**: Direct links to `.ipynb` files (automatically converts to raw URLs)
- **Raw URLs**: Direct access to notebook JSON content
- **Any HTTP/HTTPS URL** serving Jupyter notebook JSON format

### GitHub URL Conversion
The tool automatically converts GitHub blob URLs to raw content:
```
https://github.com/user/repo/blob/main/notebook.ipynb
↓ (automatically converted to)
https://raw.githubusercontent.com/user/repo/main/notebook.ipynb
```

## 🎓 Use Cases

### For Data Scientists & ML Engineers
- **Hardware Planning**: Determine optimal GPU configuration for projects
- **Cost Optimization**: Balance performance vs budget with min/optimal recommendations
- **Runtime Planning**: Estimate project completion times
- **Platform Selection**: Choose between different cloud GPU offerings

### For NVIDIA Teams
- **Content Quality Assurance**: Ensure notebooks meet company standards
- **Launchable Validation**: Verify notebooks before publication
- **Developer Experience**: Improve notebook quality for better user experience
- **Brand Consistency**: Maintain unified voice across NVIDIA content

### For DevOps & Infrastructure
- **Resource Allocation**: Plan GPU cluster requirements
- **Performance Monitoring**: Validate actual vs predicted performance
- **Platform Compatibility**: Assess ARM/Grace system compatibility

## 🚨 Limitations

- **Estimation Accuracy**: Runtime estimates are approximations based on patterns
- **Dynamic Content**: Cannot analyze notebooks with runtime-dependent behavior
- **API Dependencies**: LLM features require internet access and API availability
- **Language Support**: Primarily designed for Python notebooks
- **Complex Workflows**: May not capture intricate distributed training setups

## 🤝 Contributing

This tool is designed to be extensible. Areas for contribution:
- Additional GPU model support
- Enhanced pattern recognition
- Improved runtime estimation models
- Better compliance rule definitions
- Support for additional notebook formats

## 📄 License

This tool follows NVIDIA's internal development guidelines. For external use, please ensure compliance with relevant licensing terms.

## 🆘 Support

For issues with the tool:
1. Check verbose output for detailed analysis
2. Verify environment variable configuration
3. Test with a simple notebook first
4. Review the analysis methodology section

For NVIDIA-specific notebook guidelines, refer to the official NVIDIA notebook standards documentation.
