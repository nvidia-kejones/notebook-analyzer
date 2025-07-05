# Examples & Test Cases

This folder contains various notebook examples that serve as test cases for the NVIDIA Notebook Analyzer. Each example demonstrates different scenarios and helps verify the analyzer's GPU requirement detection capabilities.

## Test Cases

### `test_arm_problems.ipynb` - CPU-Only Detection Test
**Purpose**: Tests analyzer's ability to detect notebooks with no GPU workload  
**Expected Result**: CPU-only recommendation (no GPU required)  
**Key Features**:
- Contains only library imports and compatibility warnings
- No actual ML model training or inference
- Should trigger ARM compatibility warnings for legacy versions
- Demonstrates proper "no GPU workload" detection

### `test_gpu_workload.ipynb` - Basic GPU Training Test
**Purpose**: Tests analyzer's ability to detect legitimate GPU training workloads  
**Expected Result**: 3-tier GPU recommendations (RTX 4060 → RTX 5090 → H100 PCIe)  
**Key Features**:
- Simple PyTorch neural network training
- Clear GPU usage patterns (.cuda(), .to(device))
- Training loop with optimizer, loss, backpropagation
- Demonstrates proper 3-tier GPU requirement scaling

### `jupyter_example.ipynb` - Standard Jupyter Example
**Purpose**: Existing example notebook for general testing  
**Expected Result**: Variable depending on content  

### `marimo_example.py` - Marimo Notebook Example
**Purpose**: Tests analyzer's ability to parse Python-based notebooks  
**Expected Result**: Variable depending on content  

## Usage

Test the analyzer against these examples using the API:

```bash
# Test CPU-only detection
curl -X POST -F "file=@examples/test_arm_problems.ipynb" "http://localhost:8080/api/analyze"

# Test GPU workload detection  
curl -X POST -F "file=@examples/test_gpu_workload.ipynb" "http://localhost:8080/api/analyze"
```

## Expected Behavior - 3-Tier System

### CPU-Only Workloads
- **All Tiers**: CPU-only recommendation
- **Examples**: Basic Python, data analysis with pandas/numpy, simple plotting

### GPU Workloads - 3-Tier Recommendations

**Entry-level ML/Training:**
- **Minimum**: RTX 4060 (8GB)
- **Recommended**: RTX 4070 (12GB) 
- **Optimal**: RTX 4080 (16GB)

**Mid-scale Training:**
- **Minimum**: RTX 4070 (12GB)
- **Recommended**: RTX 4090 (24GB)
- **Optimal**: L40S (48GB)

**Large-scale Training:**
- **Minimum**: RTX 4090 (24GB)
- **Recommended**: L40S (48GB)
- **Optimal**: A100 PCIe 80G (80GB)

**Enterprise Production:**
- **Minimum**: L40S (48GB)
- **Recommended**: A100 PCIe 80G (80GB)
- **Optimal**: H100 PCIe (80GB)

## Adding New Test Cases

When adding new test cases:
1. Give descriptive filenames (e.g., `test_transformer_inference.ipynb`)
2. Document the expected GPU requirements
3. Add explanation to this README
4. Test both static analysis and LLM enhancement behavior 