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
**Expected Result**: Entry to mid-tier GPU recommendation (RTX 4070-4080)  
**Key Features**:
- Simple PyTorch neural network training
- Clear GPU usage patterns (.cuda(), .to(device))
- Training loop with optimizer, loss, backpropagation
- Demonstrates proper GPU requirement scaling

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

## Expected Behavior Changes

### Before GPU Detection Fix
- All notebooks got expensive defaults (L4 minimum, A100 SXM optimal)
- No distinction between CPU-only and GPU workloads

### After GPU Detection Fix  
- CPU-only notebooks: CPU-only recommendation
- Entry-level GPU workloads: RTX 4060-4070 
- Mid-tier workloads: RTX 4070-4080
- High-end workloads: RTX 4090-L4
- Enterprise workloads: A100 series

## Adding New Test Cases

When adding new test cases:
1. Give descriptive filenames (e.g., `test_transformer_inference.ipynb`)
2. Document the expected GPU requirements
3. Add explanation to this README
4. Test both static analysis and LLM enhancement behavior 