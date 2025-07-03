# Analysis Confidence Examples

This directory contains example notebooks that demonstrate different levels of analysis confidence in the GPU Notebook Analyzer.

## Dynamic Confidence System

The analyzer uses a dynamic confidence scoring system that evaluates multiple factors to determine how confident it is in its GPU requirements analysis. The confidence score ranges from 10% to 100% and is based on:

1. **Workload Detection Quality** (0-25%): How clearly GPU workload patterns are identified
2. **Framework Detection** (0-20%): Recognition of ML/DL frameworks
3. **Model Identification** (0-20%): Specific model architectures detected
4. **VRAM Estimation** (0-15%): Confidence in memory requirements
5. **Multi-GPU Detection** (0-10%): Multi-GPU setup identification
6. **LLM Enhancement** (variable): Agreement between static and LLM analysis
7. **Pattern Clarity** (variable): Penalties for uncertainty, bonuses for definitive patterns

## Example Notebooks

### High Confidence Examples (70-90%)

- **`test_gpu_workload.ipynb`** - 75% confidence
  - Clear training workload patterns
  - Definitive GPU requirements
  - High LLM confidence

- **`enterprise_gpu_example.ipynb`** - 85% confidence  
  - Multiple frameworks detected
  - Specific model architectures
  - Clear enterprise-scale requirements

### Medium Confidence Examples (40-70%)

- **`low_confidence_example.ipynb`** - 35% confidence
  - ML libraries imported but no active workload
  - Ambiguous GPU requirements
  - Exploratory/uncertain language

- **`very_low_confidence_example.ipynb`** - 40% confidence
  - No GPU workload detected
  - Basic data processing only
  - No ML frameworks actively used

### Low Confidence Examples (10-40%)

- **`jupyter_example.ipynb`** - 30% confidence
  - Minimal GPU patterns
  - Unclear workload type
  - Basic notebook structure

## Testing Confidence Levels

You can test these examples using the API:

```bash
# Test high confidence example
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@examples/test_gpu_workload.ipynb" | jq '.analysis.confidence'

# Test low confidence example  
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@examples/low_confidence_example.ipynb" | jq '.analysis.confidence'
```

## Confidence Factors

Each analysis includes detailed confidence factors that explain why the confidence is high or low:

```bash
# View confidence factors
curl -X POST http://localhost:8080/api/analyze \
  -F "file=@examples/low_confidence_example.ipynb" | jq '.analysis.confidence_factors'
```

Example output:
```json
[
  "No GPU workload detected",
  "No clear framework identification", 
  "No specific models identified",
  "VRAM requirement estimated",
  "Single-GPU workload",
  "LLM analysis has low confidence"
]
```

## Creating Your Own Examples

To create notebooks that demonstrate specific confidence levels:

### For Lower Confidence:
- Use commented-out ML imports
- Include uncertain language ("might", "could", "possibly")
- Avoid specific model names or training patterns
- Focus on data exploration rather than ML execution

### For Higher Confidence:
- Include active ML training/inference code
- Use specific model architectures (BERT, GPT, etc.)
- Include clear GPU operations (`.cuda()`, device assignments)
- Use definitive language about requirements
- Include memory-intensive operations

## Confidence Interpretation

- **90-100%**: Extremely confident - clear GPU workload with specific requirements
- **70-89%**: High confidence - definitive patterns with good framework detection
- **50-69%**: Moderate confidence - some uncertainty but clear ML intent
- **30-49%**: Low confidence - ambiguous patterns, exploratory code
- **10-29%**: Very low confidence - minimal or no GPU workload detected

The dynamic confidence system helps users understand how reliable the GPU recommendations are and whether they should seek additional validation for their specific use case. 