# Testing Guide

This guide covers how to test the Notebook Analyzer application using the provided test suite.

## Test Script Overview

The unified test script `test.sh` provides comprehensive testing of all application functionality with flexible options for different testing scenarios.

## Prerequisites

Before running tests, ensure you have the required dependencies installed:

- **curl**: For making HTTP requests
- **jq**: For JSON parsing and validation

### Installation

**macOS:**
```bash
brew install curl jq
```

**Ubuntu/Debian:**
```bash
sudo apt-get install curl jq
```

**CentOS/RHEL:**
```bash
sudo yum install curl jq
```

## Usage

The test script supports multiple options for different testing scenarios:

```bash
./test.sh [--quick] [--url <base_url>] [--help]
```

### Options

- `--quick, -q`: Run quick tests only (health check, MCP tools, GPU recommendations)
- `--url, -u <url>`: Specify the base URL to test against (default: http://localhost:8080)
- `--help, -h`: Show help message with usage examples

### Examples

**Run all tests against default local server:**
```bash
./test.sh
```

**Run quick tests only:**
```bash
./test.sh --quick
```

**Test against a different port:**
```bash
./test.sh --url http://localhost:5000
```

**Quick tests against a remote deployment:**
```bash
./test.sh --quick --url https://my-notebook-analyzer.vercel.app
```

**Test against production Vercel deployment:**
```bash
./test.sh --url https://your-app-name.vercel.app
```

## Test Categories

### Quick Tests (--quick)
Essential tests that verify core functionality:
1. **Health Check**: Verifies the web UI is accessible
2. **MCP Tools List**: Confirms MCP endpoint returns expected tools
3. **MCP GPU Recommendations**: Tests GPU recommendation functionality

### Full Test Suite (default)
All quick tests plus comprehensive feature testing:
4. **Jupyter Analysis**: Tests analysis of Jupyter notebook files via MCP
5. **Marimo Analysis**: Tests analysis of Marimo notebook files via MCP
6. **Streaming Analysis**: Tests the streaming endpoint with progress updates
7. **Web Form Analysis**: Tests file upload and analysis via web interface

## Setting Up Test Environment

### Local Docker Testing
```bash
# Start the service
docker compose up -d

# Run tests
./test.sh

# Quick smoke test
./test.sh --quick
```

### Local Development Testing
```bash
# Start the Flask app (assuming port 8080)
python app.py

# Test in another terminal
./test.sh --url http://localhost:8080
```

### Vercel Deployment Testing
```bash
# After deploying to Vercel
./test.sh --url https://your-app-name.vercel.app

# Quick test for CI/CD validation
./test.sh --quick --url https://your-app-name.vercel.app
```

## Understanding Test Results

The script provides color-coded output:
- ✅ **PASS** (Green): Test completed successfully
- ❌ **FAIL** (Red): Test failed, check error message
- ⚡ **Quick Mode** (Yellow): Running in quick test mode
- 🚀 **Starting** (Blue): Test execution beginning

### Sample Output

```bash
Notebook Analyzer - Test Suite
Mode: Full Test Suite
Target: http://localhost:8080

🚀 Starting Notebook Analyzer Tests
==================================================
Testing against: http://localhost:8080

✅ PASS: Health Check
   Web UI accessible
✅ PASS: MCP Tools List
   Found 2 tools: analyze_notebook get_gpu_recommendations 
✅ PASS: MCP GPU Recommendations
   Response: GPU: A100 80GB, Quantity: 4, VRAM per GPU: 80 GB...
✅ PASS: Jupyter Analysis
   Analysis completed successfully
✅ PASS: Marimo Analysis
   Analysis completed successfully
✅ PASS: Streaming Analysis
   Streaming endpoint working
✅ PASS: Web Form Analysis
   File upload and analysis working

==================================================
📊 Test Results: 7/7 passed
🎉 All tests passed! System is fully functional.
```

## Troubleshooting

### Common Issues

**Connection Refused:**
- Ensure the service is running on the specified port
- Check if Docker container is up: `docker compose ps`
- Verify the URL and port are correct

**Tool Dependencies Missing:**
- Install curl and jq as shown in prerequisites
- Verify installation: `curl --version` and `jq --version`

**Analysis Tests Failing:**
- Ensure example files exist: `examples/jupyter_example.ipynb` and `examples/marimo_example.py`
- Check file permissions are readable
- Verify notebook files contain valid content

**Timeout Errors:**
- The script uses a 30-second timeout for analysis tests
- For slower systems, this might need adjustment in the script
- Check system resources and network connectivity

### Debugging Failed Tests

When tests fail, the script provides detailed error messages. Common failure patterns:

1. **HTTP 404/500 errors**: Service endpoint issues
2. **Invalid JSON responses**: API response format problems
3. **Missing content**: Analysis results not containing expected patterns
4. **Timeout failures**: Service taking too long to respond

## Continuous Integration

The test script is designed to work well in CI/CD pipelines:

```bash
# Exit code 0 on success, 1 on failure
./test.sh --quick --url "$DEPLOYMENT_URL"
echo "Exit code: $?"
```

## File Requirements

The following files must be present for full testing:
- `examples/jupyter_example.ipynb`: Sample Jupyter notebook
- `examples/marimo_example.py`: Sample Marimo notebook

These files are included in the repository and contain example ML/AI code for analysis testing. 