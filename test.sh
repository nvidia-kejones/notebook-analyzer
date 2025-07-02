#!/bin/bash
# Unified test script for Notebook Analyzer
# Usage: ./test.sh [--quick] [--url <base_url>] [--help]

set -e  # Exit on any error unless explicitly handled

# Default configuration
DEFAULT_BASE_URL="http://localhost:8080"
TIMEOUT=30
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
            echo "  --quick, -q          Run quick tests only (health, MCP, GPU analysis)"
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
    
    local notebook_content=$(cat "examples/marimo_example.py" | jq -Rs .)
    local payload="{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"analyze_notebook\",\"arguments\":{\"notebook_content\":$notebook_content,\"source_info\":\"marimo_example.py\"}}}"
    local response_file="$TEMP_DIR/marimo_response"
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL/mcp" > "$TEMP_DIR/marimo_code" 2>/dev/null; then
        
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
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -F "file=@examples/jupyter_example.ipynb" \
        -F "analysis_type=basic" \
        "$BASE_URL/analyze-stream" > "$TEMP_DIR/streaming_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/streaming_code")
        if [ "$http_code" = "200" ]; then
            local content=$(cat "$response_file")
            if echo "$content" | grep -q "data:" && (echo "$content" | grep -qi "progress\|analyzing"); then
                log_result "Streaming Analysis" "true" "Streaming endpoint working"
                return 0
            else
                log_result "Streaming Analysis" "false" "Invalid streaming response"
                return 1
            fi
        else
            log_result "Streaming Analysis" "false" "HTTP $http_code"
            return 1
        fi
    else
        log_result "Streaming Analysis" "false" "Request failed"
        return 1
    fi
}

test_web_form_analysis() {
    if [ ! -f "examples/jupyter_example.ipynb" ]; then
        log_result "Web Form Analysis" "false" "Example file not found"
        return 1
    fi
    
    local response_file="$TEMP_DIR/webform_response"
    
    if curl -s -m "$TIMEOUT" -o "$response_file" -w "%{http_code}" \
        -F "file=@examples/jupyter_example.ipynb" \
        -F "analysis_type=basic" \
        "$BASE_URL/analyze" > "$TEMP_DIR/webform_code" 2>/dev/null; then
        
        local http_code=$(cat "$TEMP_DIR/webform_code")
        if [ "$http_code" = "200" ]; then
            local content=$(cat "$response_file")
            if echo "$content" | grep -q "GPU" && (echo "$content" | grep -qi "recommendation\|analysis"); then
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
    
    # MCP tests (always run)
    test_mcp_tools_list
    test_mcp_gpu_recommendations
    
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