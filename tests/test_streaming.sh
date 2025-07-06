#!/bin/bash
# Test script to verify streaming functionality
# Usage: ./test_streaming.sh [--url <base_url>] [--help]

set -e  # Exit on any error unless explicitly handled

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of tests/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
DEFAULT_BASE_URL="http://localhost:8080"
TIMEOUT=120
TEST_NOTEBOOK_URL="https://github.com/nvidia-kejones/launchables/blob/main/hugging-face-intro.ipynb"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
BASE_URL="$DEFAULT_BASE_URL"
while [[ $# -gt 0 ]]; do
    case $1 in
        --url|-u)
            BASE_URL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Streaming Endpoint Test Script"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --url, -u <url>      Base URL to test against (default: $DEFAULT_BASE_URL)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Test against localhost:8080"
            echo "  $0 --url http://localhost:5000       # Test against different port"
            echo "  $0 --url https://my-app.com          # Test against remote server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🧪 Streaming Endpoint Test${NC}"
echo "=========================="
echo "🎯 Testing: $BASE_URL"
echo "📓 Notebook: $TEST_NOTEBOOK_URL"
echo ""

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

# Test health endpoint first
test_health() {
    echo "🔍 Testing health endpoint..."
    local health_response=$(curl -s -m 5 -w "%{http_code}" "$BASE_URL/health" -o /dev/null)
    
    if [ "$health_response" = "200" ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Service not healthy (HTTP $health_response)${NC}"
        echo "Please ensure the service is running at $BASE_URL"
        exit 1
    fi
}

# Test streaming endpoint
test_streaming_endpoint() {
    echo ""
    echo "🚀 Testing streaming endpoint..."
    
    local temp_file="/tmp/streaming_test_$$"
    local start_time=$(date +%s)
    
    # Make streaming request and capture output
    echo "📡 Starting streaming request..."
    curl -s -m $TIMEOUT \
         -X POST \
         -F "url=$TEST_NOTEBOOK_URL" \
         "$BASE_URL/analyze-stream" \
         -H "Accept: text/event-stream" \
         > "$temp_file" &
    
    local curl_pid=$!
    local message_count=0
    local progress_count=0
    local complete_found=false
    local error_found=false
    
    echo ""
    echo "=== STREAMING DATA ==="
    
    # Monitor the streaming output in real-time
    while kill -0 $curl_pid 2>/dev/null; do
        if [ -f "$temp_file" ]; then
            # Process new lines as they arrive
            while IFS= read -r line; do
                if [[ "$line" == data:* ]]; then
                    local current_time=$(date +%s)
                    local elapsed=$((current_time - start_time))
                    
                    # Extract JSON data (remove 'data: ' prefix)
                    local json_data="${line#data: }"
                    
                    if echo "$json_data" | jq -e . >/dev/null 2>&1; then
                        local msg_type=$(echo "$json_data" | jq -r '.type // "unknown"')
                        local message=$(echo "$json_data" | jq -r '.message // ""' | cut -c1-100)
                        
                        message_count=$((message_count + 1))
                        
                        echo -e "${BLUE}[${elapsed}s]${NC} $msg_type: $message..."
                        
                        case "$msg_type" in
                            "progress")
                                progress_count=$((progress_count + 1))
                                ;;
                            "complete")
                                echo -e "${GREEN}[${elapsed}s] ANALYSIS COMPLETE!${NC}"
                                complete_found=true
                                break
                                ;;
                            "error")
                                echo -e "${RED}[${elapsed}s] ERROR: $message${NC}"
                                error_found=true
                                break
                                ;;
                        esac
                    else
                        echo -e "${YELLOW}[${elapsed}s] Invalid JSON: ${json_data:0:50}...${NC}"
                    fi
                fi
            done < <(tail -f "$temp_file" 2>/dev/null)
            
            if [ "$complete_found" = true ] || [ "$error_found" = true ]; then
                break
            fi
        fi
        sleep 0.5
    done
    
    # Wait for curl to finish
    wait $curl_pid 2>/dev/null
    local curl_exit_code=$?
    
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    echo ""
    echo "=== SUMMARY ==="
    echo "📊 Total messages: $message_count"
    echo "📈 Progress messages: $progress_count"
    echo "⏱️  Total time: ${total_time}s"
    echo "🔄 Curl exit code: $curl_exit_code"
    
    # Analyze results
    if [ "$complete_found" = true ]; then
        echo -e "${GREEN}✅ Analysis completed successfully${NC}"
        
        if [ $progress_count -gt 1 ] && [ $total_time -gt 1 ]; then
            echo -e "${GREEN}✅ REAL-TIME STREAMING WORKING!${NC}"
        elif [ $progress_count -gt 1 ]; then
            echo -e "${YELLOW}⚠️  Messages received but may not be real-time${NC}"
        else
            echo -e "${RED}❌ No progress messages received${NC}"
        fi
        
        return 0
    elif [ "$error_found" = true ]; then
        echo -e "${RED}❌ Analysis failed with error${NC}"
        return 1
    elif [ $curl_exit_code -ne 0 ]; then
        echo -e "${RED}❌ Curl failed (exit code: $curl_exit_code)${NC}"
        return 1
    else
        echo -e "${RED}❌ No completion message received${NC}"
        return 1
    fi
    
    # Cleanup
    rm -f "$temp_file"
}

# Main execution
main() {
    check_dependencies
    test_health
    
    if test_streaming_endpoint; then
        echo ""
        echo -e "${GREEN}🎉 Streaming test completed successfully!${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}❌ Streaming test failed${NC}"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    rm -f "/tmp/streaming_test_$$" 2>/dev/null || true
}
trap cleanup EXIT

# Run main function
main 