#!/bin/bash

# test-accuracy.sh - Notebook Analysis Accuracy Test Suite
# This script iterates through all notebooks in the examples/ directory 
# and evaluates the analysis results to verify no regressions have occurred.

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of tests/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
EXAMPLES_DIR="$PROJECT_ROOT/examples"
TEST_RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${TEST_RESULTS_DIR}/accuracy_test_${TIMESTAMP}.json"
LOG_FILE="${TEST_RESULTS_DIR}/accuracy_test_${TIMESTAMP}.log"

# Expected results baseline (these should be updated when analysis improves)
# Format: filename:expected_conditions
EXPECTED_RESULTS="
enterprise_gpu_example.ipynb:gpu_required=true,min_vram_gb>=16,workload_detected=true
jupyter_example.ipynb:gpu_required=true,min_vram_gb>=8,workload_detected=true
test_gpu_workload.ipynb:gpu_required=true,min_vram_gb>=8,workload_detected=true
test_arm_problems.ipynb:gpu_required=true,min_vram_gb>=8,workload_detected=true
marimo_example.py:gpu_required=true,min_vram_gb>=8,workload_detected=true
low_confidence_example.ipynb:gpu_required=false,confidence<0.5,workload_detected=false
very_low_confidence_example.ipynb:gpu_required=false,confidence<0.3,workload_detected=false
"

# Function to get expected result for a notebook
get_expected_result() {
    local notebook_name=$1
    echo "$EXPECTED_RESULTS" | grep "^${notebook_name}:" | cut -d':' -f2
}

# Performance thresholds
MAX_ANALYSIS_TIME=30  # seconds
MIN_CONFIDENCE_THRESHOLD=0.3

# Initialize
echo -e "${BLUE}=== Notebook Analysis Accuracy Test Suite ===${NC}"
echo "Starting test run at $(date)"
echo "Test results will be saved to: $RESULTS_FILE"
echo "Test logs will be saved to: $LOG_FILE"
echo

# Create results directory
mkdir -p "$TEST_RESULTS_DIR"

# Initialize results file
cat > "$RESULTS_FILE" << EOF
{
  "test_run": {
    "timestamp": "$(date -Iseconds)",
    "version": "3.0.0",
    "environment": {
      "python_version": "$(python3 --version 2>&1 || echo 'Unknown')",
      "openai_model": "${OPENAI_MODEL:-not_set}",
      "llm_enabled": $([ -n "$OPENAI_API_KEY" ] && echo "true" || echo "false")
    }
  },
  "results": [
EOF

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

# Function to analyze a single notebook
analyze_notebook() {
    local notebook_path=$1
    local notebook_name=$(basename "$notebook_path")
    local start_time=$(date +%s)
    
    log_message "INFO" "Starting analysis of $notebook_name"
    
    # Run the analysis with JSON output
    local analysis_result
    local analysis_exit_code=0
    
    # Use timeout to prevent hanging
            if timeout $MAX_ANALYSIS_TIME python3 "$PROJECT_ROOT/notebook-analyzer.py" --json "$notebook_path" > /tmp/analysis_output_$$.json 2>/tmp/analysis_error_$$.log; then
        analysis_result=$(cat /tmp/analysis_output_$$.json)
    else
        analysis_exit_code=$?
        analysis_result='{"error": "Analysis failed or timed out"}'
        log_message "ERROR" "Analysis failed for $notebook_name (exit code: $analysis_exit_code)"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Clean up temp files
    rm -f /tmp/analysis_output_$$.json /tmp/analysis_error_$$.log
    
    # Parse analysis result
    local gpu_required="unknown"
    local min_vram_gb=0
    local confidence=0.0
    local workload_detected="unknown"
    local llm_enhanced="false"
    local analysis_status="unknown"
    
    if echo "$analysis_result" | jq -e . >/dev/null 2>&1; then
        # Valid JSON response
        gpu_required=$(echo "$analysis_result" | jq -r '.min_gpu_type // "unknown"' | grep -v "CPU-only" >/dev/null && echo "true" || echo "false")
        min_vram_gb=$(echo "$analysis_result" | jq -r '.min_vram_gb // 0')
        confidence=$(echo "$analysis_result" | jq -r '.confidence // 0')
        workload_detected=$(echo "$analysis_result" | jq -r '.workload_detected // false')
        llm_enhanced=$(echo "$analysis_result" | jq -r '.llm_enhanced // false')
        analysis_status="success"
    else
        analysis_status="failed"
        log_message "ERROR" "Invalid JSON response for $notebook_name"
    fi
    
    # Evaluate against expected results
    local expected=$(get_expected_result "$notebook_name")
    local test_passed="true"
    local failure_reasons=()
    
    if [ -n "$expected" ]; then
        # Parse expected results
        IFS=',' read -ra EXPECTED_PARTS <<< "$expected"
        for part in "${EXPECTED_PARTS[@]}"; do
            IFS='=' read -ra CONDITION <<< "$part"
            local key="${CONDITION[0]}"
            local value="${CONDITION[1]}"
            
            case "$key" in
                "gpu_required")
                    if [ "$gpu_required" != "$value" ]; then
                        test_passed="false"
                        failure_reasons+=("Expected gpu_required=$value, got $gpu_required")
                    fi
                    ;;
                "min_vram_gb")
                    if [[ "$value" == *">="* ]]; then
                        local threshold=${value#*>=}
                        if (( $(echo "$min_vram_gb < $threshold" | bc -l) )); then
                            test_passed="false"
                            failure_reasons+=("Expected min_vram_gb>=$threshold, got $min_vram_gb")
                        fi
                    fi
                    ;;
                "confidence")
                    if [[ "$value" == *"<"* ]]; then
                        local threshold=${value#*<}
                        if (( $(echo "$confidence >= $threshold" | bc -l) )); then
                            test_passed="false"
                            failure_reasons+=("Expected confidence<$threshold, got $confidence")
                        fi
                    fi
                    ;;
                "workload_detected")
                    if [ "$workload_detected" != "$value" ]; then
                        test_passed="false"
                        failure_reasons+=("Expected workload_detected=$value, got $workload_detected")
                    fi
                    ;;
            esac
        done
    else
        log_message "WARN" "No expected results defined for $notebook_name"
    fi
    
    # Performance checks
    if [ "$duration" -gt "$MAX_ANALYSIS_TIME" ]; then
        test_passed="false"
        failure_reasons+=("Analysis took too long: ${duration}s > ${MAX_ANALYSIS_TIME}s")
    fi
    
    if [ "$analysis_status" = "failed" ]; then
        test_passed="false"
        failure_reasons+=("Analysis failed to complete")
    fi
    
    # Output result
    local result_json=$(cat << EOF
    {
      "notebook": "$notebook_name",
      "test_passed": $test_passed,
      "duration_seconds": $duration,
      "analysis_status": "$analysis_status",
      "results": {
        "gpu_required": $gpu_required,
        "min_vram_gb": $min_vram_gb,
        "confidence": $confidence,
        "workload_detected": $workload_detected,
        "llm_enhanced": $llm_enhanced
      },
      "expected": "$expected",
      "failure_reasons": [$(printf '"%s",' "${failure_reasons[@]}" | sed 's/,$//' || echo '""')]
    }
EOF
)
    
    # Add comma if not first result
    if [ $TOTAL_TESTS -gt 0 ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    echo "$result_json" >> "$RESULTS_FILE"
    
    # Update counters
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ "$test_passed" = "true" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ PASS${NC} $notebook_name (${duration}s)"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ FAIL${NC} $notebook_name (${duration}s)"
        for reason in "${failure_reasons[@]}"; do
            echo -e "  ${RED}└─${NC} $reason"
        done
    fi
    
    log_message "INFO" "Completed analysis of $notebook_name: $test_passed (${duration}s)"
}

# Function to check prerequisites
check_prerequisites() {
    log_message "INFO" "Checking prerequisites..."
    
    # Check if notebook-analyzer.py exists
    if [ ! -f "$PROJECT_ROOT/notebook-analyzer.py" ]; then
        log_message "ERROR" "notebook-analyzer.py not found in project root"
        exit 1
    fi
    
    # Check if examples directory exists
    if [ ! -d "$EXAMPLES_DIR" ]; then
        log_message "ERROR" "Examples directory not found: $EXAMPLES_DIR"
        exit 1
    fi
    
    # Check required commands
    for cmd in python3 jq bc timeout; do
        if ! command -v "$cmd" &> /dev/null; then
            log_message "ERROR" "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Python dependencies
    if ! python3 -c "import requests, json, re, os" 2>/dev/null; then
        log_message "ERROR" "Required Python modules not available"
        exit 1
    fi
    
    log_message "INFO" "Prerequisites check passed"
}

# Function to print summary
print_summary() {
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
    fi
    
    echo
    echo -e "${BLUE}=== Test Summary ===${NC}"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS${NC}"
    echo "Success rate: ${success_rate}%"
    echo
    
    if [ $FAILED_TESTS -gt 0 ]; then
        echo -e "${RED}❌ Some tests failed. Check the logs for details.${NC}"
        exit 1
    else
        echo -e "${GREEN}✅ All tests passed!${NC}"
    fi
}

# Main execution
main() {
    # Check prerequisites
    check_prerequisites
    
    # Find all notebook files
    local notebook_files=()
    while IFS= read -r -d '' file; do
        notebook_files+=("$file")
    done < <(find "$EXAMPLES_DIR" \( -name "*.ipynb" -o -name "*.py" \) -print0 | sort -z)
    
    if [ ${#notebook_files[@]} -eq 0 ]; then
        log_message "ERROR" "No notebook files found in $EXAMPLES_DIR"
        exit 1
    fi
    
    log_message "INFO" "Found ${#notebook_files[@]} notebook files to test"
    
    # Analyze each notebook
    for notebook_file in "${notebook_files[@]}"; do
        analyze_notebook "$notebook_file"
    done
    
    # Close JSON array
    echo -e "\n  ],\n  \"summary\": {" >> "$RESULTS_FILE"
    echo "    \"total_tests\": $TOTAL_TESTS," >> "$RESULTS_FILE"
    echo "    \"passed_tests\": $PASSED_TESTS," >> "$RESULTS_FILE"
    echo "    \"failed_tests\": $FAILED_TESTS," >> "$RESULTS_FILE"
    echo "    \"skipped_tests\": $SKIPPED_TESTS," >> "$RESULTS_FILE"
    echo "    \"success_rate\": $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)" >> "$RESULTS_FILE"
    echo "  }" >> "$RESULTS_FILE"
    echo "}" >> "$RESULTS_FILE"
    
    # Print summary
    print_summary
    
    log_message "INFO" "Test completed. Results saved to $RESULTS_FILE"
}

# Handle script arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --list         List available test notebooks"
        echo "  --clean        Clean up old test results"
        echo
        echo "Environment Variables:"
        echo "  OPENAI_API_KEY     Enable LLM enhancement"
        echo "  OPENAI_MODEL       Specify model to use"
        echo "  OPENAI_BASE_URL    Specify API endpoint"
        exit 0
        ;;
    "--list")
        echo "Available test notebooks:"
        find "$EXAMPLES_DIR" -name "*.ipynb" -o -name "*.py" | sort
        exit 0
        ;;
    "--clean")
        echo "Cleaning up old test results..."
        rm -rf "$TEST_RESULTS_DIR"
        echo "Cleanup complete."
        exit 0
        ;;
    "")
        # No arguments, run main
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac 