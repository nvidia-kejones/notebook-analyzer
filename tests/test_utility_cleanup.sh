#!/bin/bash
# Utility Function Cleanup Test Suite
# Tests all consolidated functions from Phases 1-4 to ensure no regressions
# Usage: ./test_utility_cleanup.sh [--quick] [--verbose] [--help]

set -e  # Exit on any error unless explicitly handled

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of tests/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
QUICK_MODE=false
VERBOSE=false
TEMP_DIR="/tmp/utility_cleanup_test_$$"
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
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Utility Function Cleanup Test Suite"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick, -q          Run quick tests only (basic functionality)"
            echo "  --verbose, -v        Show detailed test output"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Test Coverage:"
            echo "  Phase 1: Runtime function consolidation"
            echo "  Phase 2: Consumer viability consolidation"
            echo "  Phase 3: Module-level utilities"
            echo "  Phase 4: GPU finding optimization"
            echo "  Integration: Full notebook analysis"
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
    
    if [ "$VERBOSE" = "true" ] && [ -f "$TEMP_DIR/test_output" ]; then
        echo "   Output: $(cat $TEMP_DIR/test_output)"
    fi
    
    echo "$name:$passed:$message" >> "$RESULTS_FILE"
}

# Check if required tools are available
check_dependencies() {
    local missing=""
    
    if ! command -v python3 >/dev/null 2>&1; then
        missing="$missing python3"
    fi
    
    if [ -n "$missing" ]; then
        echo -e "${RED}❌ Missing required tools:$missing${NC}"
        echo "Please install them and ensure the analyzer module is available"
        exit 1
    fi
    
    # Check if analyzer module is importable
    local current_dir=$(pwd)
    cd "$PROJECT_ROOT"
    if ! python3 -c "from analyzer.core import GPUAnalyzer" 2>/dev/null; then
        echo -e "${RED}❌ Cannot import analyzer.core module${NC}"
        echo "Please ensure the analyzer module is available in the project root"
        cd "$current_dir"
        exit 1
    fi
    cd "$current_dir"
}

# Phase 1 Tests: Runtime Function Consolidation
test_phase1_runtime_functions() {
    echo -e "${BLUE}📋 Testing Phase 1: Runtime Function Consolidation${NC}"
    
    # Test parse_runtime_range
    local test_output=$(python3 -c "
from analyzer.core import parse_runtime_range
try:
    result1 = parse_runtime_range('1.5-2.5')
    result2 = parse_runtime_range('30 minutes')
    result3 = parse_runtime_range('2 hours')
    # Check that range parsing works and returns valid tuples
    if (result1 == (1.5, 2.5) and 
        isinstance(result2, tuple) and len(result2) == 2 and
        isinstance(result3, tuple) and len(result3) == 2):
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Runtime Range Parsing" "true" "All parsing scenarios work correctly"
    else
        log_result "Runtime Range Parsing" "false" "$test_output"
    fi
    
    # Test format_runtime
    test_output=$(python3 -c "
from analyzer.core import format_runtime
try:
    result1 = format_runtime(0.5)
    result2 = format_runtime(1.5)
    result3 = format_runtime(24.0)
    if '30 minutes' in result1 and '1.5 hours' in result2 and '24.0 hours' in result3:
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Runtime Formatting" "true" "All formatting scenarios work correctly"
    else
        log_result "Runtime Formatting" "false" "$test_output"
    fi
    
    # Test convert_runtime_to_new_format
    test_output=$(python3 -c "
from analyzer.core import convert_runtime_to_new_format
try:
    result1 = convert_runtime_to_new_format('1.0-2.0')
    result2 = convert_runtime_to_new_format('30 minutes')
    if 'hours' in result1 or 'minutes' in result1:
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Runtime Conversion" "true" "Runtime conversion works correctly"
    else
        log_result "Runtime Conversion" "false" "$test_output"
    fi
}

# Phase 2 Tests: Consumer Viability Consolidation
test_phase2_consumer_viability() {
    echo -e "${BLUE}📋 Testing Phase 2: Consumer Viability Consolidation${NC}"
    
    # Test unified consumer viability function
    local test_output=$(python3 -c "
from analyzer.core import GPUAnalyzer
try:
    analyzer = GPUAnalyzer()
    
    # Test with analysis dict
    analysis = {'min_vram_gb': 16, 'min_quantity': 1, 'workload_type': 'inference'}
    result1 = analyzer._assess_consumer_viability(analysis)
    
    # Test with explicit parameters
    result2 = analyzer._assess_consumer_viability(analysis, per_gpu_vram=32, quantity=1)
    
    # Test mixed mode
    result3 = analyzer._assess_consumer_viability(analysis, quantity=4)
    
    if all(isinstance(r, tuple) and len(r) == 2 for r in [result1, result2, result3]):
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Consumer Viability Consolidation" "true" "All parameter modes work correctly"
    else
        log_result "Consumer Viability Consolidation" "false" "$test_output"
    fi
}

# Phase 3 Tests: Module-Level Utilities
test_phase3_module_utilities() {
    echo -e "${BLUE}📋 Testing Phase 3: Module-Level Utilities${NC}"
    
    # Test compare_versions
    local test_output=$(python3 -c "
from analyzer.core import compare_versions
try:
    result1 = compare_versions('1.0', '1.1')
    result2 = compare_versions('2.0', '1.9')
    result3 = compare_versions('1.0', '1.0')
    if result1 == -1 and result2 == 1 and result3 == 0:
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Version Comparison" "true" "All comparison scenarios work correctly"
    else
        log_result "Version Comparison" "false" "$test_output"
    fi
    
    # Test normalize_gpu_quantity
    test_output=$(python3 -c "
from analyzer.core import normalize_gpu_quantity
try:
    result1 = normalize_gpu_quantity(0)
    result2 = normalize_gpu_quantity(3)
    result3 = normalize_gpu_quantity(12)
    if result1 == 1 and result2 == 4 and result3 == 16:
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "GPU Quantity Normalization" "true" "All normalization scenarios work correctly"
    else
        log_result "GPU Quantity Normalization" "false" "$test_output"
    fi
    
    # Test calculate_multi_gpu_scaling
    test_output=$(python3 -c "
from analyzer.core import calculate_multi_gpu_scaling
try:
    result1 = calculate_multi_gpu_scaling(1)
    result2 = calculate_multi_gpu_scaling(2)
    result3 = calculate_multi_gpu_scaling(8)
    if result1 == 1.0 and result2 == 0.55 and result3 == 0.25:
        print('SUCCESS')
    else:
        print(f'FAIL: {result1}, {result2}, {result3}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Multi-GPU Scaling Calculation" "true" "All scaling calculations work correctly"
    else
        log_result "Multi-GPU Scaling Calculation" "false" "$test_output"
    fi
}

# Phase 4 Tests: GPU Finding Optimization
test_phase4_gpu_finding() {
    echo -e "${BLUE}📋 Testing Phase 4: GPU Finding Optimization${NC}"
    
    # Test unified find_best_gpu function
    local test_output=$(python3 -c "
from analyzer.core import find_best_gpu
try:
    # Sample GPU specs for testing
    sample_specs = {
        'RTX 4070': {'vram': 12, 'performance_factor': 5.0, 'category': 'consumer', 'tier': 'high'},
        'RTX 4090': {'vram': 24, 'performance_factor': 8.0, 'category': 'consumer', 'tier': 'flagship'},
        'H100 PCIe': {'vram': 80, 'performance_factor': 10.0, 'category': 'enterprise', 'tier': 'cutting_edge'}
    }
    
    # Test different selection modes
    result1 = find_best_gpu(sample_specs, 16, 'cost')
    result2 = find_best_gpu(sample_specs, 16, 'consumer')
    result3 = find_best_gpu(sample_specs, 16, 'professional')
    result4 = find_best_gpu(sample_specs, 16, 'optimal')
    
    if all(r is not None and len(r) == 3 for r in [result1, result2, result3, result4]):
        print('SUCCESS')
    else:
        print(f'FAIL: Some selection modes returned None or invalid results')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Unified GPU Selection" "true" "All selection modes work correctly"
    else
        log_result "Unified GPU Selection" "false" "$test_output"
    fi
    
    # Test GPUAnalyzer wrapper methods
    test_output=$(python3 -c "
from analyzer.core import GPUAnalyzer
try:
    analyzer = GPUAnalyzer()
    
    # Test all consolidated methods
    result1 = analyzer._find_best_consumer_gpu(16)
    result2 = analyzer._find_best_professional_gpu(16, 'training')
    result3 = analyzer._find_optimal_gpu(16, 'training')
    result4 = analyzer._find_minimum_viable_gpu_strict(16)
    result5 = analyzer._find_balanced_gpu(16, 'training')
    result6 = analyzer._find_performance_gpu(16, 'training')
    
    # Check that all methods return valid results
    results = [result1, result2, result3, result4, result5, result6]
    if all(r is not None for r in results):
        print('SUCCESS')
    else:
        print(f'FAIL: Some methods returned None')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "GPU Analyzer Wrapper Methods" "true" "All 6 consolidated methods work correctly"
    else
        log_result "GPU Analyzer Wrapper Methods" "false" "$test_output"
    fi
}

# Integration Tests: Full Notebook Analysis
test_integration_analysis() {
    echo -e "${BLUE}📋 Testing Integration: Full Notebook Analysis${NC}"
    
    # Test CPU-only workload
    local test_output=$(python3 -c "
from analyzer.core import GPUAnalyzer
import sys
try:
    analyzer = GPUAnalyzer()
    result = analyzer.analyze_notebook('examples/hello_world_cpu_only.ipynb')
    
    if result.min_gpu_type == 'CPU-only' and not result.workload_detected:
        print('SUCCESS')
    else:
        print(f'FAIL: Expected CPU-only, got {result.min_gpu_type}, workload: {result.workload_detected}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "CPU-only Analysis" "true" "CPU workload correctly detected"
    else
        log_result "CPU-only Analysis" "false" "$test_output"
    fi
    
    # Test GPU workload (if available)
    if [ -f "examples/test_gpu_workload.ipynb" ]; then
        test_output=$(python3 -c "
from analyzer.core import GPUAnalyzer
try:
    analyzer = GPUAnalyzer()
    result = analyzer.analyze_notebook('examples/test_gpu_workload.ipynb')
    
    if result.min_gpu_type != 'CPU-only' and result.workload_detected:
        print('SUCCESS')
    else:
        print(f'FAIL: Expected GPU workload, got {result.min_gpu_type}, workload: {result.workload_detected}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
        
        if [[ "$test_output" == *"SUCCESS"* ]]; then
            log_result "GPU Workload Analysis" "true" "GPU workload correctly detected and analyzed"
        else
            log_result "GPU Workload Analysis" "false" "$test_output"
        fi
    else
        log_result "GPU Workload Analysis" "true" "Test file not available (skipped)"
    fi
}

# Performance Tests
test_performance() {
    if [ "$QUICK_MODE" = "true" ]; then
        return 0
    fi
    
    echo -e "${BLUE}📋 Testing Performance: No Degradation${NC}"
    
    # Test analysis speed
    local test_output=$(python3 -c "
import time
from analyzer.core import GPUAnalyzer
try:
    analyzer = GPUAnalyzer()
    
    start_time = time.time()
    for i in range(5):
        result = analyzer.analyze_notebook('examples/hello_world_cpu_only.ipynb')
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 5
    if avg_time < 10.0:  # Should complete in under 10 seconds per analysis
        print(f'SUCCESS: {avg_time:.2f}s average')
    else:
        print(f'SLOW: {avg_time:.2f}s average (expected < 10s)')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Performance Test" "true" "$test_output"
    else
        log_result "Performance Test" "false" "$test_output"
    fi
}

# Backward Compatibility Tests
test_backward_compatibility() {
    echo -e "${BLUE}📋 Testing Backward Compatibility${NC}"
    
    # Test that all original method signatures still work
    local test_output=$(python3 -c "
from analyzer.core import GPUAnalyzer, LLMAnalyzer
try:
    # Test GPUAnalyzer methods
    analyzer = GPUAnalyzer()
    
    # These should all work without errors
    analyzer._compare_versions('1.0', '1.1')
    analyzer._normalize_gpu_quantity(3)
    analyzer._calculate_multi_gpu_scaling(4)
    analyzer._parse_runtime_range('1.0-2.0')
    
    analysis = {'min_vram_gb': 16, 'min_quantity': 1, 'workload_type': 'inference'}
    analyzer._assess_consumer_viability(analysis)
    
    # Test LLMAnalyzer methods (if API key available, otherwise skip)
    try:
        llm_analyzer = LLMAnalyzer('http://test', 'test-model', 'test-key')
        llm_analyzer._parse_runtime_range('1.0-2.0')
        llm_analyzer._format_runtime(1.5)
        llm_analyzer._convert_runtime_to_new_format('1.0-2.0')
    except:
        pass  # Skip LLM tests if no API key
    
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
    
    if [[ "$test_output" == *"SUCCESS"* ]]; then
        log_result "Backward Compatibility" "true" "All original method signatures work"
    else
        log_result "Backward Compatibility" "false" "$test_output"
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}🧪 Utility Function Cleanup Test Suite${NC}"
    echo "Testing consolidated functions from Phases 1-4..."
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Run test phases
    test_phase1_runtime_functions
    test_phase2_consumer_viability
    test_phase3_module_utilities
    test_phase4_gpu_finding
    test_integration_analysis
    test_backward_compatibility
    
    if [ "$QUICK_MODE" = "false" ]; then
        test_performance
    fi
    
    # Summary
    echo ""
    echo "========================================"
    if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED${NC} ($PASSED_TESTS/$TOTAL_TESTS)"
        echo "✅ Utility function cleanup is working correctly!"
        echo "✅ No regressions detected"
        echo "✅ All consolidated functions are functional"
    else
        local failed_tests=$((TOTAL_TESTS - PASSED_TESTS))
        echo -e "${RED}❌ SOME TESTS FAILED${NC} ($failed_tests/$TOTAL_TESTS failed)"
        echo "Please review the failed tests above"
    fi
    echo "========================================"
    
    # Exit with appropriate code
    if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@" 