#!/bin/bash

# Security Headers Verification Script
# Usage: ./verify_security_headers.sh [--url <base_url>] [--help]

set -e  # Exit on any error unless explicitly handled

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of tests/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
DEFAULT_BASE_URL="http://localhost:8080"
TIMEOUT=10

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
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
            echo "Security Headers Verification Script"
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

echo "🔍 Security Headers Verification - Phase 1"
echo "=========================================="
echo "🎯 Testing: $BASE_URL"
echo ""

# Check if required tools are available
check_dependencies() {
    local missing=""
    
    if ! command -v curl >/dev/null 2>&1; then
        missing="$missing curl"
    fi
    
    if [ -n "$missing" ]; then
        echo -e "${RED}❌ Missing required tools:$missing${NC}"
        echo "Please install curl"
        exit 1
    fi
}

# Function to check if a header exists and matches expected value
check_header() {
    local header_name="$1"
    local expected_value="$2"
    local actual_value="$3"
    
    if [[ -n "$actual_value" ]]; then
        if [[ "$actual_value" == *"$expected_value"* ]]; then
            echo -e "${GREEN}✅ $header_name${NC}: Present and correct"
            return 0
        else
            echo -e "${YELLOW}⚠️  $header_name${NC}: Present but unexpected value"
            echo -e "   Expected: ${BLUE}$expected_value${NC}"
            echo -e "   Actual:   ${BLUE}$actual_value${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ $header_name${NC}: Missing"
        return 1
    fi
}

# Check dependencies first
check_dependencies

# Test if the service is accessible
echo "🔍 Testing service availability..."
if ! curl -s --connect-timeout 5 --max-time $TIMEOUT -L "$BASE_URL" > /dev/null; then
    echo -e "${RED}❌ Service not accessible at $BASE_URL${NC}"
    echo "   Please ensure the Notebook Analyzer is running"
    exit 1
fi

echo -e "${GREEN}✅ Service is accessible${NC}"
echo ""

# Fetch headers from the home page
echo "🔍 Testing security headers on home page..."
RESPONSE=$(curl -I -s --max-time $TIMEOUT -L "$BASE_URL")

# Extract headers (handle multi-line headers by joining them)
CSP_HEADER=$(echo "$RESPONSE" | sed -n '/^[Cc]ontent-[Ss]ecurity-[Pp]olicy:/,/^[A-Za-z-]*:/p' | sed '$d' | tr -d '\r\n' | sed 's/^[Cc]ontent-[Ss]ecurity-[Pp]olicy:[[:space:]]*//' | sed 's/[[:space:]]*$//')
FRAME_HEADER=$(echo "$RESPONSE" | grep -i "x-frame-options:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
CONTENT_TYPE_HEADER=$(echo "$RESPONSE" | grep -i "x-content-type-options:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')

# Phase 2 Security Headers
REFERRER_HEADER=$(echo "$RESPONSE" | grep -i "referrer-policy:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
PERMISSIONS_HEADER=$(echo "$RESPONSE" | grep -i "permissions-policy:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
XSS_HEADER=$(echo "$RESPONSE" | grep -i "x-xss-protection:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
COOP_HEADER=$(echo "$RESPONSE" | grep -i "cross-origin-opener-policy:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
CORP_HEADER=$(echo "$RESPONSE" | grep -i "cross-origin-resource-policy:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
COEP_HEADER=$(echo "$RESPONSE" | grep -i "cross-origin-embedder-policy:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')

echo ""
echo "📋 Phase 1 Security Headers Analysis:"
echo "------------------------------------"

# Check Content Security Policy
check_header "Content-Security-Policy" "default-src 'self'" "$CSP_HEADER"
CSP_SUCCESS=$?

# Check X-Frame-Options
check_header "X-Frame-Options" "DENY" "$FRAME_HEADER"
FRAME_SUCCESS=$?

# Check X-Content-Type-Options
check_header "X-Content-Type-Options" "nosniff" "$CONTENT_TYPE_HEADER"
CONTENT_TYPE_SUCCESS=$?

echo ""
echo "📋 Phase 2 Security Headers Analysis:"
echo "------------------------------------"

# Check Referrer Policy
check_header "Referrer-Policy" "strict-origin-when-cross-origin" "$REFERRER_HEADER"
REFERRER_SUCCESS=$?

# Check Permissions Policy
check_header "Permissions-Policy" "geolocation=()" "$PERMISSIONS_HEADER"
PERMISSIONS_SUCCESS=$?

# Check X-XSS-Protection
check_header "X-XSS-Protection" "1; mode=block" "$XSS_HEADER"
XSS_SUCCESS=$?

# Check Cross-Origin-Opener-Policy
check_header "Cross-Origin-Opener-Policy" "same-origin" "$COOP_HEADER"
COOP_SUCCESS=$?

# Check Cross-Origin-Resource-Policy
check_header "Cross-Origin-Resource-Policy" "same-origin" "$CORP_HEADER"
CORP_SUCCESS=$?

# Check Cross-Origin-Embedder-Policy
check_header "Cross-Origin-Embedder-Policy" "require-corp" "$COEP_HEADER"
COEP_SUCCESS=$?

# Test API endpoint if available
echo ""
echo "🔍 Testing headers on health endpoint..."
API_RESPONSE=$(curl -I -s --max-time $TIMEOUT -L "$BASE_URL/health" 2>/dev/null || echo "")

if [[ -n "$API_RESPONSE" ]]; then
    # Extract API headers (handle multi-line headers)
    API_CSP=$(echo "$API_RESPONSE" | sed -n '/^[Cc]ontent-[Ss]ecurity-[Pp]olicy:/,/^[A-Za-z-]*:/p' | sed '$d' | tr -d '\r\n' | sed 's/^[Cc]ontent-[Ss]ecurity-[Pp]olicy:[[:space:]]*//' | sed 's/[[:space:]]*$//')
    API_FRAME=$(echo "$API_RESPONSE" | grep -i "x-frame-options:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')
    API_CONTENT_TYPE=$(echo "$API_RESPONSE" | grep -i "x-content-type-options:" | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '\r')

    echo ""
    echo "📋 API Endpoint Headers Analysis:"
    echo "--------------------------------"

    check_header "Content-Security-Policy" "default-src 'self'" "$API_CSP"
    check_header "X-Frame-Options" "DENY" "$API_FRAME"
    check_header "X-Content-Type-Options" "nosniff" "$API_CONTENT_TYPE"
else
    echo -e "${YELLOW}⚠️  Health endpoint not accessible - skipping API header test${NC}"
fi

echo ""
echo "🔍 Detailed CSP Analysis:"
echo "------------------------"

if [[ -n "$CSP_HEADER" ]]; then
    echo -e "${BLUE}Full CSP Header:${NC}"
    echo "$CSP_HEADER" | sed 's/;/;\n/g' | sed 's/^/  /'
    
    echo ""
    echo "CSP Directive Check:"
    
    # Check key CSP directives
    if [[ "$CSP_HEADER" == *"frame-ancestors 'none'"* ]]; then
        echo -e "${GREEN}✅ frame-ancestors 'none'${NC} - Prevents clickjacking"
    else
        echo -e "${RED}❌ frame-ancestors directive missing${NC}"
    fi
    
    if [[ "$CSP_HEADER" == *"base-uri 'self'"* ]]; then
        echo -e "${GREEN}✅ base-uri 'self'${NC} - Prevents base tag injection"
    else
        echo -e "${RED}❌ base-uri directive missing${NC}"
    fi
    
    if [[ "$CSP_HEADER" == *"form-action 'self'"* ]]; then
        echo -e "${GREEN}✅ form-action 'self'${NC} - Prevents form hijacking"
    else
        echo -e "${RED}❌ form-action directive missing${NC}"
    fi
    
    if [[ "$CSP_HEADER" == *"script-src"* ]]; then
        echo -e "${GREEN}✅ script-src directive${NC} - Controls script sources"
    else
        echo -e "${RED}❌ script-src directive missing${NC}"
    fi
else
    echo -e "${RED}❌ No CSP header found${NC}"
fi

echo ""
echo "📊 Security Headers Test Summary:"
echo "================================"

# Count successful headers (Phase 1 + Phase 2)
PHASE1_SUCCESS=$((3 - CSP_SUCCESS - FRAME_SUCCESS - CONTENT_TYPE_SUCCESS))
PHASE2_SUCCESS=$((6 - REFERRER_SUCCESS - PERMISSIONS_SUCCESS - XSS_SUCCESS - COOP_SUCCESS - CORP_SUCCESS - COEP_SUCCESS))
TOTAL_SUCCESS=$((PHASE1_SUCCESS + PHASE2_SUCCESS))
TOTAL_COUNT=9

echo "✅ Phase 1 headers implemented: $PHASE1_SUCCESS/3"
echo "✅ Phase 2 headers implemented: $PHASE2_SUCCESS/6"
echo "✅ Total headers implemented: $TOTAL_SUCCESS/$TOTAL_COUNT"

if [[ $PHASE1_SUCCESS -eq 3 && $PHASE2_SUCCESS -eq 6 ]]; then
    echo -e "${GREEN}🎉 All Phase 1 & Phase 2 security headers are properly implemented!${NC}"
    echo ""
    echo "🛡️  Phase 1 Protection Summary:"
    echo "   • XSS attacks blocked by CSP"
    echo "   • Clickjacking prevented by X-Frame-Options"
    echo "   • MIME confusion attacks blocked by X-Content-Type-Options"
    echo ""
    echo "🛡️  Phase 2 Protection Summary:"
    echo "   • Referrer information controlled by Referrer-Policy"
    echo "   • Browser features restricted by Permissions-Policy"
    echo "   • XSS filtering enabled by X-XSS-Protection"
    echo "   • Cross-origin isolation by COOP/CORP/COEP headers"
    echo "   • Enhanced security for modern browsers"
    exit 0
elif [[ $PHASE1_SUCCESS -eq 3 ]]; then
    echo -e "${YELLOW}⚠️  Phase 1 headers complete, but Phase 2 headers missing${NC}"
    echo "   Phase 2 headers provide enhanced security for modern browsers."
    exit 1
else
    echo -e "${RED}⚠️  Some security headers are missing or incorrect${NC}"
    echo "   Please check the Flask application security configuration."
    exit 1
fi 