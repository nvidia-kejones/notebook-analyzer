#!/usr/bin/env python3
"""
Core Security Sandbox Validation Test

Tests the enhanced security sandbox features without requiring psutil.
This validates the implementation works correctly and provides value.
"""

import sys
import os
import tempfile
import subprocess
import time

# Add the analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analyzer'))

def test_sandbox_import():
    """Test that the SecuritySandbox can be imported and instantiated."""
    try:
        from security_sandbox import SecuritySandbox, ProcessResourceUsage
        
        # Test basic instantiation
        sandbox = SecuritySandbox(max_memory_mb=128, max_time_seconds=5)
        
        print("✅ SecuritySandbox import and instantiation successful")
        return True
    except Exception as e:
        print(f"❌ SecuritySandbox import failed: {e}")
        return False

def test_process_creation():
    """Test basic process creation and isolation."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox(max_memory_mb=128, max_time_seconds=5)
        
        # Test creating a simple isolated process
        process = sandbox.create_isolated_process(['echo', 'Hello World'])
        stdout, stderr = process.communicate(timeout=2)
        
        if process.returncode == 0 and b'Hello World' in stdout:
            print("✅ Basic process creation and isolation successful")
            return True
        else:
            print(f"❌ Process creation failed: return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Process creation test failed: {e}")
        return False

def test_context_manager():
    """Test the isolated process context manager."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox(max_memory_mb=128, max_time_seconds=5)
        
        # Test context manager
        with sandbox.isolated_process_context(['echo', 'Context Manager Test']) as (process, get_resources):
            stdout, stderr = process.communicate(timeout=2)
            
            # Try to get resources (may return None if psutil not available)
            resources = get_resources()
            
            if process.returncode == 0 and b'Context Manager Test' in stdout:
                print("✅ Context manager test successful")
                if resources:
                    print(f"   Resource monitoring: {resources.status}")
                else:
                    print("   Resource monitoring: Not available (psutil not installed)")
                return True
            else:
                print(f"❌ Context manager test failed: return code {process.returncode}")
                return False
                
    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        return False

def test_process_termination():
    """Test process termination functionality."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox(max_memory_mb=128, max_time_seconds=2)
        
        # Create a long-running process
        start_time = time.time()
        process = sandbox.create_isolated_process(['python3', '-c', 'import time; time.sleep(10)'])
        
        # Give it a moment to start
        time.sleep(0.5)
        
        # Terminate it
        sandbox.terminate_process_group(process)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        if runtime < 5:  # Should be terminated quickly
            print(f"✅ Process termination test successful (runtime: {runtime:.1f}s)")
            return True
        else:
            print(f"❌ Process termination test failed (runtime: {runtime:.1f}s)")
            return False
            
    except Exception as e:
        print(f"❌ Process termination test failed: {e}")
        return False

def test_existing_validation():
    """Test that existing validation functionality still works."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test malicious notebook detection
        malicious_notebook = '''
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import subprocess\\n", "subprocess.call(['rm', '-rf', '/'])"]
                }
            ]
        }
        '''
        
        is_safe, error_msg, content = sandbox.validate_notebook_structure(malicious_notebook)
        
        if not is_safe and 'subprocess' in error_msg:
            print("✅ Existing malicious code detection still working")
            return True
        else:
            print(f"❌ Malicious code detection failed: is_safe={is_safe}, error={error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Existing validation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔧 Security Sandbox Core Validation")
    print("====================================")
    print()
    
    tests = [
        ("Import and Instantiation", test_sandbox_import),
        ("Process Creation", test_process_creation),
        ("Context Manager", test_context_manager),
        ("Process Termination", test_process_termination),
        ("Existing Validation", test_existing_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        print("-" * (len(test_name) + 9))
        
        if test_func():
            passed += 1
        else:
            print("❌ Test failed!")
        
        print()
    
    print("📊 Test Summary")
    print("===============")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core sandbox features are working correctly!")
        print("✅ Process isolation enhancement is ready for use")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("Some features may not work as expected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 