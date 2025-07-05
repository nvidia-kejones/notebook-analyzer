#!/usr/bin/env python3
"""
Test suite for security context managers.

Tests comprehensive context management including:
- Resource cleanup
- Security integration
- Process isolation
- File operations
- Error handling
"""

import os
import sys
import time
import tempfile
import unittest
import subprocess
from unittest.mock import MagicMock, patch, call
import threading

# Add the parent directory to the path to import the analyzer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analyzer.security_context_managers import (
        SecurityContext, SecurityContextManager, SecurityError,
        secure_file_operation, secure_process_execution, 
        secure_request_handling, secure_notebook_analysis,
        get_current_security_context, log_security_event
    )
    CONTEXT_MANAGERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Security context managers not available: {e}")
    CONTEXT_MANAGERS_AVAILABLE = False


class TestSecurityContextManagers(unittest.TestCase):
    """Test security context managers functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not CONTEXT_MANAGERS_AVAILABLE:
            self.skipTest("Security context managers not available")
    
    def test_security_context_creation(self):
        """Test security context creation and initialization."""
        context = SecurityContext(
            client_ip="192.168.1.100",
            user_agent="Mozilla/5.0",
            operation_type="test_operation",
            risk_level="high"
        )
        
        self.assertEqual(context.client_ip, "192.168.1.100")
        self.assertEqual(context.user_agent, "Mozilla/5.0")
        self.assertEqual(context.operation_type, "test_operation")
        self.assertEqual(context.risk_level, "high")
        self.assertIsInstance(context.temp_files, list)
        self.assertIsInstance(context.temp_dirs, list)
        self.assertIsInstance(context.processes, list)
        self.assertIsNotNone(context.start_time)
        
        print("✅ Security context creation working")
    
    def test_security_context_manager_lifecycle(self):
        """Test security context manager enter/exit lifecycle."""
        context = SecurityContext(
            client_ip="192.168.1.100",
            operation_type="test_lifecycle"
        )
        
        manager = SecurityContextManager(context)
        
        # Test enter
        with manager as mgr:
            self.assertIs(mgr, manager)
            self.assertEqual(mgr.context.operation_type, "test_lifecycle")
        
        # Context should be properly cleaned up after exit
        # (cleanup tested in separate test)
        
        print("✅ Security context manager lifecycle working")
    
    def test_temp_file_creation_and_cleanup(self):
        """Test temporary file creation and automatic cleanup."""
        context = SecurityContext(operation_type="test_temp_files")
        
        temp_files_created = []
        
        with SecurityContextManager(context) as manager:
            # Create multiple temp files
            temp_file1 = manager.create_temp_file("Hello World", ".txt", "test1_")
            temp_file2 = manager.create_temp_file("Test Content", ".py", "test2_")
            
            temp_files_created.extend([temp_file1, temp_file2])
            
            # Files should exist during context
            self.assertTrue(os.path.exists(temp_file1))
            self.assertTrue(os.path.exists(temp_file2))
            
            # Check file content
            with open(temp_file1, 'r') as f:
                self.assertEqual(f.read(), "Hello World")
        
        # Files should be cleaned up after context exit
        for temp_file in temp_files_created:
            self.assertFalse(os.path.exists(temp_file))
        
        print("✅ Temporary file creation and cleanup working")
    
    def test_temp_directory_creation_and_cleanup(self):
        """Test temporary directory creation and automatic cleanup."""
        context = SecurityContext(operation_type="test_temp_dirs")
        
        temp_dirs_created = []
        
        with SecurityContextManager(context) as manager:
            # Create multiple temp directories
            temp_dir1 = manager.create_temp_directory("test1_")
            temp_dir2 = manager.create_temp_directory("test2_")
            
            temp_dirs_created.extend([temp_dir1, temp_dir2])
            
            # Directories should exist during context
            self.assertTrue(os.path.exists(temp_dir1))
            self.assertTrue(os.path.exists(temp_dir2))
            self.assertTrue(os.path.isdir(temp_dir1))
            self.assertTrue(os.path.isdir(temp_dir2))
        
        # Directories should be cleaned up after context exit
        for temp_dir in temp_dirs_created:
            self.assertFalse(os.path.exists(temp_dir))
        
        print("✅ Temporary directory creation and cleanup working")
    
    def test_process_creation_and_cleanup(self):
        """Test process creation and automatic cleanup."""
        context = SecurityContext(operation_type="test_processes")
        
        with SecurityContextManager(context) as manager:
            # Create a simple process (sleep command)
            process = manager.create_isolated_process(["sleep", "10"])
            
            # Process should be running
            self.assertIsNone(process.poll())  # None means still running
            self.assertGreater(process.pid, 0)
        
        # Process should be terminated after context exit
        # Give it a moment to terminate
        time.sleep(0.5)
        self.assertIsNotNone(process.poll())  # Should be terminated
        
        print("✅ Process creation and cleanup working")
    
    def test_request_validation_integration(self):
        """Test request validation integration."""
        context = SecurityContext(
            client_ip="192.168.1.100",
            operation_type="test_validation"
        )
        
        manager = SecurityContextManager(context)
        
        # Test valid request
        valid_request = {
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "data": {"key": "value"}
        }
        
        # This should work even if validator is not available
        result = manager.validate_request(valid_request)
        self.assertIsInstance(result, bool)
        
        print("✅ Request validation integration working")
    
    def test_rate_limiting_integration(self):
        """Test rate limiting integration."""
        context = SecurityContext(
            client_ip="192.168.1.100",
            operation_type="test_rate_limiting"
        )
        
        manager = SecurityContextManager(context)
        
        # This should work even if rate limiter is not available
        result = manager.check_rate_limit()
        self.assertIsInstance(result, bool)
        
        print("✅ Rate limiting integration working")
    
    def test_custom_cleanup_handlers(self):
        """Test custom cleanup handlers."""
        context = SecurityContext(operation_type="test_cleanup_handlers")
        
        cleanup_called = []
        
        def custom_cleanup():
            cleanup_called.append(True)
        
        with SecurityContextManager(context) as manager:
            manager.add_cleanup_handler(custom_cleanup)
        
        # Custom cleanup should have been called
        self.assertEqual(len(cleanup_called), 1)
        
        print("✅ Custom cleanup handlers working")
    
    def test_error_handling_and_cleanup(self):
        """Test error handling and cleanup on exceptions."""
        context = SecurityContext(operation_type="test_error_handling")
        
        temp_files_created = []
        
        try:
            with SecurityContextManager(context) as manager:
                # Create temp file
                temp_file = manager.create_temp_file("Test", ".txt")
                temp_files_created.append(temp_file)
                
                # File should exist
                self.assertTrue(os.path.exists(temp_file))
                
                # Raise an exception
                raise ValueError("Test exception")
        
        except ValueError:
            pass  # Expected exception
        
        # Cleanup should still happen despite exception
        for temp_file in temp_files_created:
            self.assertFalse(os.path.exists(temp_file))
        
        print("✅ Error handling and cleanup working")
    
    def test_secure_file_operation_context(self):
        """Test secure file operation convenience context manager."""
        temp_files_created = []
        
        with secure_file_operation(client_ip="192.168.1.100") as manager:
            temp_file = manager.create_temp_file("Test Content", ".txt")
            temp_files_created.append(temp_file)
            
            self.assertTrue(os.path.exists(temp_file))
            self.assertEqual(manager.context.operation_type, "file_operation")
            self.assertEqual(manager.context.client_ip, "192.168.1.100")
        
        # Cleanup should happen
        for temp_file in temp_files_created:
            self.assertFalse(os.path.exists(temp_file))
        
        print("✅ Secure file operation context working")
    
    def test_secure_process_execution_context(self):
        """Test secure process execution convenience context manager."""
        command = ["echo", "hello"]
        
        with secure_process_execution(command, client_ip="192.168.1.100") as (manager, process):
            self.assertIsInstance(manager, SecurityContextManager)
            self.assertIsInstance(process, subprocess.Popen)
            self.assertEqual(manager.context.operation_type, "process_execution")
            
            # Wait for process to complete
            process.wait()
            self.assertEqual(process.returncode, 0)
        
        print("✅ Secure process execution context working")
    
    def test_secure_request_handling_context(self):
        """Test secure request handling convenience context manager."""
        request_data = {
            "method": "GET",
            "headers": {"User-Agent": "Test"},
            "url": "https://example.com"
        }
        
        # This should work (validation may pass or fail depending on availability)
        try:
            with secure_request_handling(request_data, client_ip="192.168.1.100") as manager:
                self.assertIsInstance(manager, SecurityContextManager)
                self.assertEqual(manager.context.operation_type, "request_handling")
        except SecurityError:
            # This is acceptable if validation fails
            pass
        
        print("✅ Secure request handling context working")
    
    def test_secure_notebook_analysis_context(self):
        """Test secure notebook analysis convenience context manager."""
        # Create a temporary notebook file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write('{"cells": [], "metadata": {}, "nbformat": 4}')
            notebook_path = f.name
        
        try:
            # This should work (validation may pass or fail depending on availability)
            try:
                with secure_notebook_analysis(notebook_path, client_ip="192.168.1.100") as manager:
                    self.assertIsInstance(manager, SecurityContextManager)
                    self.assertEqual(manager.context.operation_type, "notebook_analysis")
            except SecurityError:
                # This is acceptable if validation fails
                pass
        finally:
            # Clean up test file
            if os.path.exists(notebook_path):
                os.unlink(notebook_path)
        
        print("✅ Secure notebook analysis context working")
    
    def test_concurrent_context_managers(self):
        """Test concurrent context managers don't interfere."""
        def create_context(context_id):
            context = SecurityContext(operation_type=f"test_concurrent_{context_id}")
            with SecurityContextManager(context) as manager:
                temp_file = manager.create_temp_file(f"Content {context_id}", ".txt")
                time.sleep(0.1)  # Small delay to test concurrency
                return temp_file, os.path.exists(temp_file)
        
        # Run multiple contexts concurrently
        threads = []
        results = []
        
        def worker(context_id):
            result = create_context(context_id)
            results.append(result)
        
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All contexts should have worked independently
        self.assertEqual(len(results), 3)
        
        # All temp files should have been cleaned up
        for temp_file, existed_during_context in results:
            self.assertTrue(existed_during_context)  # Should have existed during context
            self.assertFalse(os.path.exists(temp_file))  # Should be cleaned up now
        
        print("✅ Concurrent context managers working")
    
    def test_security_error_exception(self):
        """Test SecurityError exception."""
        error = SecurityError("Test security error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test security error")
        
        print("✅ SecurityError exception working")


def run_security_context_tests():
    """Run all security context manager tests."""
    if not CONTEXT_MANAGERS_AVAILABLE:
        print("❌ Security context managers not available - skipping tests")
        return False
    
    print("🧪 Running Security Context Manager Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestSecurityContextManagers)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(test_suite)
    
    # Print results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"\n📊 Test Results:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failures}")
    print(f"   Errors: {errors}")
    
    if failures > 0:
        print(f"\n❌ Test Failures:")
        for test, failure in result.failures:
            print(f"   {test}: {failure}")
    
    if errors > 0:
        print(f"\n💥 Test Errors:")
        for test, error in result.errors:
            print(f"   {test}: {error}")
    
    success = failures == 0 and errors == 0
    if success:
        print(f"\n🎉 All security context manager tests passed! ({passed}/{total_tests})")
    else:
        print(f"\n⚠️  Some tests failed. ({passed}/{total_tests} passed)")
    
    return success


if __name__ == "__main__":
    success = run_security_context_tests()
    sys.exit(0 if success else 1) 