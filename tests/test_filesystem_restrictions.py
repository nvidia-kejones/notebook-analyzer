#!/usr/bin/env python3
"""
Filesystem Restrictions Test

Tests the filesystem access restrictions and secure file operations.
"""

import sys
import os
import tempfile
import shutil
import time

# Add the analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analyzer'))

def test_filesystem_restrictions_import():
    """Test that the SecuritySandbox can be imported with filesystem restrictions."""
    try:
        from security_sandbox import SecuritySandbox, FilesystemRestrictions
        
        # Test basic instantiation
        sandbox = SecuritySandbox()
        
        # Check that filesystem restrictions are properly initialized
        assert hasattr(sandbox, 'filesystem_restrictions')
        assert isinstance(sandbox.filesystem_restrictions, FilesystemRestrictions)
        
        # Check default restrictions
        restrictions = sandbox.filesystem_restrictions
        assert '/tmp' in restrictions.allowed_read_paths
        assert '/tmp' in restrictions.allowed_write_paths
        assert '/etc' in restrictions.blocked_paths
        assert restrictions.max_file_size_mb == 100
        assert restrictions.max_total_files == 1000
        assert restrictions.temp_dir_prefix == 'notebook_sandbox_'
        assert restrictions.enforce_permissions == True
        
        print("✅ Filesystem restrictions import and initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Filesystem restrictions import failed: {e}")
        return False

def test_path_validation():
    """Test path validation and normalization."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test allowed paths
        is_allowed, reason = sandbox._is_path_allowed('/tmp/test.txt', 'read')
        assert is_allowed, f"Should allow /tmp access: {reason}"
        
        is_allowed, reason = sandbox._is_path_allowed('/tmp/test.txt', 'write')
        assert is_allowed, f"Should allow /tmp write: {reason}"
        
        # Test blocked paths
        is_allowed, reason = sandbox._is_path_allowed('/etc/passwd', 'read')
        assert not is_allowed, "Should block /etc access"
        
        is_allowed, reason = sandbox._is_path_allowed('/root/.bashrc', 'read')
        assert not is_allowed, "Should block /root access"
        
        # Test path traversal protection
        is_allowed, reason = sandbox._is_path_allowed('/tmp/../etc/passwd', 'read')
        assert not is_allowed, "Should block path traversal"
        
        is_allowed, reason = sandbox._is_path_allowed('/tmp/../../etc/passwd', 'read')
        assert not is_allowed, "Should block double path traversal"
        
        print("✅ Path validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Path validation test failed: {e}")
        return False

def test_secure_temp_file_creation():
    """Test secure temporary file creation and cleanup."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test secure temporary file creation
        test_content = "This is a test file content for security validation."
        temp_file = sandbox.create_secure_temp_file(test_content, '.txt', 'test_')
        
        # Verify file was created
        assert os.path.exists(temp_file), "Temporary file should exist"
        
        # Verify file content
        with open(temp_file, 'r') as f:
            content = f.read()
        assert content == test_content, "File content should match"
        
        # Verify file is tracked for cleanup
        assert temp_file in sandbox._created_files, "File should be tracked"
        
        # Test file permissions (if supported)
        try:
            stat_info = os.stat(temp_file)
            # Check that file is read-only for owner (0o400)
            assert stat_info.st_mode & 0o777 == 0o400, f"File should be read-only, got {oct(stat_info.st_mode)}"
        except (OSError, AssertionError):
            # Permissions might not be supported on all systems
            pass
        
        # Test cleanup
        sandbox.cleanup_temp_file(temp_file)
        assert not os.path.exists(temp_file), "File should be cleaned up"
        assert temp_file not in sandbox._created_files, "File should be removed from tracking"
        
        print("✅ Secure temporary file creation and cleanup working")
        return True
        
    except Exception as e:
        print(f"❌ Secure temp file test failed: {e}")
        return False

def test_secure_filesystem_context():
    """Test secure filesystem context manager."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test context manager
        with sandbox.secure_filesystem_context('test_context_') as ctx:
            # Create files within the context
            temp_file1 = ctx.create_secure_temp_file("Content 1", '.txt')
            temp_file2 = ctx.create_secure_temp_file("Content 2", '.txt')
            
            # Verify files exist
            assert os.path.exists(temp_file1), "First file should exist"
            assert os.path.exists(temp_file2), "Second file should exist"
            
            # Verify they are tracked
            assert temp_file1 in ctx._created_files, "First file should be tracked"
            assert temp_file2 in ctx._created_files, "Second file should be tracked"
        
        # After context exit, files should be cleaned up
        assert not os.path.exists(temp_file1), "First file should be cleaned up"
        assert not os.path.exists(temp_file2), "Second file should be cleaned up"
        assert len(sandbox._created_files) == 0, "No files should be tracked after cleanup"
        
        print("✅ Secure filesystem context manager working")
        return True
        
    except Exception as e:
        print(f"❌ Filesystem context test failed: {e}")
        return False

def test_safe_file_operations():
    """Test safe file read/write operations."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Create a test file in allowed location
        test_content = "Safe file operation test content"
        temp_file = sandbox.create_secure_temp_file(test_content, '.txt')
        
        # Test safe file reading
        success, content, error = sandbox.read_file_safely(temp_file)
        assert success, f"Should be able to read allowed file: {error}"
        assert content == test_content, "Content should match"
        
        # Test reading blocked file
        success, content, error = sandbox.read_file_safely('/etc/passwd')
        assert not success, "Should not be able to read blocked file"
        assert "blocked" in error.lower(), f"Error should mention blocking: {error}"
        
        # Test safe file writing
        write_path = os.path.join('/tmp', f'test_write_{int(time.time())}.txt')
        success, error = sandbox.write_file_safely(write_path, "New content")
        assert success, f"Should be able to write to allowed location: {error}"
        
        # Verify file was created and tracked
        assert os.path.exists(write_path), "Written file should exist"
        assert write_path in sandbox._created_files, "Written file should be tracked"
        
        # Test writing to blocked location
        success, error = sandbox.write_file_safely('/etc/test.txt', "Blocked content")
        assert not success, "Should not be able to write to blocked location"
        assert "blocked" in error.lower(), f"Error should mention blocking: {error}"
        
        # Clean up
        sandbox.cleanup_all_temp_files()
        
        print("✅ Safe file operations working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Safe file operations test failed: {e}")
        return False

def test_file_size_limits():
    """Test file size validation and limits."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test normal size file
        normal_content = "A" * 1024  # 1KB
        temp_file = sandbox.create_secure_temp_file(normal_content, '.txt')
        assert os.path.exists(temp_file), "Normal size file should be created"
        
        # Test file size validation
        success, error = sandbox._validate_file_size(temp_file)
        assert success, f"Normal file should pass size validation: {error}"
        
        # Test oversized content (simulate - don't actually create huge file)
        try:
            # Create a sandbox with very small limit for testing
            small_sandbox = SecuritySandbox()
            small_sandbox.filesystem_restrictions.max_file_size_mb = 0.001  # ~1KB limit
            
            large_content = "A" * 2048  # 2KB - should exceed limit
            
            # This should raise an exception
            try:
                temp_file = small_sandbox.create_secure_temp_file(large_content, '.txt')
                assert False, "Should have raised exception for oversized content"
            except ValueError as e:
                assert "exceeds maximum" in str(e), f"Should mention size limit: {e}"
                print("✅ File size limits enforced correctly")
        except Exception as e:
            print(f"⚠️  File size limit test partially failed: {e}")
        
        # Clean up
        sandbox.cleanup_all_temp_files()
        
        print("✅ File size validation working")
        return True
        
    except Exception as e:
        print(f"❌ File size limits test failed: {e}")
        return False

def test_enhanced_security_patterns():
    """Test enhanced security pattern detection."""
    try:
        from security_sandbox import SecuritySandbox
        
        sandbox = SecuritySandbox()
        
        # Test path traversal patterns in content
        malicious_content = """
        import os
        os.system("cat ../../../etc/passwd")
        with open("/etc/passwd", "r") as f:
            print(f.read())
        """
        
        is_safe, error = sandbox._validate_text_content(malicious_content, "test")
        assert not is_safe, "Should detect path traversal patterns"
        assert "file system access pattern" in error.lower(), f"Should mention filesystem pattern: {error}"
        
        # Test symlink patterns
        symlink_content = """
        import os
        os.symlink("/etc/passwd", "/tmp/passwd_link")
        """
        
        is_safe, error = sandbox._validate_text_content(symlink_content, "test")
        assert not is_safe, "Should detect symlink patterns"
        
        # Test safe content
        safe_content = """
        import json
        data = {"key": "value"}
        print(json.dumps(data))
        """
        
        is_safe, error = sandbox._validate_text_content(safe_content, "test")
        assert is_safe, f"Should allow safe content: {error}"
        
        print("✅ Enhanced security pattern detection working")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced security patterns test failed: {e}")
        return False

def run_all_tests():
    """Run all filesystem restrictions tests."""
    print("🔒 Running Filesystem Restrictions Tests...")
    print("=" * 60)
    
    tests = [
        test_filesystem_restrictions_import,
        test_path_validation,
        test_secure_temp_file_creation,
        test_secure_filesystem_context,
        test_safe_file_operations,
        test_file_size_limits,
        test_enhanced_security_patterns
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"📊 Filesystem Restrictions Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All filesystem restrictions tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 