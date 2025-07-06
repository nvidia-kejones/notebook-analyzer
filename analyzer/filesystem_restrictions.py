#!/usr/bin/env python3
"""
Filesystem Restrictions Module

Provides filesystem access restrictions and secure file operations
for the notebook analyzer security sandbox.
"""

import os
import tempfile
import shutil
import stat
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import re
from functools import lru_cache


class FilesystemRestrictions:
    """Configuration for filesystem access restrictions."""
    
    def __init__(self):
        """Initialize with secure default restrictions."""
        self.allowed_read_paths = ['/tmp', '/var/tmp', '/usr/lib/python*', '/usr/local/lib/python*', os.getcwd(), tempfile.gettempdir()]
        self.allowed_write_paths = ['/tmp', '/var/tmp', tempfile.gettempdir()]
        self.blocked_paths = [
            '/etc', '/root', '/home', '/var/log', '/var/lib', '/var/run',
            '/sys', '/proc', '/dev', '/boot', '/bin', '/sbin', '/usr/bin',
            '/usr/sbin', '/opt', '/mnt', '/media', '/srv'
        ]
        self.max_file_size_mb = 100
        self.max_total_files = 1000
        self.temp_dir_prefix = 'notebook_sandbox_'
        self.enforce_permissions = True
        
        # Track created files for cleanup
        self._created_files = set()
        self._created_dirs = set()
    
    @lru_cache(maxsize=128)
    def _get_compiled_pattern(self, pattern: str) -> re.Pattern:
        """Cache compiled regex patterns for better performance."""
        return re.compile(pattern, re.IGNORECASE)
    
    def _normalize_path(self, path: str) -> str:
        """Normalize and resolve path to prevent directory traversal."""
        try:
            # Convert to absolute path and resolve symlinks
            normalized = os.path.abspath(os.path.realpath(path))
            
            # Additional security: ensure no null bytes or control characters
            if '\x00' in normalized or any(ord(c) < 32 for c in normalized if c not in '\t\n\r'):
                raise ValueError("Path contains invalid characters")
            
            return normalized
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid path: {path} - {str(e)}")
    
    def _is_path_allowed(self, path: str, operation: str = 'read') -> Tuple[bool, str]:
        """
        Check if a path is allowed for the specified operation.
        
        Args:
            path: The path to check
            operation: 'read', 'write', or 'execute'
            
        Returns:
            (is_allowed, reason)
        """
        try:
            normalized_path = self._normalize_path(path)
        except ValueError as e:
            return False, str(e)
        
        # Check blocked paths first (highest priority)
        for blocked_path in self.blocked_paths:
            try:
                blocked_normalized = self._normalize_path(blocked_path)
                if normalized_path.startswith(blocked_normalized):
                    return False, f"Path is in blocked directory: {blocked_path}"
            except ValueError:
                continue
        
        # Check allowed paths based on operation
        if operation == 'read':
            allowed_paths = self.allowed_read_paths
        elif operation == 'write':
            allowed_paths = self.allowed_write_paths
        else:
            return False, f"Unknown operation: {operation}"
        
        # Check if path is in allowed directories
        for allowed_path in allowed_paths:
            try:
                # Handle glob patterns in allowed paths
                if '*' in allowed_path:
                    import glob
                    expanded_paths = glob.glob(allowed_path)
                    for expanded_path in expanded_paths:
                        expanded_normalized = self._normalize_path(expanded_path)
                        if normalized_path.startswith(expanded_normalized):
                            return True, ""
                else:
                    allowed_normalized = self._normalize_path(allowed_path)
                    if normalized_path.startswith(allowed_normalized):
                        return True, ""
            except (ValueError, OSError):
                continue
        
        return False, f"Path not in allowed directories for {operation}"
    
    def _validate_file_size(self, file_path: str) -> Tuple[bool, str]:
        """Validate file size against restrictions."""
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = self.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                return False, f"File size ({file_size} bytes) exceeds maximum ({max_size_bytes} bytes)"
            
            return True, ""
        except OSError as e:
            return False, f"Cannot check file size: {str(e)}"
    
    def _validate_text_content(self, content: str, context: str) -> Tuple[bool, str]:
        """Validate text content for security patterns."""
        # Enhanced security patterns
        dangerous_patterns = [
            # Path traversal patterns
            r'\.\./',
            r'\.\.\.',
            r'\.\.\\',
            
            # File URI schemes
            r'file://',
            r'file:\\',
            
            # System paths
            r'/etc/',
            r'/root/',
            r'/home/',
            
            # Filesystem operations
            r'\bsymlink\b',
            r'\bhardlink\b',
            r'\bmount\b',
            r'\bumount\b',
            
            # Dangerous system calls
            r'\bos\.system\b',
            r'\bsubprocess\.',
            r'\beval\b',
            r'\bexec\b',
            
            # Network operations
            r'\bsocket\.',
            r'\burllib\.',
            r'\brequests\.',
        ]
        
        for pattern in dangerous_patterns:
            compiled_pattern = self._get_compiled_pattern(pattern)
            if compiled_pattern.search(content):
                return False, f"Content contains dangerous file system access pattern: {pattern}"
        
        return True, ""
    
    def _enforce_file_permissions(self, file_path: str, mode: int = 0o400):
        """Enforce restrictive file permissions."""
        if self.enforce_permissions:
            try:
                os.chmod(file_path, mode)
            except OSError:
                pass  # Permissions might not be supported on all systems
    
    def create_secure_temp_directory(self, prefix: Optional[str] = None) -> str:
        """Create a secure temporary directory with proper permissions."""
        if prefix is None:
            prefix = self.temp_dir_prefix
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        
        # Track for cleanup
        self._created_dirs.add(temp_dir)
        
        # Set restrictive permissions
        self._enforce_file_permissions(temp_dir, 0o700)
        
        return temp_dir
    
    def create_secure_temp_file(self, content: str, suffix: str = '.tmp', 
                               prefix: Optional[str] = None) -> str:
        """Create a secure temporary file with the given content."""
        if prefix is None:
            prefix = self.temp_dir_prefix
        
        # Validate content size
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
        if content_size_mb > self.max_file_size_mb:
            raise ValueError(f"Content size ({content_size_mb:.2f} MB) exceeds maximum ({self.max_file_size_mb} MB)")
        
        # Validate content for security patterns
        is_safe, error = self._validate_text_content(content, "temp_file")
        if not is_safe:
            raise ValueError(f"Content validation failed: {error}")
        
        # Create temporary file
        fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        
        try:
            # Write content
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Set restrictive permissions (read-only for owner)
            self._enforce_file_permissions(temp_file, 0o400)
            
            # Track for cleanup
            self._created_files.add(temp_file)
            
            return temp_file
        except Exception:
            # Clean up on error
            try:
                os.unlink(temp_file)
            except OSError:
                pass
            raise
    
    def read_file_safely(self, file_path: str, max_size_mb: Optional[int] = None) -> Tuple[bool, str, str]:
        """Safely read a file with security checks."""
        try:
            # Check if path is allowed
            is_allowed, reason = self._is_path_allowed(file_path, 'read')
            if not is_allowed:
                return False, "", f"Read access denied: {reason}"
            
            # Validate file size
            if max_size_mb is None:
                max_size_mb = self.max_file_size_mb
            
            is_valid_size, size_error = self._validate_file_size(file_path)
            if not is_valid_size:
                return False, "", f"File size validation failed: {size_error}"
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Validate content
            is_safe, validation_error = self._validate_text_content(content, "read_file")
            if not is_safe:
                return False, "", f"Content validation failed: {validation_error}"
            
            return True, content, ""
            
        except Exception as e:
            return False, "", f"Error reading file: {str(e)}"
    
    def write_file_safely(self, file_path: str, content: str, 
                         create_dirs: bool = False) -> Tuple[bool, str]:
        """Safely write content to a file with security checks."""
        try:
            # Check if path is allowed
            is_allowed, reason = self._is_path_allowed(file_path, 'write')
            if not is_allowed:
                return False, f"Write access denied: {reason}"
            
            # Validate content size
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if content_size_mb > self.max_file_size_mb:
                return False, f"Content size ({content_size_mb:.2f} MB) exceeds maximum ({self.max_file_size_mb} MB)"
            
            # Validate content for security patterns
            is_safe, validation_error = self._validate_text_content(content, "write_file")
            if not is_safe:
                return False, f"Content validation failed: {validation_error}"
            
            # Create directories if requested
            if create_dirs:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Set restrictive permissions
            self._enforce_file_permissions(file_path, 0o600)
            
            # Track for cleanup
            self._created_files.add(file_path)
            
            return True, ""
            
        except Exception as e:
            return False, f"Error writing file: {str(e)}"
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up a temporary file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            self._created_files.discard(file_path)
        except OSError:
            pass  # File might already be deleted
    
    def cleanup_all_temp_files(self):
        """Clean up all created temporary files and directories."""
        # Clean up files
        for file_path in list(self._created_files):
            self.cleanup_temp_file(file_path)
        
        # Clean up directories
        for dir_path in list(self._created_dirs):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                self._created_dirs.discard(dir_path)
            except OSError:
                pass  # Directory might already be deleted
    
    @contextmanager
    def secure_filesystem_context(self, temp_dir_prefix: Optional[str] = None):
        """Context manager for secure filesystem operations with automatic cleanup."""
        if temp_dir_prefix is None:
            temp_dir_prefix = self.temp_dir_prefix
        
        # Create a new instance for this context
        context = FilesystemRestrictions()
        context.temp_dir_prefix = temp_dir_prefix
        
        try:
            yield context
        finally:
            # Clean up all files created in this context
            context.cleanup_all_temp_files() 