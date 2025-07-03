#!/usr/bin/env python3
"""
Security Sandbox Module for Notebook Analysis

Implements proper sandboxing and isolation to prevent code execution
and limit resource access during notebook analysis.
"""

import os
import tempfile
import json
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil
import resource
import signal
import time


class SecuritySandbox:
    """
    Secure sandbox for analyzing notebook content without risk of code execution.
    """
    
    def __init__(self, max_memory_mb: int = 512, max_time_seconds: int = 30):
        self.max_memory_mb = max_memory_mb
        self.max_time_seconds = max_time_seconds
        self.allowed_imports = {
            'json', 'ast', 're', 'os', 'sys', 'pathlib', 'typing',
            'dataclasses', 'collections', 'itertools', 'functools',
            'math', 'statistics', 'datetime', 'time', 'urllib.parse'
        }
        self.blocked_builtins = {
            'eval', 'exec', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
            'dir', 'getattr', 'setattr', 'hasattr', 'delattr'
        }
        
    def validate_notebook_structure(self, content: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate notebook structure without executing any code.
        Returns (is_safe, error_msg, parsed_content)
        """
        try:
            notebook_data = json.loads(content)
            
            # Basic structure validation
            if not isinstance(notebook_data, dict):
                return False, "Invalid notebook format - not a JSON object", None
            
            if 'cells' not in notebook_data:
                return False, "Invalid notebook format - missing cells", None
            
            if not isinstance(notebook_data['cells'], list):
                return False, "Invalid notebook format - cells must be an array", None
            
            # Validate each cell
            for i, cell in enumerate(notebook_data['cells']):
                if not isinstance(cell, dict):
                    return False, f"Invalid cell {i} - not an object", None
                
                if 'cell_type' not in cell:
                    return False, f"Invalid cell {i} - missing cell_type", None
                
                if cell['cell_type'] not in ['code', 'markdown', 'raw']:
                    return False, f"Invalid cell {i} - unknown cell_type", None
                
                # Validate cell source
                if 'source' in cell:
                    source = cell['source']
                    if isinstance(source, list):
                        source_text = ''.join(source)
                    else:
                        source_text = str(source)
                    
                    # Security validation for code cells
                    if cell['cell_type'] == 'code':
                        is_safe, error_msg = self._validate_code_cell(source_text, i)
                        if not is_safe:
                            return False, error_msg, None
            
            return True, "", notebook_data
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def validate_python_file(self, content: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate Python file content without executing any code.
        Returns (is_safe, error_msg, parsed_content)
        """
        try:
            # Parse AST to validate syntax
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return False, f"Invalid Python syntax: {str(e)}", None
            
            # Security validation
            is_safe, error_msg = self._validate_python_ast(tree)
            if not is_safe:
                return False, error_msg, None
            
            return True, "", {"content": content, "ast": tree}
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _validate_code_cell(self, code: str, cell_index: int) -> Tuple[bool, str]:
        """Validate a single code cell for security issues."""
        try:
            # Try to parse as Python code
            tree = ast.parse(code)
            return self._validate_python_ast(tree, f"cell {cell_index}")
        except SyntaxError:
            # Not valid Python, but might be shell commands or other content
            return self._validate_text_content(code, f"cell {cell_index}")
    
    def _validate_python_ast(self, tree: ast.AST, context: str = "file") -> Tuple[bool, str]:
        """Validate Python AST for dangerous operations."""
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, sandbox):
                self.sandbox = sandbox
                self.violations = []
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.sandbox.blocked_builtins:
                        self.violations.append(f"Blocked builtin function: {node.func.id}")
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous attribute access
                    if isinstance(node.func.value, ast.Name):
                        if (node.func.value.id == 'os' and 
                            node.func.attr in ['system', 'popen', 'spawn', 'exec']):
                            self.violations.append(f"Blocked os.{node.func.attr} call")
                        elif (node.func.value.id == 'subprocess' and 
                              node.func.attr in ['call', 'run', 'Popen', 'check_output']):
                            self.violations.append(f"Blocked subprocess.{node.func.attr} call")
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for dangerous imports
                for alias in node.names:
                    if alias.name not in self.sandbox.allowed_imports:
                        if alias.name in ['subprocess', 'os', 'sys']:
                            self.violations.append(f"Potentially dangerous import: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                # Check for dangerous from imports
                if node.module and node.module not in self.sandbox.allowed_imports:
                    if node.module in ['subprocess', 'os', 'sys']:
                        self.violations.append(f"Potentially dangerous import from: {node.module}")
                self.generic_visit(node)
            
            def visit_Exec(self, node):
                self.violations.append("Blocked exec statement")
                self.generic_visit(node)
            
            def visit_Eval(self, node):
                self.violations.append("Blocked eval statement")
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        
        if visitor.violations:
            return False, f"Security violations in {context}: {'; '.join(visitor.violations)}"
        
        return True, ""
    
    def _validate_text_content(self, content: str, context: str) -> Tuple[bool, str]:
        """Validate text content for dangerous patterns."""
        content_lower = content.lower()
        
        # Shell command patterns
        dangerous_shell_patterns = [
            'rm -rf', 'sudo ', 'chmod 777', 'wget ', 'curl ',
            'nc -l', 'netcat', '/bin/bash', '/bin/sh',
            'python -c', 'eval(', 'exec(', '$(', '`',
            'os.system', 'subprocess.', '__import__'
        ]
        
        # Network/file access patterns
        dangerous_network_patterns = [
            'socket.', 'urllib.', 'requests.', 'http.',
            'ftp.', 'ssh.', 'telnet.', 'smtp.'
        ]
        
        # File system patterns
        dangerous_file_patterns = [
            'open(', 'file(', 'with open', 'os.path.',
            'pathlib.', 'shutil.', 'tempfile.'
        ]
        
        violations = []
        
        for pattern in dangerous_shell_patterns:
            if pattern in content_lower:
                violations.append(f"Dangerous shell pattern: {pattern}")
        
        for pattern in dangerous_network_patterns:
            if pattern in content_lower:
                violations.append(f"Network access pattern: {pattern}")
        
        for pattern in dangerous_file_patterns:
            if pattern in content_lower:
                violations.append(f"File system access pattern: {pattern}")
        
        if violations:
            return False, f"Security violations in {context}: {'; '.join(violations)}"
        
        return True, ""
    
    def create_secure_temp_file(self, content: str, suffix: str = '.tmp') -> str:
        """
        Create a secure temporary file with proper permissions and cleanup.
        """
        # Create a secure temporary directory
        temp_dir = tempfile.mkdtemp(prefix='notebook_sandbox_')
        
        # Set restrictive permissions
        os.chmod(temp_dir, 0o700)
        
        # Create file with secure permissions
        temp_file = os.path.join(temp_dir, f'secure_file{suffix}')
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Set file to read-only
        os.chmod(temp_file, 0o400)
        
        return temp_file
    
    def cleanup_temp_file(self, file_path: str):
        """Securely cleanup temporary files and directories."""
        try:
            if os.path.exists(file_path):
                # Get the directory
                temp_dir = os.path.dirname(file_path)
                
                # Remove the entire temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception:
            # Ignore cleanup errors - but log them in production
            pass
    
    def set_resource_limits(self):
        """Set resource limits to prevent resource exhaustion attacks."""
        try:
            # Limit memory usage
            memory_limit = self.max_memory_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Limit CPU time
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_time_seconds, self.max_time_seconds))
            
            # Limit number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
            
            # Limit file size
            max_file_size = 100 * 1024 * 1024  # 100MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))
            
        except Exception:
            # Resource limits may not be available on all systems
            pass
    
    def timeout_handler(self, signum, frame):
        """Handle timeout signals."""
        raise TimeoutError("Operation timed out")
    
    def with_timeout(self, func, *args, **kwargs):
        """Execute function with timeout protection."""
        # Set up timeout handler
        old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.max_time_seconds)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore old handler and cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
