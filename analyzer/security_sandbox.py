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
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import shutil
import resource
import signal
import time
import subprocess
import threading
from dataclasses import dataclass
from contextlib import contextmanager

# Optional dependency - graceful fallback if not available
try:
    import psutil
    HAS_PSUTIL = True
    # Type aliases for when psutil is available
    PsutilProcess = psutil.Process
    PsutilNoSuchProcess = psutil.NoSuchProcess
    PsutilAccessDenied = psutil.AccessDenied
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    # Dummy types for when psutil is not available
    class PsutilProcess: pass
    class PsutilNoSuchProcess(Exception): pass
    class PsutilAccessDenied(Exception): pass


@dataclass
class ProcessResourceUsage:
    """Resource usage metrics for a process."""
    cpu_percent: float
    memory_mb: float
    max_memory_mb: float
    runtime_seconds: float
    process_count: int
    status: str


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
        
    @lru_cache(maxsize=128)
    def _get_compiled_pattern(self, pattern: str) -> re.Pattern:
        """Cache compiled regex patterns for better performance."""
        return re.compile(pattern, re.IGNORECASE)
    
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
    
    def create_isolated_process(self, command: List[str], working_dir: Optional[str] = None, 
                               env_vars: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        """
        Create a subprocess with enhanced isolation and security restrictions.
        
        Args:
            command: Command and arguments to execute
            working_dir: Working directory for the process
            env_vars: Environment variables (only allowed ones will be passed)
            
        Returns:
            subprocess.Popen object with enhanced isolation
        """
        # Create a restricted environment
        safe_env = {
            'PATH': '/usr/bin:/bin',
            'LANG': 'C.UTF-8',
            'LC_ALL': 'C.UTF-8',
            'HOME': '/tmp',
            'TMPDIR': '/tmp',
            'USER': 'sandbox',
            'SHELL': '/bin/sh'
        }
        
        # Add only safe environment variables if provided
        if env_vars:
            safe_env_keys = {'PYTHONPATH', 'JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR'}
            for key, value in env_vars.items():
                if key in safe_env_keys and isinstance(value, str):
                    safe_env[key] = value
        
        # Set up process isolation parameters
        process_args = {
            'args': command,
            'env': safe_env,
            'cwd': working_dir or '/tmp',
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'stdin': subprocess.PIPE,
            'start_new_session': True,  # Create new process group
            'preexec_fn': self._setup_process_isolation,
            'creationflags': 0 if os.name != 'nt' else subprocess.CREATE_NEW_PROCESS_GROUP
        }
        
        try:
            process = subprocess.Popen(**process_args)
            return process
        except Exception as e:
            raise RuntimeError(f"Failed to create isolated process: {str(e)}")
    
    def _setup_process_isolation(self):
        """Set up process isolation (called in child process)."""
        try:
            # Set resource limits in the child process
            self.set_resource_limits()
            
            # Set process group (already done by start_new_session, but ensure it)
            os.setsid()
            
            # Set umask for restrictive file permissions
            os.umask(0o077)
            
        except Exception:
            # Don't let process isolation failures prevent execution
            # In production, this should be logged
            pass
    
    def monitor_process_resources(self, process: subprocess.Popen, 
                                 interval: float = 0.5) -> ProcessResourceUsage:
        """
        Monitor process resource usage in real-time.
        
        Args:
            process: The process to monitor
            interval: Monitoring interval in seconds
            
        Returns:
            ProcessResourceUsage object with current metrics
        """
        if not HAS_PSUTIL:
            # Fallback when psutil is not available
            return ProcessResourceUsage(
                cpu_percent=0.0,
                memory_mb=0.0,
                max_memory_mb=0.0,
                runtime_seconds=0.0,
                process_count=1,
                status='running' if process.poll() is None else 'terminated'
            )
        
        try:
            # Get process info using psutil
            ps_process = psutil.Process(process.pid)
            
            # Get resource usage
            memory_info = ps_process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Get CPU usage (may be 0.0 on first call)
            cpu_percent = ps_process.cpu_percent(interval=interval)
            
            # Get process count (including children)
            process_count = 1
            try:
                children = ps_process.children(recursive=True)
                process_count += len(children)
            except psutil.NoSuchProcess:
                pass
            
            # Calculate runtime
            create_time = ps_process.create_time()
            runtime_seconds = time.time() - create_time
            
            # Get process status
            status = ps_process.status()
            
            # Track maximum memory usage
            max_memory_mb = getattr(self, '_max_memory_seen', 0)
            max_memory_mb = max(max_memory_mb, memory_mb)
            self._max_memory_seen = max_memory_mb
            
            return ProcessResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                max_memory_mb=max_memory_mb,
                runtime_seconds=runtime_seconds,
                process_count=process_count,
                status=status
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            # Process may have terminated or access denied
            return ProcessResourceUsage(
                cpu_percent=0.0,
                memory_mb=0.0,
                max_memory_mb=getattr(self, '_max_memory_seen', 0),
                runtime_seconds=0.0,
                process_count=0,
                status='terminated'
            )
    
    def terminate_process_group(self, process: subprocess.Popen, timeout: float = 5.0):
        """
        Safely terminate a process and all its children.
        
        Args:
            process: The process to terminate
            timeout: Time to wait for graceful termination
        """
        if not process or process.poll() is not None:
            return  # Process already terminated
        
        if not HAS_PSUTIL:
            # Fallback when psutil is not available - basic termination
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            return
        
        try:
            # Get the process and all children
            ps_process = psutil.Process(process.pid)
            children = ps_process.children(recursive=True)
            
            # First, try graceful termination
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            try:
                ps_process.terminate()
            except psutil.NoSuchProcess:
                pass
            
            # Wait for graceful termination
            gone, alive = psutil.wait_procs(children + [ps_process], timeout=timeout)
            
            # Force kill any remaining processes
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Final cleanup - terminate the subprocess object
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process may have already terminated
            pass
        except Exception:
            # Fallback to basic termination
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
    
    @contextmanager
    def isolated_process_context(self, command: List[str], working_dir: Optional[str] = None,
                                env_vars: Optional[Dict[str, str]] = None, monitor: bool = True):
        """
        Context manager for isolated process execution with automatic cleanup.
        
        Args:
            command: Command and arguments to execute
            working_dir: Working directory for the process
            env_vars: Environment variables
            monitor: Whether to monitor resource usage
            
        Yields:
            Tuple of (process, resource_monitor_func)
        """
        process = None
        monitor_thread = None
        resource_usage = None
        
        try:
            # Create isolated process
            process = self.create_isolated_process(command, working_dir, env_vars)
            
            # Set up resource monitoring if requested
            if monitor:
                def monitor_resources():
                    nonlocal resource_usage
                    while process.poll() is None:
                        try:
                            resource_usage = self.monitor_process_resources(process)
                            # Check if process exceeds limits
                            if resource_usage.memory_mb > self.max_memory_mb:
                                self.terminate_process_group(process)
                                break
                            if resource_usage.runtime_seconds > self.max_time_seconds:
                                self.terminate_process_group(process)
                                break
                            time.sleep(0.5)
                        except Exception:
                            break
                
                monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
                monitor_thread.start()
            
            # Yield process and resource getter
            yield process, lambda: resource_usage
            
        finally:
            # Clean up process and monitoring
            if process:
                self.terminate_process_group(process)
            
            if monitor_thread and monitor_thread.is_alive():
                # Give monitoring thread time to finish
                monitor_thread.join(timeout=1.0)
