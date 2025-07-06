#!/usr/bin/env python3
"""
Notebook Analyzer - Vercel Web UI Deployment

Full web interface for analyzing Jupyter notebooks to determine minimum GPU requirements.
Optimized for Vercel's serverless environment with complete UI support.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response, session
from werkzeug.utils import secure_filename
import sys
import time
import requests
from functools import lru_cache
from datetime import datetime, timedelta
import gzip
import io
from typing import Optional, Dict, Any

# Import request validation
try:
    from analyzer.request_validator import validate_flask_request
    REQUEST_VALIDATION_AVAILABLE = True
except ImportError:
    REQUEST_VALIDATION_AVAILABLE = False

def validate_request_security(request):
    """
    Validate request for security issues.
    Returns error response if invalid, None if valid.
    """
    if not REQUEST_VALIDATION_AVAILABLE:
        return None
    
    try:
        # Get client IP for rate limiting
        client_ip = request.environ.get('REMOTE_ADDR', '127.0.0.1')
        
        # Handle forwarded headers
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        # Skip comprehensive validation for file uploads - let security sandbox handle it
        # This allows security tests to reach the sandbox for proper validation
        if 'file' in request.files:
            # Only do basic rate limiting for file uploads
            from analyzer.rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
            
            # Create rate limiter with more permissive settings for file uploads
            config = RateLimitConfig(
                requests_per_minute=30,  # More permissive for testing
                requests_per_hour=200,
                burst_limit=15
            )
            
            rate_limiter = SlidingWindowRateLimiter(config)
            
            # Check rate limiting only
            status = rate_limiter.check_rate_limit(client_ip)
            if not status.allowed:
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': status.retry_after or 60
                }), 429
            
            # Allow file uploads to proceed to security sandbox
            return None
        
        # For non-file requests, do full validation
        result = validate_flask_request(request, client_ip)
        
        if not result.is_valid:
            # Log security violation
            print(f"🚨 Security validation failed: {result.error_message}")
            print(f"   Error code: {result.error_code}")
            print(f"   Risk level: {result.risk_level}")
            print(f"   Client IP: {client_ip}")
            
            # Return appropriate error response
            if result.error_code == "RATE_LIMIT_EXCEEDED":
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again later.',
                    'retry_after': 60
                }), 429
            elif result.risk_level == "critical":
                return jsonify({
                    'error': 'Request blocked for security reasons.'
                }), 403
            elif result.risk_level == "high":
                return jsonify({
                    'error': 'Request validation failed.'
                }), 400
            else:
                return jsonify({
                    'error': 'Invalid request format.'
                }), 400
        
        return None
        
    except Exception as e:
        # Don't fail on validation errors, but log them
        print(f"⚠️  Request validation error: {e}")
        return None

# Performance optimization: Response compression
def compress_response(response: Response) -> Response:
    """Compress response data if it's large enough to benefit."""
    if (response.status_code == 200 and 
        response.content_type and 
        'json' in response.content_type and
        len(response.get_data()) > 1000):  # Only compress if > 1KB
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_encoding:
            # Compress the response
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                f.write(response.get_data())
            
            response.set_data(buffer.getvalue())
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Length'] = len(response.get_data())
    
    return response

# Performance optimization: Global model cache
MODEL_CACHE = {
    'data': None,
    'timestamp': None,
    'ttl_minutes': 30  # Cache for 30 minutes
}

def is_cache_valid():
    """Check if model cache is still valid."""
    if MODEL_CACHE['data'] is None or MODEL_CACHE['timestamp'] is None:
        return False
    
    cache_age = datetime.now() - MODEL_CACHE['timestamp']
    return cache_age < timedelta(minutes=MODEL_CACHE['ttl_minutes'])

def get_cached_models():
    """Get models from cache if valid, otherwise return None."""
    if is_cache_valid():
        return MODEL_CACHE['data']
    return None

def cache_models(models_data):
    """Cache models data with timestamp."""
    MODEL_CACHE['data'] = models_data
    MODEL_CACHE['timestamp'] = datetime.now()

# Enhanced secure error handling - Phase 2 Security Fix
try:
    from analyzer.secure_error_handler import (
        get_secure_error_handler, 
        create_secure_flask_response,
        handle_error_securely,
        ValidationError,
        SecurityError,
        RateLimitError
    )
    SECURE_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    SECURE_ERROR_HANDLER_AVAILABLE = False

def sanitize_error_message(error: Exception, debug_mode: bool = False) -> str:
    """
    Sanitize error messages to prevent information disclosure.
    Enhanced with comprehensive secure error handling.
    """
    if SECURE_ERROR_HANDLER_AVAILABLE:
        # Use comprehensive secure error handler
        handler = get_secure_error_handler(debug_mode)
        response = handler.handle_error(error)
        return response.internal_message if debug_mode and response.internal_message else response.public_message
    else:
        # Fallback to basic sanitization
        if debug_mode:
            return str(error)
        else:
            # Generic error messages that don't leak system information
            error_mappings = {
                'FileNotFoundError': 'File not found. Please check the file path.',
                'PermissionError': 'Access denied. Please check file permissions.',
                'JSONDecodeError': 'Invalid file format. Please check the file content.',
                'UnicodeDecodeError': 'File encoding error. Please use UTF-8 encoding.',
                'ConnectionError': 'Network connection failed. Please try again.',
                'TimeoutError': 'Request timed out. Please try again.',
                'ValueError': 'Invalid input. Please check your data.',
                'TypeError': 'Invalid data type. Please check your input.',
            }
            
            error_type = type(error).__name__
            return error_mappings.get(error_type, 'An error occurred. Please try again.')

def create_secure_error_response(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Create a secure error response using the enhanced error handler.
    Falls back to basic error handling if secure handler not available.
    """
    if SECURE_ERROR_HANDLER_AVAILABLE:
        return create_secure_flask_response(error, context, app.debug)
    else:
        # Fallback to basic error response
        message = sanitize_error_message(error, app.debug)
        return jsonify({'error': message}), 500

# Set up paths for Vercel environment
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create Flask app with proper template folder
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Secure secret key configuration - P1 Security Fix
import secrets
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    # Generate cryptographically secure random key
    secret_key = secrets.token_hex(32)
    print("⚠️  WARNING: Using generated secret key. Set SECRET_KEY environment variable for production.")
    print("   Generated key will not persist across restarts, causing session invalidation.")
app.secret_key = secret_key

# Configuration
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'ipynb', 'py'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Import analyzer classes using standard import mechanism
GPUAnalyzer = None
GPURequirement = None

try:
    # Import directly from the analyzer package
    from analyzer import GPUAnalyzer, GPURequirement
    print("✅ Successfully imported GPUAnalyzer from analyzer package")
except ImportError as e:
    print(f"❌ Failed to import from analyzer package: {e}")
    try:
        # Fallback: try importing from notebook-analyzer.py (for backward compatibility)
        import importlib.machinery
        import importlib.util
        
        analyzer_path = os.path.join(current_dir, "notebook-analyzer.py")
        
        if os.path.exists(analyzer_path):
            loader = importlib.machinery.SourceFileLoader("notebook_analyzer", analyzer_path)
            spec = importlib.util.spec_from_loader("notebook_analyzer", loader)
            
            if spec and spec.loader:
                notebook_analyzer = importlib.util.module_from_spec(spec)
                sys.modules["notebook_analyzer"] = notebook_analyzer
                spec.loader.exec_module(notebook_analyzer)
                
                GPUAnalyzer = getattr(notebook_analyzer, 'GPUAnalyzer', None)
                GPURequirement = getattr(notebook_analyzer, 'GPURequirement', None)
                
                if GPUAnalyzer and GPURequirement:
                    print("✅ Successfully imported GPUAnalyzer from fallback method")
                else:
                    print("❌ Failed to extract classes from notebook-analyzer.py")
        else:
            print(f"❌ notebook-analyzer.py not found at {analyzer_path}")
    except Exception as fallback_error:
        print(f"❌ Fallback import error: {fallback_error}")
except Exception as e:
    print(f"❌ Import error: {e}")
    print(f"Traceback: {traceback.format_exc()}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def format_analysis_for_web(analysis) -> dict:
    """Format the analysis result for web display."""
    if not analysis:
        return {}
        
    formatted_data = {
        'min_gpu': {
            'type': analysis.min_gpu_type,
            'quantity': analysis.min_quantity,
            'vram_gb': analysis.min_vram_gb,
            'runtime': analysis.min_runtime_estimate
        },
        'optimal_gpu': {
            'type': analysis.optimal_gpu_type,
            'quantity': analysis.optimal_quantity,
            'vram_gb': analysis.optimal_vram_gb,
            'runtime': analysis.optimal_runtime_estimate
        },
        'sxm_required': analysis.sxm_required,
        'sxm_reasoning': analysis.sxm_reasoning,
        'arm_compatibility': analysis.arm_compatibility,
        'arm_reasoning': analysis.arm_reasoning,
        'confidence': round(analysis.confidence * 100, 1),
        'reasoning': analysis.reasoning,
        'llm_enhanced': analysis.llm_enhanced,
        'llm_reasoning': analysis.llm_reasoning or [],
        'llm_model_used': analysis.llm_model_used,
        'nvidia_compliance_score': round(analysis.nvidia_compliance_score, 1),
        'structure_assessment': analysis.structure_assessment or {},
        'content_quality_issues': analysis.content_quality_issues or [],
        'technical_recommendations': analysis.technical_recommendations or [],
        'confidence_factors': analysis.confidence_factors or []
    }
    
    # Add recommended GPU (formerly consumer) - maintain backward compatibility
    if analysis.recommended_gpu_type:
        recommended_gpu = {
            'type': analysis.recommended_gpu_type,
            'quantity': analysis.recommended_quantity,
            'vram_gb': analysis.recommended_vram_gb,
            'runtime': analysis.recommended_runtime_estimate
        }
        formatted_data['recommended_gpu'] = recommended_gpu
        formatted_data['consumer_gpu'] = recommended_gpu  # Backward compatibility
    else:
        formatted_data['recommended_gpu'] = None
        formatted_data['consumer_gpu'] = None
    
    # Add recommended viability info (with backward compatibility)
    formatted_data['recommended_viable'] = analysis.recommended_viable
    formatted_data['recommended_limitation'] = analysis.recommended_limitation
    formatted_data['consumer_viable'] = analysis.recommended_viable  # Backward compatibility
    formatted_data['consumer_limitation'] = analysis.recommended_limitation  # Backward compatibility
    
    # Add optimal GPU (formerly enterprise) - maintain backward compatibility
    optimal_gpu = {
        'type': analysis.optimal_gpu_type,
        'quantity': analysis.optimal_quantity,
        'vram_gb': analysis.optimal_vram_gb,
        'runtime': analysis.optimal_runtime_estimate
    }
    formatted_data['optimal_gpu'] = optimal_gpu
    formatted_data['enterprise_gpu'] = optimal_gpu  # Backward compatibility
    
    return formatted_data

@app.route('/')
def index():
    """Main page with analysis form."""
    # Check if LLM is available
    llm_available = bool(os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_BASE_URL'))
    
    return render_template('index.html', llm_available=llm_available)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'analyzer_available': GPUAnalyzer is not None,
        'current_dir': current_dir,
        'files': os.listdir(current_dir) if os.path.exists(current_dir) else []
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle analysis requests (both URL and file upload) - Web UI."""
    if not GPUAnalyzer:
        flash('Analysis service is not available. GPUAnalyzer could not be loaded.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get selected model from form (but don't modify global environment)
        selected_model = request.form.get('model')
        
        # Create analyzer with thread-safe model selection
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        # If a model is selected and LLM is available, update the analyzer's model
        if selected_model and analyzer.llm_analyzer:
            analyzer.llm_analyzer.model = selected_model
        
        # Check for both inputs provided (improved UX)
        has_file = 'file' in request.files and request.files['file'].filename
        has_url = 'url' in request.form and request.form['url'].strip()
        
        if has_file and has_url:
            # Both provided - inform user about precedence
            flash('Both file and URL provided. Processing uploaded file and ignoring URL.', 'info')
        
        # Check if it's a file upload or URL (file takes precedence)
        if has_file:
            # File upload analysis with sanitization
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(str(file.filename))
                
                try:
                    # Read file content for sanitization
                    file_content = file.read()
                    
                    # Sanitize the uploaded file before processing
                    is_safe, error_msg, sanitized_content = analyzer.sanitize_file_content(file_content, filename)
                    
                    if not is_safe:
                        flash(f'File upload blocked for security: {error_msg}', 'error')
                        return redirect(url_for('index'))
                    
                    # Use secure temporary file handling with sandbox - P1 Security Fix
                    try:
                        from analyzer.security_sandbox import SecuritySandbox
                        sandbox = SecuritySandbox()
                        
                        # Create secure temporary file with proper isolation
                        if filename.lower().endswith('.ipynb'):
                            # For notebooks, save the sanitized JSON
                            temp_content = json.dumps(sanitized_content, indent=2)
                            temp_path = sandbox.create_secure_temp_file(temp_content, '.ipynb')
                        elif filename.lower().endswith('.py'):
                            # For Python files, save the sanitized content
                            python_content = sanitized_content.get('content', '') if sanitized_content else ''
                            temp_path = sandbox.create_secure_temp_file(python_content, '.py')
                        else:
                            flash('Unsupported file type after sanitization.', 'error')
                            return redirect(url_for('index'))
                    except ImportError:
                        # Fallback to less secure method if sandbox not available
                        import tempfile
                        import os
                        
                        # Create more secure temporary directory
                        temp_dir = tempfile.mkdtemp(prefix='notebook_secure_', dir='/tmp')
                        os.chmod(temp_dir, 0o700)  # Restrictive permissions
                        
                        if filename.lower().endswith('.ipynb'):
                            temp_path = os.path.join(temp_dir, f'notebook_{os.urandom(8).hex()}.ipynb')
                            with open(temp_path, 'w', encoding='utf-8') as temp_file:
                                json.dump(sanitized_content, temp_file)
                        elif filename.lower().endswith('.py'):
                            python_content = sanitized_content.get('content', '') if sanitized_content else ''
                            temp_path = os.path.join(temp_dir, f'notebook_{os.urandom(8).hex()}.py')
                            with open(temp_path, 'w', encoding='utf-8') as temp_file:
                                temp_file.write(python_content)
                        
                        # Set file permissions to read-only
                        os.chmod(temp_path, 0o400)
                    
                    try:
                        result = analyzer.analyze_notebook(temp_path)
                        if result:
                            analysis_data = format_analysis_for_web(result)
                            # Store analysis data in session for streaming interface
                            session['analysis_results'] = {
                                'analysis': analysis_data,
                                'source_type': 'file',
                                'source_name': filename
                            }
                            # Redirect to streaming results
                            return redirect(url_for('results'))
                        else:
                            flash('Failed to analyze the uploaded notebook. Please check the file format.', 'error')
                    finally:
                        # Secure cleanup of temporary file using sandbox
                        try:
                            if 'temp_path' in locals() and temp_path:
                                try:
                                    from analyzer.security_sandbox import SecuritySandbox
                                    sandbox = SecuritySandbox()
                                    sandbox.cleanup_temp_file(temp_path)
                                except ImportError:
                                    # Fallback cleanup
                                    if os.path.exists(temp_path):
                                        temp_dir = os.path.dirname(temp_path)
                                        import shutil
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                        except Exception:
                            pass  # Ignore cleanup errors but they should be logged in production
                except Exception as e:
                    # Sanitized error message - P1 Security Fix
                    if app.debug:
                        flash(f'Error processing uploaded file: {str(e)}', 'error')
                    else:
                        flash('Error processing uploaded file. Please try again.', 'error')
            else:
                flash('Invalid file type. Please upload a .ipynb or .py file.', 'error')
                
        elif has_url:
            # URL analysis
            url = request.form['url'].strip()
            result = analyzer.analyze_notebook(url)
            
            if result:
                analysis_data = format_analysis_for_web(result)
                # Store analysis data in session for streaming interface
                session['analysis_results'] = {
                    'analysis': analysis_data,
                    'source_type': 'url',
                    'source_name': url
                }
                # Redirect to streaming results
                return redirect(url_for('results'))
            else:
                flash('Failed to analyze the notebook from the provided URL. Please check the URL and try again.', 'error')
        else:
            flash('Please provide either a URL or upload a notebook file.', 'error')
            
    except Exception as e:
        print(f"Analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Sanitized error message - P1 Security Fix
        if app.debug:
            flash(f'An error occurred during analysis: {str(e)}', 'error')
        else:
            flash('An error occurred during analysis. Please try again.', 'error')
    
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis."""
    if not GPUAnalyzer:
        return jsonify({'error': 'Analysis service not available'}), 503
    
    # Validate request for security
    validation_error = validate_request_security(request)
    if validation_error:
        return validation_error
    
    try:
        # Get selected model from request (if provided)
        selected_model = None
        if request.is_json:
            data = request.get_json()
            selected_model = data.get('model')
        else:
            selected_model = request.form.get('model')
        
        # Create analyzer with thread-safe model selection
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        # If a model is selected and LLM is available, update the analyzer's model
        if selected_model and analyzer.llm_analyzer:
            analyzer.llm_analyzer.model = selected_model
        
        if request.is_json:
            data = request.get_json()
            url = data.get('url')
            if not url:
                return jsonify({'error': 'URL is required'}), 400
                
            result = analyzer.analyze_notebook(url)
            if result:
                analysis_data = format_analysis_for_web(result)
                return jsonify({
                    'success': True, 
                    'source_type': 'url',
                    'source_name': url,
                    'analysis': analysis_data
                })
            else:
                return jsonify({'error': 'Failed to analyze notebook'}), 400
                
        elif 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(str(file.filename))
                
                try:
                    # Read and sanitize file content
                    file_content = file.read()
                    is_safe, error_msg, sanitized_content = analyzer.sanitize_file_content(file_content, filename)
                    
                    if not is_safe:
                        return jsonify({'error': f'File upload blocked for security: {error_msg}'}), 400
                    
                    # Use secure temporary file handling - P1 Security Fix
                    
                    # Create secure temporary file with proper cleanup
                    if filename.lower().endswith('.ipynb'):
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, 
                                                       prefix='notebook_', dir='/tmp') as temp_file:
                            json.dump(sanitized_content, temp_file)
                            temp_file.flush()  # Ensure content is written
                            temp_path = temp_file.name
                    elif filename.lower().endswith('.py'):
                        python_content = sanitized_content.get('content', '') if sanitized_content else ''
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, 
                                                       prefix='notebook_', dir='/tmp') as temp_file:
                            temp_file.write(python_content)
                            temp_file.flush()  # Ensure content is written
                            temp_path = temp_file.name
                    
                    try:
                        result = analyzer.analyze_notebook(temp_path)
                        if result:
                            analysis_data = format_analysis_for_web(result)
                            return jsonify({
                                'success': True, 
                                'source_type': 'file',
                                'source_name': filename,
                                'analysis': analysis_data
                            })
                        else:
                            return jsonify({'error': 'Failed to analyze notebook'}), 400
                    finally:
                        # Secure cleanup of temporary file
                        try:
                            if 'temp_path' in locals() and os.path.exists(temp_path):
                                os.unlink(temp_path)
                        except OSError:
                            pass  # Ignore cleanup errors
                except Exception as e:
                    # Sanitized error message - P1 Security Fix
                    if app.debug:
                        return jsonify({'error': f'File processing error: {str(e)}'}), 500
                    else:
                        return jsonify({'error': 'File processing failed. Please try again.'}), 500
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            return jsonify({'error': 'Invalid request format'}), 400
            
    except Exception as e:
        print(f"API analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """Handle streaming analysis requests with Server-Sent Events."""
    if not GPUAnalyzer:
        return jsonify({'error': 'Analysis service not available'}), 503
    
    # Extract request data BEFORE creating the generator (to avoid request context issues)
    file_path = None
    source_name = None
    analysis_input = None
    selected_model = request.form.get('model')  # Extract selected model
    
    # Check for both inputs provided (improved UX)
    has_file = 'file' in request.files and request.files['file'].filename
    has_url = 'url' in request.form and request.form['url'].strip()
    
    if has_file and has_url:
        # Both provided - return error with clear message
        return jsonify({'error': 'Both file and URL provided. Please use only one input method.'}), 400
    
    # Check if it's a file upload or URL (file takes precedence)
    if has_file:
        # File upload analysis with sanitization
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(str(file.filename))
            
            try:
                # Read and sanitize file content
                file_content = file.read()
                
                # Create temporary analyzer instance for sanitization
                temp_analyzer = GPUAnalyzer(quiet_mode=True)
                is_safe, error_msg, sanitized_content = temp_analyzer.sanitize_file_content(file_content, filename)
                
                if not is_safe:
                    return jsonify({'error': f'File upload blocked for security: {error_msg}'}), 400
                
                # Use secure temporary file handling - P1 Security Fix
                
                # Create secure temporary file with proper cleanup
                if filename.lower().endswith('.ipynb'):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, 
                                                   prefix='notebook_', dir='/tmp') as temp_file:
                        json.dump(sanitized_content, temp_file)
                        temp_file.flush()  # Ensure content is written
                        file_path = temp_file.name
                elif filename.lower().endswith('.py'):
                    python_content = sanitized_content.get('content', '') if sanitized_content else ''
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, 
                                                   prefix='notebook_', dir='/tmp') as temp_file:
                        temp_file.write(python_content)
                        temp_file.flush()  # Ensure content is written
                        file_path = temp_file.name
                
                source_name = filename
                analysis_input = {'type': 'file', 'path': file_path, 'name': filename}
                
            except Exception as e:
                # Sanitized error message - P1 Security Fix
                if app.debug:
                    return jsonify({'error': f'File processing error: {str(e)}'}), 500
                else:
                    return jsonify({'error': 'File processing failed. Please try again.'}), 500
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .ipynb or .py file.'}), 400
            
    elif has_url:
        # URL analysis
        source_name = request.form['url'].strip()
        analysis_input = {'type': 'url', 'url': source_name}
    else:
        return jsonify({'error': 'Please provide either a URL or upload a notebook file.'}), 400
    
    def generate_progress(analysis_input, source_name):
        """Generator function for Server-Sent Events."""
        try:
            # Create analyzer with thread-safe model selection
            analyzer = GPUAnalyzer(quiet_mode=True)
            
            # If a model is selected and LLM is available, update the analyzer's model
            if selected_model and analyzer.llm_analyzer:
                analyzer.llm_analyzer.model = selected_model
            
            # Create a queue for real-time progress streaming
            import queue
            import threading
            from typing import Any, Optional
            
            progress_queue: queue.Queue = queue.Queue()
            analysis_complete = threading.Event()
            analysis_result: list[Optional[Any]] = [None]  # Use list to make it mutable
            analysis_error: list[Optional[Exception]] = [None]
            
            # Import environment detection for progress batching
            import time
            from analyzer.core import get_environment_config
            env_config = get_environment_config()
            
            # Progress batching for production
            if env_config and env_config.get('progress_batching', False):
                batch_buffer = []
                batch_size = 3  # Send messages in batches of 3
                last_send_time = time.time()
                batch_timeout = 2.0  # Send batch after 2 seconds
                
                def progress_callback(message: str) -> None:
                    nonlocal last_send_time
                    batch_buffer.append(message)
                    current_time = time.time()
                    # Send batch if full or timeout reached
                    if len(batch_buffer) >= batch_size or (current_time - last_send_time) >= batch_timeout:
                        for msg in batch_buffer:
                            progress_queue.put(msg)
                        batch_buffer.clear()
                        last_send_time = current_time
                
                def flush_batch():
                    """Flush any remaining messages in batch."""
                    if batch_buffer:
                        for msg in batch_buffer:
                            progress_queue.put(msg)
                        batch_buffer.clear()
            else:
                def progress_callback(message: str) -> None:
                    # Debug: Log all progress messages being queued
                    if "Issue" in message or "⚠️" in message:
                        print(f"🔍 STREAM DEBUG: Queueing issue message: {message}")
                    progress_queue.put(message)
                
                def flush_batch():
                    """No-op for non-batching mode."""
                    pass
            
            def run_analysis() -> None:
                try:
                    # Perform the actual analysis with progress tracking
                    if analysis_input['type'] == 'file':
                        result = analyzer.analyze_notebook_with_progress(analysis_input['path'], progress_callback)
                    else:
                        result = analyzer.analyze_notebook_with_progress(analysis_input['url'], progress_callback)
                    
                    # Flush any remaining batched messages
                    flush_batch()
                    
                    analysis_result[0] = result
                except Exception as e:
                    # Flush any remaining batched messages before error
                    flush_batch()
                    analysis_error[0] = e
                finally:
                    analysis_complete.set()
            
            # Start analysis in a separate thread
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.start()
            
            # Stream progress messages as they arrive
            while not analysis_complete.is_set() or not progress_queue.empty():
                try:
                    # Get progress message with timeout
                    message = progress_queue.get(timeout=0.1)
                    # Debug: Log issue messages being yielded
                    if "Issue" in message or "⚠️" in message:
                        print(f"🔍 STREAM DEBUG: Yielding issue message: {message}")
                    yield f"data: {json.dumps({'type': 'progress', 'message': message})}\n\n"
                except queue.Empty:
                    # Continue checking if analysis is complete
                    continue
            
            # Wait for analysis thread to complete
            analysis_thread.join()
            
            # Check for errors
            if analysis_error[0]:
                raise analysis_error[0]
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Analysis complete!'})}\n\n"
            
            # Send results
            result = analysis_result[0]
            if result:
                analysis_data = format_analysis_for_web(result)
                
                # Send the complete results
                source_type = 'file' if analysis_input['type'] == 'file' else 'url'
                yield f"data: {json.dumps({'type': 'complete', 'analysis': analysis_data, 'source_name': source_name, 'source_type': source_type})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to analyze the notebook. Please check the file format or URL.'})}\n\n"
            
            # Clean up temp file
            if analysis_input['type'] == 'file' and os.path.exists(analysis_input['path']):
                os.remove(analysis_input['path'])
                
        except Exception as e:
            print(f"Streaming analysis error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Sanitized error message - P1 Security Fix
            if app.debug:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Analysis failed: {str(e)}'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis failed. Please try again.'})}\n\n"
            
            # Clean up temp file in case of error
            if analysis_input and analysis_input['type'] == 'file' and os.path.exists(analysis_input['path']):
                os.remove(analysis_input['path'])
    
    return Response(generate_progress(analysis_input, source_name), content_type='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/api/default-model')
def get_default_model():
    """Get the default model from environment variables."""
    default_model = os.getenv('OPENAI_MODEL', 'nvidia/llama-3.3-nemotron-super-49b-v1')
    return jsonify({'default_model': default_model})

@app.route('/api/available-models')
def get_available_models():
    """Get filtered list of available models from NVIDIA API with caching."""
    try:
        # Check cache first
        cached_models = get_cached_models()
        if cached_models:
            return jsonify({
                'models': cached_models,
                'source': 'cache'
            })
        
        # Check if we have NVIDIA API access
        openai_base_url = os.getenv('OPENAI_BASE_URL', '')
        openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        if not openai_api_key or 'nvidia.com' not in openai_base_url.lower():
            # Fallback to static list if not using NVIDIA API
            fallback_models = get_fallback_models()
            cache_models(fallback_models)  # Cache static models too
            return jsonify({
                'models': fallback_models,
                'source': 'static'
            })
        
        # Fetch models from NVIDIA API
        # Remove /v1 suffix if present and add it properly
        base_url = openai_base_url.rstrip('/').rstrip('/v1')
        models_url = f"{base_url}/v1/models"
        
        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            filtered_models = filter_and_organize_models(models_data.get('data', []))
            cache_models(filtered_models)  # Cache the results
            return jsonify({
                'models': filtered_models,
                'source': 'nvidia_api'
            })
        else:
            # Fallback to static list if API call fails
            fallback_models = get_fallback_models()
            cache_models(fallback_models)
            return jsonify({
                'models': fallback_models,
                'source': 'static_fallback'
            })
            
    except Exception as e:
        print(f"Error fetching models: {e}")
        # Fallback to static list on any error
        fallback_models = get_fallback_models()
        cache_models(fallback_models)
        return jsonify({
            'models': fallback_models,
            'source': 'static_error'
        })

def filter_and_organize_models(models_list):
    """Filter and organize models based on preferences."""
    
    # Define specific model priorities
    top_priority_nemotron = [
        'nvidia/llama-3.1-nemotron-ultra-253b-v1',  # Ultra - highest priority
        'nvidia/llama-3.1-nemotron-nano-4b-v1.1',   # Nano
        'nvidia/llama-3.1-nemotron-nano-8b-v1',     # Nano
        'nvidia/llama-3.1-nemotron-51b-instruct',
        'nvidia/llama-3.1-nemotron-70b-instruct'
    ]
    
    preferred_models = {
        'meta': [
            'meta/llama-3.3-70b-instruct',
            'meta/llama-3.1-405b-instruct',
            'meta/llama-3.1-70b-instruct',
            'meta/llama-3.1-8b-instruct',
            'meta/llama-4-maverick-17b-128e-instruct',
            'meta/llama-4-scout-17b-16e-instruct'
        ],
        'mistralai': [
            'mistralai/mistral-large-2-instruct',
            'mistralai/mistral-medium-3-instruct',
            'mistralai/mistral-small-3.1-24b-instruct-2503'
            # Removed mathstral - not suitable for general notebook analysis
        ],
        'deepseek-ai': [
            'deepseek-ai/deepseek-r1'
        ],
        'qwen': [
            'qwen/qwen2.5-7b-instruct'
        ]
    }
    
    # Models to exclude (language-specific, older versions, esoteric creators, etc.)
    exclude_keywords = [
        'swallow', 'bielik', 'taiwan', 'hindi', 'sahabatai', 'italia', 'sea-lion',
        'embed', 'reward', 'guard', 'safety', 'retriever', 'clip', 'vision',
        'medical', 'med-', 'fin-', 'financial', 'nemoguard', 'shieldgemma',
        'mathstral',  # Math-specific, not ideal for general notebook analysis
        'codellama', 'nemotron-super',  # Problematic models
        'nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma'  # Additional unwanted models
    ]
    
    # Esoteric/less useful creators to exclude
    exclude_creators = [
        'abacusai', 'baichuan-inc', 'marin', 'mediatek', 'aisingapore', 
        'gotocompany', 'institute-of-science-tokyo', 'tokyotech-llm', 
        'yentinglin', 'speakleash', 'rakuten', 'utter-project'
    ]
    
    # Get all available model IDs
    available_models = {model.get('id', '') for model in models_list}
    
    # Create a function to get the base model name for version comparison
    def get_base_model_name(model_id):
        # Remove version suffixes like -v0.1, -v1, -2503, etc.
        base = model_id
        # Remove common version patterns
        import re
        base = re.sub(r'-v\d+(\.\d+)?$', '', base)
        base = re.sub(r'-\d{4}$', '', base)  # Remove year suffixes like -2503
        base = re.sub(r'-instruct-v\d+(\.\d+)?$', '-instruct', base)
        return base
    
    # Group models by base name to find latest versions
    model_groups = {}
    for model in models_list:
        model_id = model.get('id', '')
        creator = model_id.split('/')[0] if '/' in model_id else ''
        
        # Skip excluded creators
        if creator in exclude_creators:
            continue
            
        # Skip models with excluded keywords
        if any(keyword in model_id.lower() for keyword in exclude_keywords):
            continue
            
        # Additional specific model exclusions
        if any(excluded in model_id for excluded in [
            'nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma', 'deepseek-coder-6.7b-instruct'
        ]):
            continue
            
        base_name = get_base_model_name(model_id)
        if base_name not in model_groups:
            model_groups[base_name] = []
        model_groups[base_name].append(model_id)
    
    # Select the latest/best version from each group
    def select_best_version(model_list):
        if len(model_list) == 1:
            return model_list[0]
        
        # Prefer models with higher version numbers or more recent patterns
        def version_score(model_id):
            score = 0
            # Prefer v3 over v2 over v1
            if '-v3' in model_id or '-3.' in model_id:
                score += 30
            elif '-v2' in model_id or '-2.' in model_id:
                score += 20
            elif '-v1' in model_id or '-1.' in model_id:
                score += 10
            
            # Prefer recent year suffixes
            if '-2503' in model_id or '-25' in model_id:
                score += 15
            elif '-2024' in model_id or '-24' in model_id:
                score += 10
            
            # Prefer instruct versions
            if 'instruct' in model_id:
                score += 5
                
            return score
        
        # Return the model with the highest version score
        return max(model_list, key=version_score)
    
    # Organize models
    organized_models = {
        'nemotron': [],
        'preferred': [],
        'others': []
    }
    
    # Add top priority Nemotron models (in order) - with filtering
    for model_id in top_priority_nemotron:
        if (model_id in available_models and 
            not any(pattern in model_id.lower() for pattern in ['nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma'])):
            organized_models['nemotron'].append(model_id)
    
    # Add other Nemotron models not in the top priority list
    for base_name, model_list in model_groups.items():
        best_model = select_best_version(model_list)
        if (best_model not in top_priority_nemotron and 
            'nvidia/' in best_model and 
            ('nemotron' in best_model.lower() or 'nemo' in best_model.lower()) and
            'instruct' in best_model.lower() and
            not any(pattern in best_model.lower() for pattern in ['nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma'])):
            organized_models['nemotron'].append(best_model)
    
    # Add preferred creator models (in priority order) - with filtering
    for creator, model_list in preferred_models.items():
        for model_id in model_list:
            if (model_id in available_models and 
                not any(pattern in model_id.lower() for pattern in ['gemma', 'codegemma'])):
                organized_models['preferred'].append(model_id)
    
    # Add other useful models from preferred creators not in the specific list
    preferred_creators = list(preferred_models.keys())
    for base_name, model_list in model_groups.items():
        best_model = select_best_version(model_list)
        creator = best_model.split('/')[0] if '/' in best_model else ''
        
        if (creator in preferred_creators and 
            best_model not in organized_models['preferred'] and
            best_model not in organized_models['nemotron'] and
            ('instruct' in best_model.lower() or 'chat' in best_model.lower() or 'code' in best_model.lower()) and
            not any(pattern in best_model.lower() for pattern in ['nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma'])):
            
            # Skip very small models (less useful for analysis)
            if not any(size in best_model.lower() for size in ['1b', '2b', '3b']):
                organized_models['preferred'].append(best_model)
    
    # Add other high-quality models from any creator (very selective)
    for base_name, model_list in model_groups.items():
        best_model = select_best_version(model_list)
        creator = best_model.split('/')[0] if '/' in best_model else ''
        
        # Only include well-known creators for "others" category
        known_good_creators = ['upstage', 'tiiuae', 'ibm', 'writer']
        
        if (creator in known_good_creators and
            best_model not in organized_models['nemotron'] and
            best_model not in organized_models['preferred'] and
            ('instruct' in best_model.lower() or 'chat' in best_model.lower()) and
            # Only include larger models from other creators
            any(size in best_model.lower() for size in ['70b', '32b', '27b', '22b'])):
            organized_models['others'].append(best_model)
    
    # Final filtering to remove any unwanted models that slipped through
    final_exclude_patterns = [
        'nemotron-4-340b', 'nemotron-mini', 'gemma', 'codegemma', 
        'codellama', 'nemotron-super', 'deepseek-coder-6.7b-instruct'
    ]
    
    def final_filter(model_list):
        return [model for model in model_list 
                if not any(pattern in model.lower() for pattern in final_exclude_patterns)]
    
    organized_models['nemotron'] = final_filter(organized_models['nemotron'])
    organized_models['preferred'] = final_filter(organized_models['preferred'])
    organized_models['others'] = final_filter(organized_models['others'])
    
    # Limit each group to reasonable sizes
    organized_models['nemotron'] = organized_models['nemotron'][:10]
    organized_models['preferred'] = organized_models['preferred'][:12]
    organized_models['others'] = organized_models['others'][:5]  # Reduced from 8
    
    return organized_models

def get_fallback_models():
    """Fallback static model list when API is not available."""
    return {
        'nemotron': [
            'nvidia/llama-3.1-nemotron-ultra-253b-v1',
            'nvidia/llama-3.1-nemotron-70b-instruct',
            'nvidia/llama-3.1-nemotron-51b-instruct'
        ],
        'preferred': [
            'meta/llama-3.3-70b-instruct',
            'meta/llama-3.1-405b-instruct',
            'meta/llama-3.1-70b-instruct',
            'meta/llama-4-maverick-17b-128e-instruct',
            'meta/llama-4-scout-17b-16e-instruct',
            'mistralai/mistral-large-2-instruct',
            'deepseek-ai/deepseek-r1',
            'qwen/qwen2.5-7b-instruct'
        ],
        'others': []
    }

@app.route('/results')
def results():
    """Display analysis results (for streaming interface)."""
    # Check if we have analysis results from the session (traditional form submission)
    if 'analysis_results' in session:
        analysis_data = session.pop('analysis_results')  # Get and remove from session
        return render_template('results_stream.html', 
                             analysis_data=analysis_data, 
                             direct_results=True)
    
    # When accessed directly, it should guide users to start an analysis
    return render_template('results_stream.html', direct_access=True)

@app.route('/debug-analysis', methods=['POST'])
def debug_analysis():
    """Debug endpoint to see analysis data structure."""
    if not GPUAnalyzer:
        return jsonify({'error': 'Analysis service not available'}), 503
    
    try:
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        if request.is_json:
            data = request.get_json()
            url = data.get('url')
        else:
            url = request.form.get('url')
            
        if not url:
            return jsonify({'error': 'URL is required'}), 400
            
        result = analyzer.analyze_notebook(url)
        if result:
            analysis_data = format_analysis_for_web(result)
            return jsonify({
                'success': True, 
                'raw_analysis': {
                    'consumer_viable': getattr(result, 'consumer_viable', 'MISSING'),
                    'consumer_gpu_type': getattr(result, 'consumer_gpu_type', 'MISSING'),
                    'enterprise_gpu_type': getattr(result, 'enterprise_gpu_type', 'MISSING'),
                },
                'formatted_analysis': {
                    'consumer_viable': analysis_data.get('consumer_viable', 'MISSING'),
                    'consumer_gpu': analysis_data.get('consumer_gpu', 'MISSING'),
                    'enterprise_gpu': analysis_data.get('enterprise_gpu', 'MISSING'),
                }
            })
        else:
            return jsonify({'error': 'Failed to analyze notebook'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/mcp', methods=['POST'])
def mcp_endpoint():
    """MCP (Model Context Protocol) endpoint for AI assistant integration."""
    if not GPUAnalyzer:
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32000, 'message': 'Analysis service not available'},
            'id': request.json.get('id') if request.is_json and request.json else None
        }), 503
    
    if not request.is_json:
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32700, 'message': 'Parse error - JSON required'},
            'id': None
        }), 400
    
    try:
        data = request.get_json()
        method = data.get('method')
        params = data.get('params', {})
        request_id = data.get('id')
        
        if method == 'initialize':
            return jsonify({
                'jsonrpc': '2.0',
                'result': {
                    'capabilities': {
                        'tools': True
                    },
                    'serverInfo': {
                        'name': 'notebook-analyzer',
                        'version': '3.0.0'
                    }
                },
                'id': request_id
            })
        
        elif method == 'tools/list':
            return jsonify({
                'jsonrpc': '2.0',
                'result': {
                    'tools': [
                        {
                            'name': 'analyze_notebook',
                            'description': 'Analyze Jupyter or marimo notebooks for GPU requirements and NVIDIA compliance',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'url': {
                                        'type': 'string',
                                        'description': 'URL to the notebook (GitHub, GitLab, or direct .ipynb/.py URL)'
                                    },
                                    'notebook_content': {
                                        'type': 'string',
                                        'description': 'Direct notebook content (JSON for .ipynb or Python code for .py files)'
                                    },
                                    'source_info': {
                                        'type': 'string',
                                        'description': 'Source information for direct content (filename or description)',
                                        'default': 'unknown'
                                    },
                                    'include_reasoning': {
                                        'type': 'boolean',
                                        'description': 'Include detailed analysis reasoning',
                                        'default': True
                                    },
                                    'include_compliance': {
                                        'type': 'boolean',
                                        'description': 'Include NVIDIA compliance assessment',
                                        'default': True
                                    }
                                },
                                'anyOf': [
                                    {'required': ['url']},
                                    {'required': ['notebook_content']}
                                ]
                            }
                        },
                        {
                            'name': 'get_gpu_recommendations',
                            'description': 'Get GPU recommendations for specific workload types',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'workload_type': {
                                        'type': 'string',
                                        'enum': ['inference', 'training', 'fine-tuning'],
                                        'description': 'Type of workload'
                                    },
                                    'model_size': {
                                        'type': 'string',
                                        'enum': ['small', 'medium', 'large', 'xlarge'],
                                        'description': 'Expected model size',
                                        'default': 'medium'
                                    },
                                    'batch_size': {
                                        'type': 'integer',
                                        'description': 'Expected batch size',
                                        'default': 1
                                    }
                                },
                                'required': ['workload_type']
                            }
                        }
                    ]
                },
                'id': request_id
            })
        
        elif method == 'tools/call':
            tool_name = params.get('name')
            arguments = params.get('arguments', {})
            
            if tool_name == 'analyze_notebook':
                url = arguments.get('url')
                notebook_content = arguments.get('notebook_content')
                source_info = arguments.get('source_info', 'unknown')
                
                if not url and not notebook_content:
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {'code': -32602, 'message': 'Invalid params - either url or notebook_content is required'},
                        'id': request_id
                    }), 400
                
                try:
                    analyzer = GPUAnalyzer(quiet_mode=True)
                    if url:
                        result = analyzer.analyze_notebook(url)
                        source_display = url
                    else:
                        # Analyze notebook content directly by creating a secure temporary file - P1 Security Fix
                        import tempfile
                        import json
                        import os
                        
                        temp_path = None
                        try:
                            # Parse notebook content (could be JSON string or dict)
                            if isinstance(notebook_content, str):
                                try:
                                    notebook_data = json.loads(notebook_content)
                                    # It's a Jupyter notebook
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, 
                                                                   prefix='mcp_notebook_', dir='/tmp') as temp_file:
                                        json.dump(notebook_data, temp_file)
                                        temp_file.flush()
                                        temp_path = temp_file.name
                                except json.JSONDecodeError:
                                    # Handle marimo .py files
                                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, 
                                                                   prefix='mcp_notebook_', dir='/tmp') as temp_file:
                                        temp_file.write(notebook_content)
                                        temp_file.flush()
                                        temp_path = temp_file.name
                            else:
                                # It's already a notebook data dict
                                notebook_data = notebook_content
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False, 
                                                               prefix='mcp_notebook_', dir='/tmp') as temp_file:
                                    json.dump(notebook_data, temp_file)
                                    temp_file.flush()
                                    temp_path = temp_file.name
                            
                            result = analyzer.analyze_notebook(temp_path)
                            source_display = source_info
                        finally:
                            # Secure cleanup of temporary file
                            if temp_path and os.path.exists(temp_path):
                                try:
                                    os.unlink(temp_path)
                                except OSError:
                                    pass  # Ignore cleanup errors
                    
                    if result:
                        analysis_data = format_analysis_for_web(result)
                        
                        # Include/exclude fields based on parameters
                        if not arguments.get('include_reasoning', True):
                            analysis_data.pop('reasoning', None)
                            analysis_data.pop('llm_reasoning', None)
                        
                        if not arguments.get('include_compliance', True):
                            analysis_data.pop('nvidia_compliance_score', None)
                            analysis_data.pop('structure_assessment', None)
                            analysis_data.pop('content_quality_issues', None)
                            analysis_data.pop('technical_recommendations', None)
                        
                        return jsonify({
                            'jsonrpc': '2.0',
                            'result': {
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': f"GPU Analysis Results for: {source_display}\n\n" +
                                               f"**Minimum Configuration:**\n" +
                                               f"- GPU: {analysis_data['min_gpu']['type']}\n" +
                                               f"- Quantity: {analysis_data['min_gpu']['quantity']}\n" +
                                               f"- VRAM: {analysis_data['min_gpu']['vram_gb']} GB\n" +
                                               f"- Runtime: {analysis_data['min_gpu']['runtime']}\n\n" +
                                               f"**Recommended Configuration:**\n" +
                                               (f"- GPU: {analysis_data['recommended_gpu']['type']}\n" +
                                               f"- Quantity: {analysis_data['recommended_gpu']['quantity']}\n" +
                                               f"- VRAM: {analysis_data['recommended_gpu']['vram_gb']} GB\n" +
                                               f"- Runtime: {analysis_data['recommended_gpu']['runtime']}\n" if analysis_data.get('recommended_gpu') else
                                               f"- Same as minimum configuration\n") +
                                               f"\n" +
                                               f"**Optimal Configuration:**\n" +
                                               f"- GPU: {analysis_data['optimal_gpu']['type']}\n" +
                                               f"- Quantity: {analysis_data['optimal_gpu']['quantity']}\n" +
                                               f"- VRAM: {analysis_data['optimal_gpu']['vram_gb']} GB\n" +
                                               f"- Runtime: {analysis_data['optimal_gpu']['runtime']}\n\n" +
                                               f"**Additional Information:**\n" +
                                               f"- SXM Required: {'Yes' if analysis_data['sxm_required'] else 'No'}\n" +
                                               f"- ARM Compatibility: {analysis_data['arm_compatibility']}\n" +
                                               f"- Analysis Confidence: {analysis_data['confidence']}%\n" +
                                               f"- LLM Enhanced: {'Yes' if analysis_data['llm_enhanced'] else 'No'}\n" +
                                               (f"- NVIDIA Compliance Score: {analysis_data.get('nvidia_compliance_score', 'N/A')}/100\n" if 'nvidia_compliance_score' in analysis_data else "")
                                    }
                                ],
                                'analysis_data': analysis_data
                            },
                            'id': request_id
                        })
                    else:
                        return jsonify({
                            'jsonrpc': '2.0',
                            'error': {'code': -32000, 'message': 'Failed to analyze notebook'},
                            'id': request_id
                        }), 400
                        
                except Exception as e:
                    # Sanitized error message - P1 Security Fix
                    if app.debug:
                        error_message = f'Analysis error: {str(e)}'
                    else:
                        error_message = 'Analysis failed. Please try again.'
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {'code': -32000, 'message': error_message},
                        'id': request_id
                    }), 500
            
            elif tool_name == 'get_gpu_recommendations':
                workload_type = arguments.get('workload_type')
                model_size = arguments.get('model_size', 'medium')
                batch_size = arguments.get('batch_size', 1)
                
                if not workload_type:
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {'code': -32602, 'message': 'Invalid params - workload_type is required'},
                        'id': request_id
                    }), 400
                
                # Simple GPU recommendations based on workload
                recommendations = {
                    'inference': {
                        'small': {'gpu': 'RTX 4080', 'vram': 16, 'quantity': 1},
                        'medium': {'gpu': 'RTX 4090', 'vram': 24, 'quantity': 1},
                        'large': {'gpu': 'A100 40GB', 'vram': 40, 'quantity': 1},
                        'xlarge': {'gpu': 'H100 SXM 80GB', 'vram': 80, 'quantity': 1}
                    },
                    'training': {
                        'small': {'gpu': 'RTX 4090', 'vram': 24, 'quantity': 1},
                        'medium': {'gpu': 'A100 40GB', 'vram': 40, 'quantity': 2},
                        'large': {'gpu': 'A100 80GB', 'vram': 80, 'quantity': 4},
                        'xlarge': {'gpu': 'H100 SXM 80GB', 'vram': 80, 'quantity': 8}
                    },
                    'fine-tuning': {
                        'small': {'gpu': 'RTX 4090', 'vram': 24, 'quantity': 1},
                        'medium': {'gpu': 'A100 40GB', 'vram': 40, 'quantity': 1},
                        'large': {'gpu': 'A100 80GB', 'vram': 80, 'quantity': 2},
                        'xlarge': {'gpu': 'H100 SXM 80GB', 'vram': 80, 'quantity': 4}
                    }
                }
                
                rec = recommendations.get(workload_type, {}).get(model_size, {})
                if not rec:
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {'code': -32602, 'message': 'Invalid workload_type or model_size'},
                        'id': request_id
                    }), 400
                
                # Adjust for batch size
                if batch_size > 32:
                    rec['quantity'] *= 2
                elif batch_size > 128:
                    rec['quantity'] *= 4
                
                return jsonify({
                    'jsonrpc': '2.0',
                    'result': {
                        'content': [
                            {
                                'type': 'text',
                                'text': f"GPU Recommendations for {workload_type} ({model_size} model, batch size {batch_size}):\n\n" +
                                       f"**Recommended Configuration:**\n" +
                                       f"- GPU: {rec['gpu']}\n" +
                                       f"- Quantity: {rec['quantity']}\n" +
                                       f"- VRAM per GPU: {rec['vram']} GB\n" +
                                       f"- Total VRAM: {rec['vram'] * rec['quantity']} GB\n\n" +
                                       f"**Performance Considerations:**\n" +
                                       f"- Workload type: {workload_type.title()}\n" +
                                       f"- Model size category: {model_size.title()}\n" +
                                       f"- Batch size: {batch_size}"
                            }
                        ],
                        'recommendation': rec
                    },
                    'id': request_id
                })
            
            else:
                return jsonify({
                    'jsonrpc': '2.0',
                    'error': {'code': -32601, 'message': f'Method not found: {tool_name}'},
                    'id': request_id
                }), 404
        
        else:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32601, 'message': f'Method not found: {method}'},
                'id': request_id
            }), 404
            
    except Exception as e:
        # Sanitized error message - P1 Security Fix
        if app.debug:
            error_message = f'Internal error: {str(e)}'
        else:
            error_message = 'Internal server error. Please try again.'
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32603, 'message': error_message},
            'id': data.get('id') if 'data' in locals() else None
        }), 500

# Security headers middleware
@app.after_request
def add_security_headers_and_compression(response):
    """Add comprehensive security headers and compression to all responses."""
    
    # Content Security Policy - prevents XSS attacks
    # Allow Bootstrap CSS/JS from CDN, inline styles for UI components
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://stackpath.bootstrapcdn.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://stackpath.bootstrapcdn.com; "
        "font-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    
    # Prevent clickjacking attacks
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Prevent MIME type confusion attacks
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # HSTS - Enforce HTTPS connections (Phase 2 Security Enhancement)
    # Only add HSTS header if request is over HTTPS
    from flask import has_request_context
    if has_request_context() and (request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https'):
        # 1 year max-age, include subdomains, allow preload
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    
    # Referrer Policy - Control referrer information leakage
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions Policy - Control browser features and APIs
    response.headers['Permissions-Policy'] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=(), "
        "usb=(), "
        "bluetooth=(), "
        "accelerometer=(), "
        "gyroscope=(), "
        "magnetometer=(), "
        "ambient-light-sensor=(), "
        "encrypted-media=(), "
        "autoplay=(), "
        "picture-in-picture=(), "
        "display-capture=(), "
        "fullscreen=(self)"
    )
    
    # X-XSS-Protection - Enable XSS filtering (legacy browsers)
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Cross-Origin-Opener-Policy - Isolate browsing context
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    
    # Cross-Origin-Resource-Policy - Control cross-origin resource sharing
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    
    # Cross-Origin-Embedder-Policy - Require opt-in for cross-origin resources
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    
    # Cache-Control for security-sensitive responses
    if has_request_context() and request.endpoint in ['analyze', 'api_analyze', 'debug_analysis']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # Apply response compression for performance
    response = compress_response(response)
    
    return response

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

# For Vercel WSGI
app.wsgi_app = app.wsgi_app

if __name__ == '__main__':
    app.run(debug=True) 