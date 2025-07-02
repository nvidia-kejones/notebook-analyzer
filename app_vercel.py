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
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
import sys
import time

# Set up paths for Vercel environment
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create Flask app with proper template folder
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'ipynb', 'py'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Safe import analyzer without dynamic code execution
GPUAnalyzer = None
GPURequirement = None

try:
    # Safely import the analyzer using sys.path manipulation
    import importlib.machinery
    import importlib.util
    
    analyzer_path = os.path.join(current_dir, "notebook-analyzer.py")
    
    if os.path.exists(analyzer_path):
        # Use SourceFileLoader for safer module loading (no arbitrary code execution)
        loader = importlib.machinery.SourceFileLoader("notebook_analyzer", analyzer_path)
        spec = importlib.util.spec_from_loader("notebook_analyzer", loader)
        
        if spec and spec.loader:
            # Create module and load it safely
            notebook_analyzer = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to prevent re-execution
            sys.modules["notebook_analyzer"] = notebook_analyzer
            
            # Execute the module (safe because it has if __name__ == "__main__" protection)
            spec.loader.exec_module(notebook_analyzer)
            
            # Extract the classes we need
            GPUAnalyzer = getattr(notebook_analyzer, 'GPUAnalyzer', None)
            GPURequirement = getattr(notebook_analyzer, 'GPURequirement', None)
            
            if GPUAnalyzer and GPURequirement:
                print("✅ Successfully imported GPUAnalyzer safely")
            else:
                print("❌ Failed to extract GPUAnalyzer or GPURequirement classes")
        else:
            print("❌ Failed to create module spec")
    else:
        print(f"❌ notebook-analyzer.py not found at {analyzer_path}")
        
except Exception as e:
    print(f"❌ Safe import error: {e}")
    print(f"Traceback: {traceback.format_exc()}")

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def format_analysis_for_web(analysis) -> dict:
    """Format the analysis result for web display."""
    if not analysis:
        return {}
        
    return {
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
        'nvidia_compliance_score': round(analysis.nvidia_compliance_score, 1),
        'structure_assessment': analysis.structure_assessment or {},
        'content_quality_issues': analysis.content_quality_issues or [],
        'technical_recommendations': analysis.technical_recommendations or []
    }

@app.route('/')
def index():
    """Main page with analysis form."""
    return render_template('index.html')

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
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        # Check if it's a file upload or URL
        if 'file' in request.files and request.files['file'].filename:
            # File upload analysis
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join('/tmp', filename)
                file.save(filepath)
                
                try:
                    result = analyzer.analyze_notebook(filepath)
                    if result:
                        analysis_data = format_analysis_for_web(result)
                        return render_template('results.html', 
                                             analysis=analysis_data, 
                                             source_type='file',
                                             source_name=filename)
                    else:
                        flash('Failed to analyze the uploaded notebook. Please check the file format.', 'error')
                finally:
                    # Clean up temp file
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                flash('Invalid file type. Please upload a .ipynb or .py file.', 'error')
                
        elif 'url' in request.form and request.form['url'].strip():
            # URL analysis
            url = request.form['url'].strip()
            result = analyzer.analyze_notebook(url)
            
            if result:
                analysis_data = format_analysis_for_web(result)
                return render_template('results.html', 
                                     analysis=analysis_data, 
                                     source_type='url',
                                     source_name=url)
            else:
                flash('Failed to analyze the notebook from the provided URL. Please check the URL and try again.', 'error')
        else:
            flash('Please provide either a URL or upload a notebook file.', 'error')
            
    except Exception as e:
        print(f"Analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        flash(f'An error occurred during analysis: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis."""
    if not GPUAnalyzer:
        return jsonify({'error': 'Analysis service not available'}), 503
    
    try:
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        if request.is_json:
            data = request.get_json()
            url = data.get('url')
            if not url:
                return jsonify({'error': 'URL is required'}), 400
                
            result = analyzer.analyze_notebook(url)
            if result:
                analysis_data = format_analysis_for_web(result)
                return jsonify({'success': True, 'analysis': analysis_data})
            else:
                return jsonify({'error': 'Failed to analyze notebook'}), 400
                
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join('/tmp', filename)
                file.save(filepath)
                
                try:
                    result = analyzer.analyze_notebook(filepath)
                    if result:
                        analysis_data = format_analysis_for_web(result)
                        return jsonify({'success': True, 'analysis': analysis_data})
                    else:
                        return jsonify({'error': 'Failed to analyze notebook'}), 400
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)
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
    
    # Check if it's a file upload or URL
    if 'file' in request.files and request.files['file'].filename:
        # File upload analysis
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('/tmp', filename)
            file.save(file_path)
            source_name = filename
            analysis_input = {'type': 'file', 'path': file_path, 'name': filename}
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .ipynb or .py file.'}), 400
            
    elif 'url' in request.form and request.form['url'].strip():
        # URL analysis
        source_name = request.form['url'].strip()
        analysis_input = {'type': 'url', 'url': source_name}
    else:
        return jsonify({'error': 'Please provide either a URL or upload a notebook file.'}), 400
    
    def generate_progress(analysis_input, source_name):
        """Generator function for Server-Sent Events."""
        try:
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Starting analysis...'})}\n\n"
            
            analyzer = GPUAnalyzer(quiet_mode=True)
            
            if analysis_input['type'] == 'file':
                filename = analysis_input['name']
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Uploaded file: {filename}'})}\n\n"
            elif analysis_input['type'] == 'url':
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Fetching notebook from URL...'})}\n\n"
            
            # Progress updates during analysis
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Loading notebook content...'})}\n\n"
            time.sleep(0.5)  # Brief pause for UI feedback
            
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Extracting code and markdown cells...'})}\n\n"
            time.sleep(0.3)
            
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Analyzing GPU requirements...'})}\n\n"
            time.sleep(0.5)
            
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Evaluating workload complexity...'})}\n\n"
            time.sleep(0.3)
            
            # Perform the actual analysis
            if analysis_input['type'] == 'file':
                result = analyzer.analyze_notebook(analysis_input['path'])
            else:
                result = analyzer.analyze_notebook(analysis_input['url'])
            
            if result:
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Generating recommendations...'})}\n\n"
                time.sleep(0.3)
                
                analysis_data = format_analysis_for_web(result)
                
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Analysis complete!'})}\n\n"
                time.sleep(0.2)
                
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
            yield f"data: {json.dumps({'type': 'error', 'message': f'Analysis failed: {str(e)}'})}\n\n"
            
            # Clean up temp file in case of error
            if analysis_input and analysis_input['type'] == 'file' and os.path.exists(analysis_input['path']):
                os.remove(analysis_input['path'])
    
    return Response(generate_progress(analysis_input, source_name), content_type='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/results')
def results():
    """Display analysis results (for streaming interface)."""
    return render_template('results_stream.html')

@app.route('/mcp', methods=['POST'])
def mcp_endpoint():
    """MCP (Model Context Protocol) endpoint for AI assistant integration."""
    if not GPUAnalyzer:
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32000, 'message': 'Analysis service not available'},
            'id': request.json.get('id') if request.is_json else None
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
                        # Analyze notebook content directly by creating a temporary file
                        import tempfile
                        import json
                        import os
                        
                        # Parse notebook content (could be JSON string or dict)
                        if isinstance(notebook_content, str):
                            try:
                                notebook_data = json.loads(notebook_content)
                            except json.JSONDecodeError:
                                # Handle marimo .py files
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                                    temp_file.write(notebook_content)
                                    temp_path = temp_file.name
                        else:
                            notebook_data = notebook_content
                        
                        if 'notebook_data' in locals():
                            # It's a Jupyter notebook
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as temp_file:
                                json.dump(notebook_data, temp_file)
                                temp_path = temp_file.name
                        
                        try:
                            result = analyzer.analyze_notebook(temp_path)
                            source_display = source_info
                        finally:
                            # Clean up temp file
                            if 'temp_path' in locals():
                                os.unlink(temp_path)
                    
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
                                               f"**Minimum Requirements:**\n" +
                                               f"- GPU: {analysis_data['min_gpu']['type']}\n" +
                                               f"- Quantity: {analysis_data['min_gpu']['quantity']}\n" +
                                               f"- VRAM: {analysis_data['min_gpu']['vram_gb']} GB\n" +
                                               f"- Runtime: {analysis_data['min_gpu']['runtime']}\n\n" +
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
                    return jsonify({
                        'jsonrpc': '2.0',
                        'error': {'code': -32000, 'message': f'Analysis error: {str(e)}'},
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
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32603, 'message': f'Internal error: {str(e)}'},
            'id': data.get('id') if 'data' in locals() else None
        }), 500

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