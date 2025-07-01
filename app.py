#!/usr/bin/env python3
"""
Notebook Analyzer - Web Interface

A web application for analyzing Jupyter notebooks to determine minimum GPU requirements.
Supports both URL input and file upload capabilities.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response, stream_template
from werkzeug.utils import secure_filename
import sys
import asyncio
from typing import Any, Dict, List

# Import the classes from the notebook analyzer script
import importlib.util
spec = importlib.util.spec_from_file_location("notebook_analyzer", "notebook-analyzer.py")
notebook_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notebook_analyzer)

GPUAnalyzer = notebook_analyzer.GPUAnalyzer
GPURequirement = notebook_analyzer.GPURequirement

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'ipynb'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_analysis_for_web(analysis: GPURequirement) -> dict:
    """Format the analysis result for web display."""
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

@app.route('/results')
def results():
    """Results page that gets data from sessionStorage."""
    return render_template('results_stream.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle analysis requests (both URL and file upload)."""
    try:
        analyzer = GPUAnalyzer(quiet_mode=True)
        
        # Check if it's a file upload or URL
        if 'file' in request.files and request.files['file'].filename:
            # File upload analysis
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Analyze the uploaded file
                result = analyzer.analyze_notebook(filepath)
                
                # Clean up the temporary file
                os.remove(filepath)
                
                if result:
                    analysis_data = format_analysis_for_web(result)
                    return render_template('results.html', 
                                         analysis=analysis_data, 
                                         source_type='file',
                                         source_name=filename)
                else:
                    flash('Failed to analyze the uploaded notebook. Please check the file format.', 'error')
                    return redirect(url_for('index'))
            else:
                flash('Invalid file type. Please upload a .ipynb file.', 'error')
                return redirect(url_for('index'))
                
        elif 'url' in request.form and request.form['url'].strip():
            # URL analysis
            url = request.form['url'].strip()
            
            # Analyze the notebook from URL
            result = analyzer.analyze_notebook(url)
            
            if result:
                analysis_data = format_analysis_for_web(result)
                return render_template('results.html', 
                                     analysis=analysis_data, 
                                     source_type='url',
                                     source_name=url)
            else:
                flash('Failed to analyze the notebook from the provided URL. Please check the URL and try again.', 'error')
                return redirect(url_for('index'))
        else:
            flash('Please provide either a URL or upload a notebook file.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        print(f"Analysis error: {e}")
        print(traceback.format_exc())
        flash(f'An error occurred during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis."""
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
                return jsonify({
                    'success': True,
                    'analysis': analysis_data
                })
            else:
                return jsonify({'error': 'Failed to analyze notebook'}), 400
                
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = analyzer.analyze_notebook(filepath)
                os.remove(filepath)
                
                if result:
                    analysis_data = format_analysis_for_web(result)
                    return jsonify({
                        'success': True,
                        'analysis': analysis_data
                    })
                else:
                    return jsonify({'error': 'Failed to analyze notebook'}), 400
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            return jsonify({'error': 'Invalid request format'}), 400
            
    except Exception as e:
        print(f"API analysis error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """Handle analysis requests with real-time streaming."""
    # Extract data from request outside of generator
    file_data = None
    url_data = None
    
    if 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_data = {'filename': filename, 'filepath': filepath}
    elif 'url' in request.form and request.form['url'].strip():
        url_data = request.form['url'].strip()
    
    def generate_analysis():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'message': 'Initializing analyzer...'})}\n\n"
            
            analyzer = GPUAnalyzer(quiet_mode=True)
            
            # Determine input type and source
            if file_data:
                # File upload analysis
                filename = file_data['filename']
                filepath = file_data['filepath']
                
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Processing uploaded file: {filename}'})}\n\n"
                
                # Create a custom analyzer that yields progress
                for progress_msg, result in analyze_with_progress(analyzer, filepath, source_type='file', source_name=filename):
                    if result is None:  # Progress message
                        yield f"data: {json.dumps({'type': 'progress', 'message': progress_msg})}\n\n"
                    else:  # Final result
                        os.remove(filepath)  # Clean up
                        analysis_data = format_analysis_for_web(result)
                        yield f"data: {json.dumps({'type': 'complete', 'analysis': analysis_data, 'source_type': 'file', 'source_name': filename})}\n\n"
                        return
                    
            elif url_data:
                # URL analysis
                yield f"data: {json.dumps({'type': 'progress', 'message': f'Fetching notebook from URL...'})}\n\n"
                
                # Create a custom analyzer that yields progress
                for progress_msg, result in analyze_with_progress(analyzer, url_data, source_type='url', source_name=url_data):
                    if result is None:  # Progress message
                        yield f"data: {json.dumps({'type': 'progress', 'message': progress_msg})}\n\n"
                    else:  # Final result
                        analysis_data = format_analysis_for_web(result)
                        yield f"data: {json.dumps({'type': 'complete', 'analysis': analysis_data, 'source_type': 'url', 'source_name': url_data})}\n\n"
                        return
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Please provide either a URL or upload a notebook file.'})}\n\n"
                return
                
        except Exception as e:
            print(f"Streaming analysis error: {e}")
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': f'Analysis failed: {str(e)}'})}\n\n"
    
    return Response(generate_analysis(), mimetype='text/plain')

def analyze_with_progress(analyzer, url_or_path, source_type, source_name):
    """Generator that yields progress messages and final result."""
    
    # Step 1: Fetch notebook
    yield "Downloading and parsing notebook...", None
    
    # Mock the internal steps by accessing the analyzer methods
    try:
        # Fetch notebook
        notebook = analyzer.fetch_notebook(url_or_path)
        if not notebook:
            yield "Failed to fetch notebook", None
            return
            
        yield "Notebook loaded successfully", None
        
        # Step 2: Extract code
        yield "Extracting code cells and content...", None
        code_cells, markdown_cells = analyzer.extract_code_cells(notebook)
        yield f"Found {len(code_cells)} code cells and {len(markdown_cells)} markdown cells", None
        
        # Step 3: Analyze imports
        yield "Analyzing library imports and dependencies...", None
        combined_code = '\n'.join(code_cells)
        imports = analyzer.analyze_imports(combined_code)
        yield f"Detected {len(imports)} unique libraries", None
        
        # Step 4: Analyze patterns
        yield "Analyzing code patterns and model usage...", None
        models = analyzer.analyze_model_usage(combined_code)
        yield f"Found {len(models)} model patterns", None
        
        # Step 5: Check for training
        yield "Detecting training vs inference patterns...", None
        is_training = analyzer.is_training_code(combined_code)
        workload_type = "training" if is_training else "inference"
        yield f"Detected {workload_type} workload", None
        
        # Step 6: Estimate requirements
        yield "Calculating GPU memory requirements...", None
        
        # Step 7: LLM Enhancement (if available)
        llm_analyzer = getattr(analyzer, 'llm_analyzer', None)
        if llm_analyzer and hasattr(llm_analyzer, 'base_url') and llm_analyzer.api_key:
            yield "Enhancing analysis with LLM insights...", None
            
        # Step 8: Compliance analysis
        yield "Evaluating NVIDIA notebook compliance...", None
        
        # Step 9: Final analysis
        yield "Generating final recommendations...", None
        
        # Perform the actual analysis
        result = analyzer.analyze_notebook(url_or_path)
        
        if result:
            yield "Analysis complete!", result
        else:
            yield "Analysis failed - unable to process notebook", None
            
    except Exception as e:
        yield f"Error during analysis: {str(e)}", None

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'notebook-analyzer-web'})

# MCP (Model Context Protocol) Support
@app.route('/mcp', methods=['POST'])
def mcp_handler():
    """MCP protocol handler for AI assistant integration."""
    try:
        mcp_request = request.get_json()
        
        if not mcp_request or 'method' not in mcp_request:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32600, 'message': 'Invalid Request'},
                'id': mcp_request.get('id') if mcp_request else None
            }), 400
        
        method = mcp_request['method']
        params = mcp_request.get('params', {})
        request_id = mcp_request.get('id')
        
        # Handle MCP protocol methods
        if method == 'initialize':
            return jsonify({
                'jsonrpc': '2.0',
                'result': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {
                        'tools': {},
                        'resources': {}
                    },
                    'serverInfo': {
                                                 'name': 'notebook-analyzer',
                        'version': '1.0.0'
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
                            'description': 'Analyze a Jupyter notebook for GPU requirements and NVIDIA compliance',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'url': {
                                        'type': 'string',
                                        'description': 'URL to the Jupyter notebook (GitHub, GitLab, or direct .ipynb URL)'
                                    },
                                    'include_reasoning': {
                                        'type': 'boolean',
                                        'description': 'Include detailed reasoning in the response',
                                        'default': True
                                    },
                                    'include_compliance': {
                                        'type': 'boolean', 
                                        'description': 'Include NVIDIA compliance assessment',
                                        'default': True
                                    }
                                },
                                'required': ['url']
                            }
                        },
                        {
                            'name': 'get_gpu_recommendations',
                            'description': 'Get GPU recommendations for a specific workload type',
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    'workload_type': {
                                        'type': 'string',
                                        'enum': ['inference', 'training', 'fine-tuning'],
                                        'description': 'Type of ML workload'
                                    },
                                    'model_size': {
                                        'type': 'string',
                                        'enum': ['small', 'medium', 'large', 'xlarge'],
                                        'description': 'Approximate model size category'
                                    },
                                    'batch_size': {
                                        'type': 'integer',
                                        'description': 'Expected batch size',
                                        'minimum': 1
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
                return handle_analyze_notebook_mcp(arguments, request_id)
            elif tool_name == 'get_gpu_recommendations':
                return handle_gpu_recommendations_mcp(arguments, request_id)
            else:
                return jsonify({
                    'jsonrpc': '2.0',
                    'error': {'code': -32601, 'message': f'Tool not found: {tool_name}'},
                    'id': request_id
                }), 404
        
        else:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32601, 'message': f'Method not found: {method}'},
                'id': request_id
            }), 404
            
    except Exception as e:
        print(f"MCP handler error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32603, 'message': f'Internal error: {str(e)}'},
            'id': request.get_json().get('id') if request.get_json() else None
        }), 500

def handle_analyze_notebook_mcp(arguments: Dict[str, Any], request_id: str):
    """Handle MCP analyze_notebook tool call."""
    try:
        url = arguments.get('url')
        include_reasoning = arguments.get('include_reasoning', True)
        include_compliance = arguments.get('include_compliance', True)
        
        if not url:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32602, 'message': 'Missing required parameter: url'},
                'id': request_id
            }), 400
        
        # Perform analysis
        analyzer = GPUAnalyzer(quiet_mode=True)
        result = analyzer.analyze_notebook(url)
        
        if not result:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32603, 'message': 'Failed to analyze notebook'},
                'id': request_id
            }), 500
        
        # Format response for MCP
        analysis_data = format_analysis_for_web(result)
        
        # Create structured response
        response_content = {
            'gpu_requirements': {
                'minimum': {
                    'gpu_type': analysis_data['min_gpu']['type'],
                    'quantity': analysis_data['min_gpu']['quantity'],
                    'vram_gb': analysis_data['min_gpu']['vram_gb'],
                    'estimated_runtime': analysis_data['min_gpu']['runtime']
                },
                'optimal': {
                    'gpu_type': analysis_data['optimal_gpu']['type'],
                    'quantity': analysis_data['optimal_gpu']['quantity'],
                    'vram_gb': analysis_data['optimal_gpu']['vram_gb'],
                    'estimated_runtime': analysis_data['optimal_gpu']['runtime']
                }
            },
            'sxm_required': analysis_data['sxm_required'],
            'arm_compatibility': analysis_data['arm_compatibility'],
            'confidence': analysis_data['confidence'],
            'llm_enhanced': analysis_data['llm_enhanced']
        }
        
        if include_reasoning:
            response_content['reasoning'] = analysis_data['reasoning']
            if analysis_data['llm_reasoning']:
                response_content['llm_reasoning'] = analysis_data['llm_reasoning']
        
        if include_compliance:
            response_content['nvidia_compliance'] = {
                'score': analysis_data['nvidia_compliance_score'],
                'structure_assessment': analysis_data['structure_assessment'],
                'content_quality_issues': analysis_data['content_quality_issues'],
                'technical_recommendations': analysis_data['technical_recommendations']
            }
        
        return jsonify({
            'jsonrpc': '2.0',
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': f"# GPU Requirements Analysis\n\n**Notebook URL:** {url}\n\n## Minimum Requirements\n- **GPU:** {analysis_data['min_gpu']['type']}\n- **Quantity:** {analysis_data['min_gpu']['quantity']}\n- **VRAM:** {analysis_data['min_gpu']['vram_gb']} GB\n- **Runtime:** {analysis_data['min_gpu']['runtime']}\n\n## Optimal Configuration\n- **GPU:** {analysis_data['optimal_gpu']['type']}\n- **Quantity:** {analysis_data['optimal_gpu']['quantity']}\n- **VRAM:** {analysis_data['optimal_gpu']['vram_gb']} GB\n- **Runtime:** {analysis_data['optimal_gpu']['runtime']}\n\n## Additional Info\n- **SXM Required:** {'Yes' if analysis_data['sxm_required'] else 'No'}\n- **ARM Compatible:** {analysis_data['arm_compatibility']}\n- **Analysis Confidence:** {analysis_data['confidence']}%\n- **NVIDIA Compliance Score:** {analysis_data['nvidia_compliance_score']}/100"
                    },
                    {
                        'type': 'resource',
                        'resource': {
                            'uri': f'notebook-analysis://{url}',
                            'name': 'Full Analysis Data',
                            'mimeType': 'application/json'
                        },
                        'text': json.dumps(response_content, indent=2)
                    }
                ]
            },
            'id': request_id
        })
        
    except Exception as e:
        print(f"MCP analyze_notebook error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32603, 'message': f'Analysis failed: {str(e)}'},
            'id': request_id
        }), 500

def handle_gpu_recommendations_mcp(arguments: Dict[str, Any], request_id: str):
    """Handle MCP get_gpu_recommendations tool call."""
    try:
        workload_type = arguments.get('workload_type')
        model_size = arguments.get('model_size', 'medium')
        batch_size = arguments.get('batch_size', 1)
        
        if not workload_type:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32602, 'message': 'Missing required parameter: workload_type'},
                'id': request_id
            }), 400
        
        # Simple recommendation logic based on workload type and model size
        recommendations = {
            'inference': {
                'small': {'gpu': 'RTX 4090', 'vram': 24, 'quantity': 1},
                'medium': {'gpu': 'L4', 'vram': 24, 'quantity': 1}, 
                'large': {'gpu': 'A100', 'vram': 80, 'quantity': 1},
                'xlarge': {'gpu': 'H100 SXM', 'vram': 80, 'quantity': 2}
            },
            'training': {
                'small': {'gpu': 'RTX 4090', 'vram': 24, 'quantity': 1},
                'medium': {'gpu': 'A100', 'vram': 80, 'quantity': 1},
                'large': {'gpu': 'A100 SXM', 'vram': 80, 'quantity': 2},
                'xlarge': {'gpu': 'H100 SXM', 'vram': 80, 'quantity': 4}
            },
            'fine-tuning': {
                'small': {'gpu': 'L4', 'vram': 24, 'quantity': 1},
                'medium': {'gpu': 'A100', 'vram': 80, 'quantity': 1},
                'large': {'gpu': 'A100 SXM', 'vram': 80, 'quantity': 2},
                'xlarge': {'gpu': 'H100 SXM', 'vram': 80, 'quantity': 2}
            }
        }
        
        if workload_type not in recommendations:
            return jsonify({
                'jsonrpc': '2.0',
                'error': {'code': -32602, 'message': f'Invalid workload_type: {workload_type}'},
                'id': request_id
            }), 400
        
        rec = recommendations[workload_type][model_size]
        
        # Adjust for batch size
        if batch_size > 1:
            vram_multiplier = 1 + (batch_size - 1) * 0.3  # Rough estimate
            rec['vram'] = int(rec['vram'] * vram_multiplier)
            if rec['vram'] > 80:  # If VRAM exceeds single GPU capacity
                rec['quantity'] = max(rec['quantity'], (rec['vram'] // 80) + 1)
                rec['vram'] = 80  # Per GPU
        
        response_text = f"# GPU Recommendations\n\n**Workload:** {workload_type.title()}\n**Model Size:** {model_size.title()}\n**Batch Size:** {batch_size}\n\n## Recommended Configuration\n- **GPU:** {rec['gpu']}\n- **Quantity:** {rec['quantity']}\n- **VRAM per GPU:** {rec['vram']} GB\n\n## Notes\n- Recommendations are based on typical workload patterns\n- For production deployments, consider testing with your specific models\n- Memory requirements may vary based on model architecture and optimization techniques"
        
        return jsonify({
            'jsonrpc': '2.0',
            'result': {
                'content': [
                    {
                        'type': 'text',
                        'text': response_text
                    }
                ]
            },
            'id': request_id
        })
        
    except Exception as e:
        print(f"MCP gpu_recommendations error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'jsonrpc': '2.0',
            'error': {'code': -32603, 'message': f'Recommendation failed: {str(e)}'},
            'id': request_id
        }), 500

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug) 