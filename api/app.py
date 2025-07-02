#!/usr/bin/env python3
"""
Notebook Analyzer - Vercel Deployment

A serverless web application for analyzing Jupyter notebooks to determine minimum GPU requirements.
Optimized for Vercel's serverless environment.
"""

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
import sys

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the classes from the notebook analyzer script
try:
    import importlib.util
    notebook_analyzer_path = os.path.join(parent_dir, "notebook-analyzer.py")
    
    if not os.path.exists(notebook_analyzer_path):
        raise ImportError(f"notebook-analyzer.py not found at {notebook_analyzer_path}")
    
    spec = importlib.util.spec_from_file_location("notebook_analyzer", notebook_analyzer_path)
    if spec is not None and spec.loader is not None:
        notebook_analyzer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notebook_analyzer)
    else:
        raise ImportError("Could not load notebook-analyzer.py module spec")
        
    GPUAnalyzer = notebook_analyzer.GPUAnalyzer
    GPURequirement = notebook_analyzer.GPURequirement
    
except Exception as e:
    print(f"Import error: {e}")
    print(f"Current dir: {current_dir}")
    print(f"Parent dir: {parent_dir}")
    print(f"Files in parent dir: {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'N/A'}")
    raise

# Set up Flask app with proper template path
templates_dir = os.path.join(parent_dir, 'templates')
app = Flask(__name__, template_folder=templates_dir)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration for Vercel (use /tmp for temporary files)
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'ipynb', 'py'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    if not ('.' in filename):
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True

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

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy', 
        'platform': 'vercel',
        'current_dir': os.path.dirname(os.path.abspath(__file__)),
        'parent_dir': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'analyzer_available': 'GPUAnalyzer' in globals()
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis - optimized for Vercel."""
    try:
        if 'GPUAnalyzer' not in globals():
            return jsonify({'error': 'GPUAnalyzer not available - import failed'}), 500
            
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
                filepath = os.path.join('/tmp', filename)
                file.save(filepath)
                
                try:
                    result = analyzer.analyze_notebook(filepath)
                    
                    if result:
                        analysis_data = format_analysis_for_web(result)
                        return jsonify({
                            'success': True,
                            'analysis': analysis_data
                        })
                    else:
                        return jsonify({'error': 'Failed to analyze notebook'}), 400
                finally:
                    # Clean up temp file
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        else:
            return jsonify({'error': 'Invalid request format'}), 400
            
    except Exception as e:
        print(f"API analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# For Vercel, we need to expose the app directly
if __name__ == '__main__':
    app.run()

# This is the WSGI application entry point for Vercel
handler = app 