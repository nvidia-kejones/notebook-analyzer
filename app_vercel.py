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
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import sys

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

# Import analyzer with error handling
GPUAnalyzer = None
GPURequirement = None

try:
    # Try to import the analyzer
    import importlib.util
    analyzer_path = os.path.join(current_dir, "notebook-analyzer.py")
    
    if os.path.exists(analyzer_path):
        spec = importlib.util.spec_from_file_location("notebook_analyzer", analyzer_path)
        if spec and spec.loader:
            notebook_analyzer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(notebook_analyzer)
            GPUAnalyzer = notebook_analyzer.GPUAnalyzer
            GPURequirement = notebook_analyzer.GPURequirement
            print("✅ Successfully imported GPUAnalyzer")
        else:
            print("❌ Failed to create module spec")
    else:
        print(f"❌ notebook-analyzer.py not found at {analyzer_path}")
        
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