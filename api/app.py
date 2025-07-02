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

# Import the classes from the notebook analyzer script
import importlib.util
spec = importlib.util.spec_from_file_location("notebook_analyzer", "../notebook-analyzer.py")
if spec is not None and spec.loader is not None:
    notebook_analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook_analyzer)
else:
    raise ImportError("Could not load notebook-analyzer.py module")

GPUAnalyzer = notebook_analyzer.GPUAnalyzer
GPURequirement = notebook_analyzer.GPURequirement

app = Flask(__name__, template_folder='../templates')
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
    return jsonify({'status': 'healthy', 'platform': 'vercel'})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis - optimized for Vercel."""
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
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# For Vercel, we need to expose the app directly
if __name__ == '__main__':
    app.run()

# This is the WSGI application entry point for Vercel
handler = app 