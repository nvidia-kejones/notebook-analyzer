#!/usr/bin/env python3
"""
Minimal debug version for Vercel deployment troubleshooting
"""

import os
import sys
import traceback
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/debug')
def debug():
    """Debug endpoint to check environment"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # List files in current and parent directories
        current_files = os.listdir(current_dir) if os.path.exists(current_dir) else []
        parent_files = os.listdir(parent_dir) if os.path.exists(parent_dir) else []
        
        return jsonify({
            'status': 'debug_ok',
            'platform': 'vercel',
            'current_dir': current_dir,
            'parent_dir': parent_dir,
            'current_files': current_files,
            'parent_files': parent_files,
            'python_path': sys.path,
            'working_directory': os.getcwd()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/test-import')
def test_import():
    """Test importing the notebook analyzer"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Add parent directory to Python path
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try to import the module
        import importlib.util
        notebook_analyzer_path = os.path.join(parent_dir, "notebook-analyzer.py")
        
        if not os.path.exists(notebook_analyzer_path):
            return jsonify({
                'error': f'File not found: {notebook_analyzer_path}',
                'parent_files': os.listdir(parent_dir) if os.path.exists(parent_dir) else []
            }), 404
        
        spec = importlib.util.spec_from_file_location("notebook_analyzer", notebook_analyzer_path)
        if spec is not None and spec.loader is not None:
            notebook_analyzer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(notebook_analyzer)
            
            return jsonify({
                'status': 'import_success',
                'module_path': notebook_analyzer_path,
                'module_attributes': dir(notebook_analyzer)
            })
        else:
            return jsonify({
                'error': 'Could not create module spec',
                'file_exists': os.path.exists(notebook_analyzer_path)
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# WSGI handler for Vercel
handler = app

if __name__ == '__main__':
    app.run(debug=True) 