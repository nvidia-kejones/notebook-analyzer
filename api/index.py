#!/usr/bin/env python3
"""
Vercel Entry Point for Notebook Analyzer

This file serves as the entry point for Vercel deployment.
It imports the main Flask application from the root directory.
"""

import sys
import os
import traceback

# Add the parent directory to the Python path so we can import from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Initialize app variable
app = None

try:
    # Import the main Flask app with detailed error handling
    from app_vercel import app as flask_app
    app = flask_app
    print("✅ Successfully imported Flask app for Vercel")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Parent directory: {parent_dir}")
    print(f"Files in parent directory: {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'Directory not found'}")
    print(f"Python path: {sys.path}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Create a minimal error app for debugging
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    @app.route('/health')
    def error_handler():
        return jsonify({
            'error': 'Import failed',
            'message': str(e),
            'parent_dir': parent_dir,
            'files': os.listdir(parent_dir) if os.path.exists(parent_dir) else 'Directory not found',
            'python_path': sys.path
        })

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    
    # Create a minimal error app for debugging
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    @app.route('/health')
    def error_handler():
        return jsonify({
            'error': 'Unexpected error',
            'message': str(e),
            'traceback': traceback.format_exc()
        })

# Ensure we have an app object for Vercel
if app is None:
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def fallback():
        return jsonify({'error': 'No app object created'})

# Vercel will automatically handle the WSGI interface
# The app object will be used directly by Vercel's Python runtime 