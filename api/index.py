#!/usr/bin/env python3
"""
Vercel Entry Point for Notebook Analyzer

This file serves as the entry point for Vercel deployment.
It imports the main Flask application from the root directory.
"""

import sys
import os

# Add the parent directory to the Python path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main Flask app
from app_vercel import app

# Vercel will automatically handle the WSGI interface
# The app object will be used directly by Vercel's Python runtime 