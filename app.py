#!/usr/bin/env python3
"""
Notebook Analyzer - Main App

For Vercel deployment: Uses app_vercel.py
For local development: Run this file directly
"""

import os

# Import the Vercel-optimized app
from app_vercel import app

# This allows both local development and Vercel deployment
if __name__ == '__main__':
    # Local development server
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

# For Vercel deployment, the app is imported automatically 