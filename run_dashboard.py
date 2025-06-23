#!/usr/bin/env python3
"""
Simple script to run the dashboard with proper environment setup
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the dashboard app
from dashboard_app import app

if __name__ == '__main__':
    print("Starting Publication Enricher Dashboard...")
    print("Dashboard will be available at: http://localhost:5001")
    print("Press Ctrl+C to stop")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)