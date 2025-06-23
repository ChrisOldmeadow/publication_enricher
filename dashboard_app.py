#!/usr/bin/env python3
"""
Flask Dashboard for Publication Enricher
"""
import os
import json
import glob
import asyncio
import subprocess
import psutil
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from threading import Thread
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to track running processes
RUNNING_PROCESSES = {}

def get_env_status():
    """Check which API keys are configured"""
    return {
        'elsevier': bool(os.getenv('ELSEVIER_API_KEY')),
        'pubmed_email': bool(os.getenv('PUBMED_EMAIL')),
        'pubmed_key': bool(os.getenv('PUBMED_API_KEY')),
        'crossref_email': bool(os.getenv('CROSSREF_EMAIL')),
        'semantic_scholar': bool(os.getenv('SEMANTIC_SCHOLAR_API_KEY'))
    }

def get_process_status(pid):
    """Check if a process is still running"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except psutil.NoSuchProcess:
        return False

def parse_checkpoint_files():
    """Parse checkpoint files to get progress information"""
    checkpoint_files = sorted(glob.glob('checkpoint_chunk_*.json'))
    total_processed = 0
    
    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                if 'processed' in data:
                    total_processed += len(data['processed'])
        except:
            continue
    
    return total_processed

def get_enrichment_stats(output_file):
    """Get statistics from enrichment output files"""
    stats = {
        'enriched': 0,
        'failed': 0,
        'fuzzy': 0,
        'total': 0
    }
    
    base_name = output_file.rsplit('.', 1)[0]
    
    # Count enriched
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            stats['enriched'] = len(df)
        except:
            pass
    
    # Count failed
    failed_file = f"{base_name}_failed.csv"
    if os.path.exists(failed_file):
        try:
            df = pd.read_csv(failed_file)
            stats['failed'] = len(df)
        except:
            pass
    
    # Count fuzzy matches
    fuzzy_file = f"{base_name}_fuzzy.csv"
    if os.path.exists(fuzzy_file):
        try:
            df = pd.read_csv(fuzzy_file)
            stats['fuzzy'] = len(df)
        except:
            pass
    
    stats['total'] = stats['enriched'] + stats['failed']
    return stats

def parse_progress_from_logs(log_file):
    """Parse progress information from log file"""
    progress_info = {
        'current': 0,
        'total': 0,
        'rate': 0,
        'enriched': 0,
        'failed': 0,
        'percentage': 0
    }
    
    if not os.path.exists(log_file):
        return progress_info
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for progress indicators in recent lines
        for line in reversed(lines[-100:]):  # Check last 100 lines
            line = line.strip()
            
            # Parse progress bars like: "Enriching publications:  33%|███▎      | 2000/6019"
            if 'Enriching publications:' in line and '|' in line:
                try:
                    # Extract numbers from format like "2000/6019 [09:55<11:06,  6.09pub/s, enriched=1300, failed=656]"
                    
                    # Extract current/total
                    match = re.search(r'(\d+)/(\d+)', line)
                    if match:
                        progress_info['current'] = int(match.group(1))
                        progress_info['total'] = int(match.group(2))
                        progress_info['percentage'] = (progress_info['current'] / progress_info['total']) * 100
                    
                    # Extract rate
                    rate_match = re.search(r'(\d+\.?\d*)pub/s', line)
                    if rate_match:
                        progress_info['rate'] = float(rate_match.group(1))
                    
                    # Extract enriched count
                    enriched_match = re.search(r'enriched=(\d+)', line)
                    if enriched_match:
                        progress_info['enriched'] = int(enriched_match.group(1))
                    
                    # Extract failed count
                    failed_match = re.search(r'failed=(\d+)', line)
                    if failed_match:
                        progress_info['failed'] = int(failed_match.group(1))
                    
                    break  # Use the most recent progress line
                except:
                    continue
            
            # Look for total count in lines like "Found 6019 publications to process"
            elif 'publications to process' in line:
                try:
                    match = re.search(r'Found (\d+) publications', line)
                    if match and progress_info['total'] == 0:
                        progress_info['total'] = int(match.group(1))
                except:
                    continue
    
    except:
        pass
    
    return progress_info

def get_recent_logs(log_file, lines=50):
    """Get recent lines from log file"""
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            # Filter out progress bar lines for cleaner display
            filtered_lines = []
            for line in all_lines[-lines:]:
                if not ('Enriching publications:' in line and '|' in line):
                    filtered_lines.append(line)
            return filtered_lines[-10:]  # Return last 10 non-progress lines
    except:
        return []

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', env_status=get_env_status())

@app.route('/api/status')
def api_status():
    """Get current status of all processes"""
    status = {
        'processes': [],
        'env_status': get_env_status(),
        'cache_info': {}
    }
    
    # Check running processes
    for job_id, info in list(RUNNING_PROCESSES.items()):
        process_info = {
            'job_id': job_id,
            'filename': info['filename'],
            'start_time': info['start_time'],
            'output_file': info['output_file'],
            'log_file': info['log_file'],
            'pid': info.get('pid'),
            'running': False,
            'progress': 0,
            'stats': {}
        }
        
        # Check if process is still running
        if info.get('pid'):
            process_info['running'] = get_process_status(info['pid'])
        
        # Get progress from logs and checkpoints
        if process_info['running']:
            log_progress = parse_progress_from_logs(info['log_file'])
            checkpoint_progress = parse_checkpoint_files()
            
            # Use log progress if available, otherwise checkpoint progress
            if log_progress['total'] > 0:
                process_info['progress'] = log_progress
            else:
                process_info['progress'] = {
                    'current': checkpoint_progress,
                    'total': 0,
                    'percentage': 0,
                    'rate': 0,
                    'enriched': 0,
                    'failed': 0
                }
        else:
            # For completed processes, use checkpoint data
            process_info['progress'] = {
                'current': parse_checkpoint_files(),
                'total': parse_checkpoint_files(),
                'percentage': 100,
                'rate': 0,
                'enriched': 0,
                'failed': 0
            }
        
        # Get stats from output files
        process_info['stats'] = get_enrichment_stats(info['output_file'])
        
        # Get recent logs
        process_info['recent_logs'] = get_recent_logs(info['log_file'], 10)
        
        status['processes'].append(process_info)
        
        # Clean up finished processes
        if not process_info['running'] and job_id in RUNNING_PROCESSES:
            del RUNNING_PROCESSES[job_id]
    
    # Check cache database
    if os.path.exists('api_cache.db'):
        status['cache_info'] = {
            'exists': True,
            'size': os.path.getsize('api_cache.db'),
            'modified': datetime.fromtimestamp(os.path.getmtime('api_cache.db')).isoformat()
        }
    else:
        status['cache_info'] = {'exists': False}
    
    return jsonify(status)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get CSV info
        try:
            df = pd.read_csv(filepath)
            info = {
                'filename': filename,
                'filepath': filepath,
                'rows': len(df),
                'columns': list(df.columns)
            }
            return jsonify({'success': True, 'file_info': info})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/start_enrichment', methods=['POST'])
def start_enrichment():
    """Start enrichment process"""
    data = request.json
    filepath = data.get('filepath')
    options = data.get('options', {})
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 400
    
    # Generate job ID
    job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare command
    cmd = ['python', 'multi_process_enricher.py', filepath]
    
    # Add options
    if options.get('processes'):
        cmd.extend(['--processes', str(options['processes'])])
    if options.get('batch_size'):
        cmd.extend(['--batch-size', str(options['batch_size'])])
    if options.get('max_concurrent'):
        cmd.extend(['--max-concurrent', str(options['max_concurrent'])])
    if options.get('disable_pubmed'):
        cmd.append('--disable-pubmed')
    if options.get('disable_semantic'):
        cmd.append('--disable-semantic')
    
    # Set up log file
    log_file = f'logs/enrichment_{job_id}.log'
    os.makedirs('logs', exist_ok=True)
    
    # Start process in background
    with open(log_file, 'w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    
    # Determine output file name
    base_name = os.path.basename(filepath).rsplit('.', 1)[0]
    output_file = f"{base_name}_enriched_mp.csv"
    
    # Track process
    RUNNING_PROCESSES[job_id] = {
        'filename': os.path.basename(filepath),
        'start_time': datetime.now().isoformat(),
        'pid': process.pid,
        'output_file': output_file,
        'log_file': log_file
    }
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'pid': process.pid
    })

@app.route('/api/stop_process/<job_id>', methods=['POST'])
def stop_process(job_id):
    """Stop a running process"""
    if job_id not in RUNNING_PROCESSES:
        return jsonify({'error': 'Job not found'}), 404
    
    info = RUNNING_PROCESSES[job_id]
    if info.get('pid'):
        try:
            # Terminate the process and its children
            parent = psutil.Process(info['pid'])
            children = parent.children(recursive=True)
            
            for child in children:
                child.terminate()
            parent.terminate()
            
            # Wait a bit and force kill if necessary
            time.sleep(2)
            for child in children:
                if child.is_running():
                    child.kill()
            if parent.is_running():
                parent.kill()
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Process not running'}), 400

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download result file"""
    safe_filename = secure_filename(filename)
    filepath = os.path.join('.', safe_filename)
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the API cache"""
    try:
        if os.path.exists('api_cache.db'):
            os.remove('api_cache.db')
        
        # Also clear checkpoint files
        checkpoint_files = glob.glob('checkpoint_chunk_*.json')
        for f in checkpoint_files:
            os.remove(f)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files')
def list_files():
    """List available CSV files"""
    files = []
    
    # Check uploads folder
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            if f.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
                files.append({
                    'filename': f,
                    'filepath': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
    
    # Check current directory
    for f in os.listdir('.'):
        if f.endswith('.csv') and not f.startswith('.'):
            files.append({
                'filename': f,
                'filepath': f,
                'size': os.path.getsize(f),
                'modified': datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
            })
    
    return jsonify(files)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)