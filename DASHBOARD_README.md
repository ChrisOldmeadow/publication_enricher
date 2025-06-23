# Publication Enricher Dashboard

A Flask web dashboard for monitoring and managing publication enrichment processes.

## Features

### 🖥️ **Real-time Monitoring**
- Live status updates of running enrichment processes
- Progress tracking with detailed statistics
- Real-time log output monitoring
- API configuration status checking

### 📁 **File Management**
- Drag & drop CSV file upload
- Browse and select existing files
- File validation and preview
- Download enriched results

### ⚙️ **Process Control**
- Start enrichment with customizable parameters
- Stop running processes
- Resume from checkpoints
- Multiple concurrent process support

### 🗄️ **Cache Management**
- View cache size and status
- Clear API cache and checkpoints
- Monitor cache utilization

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Start the dashboard:**
   ```bash
   python run_dashboard.py
   ```

3. **Access the dashboard:**
   Open your browser to `http://localhost:5001`

## Usage

### Starting an Enrichment Process

1. **Upload or Select File:**
   - Drag & drop a CSV file onto the upload area, OR
   - Use the file picker to select a file, OR
   - Choose from existing files in the dropdown

2. **Configure Options:**
   - **Processes:** Number of parallel processes (1-8)
   - **Batch Size:** Publications per batch (10-200)
   - **Max Concurrent:** Concurrent API requests (1-20)
   - **API Options:** Disable specific APIs if needed

3. **Start Processing:**
   - Click "Start Enrichment" to begin
   - Monitor progress in real-time
   - Download results when complete

### Monitoring Progress

The dashboard shows:
- **Overall Progress:** Percentage complete with progress bar
- **Statistics:** Enriched, failed, and fuzzy match counts
- **Recent Logs:** Latest processing messages
- **Runtime:** Duration and status

### Managing Processes

- **Stop Process:** Terminate running enrichment
- **Download Results:** Get enriched CSV files
- **View Details:** Expand process information

## API Endpoints

The dashboard provides REST APIs for integration:

- `GET /api/status` - Get current system status
- `POST /api/upload` - Upload CSV file
- `POST /api/start_enrichment` - Start enrichment process
- `POST /api/stop_process/<job_id>` - Stop specific process
- `GET /api/download/<filename>` - Download result file
- `POST /api/clear_cache` - Clear API cache
- `GET /api/files` - List available files

## Configuration

### Environment Variables

The dashboard checks for these API keys:
- `ELSEVIER_API_KEY` - Elsevier Scopus API
- `PUBMED_EMAIL` - PubMed API email
- `PUBMED_API_KEY` - PubMed API key (optional)
- `CROSSREF_EMAIL` - Crossref API email (optional)
- `SEMANTIC_SCHOLAR_API_KEY` - Semantic Scholar API (optional)

### File Locations

- **Uploads:** `uploads/` directory
- **Logs:** `logs/` directory
- **Cache:** `api_cache.db` in current directory
- **Checkpoints:** `checkpoint_chunk_*.json` files

## Integration with Existing Flask Apps

To integrate with your existing Flask application:

```python
from dashboard_app import app as dashboard_app

# Mount as blueprint
your_app.register_blueprint(dashboard_app, url_prefix='/enricher')
```

Or include the routes directly:

```python
from dashboard_app import api_status, start_enrichment
# Add specific routes to your app
```

## Security Notes

- Dashboard runs on port 5001 by default
- File uploads limited to 100MB
- Only CSV files accepted for upload
- Process control requires confirmation
- All file operations use secure filenames

## Troubleshooting

### Common Issues

1. **"No module named 'psutil'" Error:**
   ```bash
   pip install psutil
   ```

2. **Permission denied for port 5001:**
   - Change port in `dashboard_app.py`
   - Or run with different port: `app.run(port=5002)`

3. **File upload fails:**
   - Check file is valid CSV
   - Ensure uploads directory exists
   - Verify file size under 100MB

4. **Process won't start:**
   - Check API keys are configured
   - Verify input file exists and is readable
   - Check logs directory is writable

### Debug Mode

Enable debug logging by setting `debug=True` in the app.run() call.

## Architecture

- **Backend:** Flask + Python
- **Frontend:** Bootstrap 5 + Vanilla JavaScript
- **Real-time Updates:** AJAX polling every 5 seconds
- **Process Management:** Python subprocess + psutil
- **File Handling:** Werkzeug secure uploads
- **Progress Tracking:** JSON checkpoint files