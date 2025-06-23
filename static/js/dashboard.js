// Dashboard JavaScript

class EnrichmentDashboard {
    constructor() {
        this.selectedFile = null;
        this.statusInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadFiles();
        this.startStatusUpdates();
    }

    setupEventListeners() {
        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Existing file selection
        document.getElementById('existing-files').addEventListener('change', (e) => {
            this.handleExistingFileSelect(e.target.value);
        });

        // Start enrichment button
        document.getElementById('start-enrichment-btn').addEventListener('click', () => {
            this.startEnrichment();
        });

        // Clear cache button
        document.getElementById('clear-cache-btn').addEventListener('click', () => {
            this.clearCache();
        });

        // Drag and drop for file upload
        const fileInput = document.getElementById('file-input');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileInput.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            fileInput.addEventListener(eventName, this.highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileInput.addEventListener(eventName, this.unhighlight, false);
        });

        fileInput.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(e) {
        e.target.closest('.upload-area')?.classList.add('dragover');
    }

    unhighlight(e) {
        e.target.closest('.upload-area')?.classList.remove('dragover');
    }

    async handleFileSelect(file) {
        if (!file || !file.name.endsWith('.csv')) {
            this.showAlert('Please select a CSV file', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.selectedFile = result.file_info;
                this.updateFileInfo();
                document.getElementById('start-enrichment-btn').disabled = false;
                this.showAlert(`File uploaded: ${result.file_info.rows} rows`, 'success');
                this.loadFiles(); // Refresh file list
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Upload failed: ' + error.message, 'danger');
        }
    }

    handleExistingFileSelect(filepath) {
        if (!filepath) {
            this.selectedFile = null;
            document.getElementById('start-enrichment-btn').disabled = true;
            return;
        }

        // Set selected file info
        this.selectedFile = {
            filepath: filepath,
            filename: filepath.split('/').pop()
        };
        
        this.updateFileInfo();
        document.getElementById('start-enrichment-btn').disabled = false;
    }

    updateFileInfo() {
        if (!this.selectedFile) return;

        const fileInfo = document.querySelector('.file-info') || this.createFileInfoElement();
        fileInfo.innerHTML = `
            <strong>Selected:</strong> ${this.selectedFile.filename}<br>
            ${this.selectedFile.rows ? `<strong>Rows:</strong> ${this.selectedFile.rows}<br>` : ''}
            ${this.selectedFile.columns ? `<strong>Columns:</strong> ${this.selectedFile.columns.join(', ')}` : ''}
        `;
    }

    createFileInfoElement() {
        const element = document.createElement('div');
        element.className = 'file-info';
        document.querySelector('#file-input').parentNode.appendChild(element);
        return element;
    }

    async loadFiles() {
        try {
            const response = await fetch('/api/files');
            const files = await response.json();
            
            const select = document.getElementById('existing-files');
            select.innerHTML = '<option value="">Select existing file...</option>';
            
            files.forEach(file => {
                const option = document.createElement('option');
                option.value = file.filepath;
                option.textContent = `${file.filename} (${this.formatFileSize(file.size)})`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load files:', error);
        }
    }

    async startEnrichment() {
        if (!this.selectedFile) {
            this.showAlert('Please select a file first', 'warning');
            return;
        }

        const options = {
            processes: parseInt(document.getElementById('processes').value),
            batch_size: parseInt(document.getElementById('batch-size').value),
            max_concurrent: parseInt(document.getElementById('max-concurrent').value),
            disable_pubmed: document.getElementById('disable-pubmed').checked,
            disable_semantic: document.getElementById('disable-semantic').checked
        };

        try {
            const response = await fetch('/api/start_enrichment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filepath: this.selectedFile.filepath,
                    options: options
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`Enrichment started (Job ID: ${result.job_id})`, 'success');
                this.updateStatus(); // Immediate update
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Failed to start enrichment: ' + error.message, 'danger');
        }
    }

    async clearCache() {
        if (!confirm('Are you sure you want to clear the cache? This will remove all cached API responses and checkpoints.')) {
            return;
        }

        try {
            const response = await fetch('/api/clear_cache', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Cache cleared successfully', 'success');
                this.updateStatus();
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Failed to clear cache: ' + error.message, 'danger');
        }
    }

    startStatusUpdates() {
        this.updateStatus();
        this.statusInterval = setInterval(() => {
            this.updateStatus();
        }, 5000); // Update every 5 seconds
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateStatusIndicator(status);
            this.updateCacheInfo(status.cache_info);
            this.updateProcesses(status.processes);
            
        } catch (error) {
            console.error('Failed to update status:', error);
            this.updateStatusIndicator({ processes: [] });
        }
    }

    updateStatusIndicator(status) {
        const indicator = document.getElementById('status-indicator');
        const runningProcesses = status.processes.filter(p => p.running).length;
        
        if (runningProcesses > 0) {
            indicator.className = 'badge bg-success';
            indicator.textContent = `${runningProcesses} Running`;
        } else {
            indicator.className = 'badge bg-secondary';
            indicator.textContent = 'Idle';
        }
    }

    updateCacheInfo(cacheInfo) {
        const container = document.getElementById('cache-info');
        
        if (cacheInfo.exists) {
            container.innerHTML = `
                <div class="d-flex justify-content-between">
                    <span>Size:</span>
                    <span class="cache-size">${this.formatFileSize(cacheInfo.size)}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <span>Modified:</span>
                    <span class="text-muted">${new Date(cacheInfo.modified).toLocaleString()}</span>
                </div>
            `;
        } else {
            container.innerHTML = '<p class="text-muted mb-0">No cache found</p>';
        }
    }

    updateProcesses(processes) {
        const container = document.getElementById('processes-container');
        
        if (processes.length === 0) {
            container.innerHTML = '<p class="text-muted">No processes running</p>';
            return;
        }

        container.innerHTML = processes.map(process => this.renderProcess(process)).join('');
        
        // Add event listeners for process actions
        container.querySelectorAll('.stop-process-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const jobId = e.target.dataset.jobId;
                this.stopProcess(jobId);
            });
        });

        container.querySelectorAll('.download-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filename = e.target.dataset.filename;
                this.downloadFile(filename);
            });
        });
    }

    renderProcess(process) {
        const startTime = new Date(process.start_time);
        const duration = Math.round((Date.now() - startTime.getTime()) / 1000 / 60); // minutes
        
        const statusClass = process.running ? 'running' : 'completed';
        const statusText = process.running ? 'Running' : 'Completed';
        const statusIcon = process.running ? 'fa-play' : 'fa-check';
        
        // Use progress info if available, otherwise fall back to stats
        let progressPercent = 0;
        let currentCount = 0;
        let totalCount = 0;
        let enrichedCount = 0;
        let failedCount = 0;
        let processingRate = 0;
        
        if (process.progress && process.progress.total > 0) {
            progressPercent = Math.round(process.progress.percentage);
            currentCount = process.progress.current;
            totalCount = process.progress.total;
            enrichedCount = process.progress.enriched;
            failedCount = process.progress.failed;
            processingRate = process.progress.rate;
        } else if (process.stats.total > 0) {
            progressPercent = Math.round((process.stats.enriched / process.stats.total) * 100);
            currentCount = process.stats.enriched;
            totalCount = process.stats.total;
            enrichedCount = process.stats.enriched;
            failedCount = process.stats.failed;
        }

        return `
            <div class="card process-card ${statusClass}">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="card-title">
                                <i class="fas ${statusIcon}"></i>
                                ${process.filename}
                            </h6>
                            <p class="card-text">
                                <small class="text-muted">
                                    Job ID: ${process.job_id} | 
                                    Started: ${startTime.toLocaleString()} |
                                    Duration: ${duration}m
                                    ${processingRate > 0 ? ` | Rate: ${processingRate.toFixed(1)} pub/s` : ''}
                                </small>
                            </p>
                        </div>
                        <span class="badge ${process.running ? 'bg-success' : 'bg-secondary'}">
                            ${statusText}
                        </span>
                    </div>
                    
                    ${totalCount > 0 ? `
                        <div class="progress mb-2">
                            <div class="progress-bar" style="width: ${progressPercent}%">
                                ${progressPercent}% (${currentCount}/${totalCount})
                            </div>
                        </div>
                        
                        <div class="stats-row">
                            <div class="stat-item">
                                <div class="stat-value text-success">${enrichedCount}</div>
                                <div class="stat-label">Enriched</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value text-warning">${failedCount}</div>
                                <div class="stat-label">Failed</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value text-info">${process.stats.fuzzy || 0}</div>
                                <div class="stat-label">Fuzzy</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${totalCount}</div>
                                <div class="stat-label">Total</div>
                            </div>
                        </div>
                        
                        ${process.running && processingRate > 0 ? `
                            <div class="mt-2">
                                <small class="text-muted">
                                    ETA: ${this.calculateETA(currentCount, totalCount, processingRate)}
                                </small>
                            </div>
                        ` : ''}
                    ` : ''}
                    
                    ${process.recent_logs && process.recent_logs.length > 0 ? `
                        <div class="mt-3">
                            <small class="text-muted">Recent logs:</small>
                            <div class="log-output">
                                ${process.recent_logs.slice(-3).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div class="process-actions">
                        ${process.running ? `
                            <button class="btn btn-danger btn-sm stop-process-btn" data-job-id="${process.job_id}">
                                <i class="fas fa-stop"></i> Stop
                            </button>
                        ` : ''}
                        
                        ${process.stats.enriched > 0 ? `
                            <button class="btn btn-success btn-sm download-btn" data-filename="${process.output_file}">
                                <i class="fas fa-download"></i> Download Results
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    async stopProcess(jobId) {
        if (!confirm('Are you sure you want to stop this process?')) {
            return;
        }

        try {
            const response = await fetch(`/api/stop_process/${jobId}`, {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Process stopped', 'success');
                this.updateStatus();
            } else {
                this.showAlert(result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Failed to stop process: ' + error.message, 'danger');
        }
    }

    downloadFile(filename) {
        window.open(`/api/download/${filename}`, '_blank');
    }

    calculateETA(current, total, rate) {
        if (rate <= 0 || current >= total) return 'N/A';
        
        const remaining = total - current;
        const secondsRemaining = remaining / rate;
        
        if (secondsRemaining < 60) {
            return `${Math.round(secondsRemaining)}s`;
        } else if (secondsRemaining < 3600) {
            return `${Math.round(secondsRemaining / 60)}m`;
        } else {
            const hours = Math.floor(secondsRemaining / 3600);
            const minutes = Math.round((secondsRemaining % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showAlert(message, type) {
        // Create alert element
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert at top of container
        const container = document.querySelector('.container-fluid');
        container.insertBefore(alert, container.firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new EnrichmentDashboard();
});