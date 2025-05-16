# Publication Enricher

A Python tool to efficiently enrich large CSV files of publications with abstracts from Elsevier's API.

## Features

- Asynchronous API requests for high throughput
- SQLite-based caching to avoid redundant API calls
- Batch processing with checkpointing
- Progress tracking with statistics
- Configurable concurrency and batch sizes
- Automatic retry with exponential backoff
- Cache expiration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/publication_enricher.git
cd publication_enricher
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:
Create a `.env` file in the project root with your API credentials:
```
ELSEVIER_API_KEY=your_elsevier_key
```

## Usage

Basic usage:
```bash
python enrich_csv.py input.csv
```

This will create `input_enriched.csv` in the same directory.

Advanced options:
```bash
python enrich_csv.py input.csv \
  -o output.csv \
  --batch-size 100 \
  --max-concurrent 20 \
  --cache-db cache.db \
  --checkpoint checkpoint.json \
  --elsevier-key YOUR_API_KEY \
  -v
```

Options:
- `-o, --output`: Output file path
- `--batch-size`: Number of publications to process in each batch (default: 50)
- `--max-concurrent`: Maximum number of concurrent API requests (default: 10)
- `--cache-db`: Path to cache database (default: api_cache.db)
- `--checkpoint`: Path to save/load checkpoint file
- `--elsevier-key`: Elsevier API key (can also be set via env var)
- `-v, --verbose`: Enable verbose logging

## Input CSV Format

The input CSV file should contain at least these columns:
- `title`: Publication title
- `doi` (optional): DOI of the publication

Example:
```csv
title,doi
"Example Paper Title","10.1234/example.doi"
"Another Paper Title",""
```

## Output Format

The output CSV will contain all original columns plus:
- `abstract`: The publication abstract (if found)
- Additional metadata from Elsevier (when available)

## Performance Considerations

- The tool uses async/await for efficient API requests
- Caching significantly reduces API calls for repeated runs
- Batch processing allows for checkpointing and resume
- Configurable concurrency helps optimize for rate limits
- For 15,000 articles, expect processing time of 1-2 hours depending on cache hits

## Error Handling

- Failed API requests are automatically retried with exponential backoff
- Progress is saved in checkpoint files for resume capability
- Detailed logging helps track issues
- Cache entries expire after 30 days by default

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 