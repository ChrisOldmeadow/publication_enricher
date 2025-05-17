# Publication Enricher

A Python tool to efficiently enrich large CSV files of publications with abstracts and metadata from multiple scientific publication APIs, including Elsevier, PubMed, Crossref, and Semantic Scholar.

## Features

- Multi-API support for maximum data coverage:
  - Elsevier API
  - PubMed API
  - Crossref API
  - Semantic Scholar API
- Advanced matching capabilities:
  - DOI-based exact matching
  - Title-based exact matching
  - Fuzzy title matching with configurable thresholds
  - Detailed match statistics tracking
- High-performance processing:
  - Asynchronous API requests for high throughput
  - Multi-process support for parallel processing
  - SQLite-based caching to avoid redundant API calls
  - Configurable concurrency and batch sizes
- Robust error handling and recovery:
  - Batch processing with checkpointing
  - Automatic retry with exponential backoff
  - API-specific rate limiting
  - Cache expiration management
- Comprehensive statistics reporting:
  - Match success rates by API source
  - Exact vs. fuzzy match performance metrics
  - Fuzzy match score distribution analysis

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
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_pubmed_key  # Optional but recommended
CROSSREF_EMAIL=your_email@example.com  # For polite usage
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key  # Optional
```

Notes on API credentials:
- Elsevier API key is required for Elsevier access
- PubMed requires an email address, and optionally an API key for higher rate limits
- Crossref works without authentication, but providing an email is considered "polite" usage and may get higher rate limits
- Semantic Scholar has a public API with rate limits, but an API key can increase these limits

## Usage

### Single-Process Mode

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
  --pubmed-email YOUR_EMAIL \
  --crossref-email YOUR_EMAIL \
  -v
```

### Multi-Process Mode (Higher Performance)

For larger datasets, you can use the multi-process enricher for parallel processing:

```bash
python multi_process_enricher.py input.csv \
  -o output.csv \
  --batch-size 100 \
  --max-concurrent 20 \
  --processes 4 \
  -v
```

### Command-Line Options

#### Common Options:
- `-o, --output`: Output file path
- `--batch-size`: Number of publications to process in each batch (default: 50)
- `--max-concurrent`: Maximum number of concurrent API requests per process (default: 10)
- `--cache-db`: Path to cache database (default: api_cache.db) 
- `-v, --verbose`: Enable verbose logging

#### Single-Process Options:
- `--checkpoint`: Path to save/load checkpoint file
- `--elsevier-key`: Elsevier API key (can also be set via env var)
- `--pubmed-email`: Email for PubMed API (can also be set via env var)
- `--pubmed-key`: API key for PubMed (can also be set via env var)
- `--crossref-email`: Email for Crossref API (can also be set via env var)
- `--semantic-scholar-key`: API key for Semantic Scholar (can also be set via env var)

#### Multi-Process Options:
- `--processes`: Number of parallel processes to use (default: 2)

### Configuration and Optimization

The enricher can be configured for different performance profiles:

#### Maximum Throughput
```bash
python multi_process_enricher.py input.csv \
  --processes 8 \
  --batch-size 50 \
  --max-concurrent 30
```

#### Low Memory Usage
```bash
python multi_process_enricher.py input.csv \
  --processes 2 \
  --batch-size 25 \
  --max-concurrent 10
```

#### Balanced Performance
```bash
python multi_process_enricher.py input.csv \
  --processes 4 \
  --batch-size 50 \
  --max-concurrent 20
```

## Input CSV Format

The input CSV file should contain publication information with at least one of these column pairs:

**Standard Column Names:**
- `title`: Publication title
- `doi` (optional): DOI of the publication

**Alternative Column Names (automatically mapped):**
- `Output_Title`: Alternative column name for publication title
- `Ref_DOI`: Alternative column name for publication DOI

The tool will automatically detect and map these alternative column names if present.

Example:
```csv
title,doi
"Example Paper Title","10.1234/example.doi"
"Another Paper Title",""
```

OR

```csv
Output_Title,Ref_DOI,additional_column
"Example Paper Title","10.1234/example.doi","extra data"
"Another Paper Title","","more data"
```

## Output Format

The output CSV will contain all original columns plus:

- `abstract`: The publication abstract (if found)
- `source`: The API source that provided the data (elsevier, pubmed, crossref, or semantic_scholar)
- `match_info`: Information about how the match was made, including:
  - `match_type`: 'exact' or 'fuzzy'
  - `fuzzy_matched`: Boolean indicating if fuzzy matching was used
  - `query_title`: The original title used for matching
  - `matched_title`: The title that was matched in the API
  - `match_score`: For fuzzy matches, the similarity score (90-100)
- Additional metadata specific to each API source (when available)

## Performance Considerations

### Multi-process vs Single-process

- The multi-process enricher (`multi_process_enricher.py`) offers significantly better performance for large datasets
- For best results on multi-core systems, set the number of processes to match available CPU cores
- Each process manages its own API concurrency, so total concurrent requests will be `processes × max-concurrent`

### Processing Speed Factors

- Caching significantly reduces API calls for repeated runs
- Batch processing allows for checkpointing and resume capability
- Configurable concurrency helps optimize for rate limits
- API rate limits vary by source and can affect overall throughput
- For 15,000 articles with multi-process enrichment, expect processing time of:
  - ~1-2 hours with 2 processes on first run
  - ~15-30 minutes on subsequent runs with cached results
  - ~30-60 minutes with 4 processes on first run

### Memory Usage

- Memory usage scales with `batch-size` and `max-concurrent` parameters
- For systems with limited memory, reduce batch size and increase processes
- Each API has different response sizes; Crossref typically returns smaller responses than Elsevier

### Recommended Configurations

- **Desktop/Laptop (4-8 cores)**: 2-4 processes, batch size 50, max-concurrent 15
- **Server (8+ cores)**: 4-8 processes, batch size 50, max-concurrent 20
- **Low Memory System**: 1-2 processes, batch size 25, max-concurrent 10

## Fuzzy Matching and Match Statistics

### Fuzzy Matching Capabilities

- The enricher uses the `fuzzywuzzy` library for title-based fuzzy matching
- Default matching threshold is 90% similarity (configurable)
- Matching process:
  1. Try exact DOI matching first (if DOI is available)
  2. Try exact title matching next (case-insensitive)
  3. Fall back to fuzzy title matching if exact matching fails
- Each API source is tried in sequence until a match is found

### Match Statistics

The enricher provides detailed statistics at the end of processing:

- Overall success rate with percentage breakdown
- Per-API source match statistics:
  - Exact match attempts and success rates
  - Fuzzy match attempts and success rates
- Fuzzy match score distribution showing quality of matches
- Source breakdown showing which APIs provided the data

These statistics help identify which API sources are most effective for your specific dataset and can guide optimization efforts.

## Error Handling

- Failed API requests are automatically retried with exponential backoff
- Progress is saved in checkpoint files for resume capability
- Detailed logging helps track issues
- Cache entries expire after 30 days by default
- API-specific error handling with adaptive disabling if rate limits are exceeded

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 