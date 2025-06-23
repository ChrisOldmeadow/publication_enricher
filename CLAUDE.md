# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python tool for enriching CSV files containing publication data (titles, DOIs) with abstracts and metadata from scientific databases (Elsevier, PubMed, Crossref, Semantic Scholar). It processes large datasets efficiently using async operations, multi-processing, and intelligent caching.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running the Tool
```bash
# Single-process mode
python enrich_csv.py input.csv

# Multi-process mode (recommended for large datasets)
python multi_process_enricher.py input.csv

# Test API connectivity
python test_elsevier_api.py
```

### Development Tools
```bash
# Create test dataset subset
python create_test_subset.py

# No formal test suite - use test_elsevier_api.py for API validation
# No linting/formatting commands configured
```

## Architecture Overview

The codebase follows a three-layer architecture:

### Core Components
- **`PublicationProcessor`** (`processor.py`): Main orchestrator that handles CSV processing, batching, checkpointing, and result aggregation
- **`APIClient`** (`api_client.py`): Manages all external API calls with caching, rate limiting, retry logic, and response parsing
- **`ngram_matcher.py`**: Advanced fuzzy matching using character n-grams for improved publication matching

### Processing Flow
1. **CSV Input Processing**: Normalizes column names (`title`/`Output_Title`, `doi`/`Ref_DOI`)
2. **Batch Processing**: Groups publications into batches for efficient processing with checkpointing
3. **Multi-API Search Strategy**: For each pub, tries APIs in sequence until match found:
   - DOI exact match (most reliable)
   - Title exact match
   - Fuzzy title matching (multiple algorithms)
4. **Result Aggregation**: Generates multiple output files (`*_enriched.csv`, `*_failed.csv`, `*_fuzzy.csv`)

### API Integration Architecture
- **Async HTTP client** with connection pooling and session management
- **SQLite caching layer** with 30-day expiration for API responses
- **Rate limiting** with adaptive backoff per API source
- **API reliability tracking** with automatic disabling of problematic APIs
- **Retry logic** with exponential backoff for transient failures

### Multi-Processing Design
- **Process-level parallelism** via `multi_process_enricher.py`
- **Chunk-based distribution** where each process handles CSV chunks independently
- **Shared caching database** accessible across all processes
- **Result aggregation** that combines outputs from all processes

## Configuration

### API Keys (required in `.env` file)
```
ELSEVIER_API_KEY=your_key
PUBMED_EMAIL=your_email@domain.com
PUBMED_API_KEY=your_key  # optional
CROSSREF_EMAIL=your_email@domain.com  # optional but recommended
SEMANTIC_SCHOLAR_API_KEY=your_key  # optional
```

### Performance Tuning Parameters
- `--batch-size`: Publications per batch (default: 50)
- `--max-concurrent`: Concurrent API requests per process (default: 10)
- `--processes`: Number of parallel processes (default: CPU count)
- `--cache-db`: SQLite cache file path (default: api_cache.db)

## Key Implementation Details

### Fuzzy Matching Strategy
The tool uses a sophisticated multi-tier matching approach:
1. **Title normalization**: Removes HTML tags, standardizes punctuation, handles academic notation
2. **Multiple fuzzy algorithms**: Combines `fuzzywuzzy` ratios with custom n-gram character matching
3. **Prefix matching**: Handles truncated titles and abstract-as-title cases
4. **Match scoring**: 90% similarity threshold with detailed match metadata

### Caching and Performance
- **Response caching**: All API responses cached in SQLite with metadata
- **Cache expiration**: 30-day TTL with automatic cleanup
- **Checkpoint system**: Batch progress saved for resume capability
- **Memory management**: Configurable batch sizes to control memory usage

### Error Handling and Reliability
- **API failure detection**: Tracks consecutive failures per API source
- **Automatic API disabling**: Temporarily disables problematic APIs
- **Manual API control**: `--disable-pubmed`, `--disable-semantic` flags
- **Graceful degradation**: Continues processing with remaining APIs when some fail

## File Structure Context

- **Entry points**: `enrich_csv.py` (single-process), `multi_process_enricher.py` (multi-process)
- **Core library**: `publication_enricher/` package with processor, API client, and matching logic
- **Utilities**: `test_elsevier_api.py` for API validation, `create_test_subset.py` for test data
- **Output patterns**: Generates `*_enriched.csv`, `*_failed.csv`, `*_fuzzy.csv`, `*_missing_data.csv`