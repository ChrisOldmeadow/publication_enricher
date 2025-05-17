#!/usr/bin/env python3
"""
Command-line tool for enriching publication CSV files with abstracts.
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from publication_enricher import PublicationProcessor

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Set up file handler
    fh = logging.FileHandler('enrich_csv.log')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # Set up console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)
    
    # Configure package logger
    pkg_logger = logging.getLogger('publication_enricher')
    pkg_logger.setLevel(level)
    
    logging.info("Logging setup complete")

async def main():
    try:
        # Load environment variables
        load_dotenv()
        print("Environment variables loaded", file=sys.stdout)
        sys.stdout.flush()
        
        parser = argparse.ArgumentParser(
            description="Enrich CSV files with publication abstracts from Elsevier"
        )
        
        parser.add_argument(
            "input_file",
            type=str,
            help="Input CSV file path"
        )
        
        parser.add_argument(
            "-o", "--output",
            type=str,
            help="Output CSV file path (default: input_enriched.csv)"
        )
        
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Number of publications to process in each batch (default: 50)"
        )
        
        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=10,
            help="Maximum number of concurrent API requests (default: 10)"
        )
        
        parser.add_argument(
            "--cache-db",
            type=str,
            default="api_cache.db",
            help="Path to cache database (default: api_cache.db)"
        )
        
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="Path to save/load checkpoint file"
        )
        
        parser.add_argument(
            "--elsevier-key",
            type=str,
            help="Elsevier API key (can also be set via ELSEVIER_API_KEY env var)"
        )
        
        parser.add_argument(
            "--pubmed-key",
            type=str,
            help="PubMed API key (can also be set via PUBMED_API_KEY env var)"
        )
        
        parser.add_argument(
            "--crossref-email",
            type=str,
            help="Email for Crossref API (can also be set via CROSSREF_EMAIL env var)"
        )
        
        parser.add_argument(
            "--semantic-scholar-key",
            type=str,
            help="Semantic Scholar API key (can also be set via SEMANTIC_SCHOLAR_API_KEY env var)"
        )
        
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        args = parser.parse_args()
        print(f"Arguments parsed: {args}", file=sys.stdout)
        sys.stdout.flush()
        
        # Setup logging
        setup_logging(args.verbose)
        
        # Get API keys
        elsevier_key = args.elsevier_key or os.environ.get('ELSEVIER_API_KEY')
        pubmed_key = args.pubmed_key or os.environ.get('PUBMED_API_KEY')
        crossref_email = args.crossref_email or os.environ.get('CROSSREF_EMAIL')
        semantic_scholar_key = args.semantic_scholar_key or os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        
        print(f"Elsevier API key found: {'yes' if elsevier_key else 'no'}", file=sys.stdout)
        print(f"PubMed API key found: {'yes' if pubmed_key else 'no'}", file=sys.stdout)
        print(f"Crossref email found: {'yes' if crossref_email else 'no'}", file=sys.stdout)
        print(f"Semantic Scholar API key found: {'yes' if semantic_scholar_key else 'no'}", file=sys.stdout)
        sys.stdout.flush()
        
        if not elsevier_key:
            parser.error("Elsevier API key must be provided via --elsevier-key or ELSEVIER_API_KEY env var")
        
        # Set output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_file)
            output_path = str(input_path.parent / f"{input_path.stem}_enriched.csv")
        
        print(f"Output will be written to: {output_path}", file=sys.stdout)
        sys.stdout.flush()
        
        # Initialize processor
        processor = PublicationProcessor(
            elsevier_api_key=elsevier_key,
            pubmed_email=pubmed_key,
            crossref_email=crossref_email,
            semantic_scholar_api_key=semantic_scholar_key,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            cache_db=args.cache_db
        )
        
        print("Processor initialized", file=sys.stdout)
        sys.stdout.flush()
        
        # Setup processor
        await processor.setup()
        print("Processor setup complete", file=sys.stdout)
        sys.stdout.flush()
        
        # Process file
        stats = await processor.process_csv(
            args.input_file,
            output_path,
            checkpoint_path=args.checkpoint
        )
        
        print(f"Processing complete. Stats: {stats}", file=sys.stdout)
        sys.stdout.flush()
        
        return stats
    
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1) 