#!/usr/bin/env python3
"""
Command-line tool for enriching publication CSV files with abstracts.
"""
import argparse
import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from publication_enricher import PublicationProcessor

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

async def main():
    # Load environment variables
    load_dotenv()
    
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
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Get API key
    api_key = args.elsevier_key or os.environ.get('ELSEVIER_API_KEY')
    if not api_key:
        parser.error("Elsevier API key must be provided via --elsevier-key or ELSEVIER_API_KEY env var")
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = str(input_path.parent / f"{input_path.stem}_enriched.csv")
    
    # Initialize processor
    processor = PublicationProcessor(
        elsevier_api_key=api_key,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        cache_db=args.cache_db
    )
    
    # Setup processor
    await processor.setup()
    
    # Process file
    stats = await processor.process_csv(
        args.input_file,
        output_path,
        checkpoint_path=args.checkpoint
    )
    
    return stats

if __name__ == "__main__":
    asyncio.run(main()) 