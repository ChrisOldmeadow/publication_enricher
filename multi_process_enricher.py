#!/usr/bin/env python3
"""
Multiprocessing wrapper for enriching publication CSV files.
"""
import argparse
import asyncio
import logging
import os
import sys
import time
import multiprocessing
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from publication_enricher import PublicationProcessor

def setup_logging(verbose: bool, process_id=None):
    """Configure logging based on verbosity level."""
    # Reset handlers on root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up format based on process ID
    if process_id is not None:
        log_format = f"%(asctime)s - [Process {process_id}] %(message)s"
    else:
        log_format = "%(asctime)s - %(message)s"
    date_format = "%H:%M:%S"
    
    # Console handler for progress info
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)
    
    # Set minimal console level regardless of verbose setting
    root_logger.setLevel(logging.INFO)
    
    # Suppress verbose logging from all modules
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    logging.getLogger('aiosqlite').setLevel(logging.ERROR)
    logging.getLogger('publication_enricher.api_client').setLevel(logging.WARNING)
    
    # Turn off all the API error logging except for critical errors
    api_logger = logging.getLogger('publication_enricher.api_client')
    api_logger.setLevel(logging.ERROR)

async def process_chunk(chunk_df, output_path, elsevier_key, pubmed_email, crossref_email, semantic_scholar_key, batch_size, max_concurrent, cache_db, checkpoint_path=None, process_id=None):
    """Process a chunk of publications asynchronously."""
    # Initialize processor
    processor = PublicationProcessor(
        elsevier_api_key=elsevier_key,
        pubmed_email=pubmed_email,
        crossref_email=crossref_email,
        semantic_scholar_api_key=semantic_scholar_key,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        cache_db=cache_db
    )
    
    # Setup processor
    await processor.setup()
    
    # Save chunk to temp file
    temp_input = f"temp_chunk_{process_id}.csv"
    temp_output = f"temp_chunk_{process_id}_out.csv"
    chunk_df.to_csv(temp_input, index=False)
    
    try:
        # Process chunk
        stats = await processor.process_csv(
            temp_input,
            temp_output,
            checkpoint_path=checkpoint_path
        )
        
        # Read processed data
        if os.path.exists(temp_output):
            return pd.read_csv(temp_output), stats
        else:
            return chunk_df, {"processed": 0, "enriched": 0, "failed": len(chunk_df)}
    
    finally:
        # Clean up temp files
        for f in [temp_input, temp_output]:
            if os.path.exists(f):
                os.remove(f)

def worker_process(chunk_df, output_queue, elsevier_key, pubmed_email, crossref_email, semantic_scholar_key, batch_size, max_concurrent, cache_db, process_id):
    """Worker process to handle a chunk of publications."""
    setup_logging(True, process_id)
    logging.info(f"Process {process_id} starting with {len(chunk_df)} publications")
    
    try:
        # Run the async processing
        result_df, stats = asyncio.run(process_chunk(
            chunk_df, 
            None, 
            elsevier_key, 
            pubmed_email,
            crossref_email,
            semantic_scholar_key,
            batch_size, 
            max_concurrent, 
            cache_db,
            checkpoint_path=f"checkpoint_chunk_{process_id}.json",
            process_id=process_id
        ))
        
        logging.info(f"Process {process_id} completed. Stats: {stats}")
        output_queue.put((process_id, result_df, stats))
    
    except Exception as e:
        logging.error(f"Process {process_id} failed: {str(e)}")
        output_queue.put((process_id, chunk_df, {"error": str(e)}))

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Enrich CSV files with publication abstracts using multiprocessing"
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
        help="Maximum number of concurrent API requests per process (default: 10)"
    )
    
    parser.add_argument(
        "--cache-db",
        type=str,
        default="api_cache.db",
        help="Path to cache database (default: api_cache.db)"
    )
    
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Number of parallel processes to use (default: 2)"
    )
    
    parser.add_argument(
        "--elsevier-key",
        type=str,
        help="Elsevier API key (can also be set via ELSEVIER_API_KEY env var)"
    )
    
    parser.add_argument(
        "--pubmed-email",
        type=str,
        help="PubMed email address (can also be set via PUBMED_EMAIL env var)"
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
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Get API keys
    elsevier_key = args.elsevier_key or os.environ.get('ELSEVIER_API_KEY')
    pubmed_email = args.pubmed_email or os.environ.get('PUBMED_EMAIL')
    crossref_email = args.crossref_email or os.environ.get('CROSSREF_EMAIL')
    semantic_scholar_key = args.semantic_scholar_key or os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
    
    logging.info(f"Elsevier API key found: {'yes' if elsevier_key else 'no'}")
    logging.info(f"PubMed email found: {'yes' if pubmed_email else 'no'}")
    logging.info(f"Crossref email found: {'yes' if crossref_email else 'no'}")
    logging.info(f"Semantic Scholar API key found: {'yes' if semantic_scholar_key else 'no'}")
    
    
    if not elsevier_key:
        parser.error("Elsevier API key must be provided via --elsevier-key or ELSEVIER_API_KEY env var")
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = str(input_path.parent / f"{input_path.stem}_enriched_mp.csv")
    
    logging.info(f"Output will be written to: {output_path}")
    
    # Read the input file
    logging.info(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    total_pubs = len(df)
    logging.info(f"Found {total_pubs} publications to process")
    
    # Check and map column names if necessary (same as in processor.py)
    column_mapping = {}
    if 'Output_Title' in df.columns and 'title' not in df.columns:
        column_mapping['Output_Title'] = 'title'
        logging.debug("Mapped 'Output_Title' to 'title'")
    if 'Ref_DOI' in df.columns and 'doi' not in df.columns:
        column_mapping['Ref_DOI'] = 'doi'
        logging.debug("Mapped 'Ref_DOI' to 'doi'")
        
    # Apply column mapping if needed
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logging.info(f"Renamed columns using mapping: {column_mapping}")
    
    # Limit the number of processes based on data size
    num_processes = min(args.processes, max(1, total_pubs // 100))
    logging.info(f"Using {num_processes} parallel processes")
    
    # Split dataframe into chunks
    chunk_size = total_pubs // num_processes
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, total_pubs, chunk_size)]
    
    # Create a queue for results
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    
    # Start the worker processes
    start_time = time.time()
    processes = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            p = multiprocessing.Process(
                target=worker_process,
                args=(chunk, result_queue, elsevier_key, pubmed_email,
                      crossref_email, semantic_scholar_key, args.batch_size, 
                      args.max_concurrent, args.cache_db, i)
            )
            processes.append(p)
            p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    stats = {
        "total": total_pubs, 
        "processed": 0, 
        "enriched": 0, 
        "failed": 0, 
        "sources": {
            "elsevier": 0,
            "pubmed": 0,
            "crossref": 0,
            "semantic_scholar": 0
        }
    }
    
    for _ in range(len(processes)):
        process_id, result_df, process_stats = result_queue.get()
        results.append((process_id, result_df))
        
        # Update overall stats
        if "processed" in process_stats:
            stats["processed"] += process_stats["processed"]
        if "enriched" in process_stats:
            stats["enriched"] += process_stats["enriched"]
        if "failed" in process_stats:
            stats["failed"] += process_stats["failed"]
        
        # Update source counts
        if "sources" in process_stats:
            for source, count in process_stats["sources"].items():
                if source in stats["sources"]:
                    stats["sources"][source] += count
    
    # Sort results by process_id to maintain order
    results.sort(key=lambda x: x[0])
    
    # Combine results
    combined_df = pd.concat([r[1] for r in results], ignore_index=True)
    
    # Save final results
    combined_df.to_csv(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    logging.info(f"\nProcessing complete in {elapsed_time:.2f} seconds!")
    logging.info(f"Total publications: {stats['total']}")
    logging.info(f"Successfully enriched: {stats['enriched']}")
    logging.info(f"Failed to enrich: {stats['failed']}")
    
    # Display source breakdown
    source_stats = stats['sources']
    non_zero_sources = {k: v for k, v in source_stats.items() if v > 0}
    if non_zero_sources:
        logging.info("\nEnrichment sources breakdown:")
        for source, count in non_zero_sources.items():
            percentage = (count / stats['enriched'] * 100) if stats['enriched'] > 0 else 0
            logging.info(f"  - {source}: {count} ({percentage:.1f}%)")
    else:
        logging.info("No enrichments from any source.")
    
    return stats

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
