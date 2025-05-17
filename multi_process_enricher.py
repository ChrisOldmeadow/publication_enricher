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
        
        # Extract match stats for logging
        match_stats = processor.api_client.get_match_statistics()
        
        # Include match stats explicitly in the returned stats
        if 'match_attempts' not in stats:
            stats['match_attempts'] = match_stats
        
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
        # Return a properly structured stats dictionary even in case of error
        error_stats = {
            "error": str(e),
            "processed": 0,
            "enriched": 0,
            "failed": len(chunk_df),
            "sources": {
                "elsevier": 0,
                "pubmed": 0,
                "crossref": 0,
                "semantic_scholar": 0
            },
            "match_types": {
                "exact_matches": 0,
                "fuzzy_matches": 0
            },
            "match_attempts": {
                "by_source": {
                    "elsevier": {"exact_match_attempts": 0, "exact_match_successes": 0, "fuzzy_match_attempts": 0, "fuzzy_match_successes": 0},
                    "pubmed": {"exact_match_attempts": 0, "exact_match_successes": 0, "fuzzy_match_attempts": 0, "fuzzy_match_successes": 0},
                    "crossref": {"exact_match_attempts": 0, "exact_match_successes": 0, "fuzzy_match_attempts": 0, "fuzzy_match_successes": 0},
                    "semantic_scholar": {"exact_match_attempts": 0, "exact_match_successes": 0, "fuzzy_match_attempts": 0, "fuzzy_match_successes": 0}
                },
                "totals": {
                    "exact_match_attempts": 0,
                    "exact_match_successes": 0,
                    "fuzzy_match_attempts": 0,
                    "fuzzy_match_successes": 0
                },
                "rates": {
                    "exact_success_rate": 0,
                    "fuzzy_success_rate": 0
                }
            }
        }
        output_queue.put((process_id, chunk_df, error_stats))

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
    all_process_stats = []
    
    # For stats aggregation, start with basic information
    stats = {
        "total": total_pubs
    }
    
    # Collect results from all worker processes
    for _ in range(len(processes)):
        process_id, result_df, process_stats = result_queue.get()
        results.append((process_id, result_df))
        all_process_stats.append(process_stats)
        
        # We'll use get_stats_summary instead of manually aggregating here
    
    # Sort results by process_id to maintain order
    results.sort(key=lambda x: x[0])
    
    # Combine results dataframes
    combined_df = pd.concat([r[1] for r in results], ignore_index=True)
    
    # Generate a clean summary of the stats using our helper
    stats = get_stats_summary(all_process_stats, total_pubs)
    
    # Double-check enrichment success based on the combined dataframe
    has_abstract = combined_df['abstract'].notna()
    actual_enriched_count = has_abstract.sum()
    
    # If there's a discrepancy, update the stats
    if actual_enriched_count > 0 and stats['enriched'] == 0:
        logging.info(f"Updated enrichment count from dataframe: {actual_enriched_count}")
        stats['enriched'] = actual_enriched_count
        stats['failed'] = len(combined_df) - actual_enriched_count
    
    # Save final results
    combined_df.to_csv(output_path, index=False)
    
    end_time = time.time()
    logging.info(f"\nProcessing complete in {end_time - start_time:.2f} seconds!")
    logging.info(f"Total publications: {stats['total']}")
    
    # Calculate and log the enrichment success percentage
    enrichment_percentage = 0.0
    if stats['total'] > 0:
        enrichment_percentage = (stats['enriched'] / stats['total']) * 100
    
    logging.info(f"Successfully enriched: {stats['enriched']} ({enrichment_percentage:.1f}%)")
    logging.info(f"Failed to enrich: {stats['failed']} ({100 - enrichment_percentage:.1f}%)")
    
    
    # Log source breakdown
    sources_sum = sum(stats['sources'].values())
    if sources_sum > 0:
        logging.info(f"Source breakdown: {stats['sources']}")
    elif stats['enriched'] > 0:
        # If we have enrichments but no source counts, infer sources from match statistics
        inferred_sources = {}
        if 'match_attempts' in stats and 'by_source' in stats['match_attempts']:
            for source, source_stats in stats['match_attempts']['by_source'].items():
                exact_successes = source_stats.get('exact_match_successes', 0)
                fuzzy_successes = source_stats.get('fuzzy_match_successes', 0)
                if exact_successes > 0 or fuzzy_successes > 0:
                    inferred_sources[source] = exact_successes + fuzzy_successes
        
        if inferred_sources:
            logging.info(f"Source breakdown (inferred from match statistics): {inferred_sources}")
        else:
            logging.info(f"Successfully enriched {stats['enriched']} publications, but source breakdown unavailable.")
    else:
        logging.info(f"No enrichments from any source.")
        
    # Log match statistics
    if 'match_types' in stats:
        logging.info(f"\nMatch statistics:")
        logging.info(f"  - Exact matches: {stats['match_types']['exact_matches']}")
        logging.info(f"  - Fuzzy matches: {stats['match_types']['fuzzy_matches']}")
        
    # Log match attempt statistics - with defensive checks
    try:
        if 'match_attempts' in stats:
            if 'totals' in stats['match_attempts']:
                totals = stats['match_attempts']['totals']
                rates = stats['match_attempts'].get('rates', {'exact_success_rate': 0, 'fuzzy_success_rate': 0})
                
                logging.info(f"\nMatching attempt statistics:")
                logging.info(f"  - Exact match attempts: {totals.get('exact_match_attempts', 0)}")
                logging.info(f"  - Exact match successes: {totals.get('exact_match_successes', 0)}")
                logging.info(f"  - Exact match success rate: {rates.get('exact_success_rate', 0)}%")
                logging.info(f"  - Fuzzy match attempts: {totals.get('fuzzy_match_attempts', 0)}")
                logging.info(f"  - Fuzzy match successes: {totals.get('fuzzy_match_successes', 0)}")
                logging.info(f"  - Fuzzy match success rate: {rates.get('fuzzy_success_rate', 0)}%")
                
                # Per-source match statistics
                if 'by_source' in stats['match_attempts']:
                    logging.info(f"\nPer-source match statistics:")
                    for source, source_stats in stats['match_attempts']['by_source'].items():
                        # Safely get the statistics with defaults
                        exact_attempts = source_stats.get('exact_match_attempts', 0)
                        exact_successes = source_stats.get('exact_match_successes', 0)
                        fuzzy_attempts = source_stats.get('fuzzy_match_attempts', 0)
                        fuzzy_successes = source_stats.get('fuzzy_match_successes', 0)
                        
                        if exact_attempts > 0 or fuzzy_attempts > 0:
                            exact_rate = (exact_successes / exact_attempts * 100) if exact_attempts > 0 else 0
                            fuzzy_rate = (fuzzy_successes / fuzzy_attempts * 100) if fuzzy_attempts > 0 else 0
                            logging.info(f"  - {source.capitalize()}:")
                            logging.info(f"    * Exact: {exact_successes}/{exact_attempts} ({round(exact_rate, 1)}%)")
                            logging.info(f"    * Fuzzy: {fuzzy_successes}/{fuzzy_attempts} ({round(fuzzy_rate, 1)}%)")
    except Exception as e:
        logging.error(f"Error when logging match statistics: {str(e)}")
        # Continue execution even if there's an error with the statistics reporting

    return stats

def get_stats_summary(all_worker_stats, total_pubs=0):
    """Generate a clean, comprehensive summary of all statistics from worker processes.
    
    Args:
        all_worker_stats: List of statistics dictionaries from worker processes
        total_pubs: The total number of publications being processed (may not be available in worker stats)
    """
    # Start with a fresh stats dictionary
    calculated_total = sum(s.get('total', 0) for s in all_worker_stats)
    
    summary = {
        # Use the passed total_pubs if greater than 0, otherwise use the calculated total
        "total": total_pubs if total_pubs > 0 else calculated_total,
        "processed": sum(s.get('processed', 0) for s in all_worker_stats),
        "enriched": sum(s.get('enriched', 0) for s in all_worker_stats),
        "failed": sum(s.get('failed', 0) for s in all_worker_stats),
        "sources": {}
    }
    
    # Aggregate source counts
    source_types = ['elsevier', 'pubmed', 'crossref', 'semantic_scholar']
    for source in source_types:
        summary['sources'][source] = sum(s.get('sources', {}).get(source, 0) for s in all_worker_stats)
    
    # Aggregate match type counts (with defaults if missing)
    summary['match_types'] = {
        'exact_matches': sum(s.get('match_types', {}).get('exact_matches', 0) for s in all_worker_stats),
        'fuzzy_matches': sum(s.get('match_types', {}).get('fuzzy_matches', 0) for s in all_worker_stats)
    }
    
    # Initialize match attempt stats with a structure guaranteeing all expected fields
    summary['match_attempts'] = {
        'totals': {
            'exact_match_attempts': 0,
            'exact_match_successes': 0,
            'fuzzy_match_attempts': 0,
            'fuzzy_match_successes': 0
        },
        'rates': {
            'exact_success_rate': 0.0,
            'fuzzy_success_rate': 0.0
        },
        'by_source': {}
    }
    
    # Initialize source-specific stats
    for source in source_types:
        summary['match_attempts']['by_source'][source] = {
            'exact_match_attempts': 0,
            'exact_match_successes': 0,
            'fuzzy_match_attempts': 0,
            'fuzzy_match_successes': 0
        }
    
    # Aggregate match attempt statistics
    for stats in all_worker_stats:
        # Add match_attempts with defensive access using get() to avoid KeyError
        match_attempts = stats.get('match_attempts', {})
        
        # Aggregate by-source stats safely
        for source in source_types:
            source_stats = match_attempts.get('by_source', {}).get(source, {})
            for stat_key in ['exact_match_attempts', 'exact_match_successes', 'fuzzy_match_attempts', 'fuzzy_match_successes']:
                value = source_stats.get(stat_key, 0)
                summary['match_attempts']['by_source'][source][stat_key] += value
        
        # Aggregate totals safely
        totals_stats = match_attempts.get('totals', {})
        for key in ['exact_match_attempts', 'exact_match_successes', 'fuzzy_match_attempts', 'fuzzy_match_successes']:
            value = totals_stats.get(key, 0)
            summary['match_attempts']['totals'][key] += value
    
    # Calculate rates
    totals = summary['match_attempts']['totals']
    if totals['exact_match_attempts'] > 0:
        summary['match_attempts']['rates']['exact_success_rate'] = round(
            totals['exact_match_successes'] / totals['exact_match_attempts'] * 100, 1)
    if totals['fuzzy_match_attempts'] > 0:
        summary['match_attempts']['rates']['fuzzy_success_rate'] = round(
            totals['fuzzy_match_successes'] / totals['fuzzy_match_attempts'] * 100, 1)
    
    return summary

if __name__ == "__main__":
    try:
        # Add an exception handler for the entire main function
        try:
            main()
        except KeyError as ke:
            # Handle specific key errors that might occur during stats processing
            logging.error(f"Key error in statistics processing: {str(ke)}")
            logging.info("Continuing execution despite statistics error...")
            sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)
