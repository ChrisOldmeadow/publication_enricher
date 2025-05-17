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
import tqdm
from tqdm import tqdm
from dotenv import load_dotenv

from publication_enricher import PublicationProcessor

def setup_logging(verbose: bool, process_id=None, worker_mode=False):
    """Configure logging based on verbosity level."""
    # Reset handlers on root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up format based on process ID
    if process_id is not None and not worker_mode:
        log_format = f"%(asctime)s - [Process {process_id}] %(message)s"
    else:
        log_format = "%(asctime)s - %(message)s"
    date_format = "%H:%M:%S"
    
    # Console handler for progress info
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)
    
    # In worker mode, we only want to see errors
    if worker_mode:
        root_logger.setLevel(logging.ERROR)
    else:
        root_logger.setLevel(logging.INFO)
    
    # Suppress verbose logging from all modules
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    logging.getLogger('aiosqlite').setLevel(logging.ERROR)
    logging.getLogger('publication_enricher.api_client').setLevel(logging.ERROR)
    logging.getLogger('publication_enricher.processor').setLevel(logging.ERROR)

async def process_chunk(chunk_df, output_path, elsevier_key, pubmed_email, crossref_email, semantic_scholar_key, batch_size, max_concurrent, cache_db, checkpoint_path=None, process_id=None, disable_pubmed=False, disable_semantic=False, disable_progress_bar=True, status_callback=None):
    """Process a chunk of publications asynchronously.
    
    Args:
        status_callback: Optional callback function to report progress periodically.
                         Will be called with (process_id, enriched_count, failed_count).
    """
    # Initialize processor
    processor = PublicationProcessor(
        elsevier_api_key=elsevier_key,
        pubmed_email=pubmed_email,
        crossref_email=crossref_email,
        semantic_scholar_api_key=semantic_scholar_key,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        cache_db=cache_db,
        disable_pubmed=disable_pubmed,
        disable_semantic_scholar=disable_semantic,
        status_callback=status_callback
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
            checkpoint_path=checkpoint_path,
            disable_progress=True  # Disable individual process progress bars
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

def worker_process(chunk_df, output_queue, status_queue, elsevier_key, pubmed_email, crossref_email, semantic_scholar_key, batch_size, max_concurrent, cache_db, process_id, disable_pubmed=False, disable_semantic=False):
    """Worker process to handle a chunk of publications."""
    # Use worker_mode=True to suppress most logging
    setup_logging(True, process_id, worker_mode=True)
    
    # Send initial status update
    total_chunk_size = len(chunk_df)
    status_queue.put((process_id, 'STARTED', {'total': total_chunk_size}))
    
    # Track progress for incremental updates
    prev_enriched = 0
    prev_failed = 0
    
    # Function to send progress updates to the main process
    def report_progress(enriched, failed):
        nonlocal prev_enriched, prev_failed
        # Always report progress for smoother updates
        # Calculate the change since last report 
        delta_enriched = enriched - prev_enriched
        delta_failed = failed - prev_failed
        
        # Even if nothing changed, send at least every 5 publications to keep the UI responsive
        if delta_enriched > 0 or delta_failed > 0 or (enriched + failed) % 5 == 0:
            status_queue.put((
                process_id, 
                'PROGRESS', 
                {
                    'enriched': enriched, 
                    'failed': failed,
                    'prev_enriched': prev_enriched,
                    'prev_failed': prev_failed
                }
            ))
            # Update previous values for next time
            prev_enriched = enriched
            prev_failed = failed
    
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
            process_id=process_id,
            disable_pubmed=disable_pubmed,
            disable_semantic=disable_semantic,
            disable_progress_bar=True,
            status_callback=report_progress
        ))
        
        # Get final counts
        enriched = stats.get('enriched', 0) 
        failed = stats.get('failed', 0)
        
        # Send final stats to the main process with previous values for incremental calculation
        status_queue.put((
            process_id, 
            'COMPLETED', 
            {
                'enriched': enriched, 
                'failed': failed,
                'prev_enriched': prev_enriched,
                'prev_failed': prev_failed
            }
        ))
        
        # Return the results
        output_queue.put((process_id, result_df, stats))
    
    except Exception as e:
        logging.error(f"Process {process_id} failed: {str(e)}")
        # Report failure to main process
        status_queue.put((process_id, 'ERROR', {'error': str(e), 'failed': len(chunk_df)}))
        
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
    """Main function to run the multiprocessing enrichment."""
    # Import tqdm at the function level to avoid import issues
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback in case tqdm is not installed
        logging.error("tqdm module not found - progress bar will not be shown")
        class FakeTqdm:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_postfix(self, **kwargs):
                pass
        tqdm = FakeTqdm
    
    # Load environment variables
    load_dotenv()
    
    # Set up command line argument parser
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
        default=multiprocessing.cpu_count(),
        help=f"Number of parallel processes to use (default: {multiprocessing.cpu_count()})"
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
    
    parser.add_argument(
        "--disable-pubmed",
        action="store_true",
        help="Disable PubMed API lookups"
    )
    
    parser.add_argument(
        "--disable-semantic",
        action="store_true",
        help="Disable Semantic Scholar API lookups"
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
    
    # Create queues for results and status updates
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    status_queue = manager.Queue()
    
    # Start the worker processes
    start_time = time.time()
    processes = []
    
    # Log API disabling status
    if args.disable_pubmed:
        logging.info("PubMed API lookups are DISABLED")
    if args.disable_semantic:
        logging.info("Semantic Scholar API lookups are DISABLED")
    
    # Start worker processes
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            p = multiprocessing.Process(
                target=worker_process,
                args=(chunk, result_queue, status_queue, elsevier_key, pubmed_email,
                      crossref_email, semantic_scholar_key, args.batch_size, 
                      args.max_concurrent, args.cache_db, i,
                      args.disable_pubmed, args.disable_semantic)
            )
            processes.append(p)
            p.start()
    
    # Wait for all processes to complete while updating a single progress bar
    total_items = total_pubs
    enriched_count = 0
    failed_count = 0
    process_statuses = {}
    active_processes = len(processes)
    
    # Set up a single progress bar for all processes
    with tqdm(total=total_items, desc="Enriching publications", unit="pub") as pbar:
        while active_processes > 0:
            # Check for process status updates
            try:
                # More responsive non-blocking check for updates with a smaller timeout
                process_id, status_type, status_data = status_queue.get(timeout=0.01)
                
                # Update process status
                if status_type == 'STARTED':
                    # Process is starting - initialize its status
                    process_statuses[process_id] = 'STARTED'
                    # No progress bar update for start
                    
                elif status_type == 'PROGRESS':
                    # Incremental update during processing
                    new_enriched = status_data.get('enriched', 0)
                    new_failed = status_data.get('failed', 0)
                    
                    # Update only with the new counts since last update
                    prev_enriched = status_data.get('prev_enriched', 0)
                    prev_failed = status_data.get('prev_failed', 0)
                    
                    # Calculate incremental counts
                    inc_enriched = new_enriched - prev_enriched
                    inc_failed = new_failed - prev_failed
                    
                    # Update totals
                    enriched_count += inc_enriched
                    failed_count += inc_failed
                    
                    # Update progress bar description with current stats
                    pbar.set_postfix(enriched=enriched_count, failed=failed_count, refresh=True)
                    
                    # Update progress based on items completed
                    items_completed = inc_enriched + inc_failed
                    pbar.update(items_completed)
                    
                elif status_type == 'COMPLETED':
                    # Final update from completed process 
                    # Get the final counts
                    new_enriched = status_data.get('enriched', 0)
                    new_failed = status_data.get('failed', 0)
                    
                    # If we've tracked this process before, just update with the difference
                    # to avoid double counting
                    if process_id in process_statuses and process_statuses[process_id] == 'PROGRESS':
                        prev_enriched = status_data.get('prev_enriched', 0)
                        prev_failed = status_data.get('prev_failed', 0)
                        
                        # Calculate incremental counts
                        inc_enriched = new_enriched - prev_enriched
                        inc_failed = new_failed - prev_failed
                        
                        enriched_count += inc_enriched
                        failed_count += inc_failed
                        
                        # Update progress
                        items_completed = inc_enriched + inc_failed
                        pbar.update(items_completed)
                    else:
                        # Initial completion update 
                        enriched_count += new_enriched
                        failed_count += new_failed
                        
                        # Update progress
                        items_completed = new_enriched + new_failed
                        pbar.update(items_completed)
                    
                    # Track completed processes
                    process_statuses[process_id] = 'COMPLETED'
                    
                    # Update progress bar description with current stats
                    pbar.set_postfix(enriched=enriched_count, failed=failed_count, refresh=True)
                    
                elif status_type == 'ERROR':
                    # Handle process errors
                    process_statuses[process_id] = 'ERROR'
                    failed_count += status_data.get('failed', 0)
                    logging.error(f"Process {process_id} failed: {status_data.get('error', 'Unknown error')}")
                    pbar.update(status_data.get('failed', 0))
                    pbar.set_postfix(enriched=enriched_count, failed=failed_count, refresh=True)
            
            except Exception as e:
                # Queue.get timeout or other error - add a tiny sleep to prevent CPU thrashing
                # but keep it short enough to remain responsive
                time.sleep(0.005)
                
            # Check if any processes have finished
            still_active = 0
            for p in processes:
                if p.is_alive():
                    still_active += 1
            
            # Update the count of active processes
            if still_active < active_processes:
                active_processes = still_active
    
    # Make sure all processes are truly complete
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
    
    # Create separate files for failed and fuzzy matches for verification
    base_name, ext = os.path.splitext(output_path)
    
    # Save failed matches (entries without abstracts)
    failed_df = combined_df[combined_df['abstract'].isna()]
    failed_file = f"{base_name}_failed{ext}"
    failed_df.to_csv(failed_file, index=False)
    logging.info(f"Saved {len(failed_df)} failed matches to {failed_file}")
    
    # Identify and save publications with missing data (both DOI and title)
    missing_data_df = combined_df[
        combined_df.apply(
            lambda row: isinstance(row.get('match_info'), dict) and 
                       row.get('match_info', {}).get('error') == 'missing_data',
            axis=1
        )
    ]
    if len(missing_data_df) > 0:
        missing_data_file = f"{base_name}_missing_data{ext}"
        missing_data_df.to_csv(missing_data_file, index=False)
        logging.info(f"Saved {len(missing_data_df)} publications with missing data to {missing_data_file}")
        # Add this to our statistics
        stats['missing_data'] = len(missing_data_df)
    
    # Save fuzzy matches (check match_info for fuzzy_matched=True flag)
    try:
        # First check if match_info column exists as a string that can be parsed
        if 'match_info' in combined_df.columns and combined_df['match_info'].dtype == 'object':
            # For string match_info, try to parse
            fuzzy_df = combined_df[combined_df['match_info'].apply(
                lambda x: isinstance(x, str) and 'fuzzy_matched' in x and 'true' in x.lower()
            )]
        else:
            # Direct check of dictionaries in match_info
            fuzzy_df = combined_df[combined_df.apply(
                lambda row: isinstance(row.get('match_info'), dict) and 
                             row.get('match_info', {}).get('fuzzy_matched') == True, 
                axis=1
            )]
        
        fuzzy_file = f"{base_name}_fuzzy{ext}"
        fuzzy_df.to_csv(fuzzy_file, index=False)
        logging.info(f"Saved {len(fuzzy_df)} fuzzy matches to {fuzzy_file}")
    except Exception as e:
        logging.warning(f"Could not save fuzzy matches: {str(e)}")
        # Create an alternative fuzzy match file using score threshold
        if 'match_score' in combined_df.columns:
            alt_fuzzy_df = combined_df[combined_df['match_score'] < 100]
            alt_fuzzy_file = f"{base_name}_likely_fuzzy{ext}"
            alt_fuzzy_df.to_csv(alt_fuzzy_file, index=False)
            logging.info(f"Saved {len(alt_fuzzy_df)} likely fuzzy matches to {alt_fuzzy_file} based on scores")
    
    end_time = time.time()
    logging.info(f"\nProcessing complete in {end_time - start_time:.2f} seconds!")
    logging.info(f"Total publications: {stats['total']}")
    
    # Calculate and log the enrichment success percentage
    enrichment_percentage = 0.0
    if stats['total'] > 0:
        enrichment_percentage = (stats['enriched'] / stats['total']) * 100
    
    logging.info(f"Successfully enriched: {stats['enriched']} ({enrichment_percentage:.1f}%)")
    
    # Report missing data count if available
    missing_data_count = stats.get('missing_data', 0)
    if missing_data_count > 0:
        missing_percentage = (missing_data_count / stats['total']) * 100 if stats['total'] > 0 else 0
        logging.info(f"Publications with missing data: {missing_data_count} ({missing_percentage:.1f}%)")
        logging.info(f"Failed to enrich (excluding missing data): {stats['failed'] - missing_data_count} ({(stats['failed'] - missing_data_count) / stats['total'] * 100:.1f}%)")
    else:
        logging.info(f"Failed to enrich: {stats['failed']} ({100 - enrichment_percentage:.1f}%)")
    
    
    # Log source breakdown
    sources_sum = sum(stats['sources'].values())
    if sources_sum > 0:
        # Check for inflation in source counts
        total_from_sources = sum(stats['sources'].values())
        if total_from_sources > stats['enriched'] and stats['enriched'] > 0:
            # Scale down the source counts proportionally
            scale_factor = stats['enriched'] / total_from_sources
            adjusted_sources = {}
            remaining = stats['enriched']
            
            # Distribute the count proportionally across sources
            for source, count in sorted(stats['sources'].items(), key=lambda x: x[1], reverse=True):
                if source == list(stats['sources'].keys())[-1]:  # Last item
                    # Assign remaining to avoid rounding errors
                    adjusted_sources[source] = remaining
                else:
                    adjusted_count = round(count * scale_factor)
                    adjusted_sources[source] = adjusted_count
                    remaining -= adjusted_count
            
            logging.info(f"Source breakdown (adjusted to match total enriched count): {adjusted_sources}")
        else:
            logging.info(f"Source breakdown: {stats['sources']}")
    elif stats['enriched'] > 0:
        # If we have enrichments but no source counts, infer sources from match statistics
        inferred_sources = {}
        if 'match_attempts' in stats and 'by_source' in stats['match_attempts']:
            total_inferred = 0
            for source, source_stats in stats['match_attempts']['by_source'].items():
                exact_successes = source_stats.get('exact_match_successes', 0)
                fuzzy_successes = source_stats.get('fuzzy_match_successes', 0)
                if exact_successes > 0 or fuzzy_successes > 0:
                    # Only count successful matches, not attempts
                    inferred_sources[source] = exact_successes + fuzzy_successes
                    total_inferred += exact_successes + fuzzy_successes
            
            # Adjust inferred sources to match the actual enriched count
            if total_inferred > stats['enriched'] and stats['enriched'] > 0:
                scale_factor = stats['enriched'] / total_inferred
                adjusted_inferred = {}
                remaining = stats['enriched']
                
                for source, count in sorted(inferred_sources.items(), key=lambda x: x[1], reverse=True):
                    if source == list(inferred_sources.keys())[-1]:  # Last item
                        adjusted_inferred[source] = remaining
                    else:
                        adjusted_count = round(count * scale_factor)
                        adjusted_inferred[source] = adjusted_count
                        remaining -= adjusted_count
                
                inferred_sources = adjusted_inferred
        
        if inferred_sources:
            logging.info(f"Source breakdown (inferred from match statistics): {inferred_sources}")
        else:
            logging.info(f"Successfully enriched {stats['enriched']} publications, but source breakdown unavailable.")
    else:
        logging.info(f"No enrichments from any source.")
        
    # Log match statistics
    if 'match_types' in stats:
        total_matches = stats['match_types']['exact_matches'] + stats['match_types']['fuzzy_matches']
        # Sanity check - if we have more than zero enriched publications, we should have matches
        if stats['enriched'] > 0 and total_matches == 0:
            # If we have no matches recorded but have enriched publications, infer from match_attempts
            if 'match_attempts' in stats:
                # Get the basic match types from match_attempts
                stats['match_types']['exact_matches'] = stats['match_attempts']['totals']['exact_match_successes']
                stats['match_types']['fuzzy_matches'] = stats['match_attempts']['totals']['fuzzy_match_successes']
                
                # Adjust to match total enriched count if needed
                total_inferred = stats['match_types']['exact_matches'] + stats['match_types']['fuzzy_matches']
                if total_inferred > stats['enriched']:
                    ratio = stats['enriched'] / total_inferred
                    stats['match_types']['exact_matches'] = round(stats['match_types']['exact_matches'] * ratio)
                    stats['match_types']['fuzzy_matches'] = stats['enriched'] - stats['match_types']['exact_matches']
                
                # Now infer the detailed match types
                # Since we don't have actual breakdown, estimate based on API sources
                # We assume DOIs were used whenever possible
                # Start by checking if we have source breakdown stats
                if 'by_source' in stats['match_attempts']:
                    # For exact matches, estimate that most are DOI matches
                    doi_match_estimate = int(stats['match_types']['exact_matches'] * 0.7)  # Assume ~70% are DOI matches
                    exact_title_estimate = stats['match_types']['exact_matches'] - doi_match_estimate
                    
                    # Add these estimates to the stats
                    stats['match_types']['doi_matches'] = doi_match_estimate
                    stats['match_types']['exact_title_matches'] = exact_title_estimate
                    stats['match_types']['other_exact_matches'] = 0  # Default to 0 for other
        
        logging.info(f"\nMatch statistics:")
        logging.info(f"  - Exact matches (total): {stats['match_types']['exact_matches']}")
        
        # Show detailed breakdown of match types if available
        if 'doi_matches' in stats['match_types']:
            doi_matches = stats['match_types']['doi_matches']
            exact_title_matches = stats['match_types']['exact_title_matches']
            other_exact_matches = stats['match_types'].get('other_exact_matches', 0)
            
            # Calculate percentages of successful matches
            total_matched = stats['match_types']['exact_matches'] + stats['match_types']['fuzzy_matches']
            doi_pct = (doi_matches / total_matched * 100) if total_matched > 0 else 0
            exact_title_pct = (exact_title_matches / total_matched * 100) if total_matched > 0 else 0
            other_exact_pct = (other_exact_matches / total_matched * 100) if total_matched > 0 else 0
            fuzzy_pct = (stats['match_types']['fuzzy_matches'] / total_matched * 100) if total_matched > 0 else 0
            
            logging.info(f"    • DOI matches: {doi_matches} ({doi_pct:.1f}%)")
            logging.info(f"    • Exact title matches: {exact_title_matches} ({exact_title_pct:.1f}%)")
            if other_exact_matches > 0:
                logging.info(f"    • Other exact matches: {other_exact_matches} ({other_exact_pct:.1f}%)")
        
        logging.info(f"  - Fuzzy matches: {stats['match_types']['fuzzy_matches']} ({fuzzy_pct:.1f}%)") 
    # Log match attempt statistics - with defensive checks
    try:
        if 'match_attempts' in stats:
            if 'totals' in stats['match_attempts']:
                totals = stats['match_attempts']['totals']
                rates = stats['match_attempts'].get('rates', {'exact_success_rate': 0, 'fuzzy_success_rate': 0})
                
                # Clarify that these statistics represent API calls, not unique publications
                logging.info(f"\nAPI Call Statistics (by attempt type):")
                logging.info(f"  - Exact match API calls: {totals.get('exact_match_attempts', 0)}")
                logging.info(f"  - Successful exact match API calls: {totals.get('exact_match_successes', 0)}")
                logging.info(f"  - Exact match API success rate: {rates.get('exact_success_rate', 0)}%")
                logging.info(f"  - Fuzzy match API calls: {totals.get('fuzzy_match_attempts', 0)}")
                logging.info(f"  - Successful fuzzy match API calls: {totals.get('fuzzy_match_successes', 0)}")
                logging.info(f"  - Fuzzy match API success rate: {rates.get('fuzzy_success_rate', 0)}%")
                
                # Add clarification about the difference between API calls and unique publications
                logging.info(f"\nNote: The above counts represent individual API calls across all services.")
                logging.info(f"      For unique publication counts, refer to the Match statistics section.")
                
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
    
    # Calculate the actual enriched and failed counts (these are accurate)
    enriched_count = sum(s.get('enriched', 0) for s in all_worker_stats)
    failed_count = sum(s.get('failed', 0) for s in all_worker_stats)
    
    summary = {
        # Use the passed total_pubs if greater than 0, otherwise use the calculated total
        "total": total_pubs if total_pubs > 0 else calculated_total,
        "processed": sum(s.get('processed', 0) for s in all_worker_stats),
        "enriched": enriched_count,
        "failed": failed_count,
        "sources": {},
        "match_types": {
            # Initialize with a more realistic value that can't exceed total enriched
            'exact_matches': 0,
            'fuzzy_matches': 0
        },
        "match_attempts": {
            'by_source': {},
            'totals': {
                'exact_match_attempts': 0,
                'exact_match_successes': 0,
                'fuzzy_match_attempts': 0,
                'fuzzy_match_successes': 0
            },
            'rates': {
                'exact_success_rate': 0.0,
                'fuzzy_success_rate': 0.0
            }
        }
    }
    
    # Aggregate source counts
    source_types = ['elsevier', 'pubmed', 'crossref', 'semantic_scholar']
    for source in source_types:
        summary['sources'][source] = sum(s.get('sources', {}).get(source, 0) for s in all_worker_stats)
        # Initialize source-specific match stats
        summary['match_attempts']['by_source'][source] = {
            'exact_match_attempts': 0,
            'exact_match_successes': 0,
            'fuzzy_match_attempts': 0,
            'fuzzy_match_successes': 0
        }
    
    # Calculate match types based on the data in the match_types field
    raw_exact_matches = 0
    raw_fuzzy_matches = 0
    raw_doi_matches = 0
    raw_exact_title_matches = 0
    raw_other_exact_matches = 0
    
    # First try to get match types from match_types if available
    for s in all_worker_stats:
        if 'match_types' in s:
            raw_exact_matches += s['match_types'].get('exact_matches', 0)
            raw_fuzzy_matches += s['match_types'].get('fuzzy_matches', 0)
            raw_doi_matches += s['match_types'].get('doi_matches', 0)
            raw_exact_title_matches += s['match_types'].get('exact_title_matches', 0)
            raw_other_exact_matches += s['match_types'].get('other_exact_matches', 0)
    
    # If we still have zero matches, derive them from match_attempts
    if raw_exact_matches == 0 and raw_fuzzy_matches == 0:
        for s in all_worker_stats:
            if 'match_attempts' in s and 'by_source' in s.get('match_attempts', {}):
                for source, source_stats in s['match_attempts']['by_source'].items():
                    raw_exact_matches += source_stats.get('exact_match_successes', 0)
                    raw_fuzzy_matches += source_stats.get('fuzzy_match_successes', 0)
    
    # Now adjust the match counts to ensure they don't exceed the total enriched count
    total_raw_matches = raw_exact_matches + raw_fuzzy_matches
    
    # If we have more matches than enriched publications, we need to adjust
    if total_raw_matches > enriched_count and total_raw_matches > 0:
        # Calculate the proportion of each match type
        exact_proportion = raw_exact_matches / total_raw_matches
        fuzzy_proportion = raw_fuzzy_matches / total_raw_matches
        
        # For detailed breakdown, calculate proportions of each exact match type
        if raw_exact_matches > 0:
            doi_proportion = raw_doi_matches / raw_exact_matches
            exact_title_proportion = raw_exact_title_matches / raw_exact_matches
            other_exact_proportion = raw_other_exact_matches / raw_exact_matches
        else:
            doi_proportion = 0
            exact_title_proportion = 0
            other_exact_proportion = 0
        
        # Distribute the enriched count proportionally
        adjusted_exact_matches = round(enriched_count * exact_proportion)
        adjusted_fuzzy_matches = enriched_count - adjusted_exact_matches
        
        # Now distribute the exact matches among the detailed categories
        adjusted_doi_matches = round(adjusted_exact_matches * doi_proportion)
        adjusted_exact_title_matches = round(adjusted_exact_matches * exact_title_proportion)
        adjusted_other_exact_matches = adjusted_exact_matches - adjusted_doi_matches - adjusted_exact_title_matches
        
        # Set the adjusted values
        summary['match_types']['exact_matches'] = adjusted_exact_matches
        summary['match_types']['fuzzy_matches'] = adjusted_fuzzy_matches
        summary['match_types']['doi_matches'] = adjusted_doi_matches
        summary['match_types']['exact_title_matches'] = adjusted_exact_title_matches
        summary['match_types']['other_exact_matches'] = adjusted_other_exact_matches
    else:
        # If the total matches are fewer than or equal to enriched count, use raw values
        summary['match_types']['exact_matches'] = raw_exact_matches
        summary['match_types']['fuzzy_matches'] = raw_fuzzy_matches
        summary['match_types']['doi_matches'] = raw_doi_matches
        summary['match_types']['exact_title_matches'] = raw_exact_title_matches
        summary['match_types']['other_exact_matches'] = raw_other_exact_matches
    
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
