"""
Main module for processing publication CSV files.
"""
import pandas as pd
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import aiofiles
from tqdm.asyncio import tqdm

from .api_client import APIClient

logger = logging.getLogger(__name__)

class PublicationProcessor:
    def __init__(self, 
                 elsevier_api_key: str,
                 pubmed_email: str = None,
                 pubmed_api_key: str = None,
                 crossref_email: str = None,
                 semantic_scholar_api_key: str = None,
                 batch_size: int = 50,
                 max_concurrent: int = 10,
                 cache_db: str = "api_cache.db",
                 disable_pubmed: bool = False,
                 disable_semantic_scholar: bool = False,
                 status_callback = None):
        """
        Initialize the publication processor.
        
        Args:
            elsevier_api_key: API key for Elsevier
            pubmed_email: Email for PubMed (optional)
            pubmed_api_key: API key for PubMed (optional)
            crossref_email: Email for Crossref API (optional)
            semantic_scholar_api_key: API key for Semantic Scholar (optional)
            batch_size: Number of publications to process in each batch
            max_concurrent: Maximum number of concurrent API requests
            cache_db: Path to SQLite cache database
            disable_pubmed: Set to True to disable PubMed API lookups
            disable_semantic_scholar: Set to True to disable Semantic Scholar API lookups
        """
        self.api_client = APIClient(
            elsevier_api_key=elsevier_api_key,
            pubmed_email=pubmed_email,
            pubmed_api_key=pubmed_api_key,
            crossref_email=crossref_email,
            semantic_scholar_api_key=semantic_scholar_api_key,
            max_concurrent=max_concurrent,
            cache_db=cache_db,
            disable_pubmed=disable_pubmed,
            disable_semantic_scholar=disable_semantic_scholar
        )
        
        # Store callback for progress reporting
        self.status_callback = status_callback
        self.batch_size = batch_size
    
    async def setup(self):
        """Initialize the processor."""
        await self.api_client.setup()
        await self.api_client.cleanup_cache()
    
    async def save_checkpoint(self, 
                            processed_data: List[Dict],
                            checkpoint_file: str):
        """Save processing progress to a checkpoint file."""
        async with aiofiles.open(checkpoint_file, 'w') as f:
            await f.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'processed': processed_data
            }))
    
    async def load_checkpoint(self, checkpoint_file: str) -> Optional[List[Dict]]:
        """Load processing progress from a checkpoint file."""
        try:
            async with aiofiles.open(checkpoint_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return data['processed']
        except:
            return None
    
    async def process_csv(self,
                         input_path: str,
                         output_path: str,
                         checkpoint_path: Optional[str] = None,
                         disable_progress: bool = False) -> Dict:
        """
        Process CSV file to enrich publications with abstracts.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save enriched CSV file
            checkpoint_path: Optional path to save/load checkpoints
            
        Returns:
            Dictionary with processing statistics
        """
        # Read CSV file
        df = pd.read_csv(input_path)
        total_pubs = len(df)
        logger.info(f"Found {total_pubs} publications to process")
        
        # Check and map column names if necessary
        column_mapping = {}
        if 'Output_Title' in df.columns and 'title' not in df.columns:
            column_mapping['Output_Title'] = 'title'
        if 'Ref_DOI' in df.columns and 'doi' not in df.columns:
            column_mapping['Ref_DOI'] = 'doi'
            
        # Apply column mapping if needed
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Initialize or load from checkpoint
        processed_data = []
        start_idx = 0
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint_data = await self.load_checkpoint(checkpoint_path)
            if checkpoint_data:
                processed_data = checkpoint_data
                start_idx = len(processed_data)
                logger.info(f"Resuming from checkpoint with {start_idx} publications already processed")
        
        # Process in batches
        self.total_count = len(df)
        self.processed_count = 0
        self.enriched_count = 0
        self.failed_count = 0
        self.source_counts = {
            'elsevier': 0,
            'pubmed': 0,
            'crossref': 0,
            'semantic_scholar': 0
        }
        
        # Initialize stats dictionary
        stats = {
            'total': total_pubs,
            'processed': 0,
            'enriched': 0,
            'failed': 0,
            'sources': self.source_counts
        }
        
        # Create a progress bar without using async context manager (if not disabled)
        if not disable_progress:
            pbar = tqdm(total=total_pubs, initial=start_idx)
        
        for i in range(start_idx, total_pubs, self.batch_size):
            batch = df.iloc[i:i + self.batch_size].to_dict('records')
            
            # Process batch
            enriched_batch = await self.api_client.verify_publications(batch)
            
            # Update statistics - and report progress for EACH publication
            for idx, pub in enumerate(enriched_batch):
                if pub.get('abstract'):
                    self.enriched_count += 1
                    # Track the source that provided this enrichment
                    source = pub.get('source', 'unknown')
                    if source in self.source_counts:
                        self.source_counts[source] += 1
                else:
                    self.failed_count += 1
                    
                # Call status callback frequently - on every publication
                if self.status_callback and (idx % 1 == 0):  # Update on every publication
                    self.status_callback(self.enriched_count, self.failed_count)
                
                # Update progress bar frequently too
                if not disable_progress:
                    pbar.update(1)  # Update one at a time for smoother progress
                    if idx % 5 == 0:  # Only update the postfix text occasionally to avoid slowdown
                        pbar.set_postfix(
                            enriched=self.enriched_count,
                            failed=self.failed_count
                        )
            
            processed_data.extend(enriched_batch)
            stats['processed'] += len(enriched_batch)
            
            # Save checkpoint after each batch
            if checkpoint_path:
                await self.save_checkpoint(processed_data, checkpoint_path)
            
            # Update processed count
            self.processed_count += len(enriched_batch)
        
        # Close the progress bar if it exists
        if not disable_progress:
            pbar.close()
        
        # Save final results to output file
        result_df = pd.DataFrame(processed_data)
        
        # Extract match information into separate columns for easier analysis
        result_df['match_type'] = None
        result_df['fuzzy_matched'] = False
        result_df['match_score'] = None
        result_df['original_query_title'] = None
        
        # Extract match information from nested dictionaries
        for idx, row in result_df.iterrows():
            match_info = row.get('match_info')
            if isinstance(match_info, dict):
                # Get the match type and fuzzy matched flag
                match_type = match_info.get('match_type')
                fuzzy_matched = match_info.get('fuzzy_matched', False)
                
                # Store in the dataframe
                result_df.at[idx, 'match_type'] = match_type
                result_df.at[idx, 'fuzzy_matched'] = fuzzy_matched
                result_df.at[idx, 'match_score'] = match_info.get('match_score')
                result_df.at[idx, 'original_query_title'] = match_info.get('query_title')
                
                # Log some samples to help debug
                if idx < 5 and match_type:  # Just log a few samples
                    logger.debug(f"Sample match_info: Publication {idx}, match_type={match_type}, fuzzy={fuzzy_matched}")
            elif row.get('source') and row.get('abstract'):  # Fallback for old format records
                # Default to 'exact' if we have a source and abstract but no match_info
                result_df.at[idx, 'match_type'] = 'exact'
                result_df.at[idx, 'fuzzy_matched'] = False
        
        # Count match types for reporting with more detail
        # Track DOI matches, exact title matches, and fuzzy matches separately
        doi_matches = ((result_df['match_type'] == 'doi') & result_df['abstract'].notna()).sum()
        exact_title_matches = ((result_df['match_type'] == 'exact') & result_df['abstract'].notna()).sum()
        other_exact_matches = ((~result_df['fuzzy_matched']) & 
                             (result_df['match_type'] != 'doi') & 
                             (result_df['match_type'] != 'exact') & 
                             result_df['abstract'].notna()).sum()
        fuzzy_matches = result_df['fuzzy_matched'].sum()
        
        # Total exact matches is the sum of DOI, exact title, and other exact matches
        exact_matches = doi_matches + exact_title_matches + other_exact_matches
        
        match_stats = {
            'exact_matches': exact_matches,
            'doi_matches': doi_matches,
            'exact_title_matches': exact_title_matches,
            'other_exact_matches': other_exact_matches,
            'fuzzy_matches': fuzzy_matches
        }
        
        # Save with match information columns
        result_df.to_csv(output_path, index=False)
        
        # Prepare final stats with source breakdown
        stats = {
            'total': self.total_count,
            'processed': self.processed_count,
            'enriched': self.enriched_count,
            'failed': self.failed_count,
            'sources': self.source_counts
        }
        
        # Add match stats to the overall stats
        stats['match_types'] = match_stats
        
        # Get matching attempt statistics from API client
        match_attempt_stats = self.api_client.get_match_statistics()
        stats['match_attempts'] = match_attempt_stats
        
        logger.info("\nProcessing complete!")
        logger.info(f"Total publications: {stats['total']}")
        logger.info(f"Successfully enriched: {stats['enriched']}")
        logger.info(f"Failed to enrich: {stats['failed']}")
        logger.info(f"Source breakdown: {stats['sources']}")
        
        # Matching statistics section
        logger.info(f"\nMatch statistics:")
        logger.info(f"  - Exact matches: {match_stats['exact_matches']}")
        logger.info(f"  - Fuzzy matches: {match_stats['fuzzy_matches']}")
        
        # Detailed match attempt statistics
        logger.info(f"\nMatching attempt statistics:")
        logger.info(f"  - Exact match attempts: {match_attempt_stats['totals']['exact_match_attempts']}")
        logger.info(f"  - Exact match successes: {match_attempt_stats['totals']['exact_match_successes']}")
        logger.info(f"  - Exact match success rate: {match_attempt_stats['rates']['exact_success_rate']}%")
        logger.info(f"  - Fuzzy match attempts: {match_attempt_stats['totals']['fuzzy_match_attempts']}")
        logger.info(f"  - Fuzzy match successes: {match_attempt_stats['totals']['fuzzy_match_successes']}")
        logger.info(f"  - Fuzzy match success rate: {match_attempt_stats['rates']['fuzzy_success_rate']}%")
        
        # Per-source statistics
        logger.info(f"\nPer-source match statistics:")
        for source, stats in match_attempt_stats['by_source'].items():
            if stats['exact_match_attempts'] > 0 or stats['fuzzy_match_attempts'] > 0:
                exact_rate = (stats['exact_match_successes'] / stats['exact_match_attempts'] * 100) if stats['exact_match_attempts'] > 0 else 0
                fuzzy_rate = (stats['fuzzy_match_successes'] / stats['fuzzy_match_attempts'] * 100) if stats['fuzzy_match_attempts'] > 0 else 0
                logger.info(f"  - {source.capitalize()}:")
                logger.info(f"    * Exact: {stats['exact_match_successes']}/{stats['exact_match_attempts']} ({round(exact_rate, 1)}%)")
                logger.info(f"    * Fuzzy: {stats['fuzzy_match_successes']}/{stats['fuzzy_match_attempts']} ({round(fuzzy_rate, 1)}%)")
        
        # If there were fuzzy matches, give information about score distribution
        if match_stats['fuzzy_matches'] > 0:
            # Calculate basic stats about fuzzy match scores
            fuzzy_scores = result_df[result_df['fuzzy_matched']]['match_score']
            if len(fuzzy_scores) > 0:
                logger.info(f"\nFuzzy match score details:")
                logger.info(f"  - Score range: {fuzzy_scores.min()}-{fuzzy_scores.max()}")
                logger.info(f"  - Average score: {fuzzy_scores.mean():.1f}")
                # Log number of matches in different score ranges
                score_ranges = {
                    '90-94': ((fuzzy_scores >= 90) & (fuzzy_scores < 95)).sum(),
                    '95-99': ((fuzzy_scores >= 95) & (fuzzy_scores < 100)).sum(),
                    '100': (fuzzy_scores == 100).sum()
                }
                logger.info(f"  - Score distribution: {score_ranges}")
        
        
        return stats