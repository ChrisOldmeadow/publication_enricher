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
                 cache_db: str = "api_cache.db"):
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
        """
        self.api_client = APIClient(
            elsevier_api_key=elsevier_api_key,
            pubmed_email=pubmed_email,
            pubmed_api_key=pubmed_api_key,
            crossref_email=crossref_email,
            semantic_scholar_api_key=semantic_scholar_api_key,
            max_concurrent=max_concurrent,
            cache_db=cache_db
        )
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
                         checkpoint_path: Optional[str] = None) -> Dict:
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
        
        # Create a progress bar without using async context manager
        pbar = tqdm(total=total_pubs, initial=start_idx)
        for i in range(start_idx, total_pubs, self.batch_size):
            batch = df.iloc[i:i + self.batch_size].to_dict('records')
            
            # Process batch
            enriched_batch = await self.api_client.verify_publications(batch)
            
            # Update statistics
            for pub in enriched_batch:
                if pub.get('abstract'):
                    self.enriched_count += 1
                    # Track the source that provided this enrichment
                    source = pub.get('source', 'unknown')
                    if source in self.source_counts:
                        self.source_counts[source] += 1
                else:
                    self.failed_count += 1
            
            processed_data.extend(enriched_batch)
            stats['processed'] += len(enriched_batch)
            
            # Save checkpoint
            if checkpoint_path:
                await self.save_checkpoint(processed_data, checkpoint_path)
            
            # Update progress bar
            pbar.update(len(enriched_batch))
            
            # Update processed count
            self.processed_count += len(enriched_batch)
            pbar.set_postfix(
                enriched=self.enriched_count,
                failed=self.failed_count
            )
        
        # Close the progress bar
        pbar.close()
        
        # Save final results to output file
        result_df = pd.DataFrame(processed_data)
        result_df.to_csv(output_path, index=False)
        
        # Prepare final stats with source breakdown
        stats = {
            'total': self.total_count,
            'processed': self.processed_count,
            'enriched': self.enriched_count,
            'failed': self.failed_count,
            'sources': self.source_counts
        }
        
        logger.info("\nProcessing complete!")
        logger.info(f"Total publications: {stats['total']}")
        logger.info(f"Successfully enriched: {stats['enriched']}")
        logger.info(f"Failed to enrich: {stats['failed']}")
        logger.info(f"Source breakdown: {stats['sources']}")
        
        return stats