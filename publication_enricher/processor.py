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
                 batch_size: int = 50,
                 max_concurrent: int = 10,
                 cache_db: str = "api_cache.db"):
        """
        Initialize the publication processor.
        
        Args:
            elsevier_api_key: API key for Elsevier
            batch_size: Number of publications to process in each batch
            max_concurrent: Maximum number of concurrent API requests
            cache_db: Path to SQLite cache database
        """
        self.api_client = APIClient(
            elsevier_api_key=elsevier_api_key,
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
        stats = {
            'total': total_pubs,
            'processed': start_idx,
            'enriched': 0,
            'failed': 0
        }
        
        async with tqdm(total=total_pubs, initial=start_idx) as pbar:
            for i in range(start_idx, total_pubs, self.batch_size):
                batch = df.iloc[i:i + self.batch_size].to_dict('records')
                
                # Process batch
                enriched_batch = await self.api_client.verify_publications(batch)
                
                # Update statistics
                for pub in enriched_batch:
                    if pub.get('abstract'):
                        stats['enriched'] += 1
                    else:
                        stats['failed'] += 1
                
                processed_data.extend(enriched_batch)
                stats['processed'] += len(enriched_batch)
                
                # Save checkpoint
                if checkpoint_path:
                    await self.save_checkpoint(processed_data, checkpoint_path)
                
                # Update progress bar
                pbar.update(len(enriched_batch))
                pbar.set_postfix(
                    enriched=stats['enriched'],
                    failed=stats['failed']
                )
        
        # Save final results
        result_df = pd.DataFrame(processed_data)
        result_df.to_csv(output_path, index=False)
        
        logger.info("\nProcessing complete!")
        logger.info(f"Total publications: {stats['total']}")
        logger.info(f"Successfully enriched: {stats['enriched']}")
        logger.info(f"Failed to enrich: {stats['failed']}")
        
        return stats 