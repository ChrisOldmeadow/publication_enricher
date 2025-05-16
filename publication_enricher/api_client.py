"""
Asynchronous API client for fetching publication metadata from multiple sources.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, Optional, List
import json
import aiosqlite
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, 
                 elsevier_api_key: str,
                 cache_db: str = "api_cache.db",
                 max_concurrent: int = 10,
                 cache_ttl_days: int = 30):
        """
        Initialize the API client.
        
        Args:
            elsevier_api_key: API key for Elsevier
            cache_db: Path to SQLite cache database
            max_concurrent: Maximum number of concurrent API requests
            cache_ttl_days: Number of days to keep cache entries
        """
        self.elsevier_api_key = elsevier_api_key
        self.cache_db = cache_db
        self.max_concurrent = max_concurrent
        self.cache_ttl_days = cache_ttl_days
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def setup(self):
        """Initialize the cache database."""
        async with aiosqlite.connect(self.cache_db) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    query_key TEXT PRIMARY KEY,
                    response JSON,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
    
    async def cleanup_cache(self):
        """Remove old cache entries."""
        cutoff = datetime.now() - timedelta(days=self.cache_ttl_days)
        async with aiosqlite.connect(self.cache_db) as db:
            await db.execute(
                "DELETE FROM api_cache WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            await db.commit()
    
    async def get_from_cache(self, query_key: str) -> Optional[Dict]:
        """Get cached response if available."""
        async with aiosqlite.connect(self.cache_db) as db:
            async with db.execute(
                "SELECT response FROM api_cache WHERE query_key = ?",
                (query_key,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return json.loads(row[0])
        return None
    
    async def save_to_cache(self, query_key: str, response: Dict):
        """Save API response to cache."""
        async with aiosqlite.connect(self.cache_db) as db:
            await db.execute(
                "INSERT OR REPLACE INTO api_cache (query_key, response) VALUES (?, ?)",
                (query_key, json.dumps(response))
            )
            await db.commit()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, session: aiohttp.ClientSession, url: str, params: Dict = None) -> Dict:
        """Make HTTP request with retry logic."""
        headers = {
            'X-ELS-APIKey': self.elsevier_api_key,
            'Accept': 'application/json'
        }
        async with self.semaphore:  # Limit concurrent requests
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return await response.json()
    
    async def get_abstract_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata by DOI."""
        cache_key = f"doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for DOI: {doi}")
            return cached
        
        try:
            url = f"https://api.elsevier.com/content/article/doi/{doi}"
            data = await self._make_request(session, url)
            
            if 'full-text-retrieval-response' in data:
                result = {
                    'title': data['full-text-retrieval-response'].get('coredata', {}).get('dc:title'),
                    'doi': doi,
                    'abstract': data['full-text-retrieval-response'].get('coredata', {}).get('dc:description')
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.error(f"Error fetching DOI {doi}: {str(e)}")
        
        return None
    
    async def get_abstract_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title."""
        cache_key = f"title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for title: {title}")
            return cached
        
        try:
            url = "https://api.elsevier.com/content/search/scopus"
            params = {
                'query': f'title("{title}")',
                'field': 'dc:title,dc:identifier,abstract'
            }
            
            data = await self._make_request(session, url, params)
            
            if 'search-results' in data and data['search-results'].get('entry'):
                entry = data['search-results']['entry'][0]
                result = {
                    'title': entry.get('dc:title'),
                    'doi': entry.get('prism:doi'),
                    'abstract': entry.get('dc:description')
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.error(f"Error searching title {title}: {str(e)}")
        
        return None
    
    async def verify_publications(self, publications: List[Dict]) -> List[Dict]:
        """
        Verify a batch of publications in parallel.
        
        Args:
            publications: List of dictionaries with 'title' and optional 'doi' keys
            
        Returns:
            List of enriched publication dictionaries
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for pub in publications:
                if pub.get('doi'):
                    task = self.get_abstract_by_doi(session, pub['doi'])
                else:
                    task = self.get_abstract_by_title(session, pub['title'])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Merge results with original publications
            enriched = []
            for pub, result in zip(publications, results):
                if result:
                    pub.update(result)
                enriched.append(pub)
            
            return enriched 