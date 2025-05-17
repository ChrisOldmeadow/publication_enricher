"""
Asynchronous API client for fetching publication metadata from multiple sources.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, Optional, List, Union, Any, Tuple
import json
import aiosqlite
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import urllib.parse
import re
import random
from fuzzywuzzy import fuzz, process

logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalize title for better matching."""
    if not title:
        return ""
    
    # Convert to lowercase
    title = title.lower()
    
    # Replace common punctuation and special characters
    title = re.sub(r'[\-_:;,.!?\[\](){}"\'\\]', ' ', title)
    
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Remove common words that don't contribute to matching
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for', 'with']
    title_words = title.split()
    title_words = [word for word in title_words if word not in stopwords]
    
    return ' '.join(title_words)

def fuzzy_title_match(title1: str, title2: str, threshold: int = 90) -> bool:
    """Check if two titles match using fuzzy matching.
    
    Args:
        title1: First title
        title2: Second title
        threshold: Minimum score to consider a match (0-100)
        
    Returns:
        bool: True if titles match above threshold
    """
    # Normalize titles first
    norm_title1 = normalize_title(title1)
    norm_title2 = normalize_title(title2)
    
    # Get both token sort and token set ratios (handles word order and subset matches)
    token_sort_ratio = fuzz.token_sort_ratio(norm_title1, norm_title2)
    token_set_ratio = fuzz.token_set_ratio(norm_title1, norm_title2)
    
    # Use the higher of the two scores
    score = max(token_sort_ratio, token_set_ratio)
    
    logger.debug(f"Fuzzy match score: {score} for '{title1}' and '{title2}'")
    return score >= threshold

def find_best_title_match(search_title: str, candidates: List[Dict], threshold: int = 90) -> Optional[Dict]:
    """Find best matching title in a list of candidates.
    
    Args:
        search_title: Title to search for
        candidates: List of dictionaries containing candidate titles
        threshold: Minimum score to consider a match (0-100)
        
    Returns:
        Optional[Dict]: Best matching candidate or None if no match found
    """
    if not candidates:
        return None
    
    # Normalize search title
    norm_search_title = normalize_title(search_title)
    
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        if 'title' in candidate and candidate['title']:
            # Get both token sort and token set ratios
            token_sort_ratio = fuzz.token_sort_ratio(norm_search_title, normalize_title(candidate['title']))
            token_set_ratio = fuzz.token_set_ratio(norm_search_title, normalize_title(candidate['title']))
            
            # Use the higher of the two scores
            score = max(token_sort_ratio, token_set_ratio)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
    
    if best_match:
        logger.debug(f"Found title match with score {best_score}: '{search_title}' -> '{best_match.get('title')}'")  
    
    return best_match

class APIClient:
    def __init__(self, 
                 elsevier_api_key: str,
                 pubmed_email: str = None,
                 pubmed_api_key: str = None,
                 crossref_email: str = None,
                 semantic_scholar_api_key: str = None,
                 cache_db: str = "api_cache.db",
                 max_concurrent: int = 10,
                 cache_ttl_days: int = 30):
        """
        Initialize the API client.
        
        Args:
            elsevier_api_key: API key for Elsevier
            pubmed_email: Email address for NCBI/PubMed (optional)
            crossref_email: Email for Crossref API (optional)
            semantic_scholar_api_key: API key for Semantic Scholar (optional)
            cache_db: Path to SQLite cache database
            max_concurrent: Maximum number of concurrent API requests
            cache_ttl_days: Number of days to keep cache entries
        """
        # API keys and credentials
        self.elsevier_api_key = elsevier_api_key
        self.pubmed_email = pubmed_email
        self.pubmed_api_key = pubmed_api_key
        self.crossref_email = crossref_email
        self.semantic_scholar_api_key = semantic_scholar_api_key
        
        # API availability flags
        self.has_pubmed = (self.pubmed_email is not None) or (self.pubmed_api_key is not None)
        self.has_crossref = True  # Crossref has a public API that works without auth
        self.has_semantic_scholar = True  # Semantic Scholar has a public API with rate limits
        
        # Error tracking
        self.pubmed_error_count = 0
        self.pubmed_error_threshold = 50  # Disable PubMed after this many errors
        self.pubmed_disabled = False
        
        # API rate limiting configuration
        self.rate_limits = {
            'crossref': {'requests_per_second': 50, 'requests_per_minute': 200},  # Crossref allows ~3000 requests per 5 min window for registered users
            'elsevier': {'requests_per_second': 5, 'requests_per_minute': 200},  # Elsevier API typically allows ~5 req/sec
            'pubmed': {'requests_per_second': 3, 'requests_per_minute': 100},    # PubMed API has stricter limits
            'semantic_scholar': {'requests_per_second': 5, 'requests_per_minute': 100}  # Semantic Scholar limits to 100 req/min
        }
        
        # Track API request timestamps to implement rate limiting
        self.request_history = {
            'crossref': [],
            'elsevier': [],
            'pubmed': [],
            'semantic_scholar': []
        }
        
        # Adaptive backoff delays when rate limits are hit
        self.backoff_delays = {
            'crossref': 0,
            'elsevier': 0, 
            'pubmed': 0,
            'semantic_scholar': 0
        }
        
        # API-specific semaphores to control concurrent requests per API
        self.api_semaphores = {
            'crossref': asyncio.Semaphore(max(3, max_concurrent // 2)),  # Allow more Crossref requests
            'elsevier': asyncio.Semaphore(max(2, max_concurrent // 3)),
            'pubmed': asyncio.Semaphore(max(2, max_concurrent // 3)),
            'semantic_scholar': asyncio.Semaphore(max(2, max_concurrent // 3))
        }
        
        # Configuration
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
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()
        
        # Cleanup old cache entries on startup
        await self.cleanup_cache()
        
    async def check_rate_limit(self, source: str) -> float:
        """Check if we're exceeding rate limits for a source and return delay time if needed.
        
        Args:
            source: The API source name (e.g., 'elsevier', 'crossref')
            
        Returns:
            Delay time in seconds (0 if no delay needed)
        """
        now = datetime.now()
        history = self.request_history.get(source, [])
        limits = self.rate_limits.get(source, {'requests_per_second': 5, 'requests_per_minute': 100})        
        
        # Remove timestamps older than 60 seconds
        history = [ts for ts in history if (now - ts).total_seconds() < 60]
        self.request_history[source] = history
        
        # Check if we're within rate limits
        if len(history) >= limits['requests_per_minute']:
            # We've exceeded the per-minute limit
            oldest = history[0]
            time_diff = 60 - (now - oldest).total_seconds()
            if time_diff > 0:
                # Adaptive backoff: increase delay if we're consistently hitting limits
                self.backoff_delays[source] = min(5.0, self.backoff_delays[source] + 0.2)  # max 5 sec backoff
                return max(time_diff, self.backoff_delays[source])
        
        # Check requests per second rate
        last_second = [ts for ts in history if (now - ts).total_seconds() < 1.0]
        if len(last_second) >= limits['requests_per_second']:
            # We're sending too many requests per second
            self.backoff_delays[source] = min(1.0, self.backoff_delays[source] + 0.1)  # max 1 sec backoff for per-second rate
            return self.backoff_delays[source]
        
        # If we've successfully stayed under limits, gradually reduce backoff
        if self.backoff_delays[source] > 0:
            self.backoff_delays[source] = max(0, self.backoff_delays[source] - 0.05)
            
        return 0.0
    
    async def track_request(self, source: str):
        """Track a request to a specific API source."""
        self.request_history[source].append(datetime.now())
        
    async def with_rate_limit(self, source: str, coroutine):
        """Execute a coroutine with rate limiting for a specific API source.
        
        Args:
            source: API source name
            coroutine: The coroutine to execute
            
        Returns:
            The result of the coroutine
        """
        # Check if we need to delay
        delay = await self.check_rate_limit(source)
        if delay > 0:
            # Add jitter to prevent synchronized requests
            jitter = delay * 0.1 * (0.5 - random.random())  # ±10% random jitter
            await asyncio.sleep(delay + jitter)
        
        # Use the API-specific semaphore to control concurrent requests
        async with self.api_semaphores[source]:
            # Track this request
            await self.track_request(source)
            # Execute the actual request
            return await coroutine
    
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, session: aiohttp.ClientSession, url: str, params: Dict = None, headers: Dict = None, source: str = 'elsevier'):
        """Make HTTP request with retry logic and adaptive rate limiting."""
        # Set default headers based on the API source if not provided
        if headers is None:
            headers = {}
            
        # Add source-specific headers if not present
        if source == 'elsevier' and 'X-ELS-APIKey' not in headers:
            headers['X-ELS-APIKey'] = self.elsevier_api_key
            if 'Accept' not in headers:
                headers['Accept'] = 'application/json'
        elif source == 'pubmed':
            if 'Accept' not in headers:
                headers['Accept'] = 'application/json'
            if self.pubmed_email and 'User-Agent' not in headers:
                headers['User-Agent'] = f'Publication Enricher (mailto:{self.pubmed_email})'
        elif source == 'crossref':
            if 'Accept' not in headers:
                headers['Accept'] = 'application/json'
            if self.crossref_email and 'User-Agent' not in headers:
                headers['User-Agent'] = f'Publication Enricher (mailto:{self.crossref_email})'
        elif source == 'semantic_scholar':
            if 'Accept' not in headers:
                headers['Accept'] = 'application/json'
            if self.semantic_scholar_api_key and 'x-api-key' not in headers:
                headers['x-api-key'] = self.semantic_scholar_api_key
                
        # Create the actual request function
        async def do_request():
            try:
                async with session.get(url, params=params, headers=headers, timeout=30) as response:
                    # Check for rate limit headers and update our limits if present
                    if source == 'crossref':
                        # Crossref uses X-Rate-Limit headers
                        if 'X-Rate-Limit-Limit' in response.headers and 'X-Rate-Limit-Interval' in response.headers:
                            try:
                                limit = int(response.headers.get('X-Rate-Limit-Limit'))
                                interval = response.headers.get('X-Rate-Limit-Interval')
                                if 'm' in interval:  # Minutes
                                    minutes = int(interval.replace('m', ''))
                                    self.rate_limits[source]['requests_per_minute'] = limit // minutes
                            except (ValueError, TypeError):
                                pass
                    
                    # Process response based on status
                    if response.status == 200:
                        if response.content_type.startswith('application/json'):
                            return await response.json()
                        else:
                            return await response.text()
                    else:
                        error_reason = 'unknown'
                        if response.status == 429:
                            error_reason = 'rate_limit_exceeded'
                            logger.warning(f"{source.capitalize()} API rate limit exceeded: {response.status} - {url}")
                            
                            # Increase backoff for this source
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    # If Retry-After is in seconds
                                    backoff = float(retry_after)
                                    self.backoff_delays[source] = max(self.backoff_delays[source], backoff)
                                except (ValueError, TypeError):
                                    # If it couldn't be parsed as a number, use default backoff increase
                                    self.backoff_delays[source] = min(10.0, self.backoff_delays[source] * 2 + 1)
                            else:
                                # No Retry-After header, use default backoff increase
                                self.backoff_delays[source] = min(10.0, self.backoff_delays[source] * 2 + 1)
                        elif response.status == 403:
                            error_reason = 'access_forbidden'
                            logger.warning(f"{source.capitalize()} API access forbidden: {response.status} - {url}")
                        elif response.status == 401:
                            error_reason = 'unauthorized'
                            logger.warning(f"{source.capitalize()} API unauthorized: {response.status} - {url}")
                        elif response.status == 404:
                            error_reason = 'not_found'
                            logger.debug(f"{source.capitalize()} API resource not found: {response.status} - {url}")
                        else:
                            logger.debug(f"{source.capitalize()} API request failed: {response.status} - {url}")
                        
                        # Track error count for this source
                        attr_name = f"{source}_error_count"
                        if hasattr(self, attr_name):
                            current_count = getattr(self, attr_name)
                            setattr(self, attr_name, current_count + 1)
                        
                        return None
                        
            except aiohttp.ClientConnectorError:
                logger.warning(f"{source.capitalize()} API connection error - possibly offline or network issues")
                raise
            except asyncio.TimeoutError:
                logger.warning(f"{source.capitalize()} API timeout error - server may be slow or overloaded")
                raise
            except Exception as e:
                logger.debug(f"{source.capitalize()} API request error: {str(e)}")
                raise
        
        # Use the general semaphore as a global limit
        async with self.semaphore:
            # Then apply the per-API rate limiting
            return await self.with_rate_limit(source, do_request())
    
    async def get_abstract_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata by DOI from Elsevier API."""
        cache_key = f"doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.elsevier.com/content/article/doi/{doi}"
            data = await self._make_request(session, url, source='elsevier')
            
            if 'full-text-retrieval-response' in data:
                result = {
                    'title': data['full-text-retrieval-response'].get('coredata', {}).get('dc:title'),
                    'doi': doi,
                    'abstract': data['full-text-retrieval-response'].get('coredata', {}).get('dc:description'),
                    'source': 'elsevier'
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None

    async def get_abstract_from_crossref_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from Crossref by DOI."""
        if not self.has_crossref:
            return None
            
        cache_key = f"crossref_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.crossref.org/works/{doi}"
            data = await self._make_request(session, url, source='crossref')
            
            if data.get('message') and data['message'].get('title'):
                # Crossref doesn't typically provide abstracts, but we can get other metadata
                title = data['message']['title'][0] if isinstance(data['message']['title'], list) else data['message']['title']
                
                # The abstract might be in 'abstract' field if available
                abstract = data['message'].get('abstract', None)
                
                result = {
                    'title': title,
                    'doi': doi,
                    'abstract': abstract,  # Often None from Crossref
                    'source': 'crossref'
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None
        
    async def get_abstract_from_semantic_scholar_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from Semantic Scholar by DOI."""
        cache_key = f"semantic_scholar_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
            params = {
                'fields': 'title,abstract,doi'
            }
            
            data = await self._make_request(session, url, params, source='semantic_scholar')
            
            if data and data.get('abstract'):
                result = {
                    'title': data.get('title'),
                    'doi': doi,
                    'abstract': data.get('abstract'),
                    'source': 'semantic_scholar'
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None

    async def get_abstract_from_semantic_scholar_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title using Semantic Scholar API."""
        cache_key = f"semantic_scholar_title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': title,
                'fields': 'title,abstract,doi',
                'limit': 1
            }
            
            data = await self._make_request(session, url, params, source='semantic_scholar')
            
            if data and data.get('data') and len(data['data']) > 0:
                paper = data['data'][0]
                if paper.get('abstract'):
                    result = {
                        'title': paper.get('title'),
                        'doi': paper.get('doi'),
                        'abstract': paper.get('abstract'),
                        'source': 'semantic_scholar'
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
                
        except Exception as e:
            # Silent fail for individual title lookups
            pass
        
        return None
    
    async def get_abstract_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title using Elsevier API."""
        cache_key = f"title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://api.elsevier.com/content/search/scopus"
            params = {
                'query': f'title("{title}")',
                'field': 'dc:title,dc:identifier,abstract'
            }
            
            data = await self._make_request(session, url, params, source='elsevier')
            
            if 'search-results' in data and data['search-results'].get('entry'):
                entry = data['search-results']['entry'][0]
                result = {
                    'title': entry.get('dc:title'),
                    'doi': entry.get('prism:doi'),
                    'abstract': entry.get('dc:description'),
                    'source': 'elsevier'
                }
                await self.save_to_cache(cache_key, result)
                return result
                
        except Exception as e:
            # Silent fail for individual title lookups
            pass
        
        return None
    
    async def get_abstract_from_pubmed_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from PubMed by DOI."""
        if not self.has_pubmed or self.pubmed_disabled:
            return None
            
        # Check if we've hit the error threshold
        if self.pubmed_error_count >= self.pubmed_error_threshold:
            logger.warning(f"PubMed API disabled due to {self.pubmed_error_count} consecutive errors")
            self.pubmed_disabled = True
            return None
            
        cache_key = f"pubmed_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # First search for the PMID using the DOI
            encoded_doi = urllib.parse.quote(doi)
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"{encoded_doi}[doi]",
                'retmode': 'json',
                'retmax': 1
            }
            
            if self.pubmed_email:
                search_params['email'] = self.pubmed_email
                search_params['tool'] = 'PublicationEnricher'
                
            # Add API key as parameter if available
            if self.pubmed_api_key:
                search_params['api_key'] = self.pubmed_api_key
                
            search_data = await self._make_request(session, search_url, search_params, source='pubmed')
            
            if search_data.get('esearchresult', {}).get('idlist') and len(search_data['esearchresult']['idlist']) > 0:
                pmid = search_data['esearchresult']['idlist'][0]
                
                # Then fetch the abstract using the PMID
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': pmid,
                    'retmode': 'xml',
                }
                
                if self.pubmed_email:
                    fetch_params['email'] = self.pubmed_email
                    fetch_params['tool'] = 'PublicationEnricher'
                
                # Add API key as parameter if available
                if self.pubmed_api_key:
                    fetch_params['api_key'] = self.pubmed_api_key
                
                # For XML, we'll use text() instead of json()
                async with self.semaphore:
                    async with session.get(fetch_url, params=fetch_params) as response:
                        response.raise_for_status()
                        xml_text = await response.text()
                        
                # Basic extraction of abstract from XML
                abstract_match = re.search(r'<AbstractText[^>]*>([\s\S]*?)</AbstractText>', xml_text)
                title_match = re.search(r'<ArticleTitle>([\s\S]*?)</ArticleTitle>', xml_text)
                
                if abstract_match:
                    result = {
                        'title': title_match.group(1) if title_match else None,
                        'doi': doi,
                        'abstract': abstract_match.group(1),
                        'pmid': pmid,
                        'source': 'pubmed'
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
        
        except aiohttp.ClientResponseError as e:
            # Check for rate limiting (HTTP 429)
            if e.status == 429:
                # Don't log every rate limit, just count it
                self.pubmed_error_count += 1
                if self.pubmed_error_count >= self.pubmed_error_threshold:
                    logger.warning(f"Disabling PubMed API after {self.pubmed_error_count} rate limit errors")
                    self.pubmed_disabled = True
            # No need to log every error
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None
        
    async def get_abstract_from_pubmed_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Get publication metadata from PubMed by title."""
        if not self.has_pubmed or self.pubmed_disabled:
            return None
            
        # Check if we've hit the error threshold
        if self.pubmed_error_count >= self.pubmed_error_threshold:
            logger.warning(f"PubMed API disabled due to {self.pubmed_error_count} consecutive errors")
            self.pubmed_disabled = True
            return None
            
        cache_key = f"pubmed_title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            return cached
        
        try:
            # Search for articles by title
            encoded_title = urllib.parse.quote(title)
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"\"{encoded_title}\"[Title]",
                'retmode': 'json',
                'retmax': 1
            }
            
            if self.pubmed_email:
                search_params['email'] = self.pubmed_email
                search_params['tool'] = 'PublicationEnricher'
                
            # Add API key as parameter if available
            if self.pubmed_api_key:
                search_params['api_key'] = self.pubmed_api_key
                
            search_data = await self._make_request(session, search_url, search_params, source='pubmed')
            
            if search_data.get('esearchresult', {}).get('idlist') and len(search_data['esearchresult']['idlist']) > 0:
                pmid = search_data['esearchresult']['idlist'][0]
                
                # Then fetch the abstract using the PMID
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': pmid,
                    'retmode': 'xml',
                }
                
                if self.pubmed_email:
                    fetch_params['email'] = self.pubmed_email
                    fetch_params['tool'] = 'PublicationEnricher'
                
                # For XML, we'll use text() instead of json()
                async with self.semaphore:
                    async with session.get(fetch_url, params=fetch_params) as response:
                        response.raise_for_status()
                        xml_text = await response.text()
                        
                # Basic extraction of abstract from XML
                abstract_match = re.search(r'<AbstractText[^>]*>([\s\S]*?)</AbstractText>', xml_text)
                title_match = re.search(r'<ArticleTitle>([\s\S]*?)</ArticleTitle>', xml_text)
                doi_match = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml_text)
                
                if abstract_match:
                    result = {
                        'title': title_match.group(1) if title_match else title,
                        'doi': doi_match.group(1) if doi_match else None,
                        'abstract': abstract_match.group(1),
                        'pmid': pmid,
                        'source': 'pubmed'
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
        
        except aiohttp.ClientResponseError as e:
            # Check for rate limiting (HTTP 429)
            if e.status == 429:
                # Just count rate limits without logging each one
                self.pubmed_error_count += 1
                if self.pubmed_error_count >= self.pubmed_error_threshold:
                    logger.warning(f"Disabling PubMed API after {self.pubmed_error_count} rate limit errors")
                    self.pubmed_disabled = True
            # No need to log every error
        except Exception as e:
            # Silent fail for individual title lookups
            pass
        
        return None
    
    async def verify_publications(self, publications: List[Dict]) -> List[Dict]:
        """
        Verify a batch of publications in parallel, with multi-API fallback strategy.
        
        Args:
            publications: List of dictionaries with 'title' and optional 'doi' keys
            
        Returns:
            List of enriched publication dictionaries
        """
        async with aiohttp.ClientSession() as session:
            # Initialize results for each publication
            results = [None] * len(publications)
            
            # Track which publications need lookup by each API
            need_lookup = list(range(len(publications)))
            
            #-----------------------------------------------------
            # PHASE 1: Try Crossref first for publications with DOIs
            #-----------------------------------------------------
            if self.has_crossref and need_lookup:
                crossref_tasks = []
                crossref_indices = []
                
                for i in need_lookup:
                    pub = publications[i]
                    if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                        task = self.get_abstract_from_crossref_by_doi(session, pub['doi'])
                        crossref_tasks.append(task)
                        crossref_indices.append(i)
                
                if crossref_tasks:
                    crossref_results = await asyncio.gather(*crossref_tasks, return_exceptions=True)
                    
                    # Process results
                    still_need_lookup = []
                    for task_idx, pub_idx in enumerate(crossref_indices):
                        result = crossref_results[task_idx]
                        if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                            results[pub_idx] = result
                        else:
                            still_need_lookup.append(pub_idx)
                    
                    # Only keep indices that were in crossref_indices and need further lookup
                    need_lookup = [idx for idx in need_lookup if idx in crossref_indices and idx in still_need_lookup] + [idx for idx in need_lookup if idx not in crossref_indices]
            
            #-----------------------------------------------------
            # PHASE 2: Try Elsevier for publications that still need it
            #-----------------------------------------------------
            elsevier_tasks = []
            elsevier_indices = []
            
            for i in need_lookup:
                pub = publications[i]
                if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                    task = self.get_abstract_by_doi(session, pub['doi'])
                else:
                    task = self.get_abstract_by_title(session, pub['title'])
                elsevier_tasks.append(task)
                elsevier_indices.append(i)
            
            if elsevier_tasks:
                elsevier_results = await asyncio.gather(*elsevier_tasks, return_exceptions=True)
                
                # Process results and determine which publications still need lookup
                still_need_lookup = []
                for task_idx, pub_idx in enumerate(elsevier_indices):
                    result = elsevier_results[task_idx]
                    if not isinstance(result, Exception) and result is not None:
                        results[pub_idx] = result
                    else:
                        still_need_lookup.append(pub_idx)
                
                need_lookup = still_need_lookup
                
            #-----------------------------------------------------
            # PHASE 3: Try PubMed for publications that still need it
            #-----------------------------------------------------
            if self.has_pubmed and need_lookup:
                pubmed_tasks = []
                pubmed_indices = []
                
                for i in need_lookup:
                    pub = publications[i]
                    if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                        task = self.get_abstract_from_pubmed_by_doi(session, pub['doi'])
                    else:
                        task = self.get_abstract_from_pubmed_by_title(session, pub['title'])
                    pubmed_tasks.append(task)
                    pubmed_indices.append(i)
                
                if pubmed_tasks:
                    pubmed_results = await asyncio.gather(*pubmed_tasks, return_exceptions=True)
                    
                    # Process results
                    still_need_lookup = []
                    for task_idx, pub_idx in enumerate(pubmed_indices):
                        result = pubmed_results[task_idx]
                        if not isinstance(result, Exception) and result is not None:
                            results[pub_idx] = result
                        else:
                            still_need_lookup.append(pub_idx)
                    
                    need_lookup = still_need_lookup
            
            #-----------------------------------------------------
            # PHASE 4: Try Semantic Scholar as last resort
            #-----------------------------------------------------
            semantic_tasks = []
            semantic_indices = []
            
            for i in need_lookup:
                pub = publications[i]
                if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                    task = self.get_abstract_from_semantic_scholar_by_doi(session, pub['doi'])
                else:
                    task = self.get_abstract_from_semantic_scholar_by_title(session, pub['title'])
                semantic_tasks.append(task)
                semantic_indices.append(i)
            
            if semantic_tasks:
                semantic_results = await asyncio.gather(*semantic_tasks, return_exceptions=True)
                
                # Process results
                for task_idx, pub_idx in enumerate(semantic_indices):
                    result = semantic_results[task_idx]
                    if not isinstance(result, Exception) and result is not None:
                        results[pub_idx] = result
            
            #-----------------------------------------------------
            # MERGE RESULTS back into original publications
            #-----------------------------------------------------
            enriched = []
            for i, pub in enumerate(publications):
                if results[i] is not None:
                    pub.update(results[i])
                enriched.append(pub)
            
            return enriched 