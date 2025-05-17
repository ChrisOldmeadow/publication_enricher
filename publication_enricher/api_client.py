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
import copy
from fuzzywuzzy import fuzz, process

# Import our custom n-gram matching module
from .ngram_matcher import enhanced_title_similarity

logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalize title for better matching.
    
    Enhanced to handle various special cases like XML/HTML tags,
    special characters, different quote styles, academic notation, etc.
    """
    if not title:
        return ""
    
    # Remove all HTML/XML tags
    title = re.sub(r'<[^>]+>', '', title)
    
    # Convert to lowercase
    title = title.lower()
    
    # Remove HTML entities
    title = re.sub(r'&[a-z0-9]+;', ' ', title)
    
    # Standardize common academic notation
    title = re.sub(r'\b(?:vs\.|vs|versus)\b', 'vs', title)  # Standardize versus/vs./vs
    title = re.sub(r'\bp\s*(?:<|=|>)\s*[0-9]\.?[0-9]*\b', '', title)  # Remove p-values
    title = re.sub(r'\bdoi:\s*[^\s]+\b', '', title)  # Remove DOI references
    title = re.sub(r'\b[0-9]+-[0-9]+\b', '', title)  # Remove page ranges
    title = re.sub(r'\b[0-9]+(?:\.[0-9]+)?%\b', '', title)  # Remove percentages
    
    # Handle common academic notations
    title = re.sub(r'\([^)]*\)', '', title)  # Remove parenthetical phrases
    title = re.sub(r'\[[^]]*\]', '', title)  # Remove bracketed text
    
    # Standardize common variations
    replacements = [
        # Standardize dashes and hyphens
        ('–', '-'),  # en-dash
        ('—', '-'),  # em-dash
        ('‐', '-'),  # hyphen
        # Standardize quotes
        ('"', ''),  # left double quote
        ('"', ''),  # right double quote
        (''', ''),  # left single quote
        (''', ''),  # right single quote
        ('\"', ''),  # double quote
        ('\'', ''),  # single quote
        # Standardize spaces
        ('\t', ' '),  # tab
        ('\n', ' '),  # newline
        ('\r', ' '),  # carriage return
        # Other common characters
        ('\u2019', ''),  # right single quotation mark
        ('\u2018', ''),  # left single quotation mark
        # Common academic terminology replacements
        ('randomized controlled trial', 'rct'),
        ('systematic review', 'review'),
        ('meta analysis', 'metaanalysis'),
        ('meta-analysis', 'metaanalysis'),
        ('cross-sectional', 'crosssectional'),
        ('double-blind', 'doubleblind'),
        ('cohort study', 'cohort'),
        ('case control', 'casecontrol'),
        ('randomised', 'randomized'),  # UK/US spelling differences
        ('analysing', 'analyzing'),
        ('colour', 'color'),
        ('centre', 'center'),
    ]
    
    for old, new in replacements:
        title = title.replace(old, new)
    
    # Normalize unicode characters (accents, special chars)
    import unicodedata
    title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf-8')
    
    # Replace all punctuation with spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Remove common words that don't contribute to matching
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
                'by', 'as', 'is', 'are', 'was', 'were', 'be', 'this', 'that', 'from',
                # Additional academic stopwords
                'review', 'study', 'analysis', 'research', 'article', 'journal', 'paper',
                'versus', 'vs', 'using', 'based', 'between', 'among', 'among', 'within',
                'evaluation', 'assessment', 'analysis', 'role', 'impact', 'effect', 'effects',
                'outcomes', 'results', 'method', 'methods', 'approach', 'approaches',
                'evidence', 'model', 'models', 'framework', 'frameworks',
                'part', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    title_words = title.split()
    title_words = [word for word in title_words if word not in stopwords]
    
    return ' '.join(title_words)

def fuzzy_title_match(title1: str, title2: str, threshold: int = 85) -> bool:
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


def fuzzy_title_match_with_year(title1: str, title2: str, year1=None, year2=None, threshold: int = 85) -> tuple:
    """Check if two titles match using fuzzy matching, with optional publication year filtering.
    
    Args:
        title1: First title
        title2: Second title
        year1: Publication year for title1 (optional)
        year2: Publication year for title2 (optional)
        threshold: Minimum score to consider a match (0-100)
        
    Returns:
        tuple: (is_match, score) - True/False if titles match above threshold and the matching score
    """
    # If both years are provided, filter out matches with year difference > 1
    if year1 and year2:
        try:
            # Handle various formats and potential non-numeric years
            year1_int = int(re.search(r'\d{4}', str(year1)).group(0)) if re.search(r'\d{4}', str(year1)) else None
            year2_int = int(re.search(r'\d{4}', str(year2)).group(0)) if re.search(r'\d{4}', str(year2)) else None
            
            if year1_int and year2_int and abs(year1_int - year2_int) > 1:
                logger.debug(f"Year mismatch: {year1_int} vs {year2_int}")
                return False, 0
        except (ValueError, AttributeError, TypeError):
            # If there's any error parsing years, skip year filtering
            pass
    
    # Normalize titles first
    norm_title1 = normalize_title(title1)
    norm_title2 = normalize_title(title2)
    
    # Get both token sort and token set ratios (handles word order and subset matches)
    token_sort_ratio = fuzz.token_sort_ratio(norm_title1, norm_title2)
    token_set_ratio = fuzz.token_set_ratio(norm_title1, norm_title2)
    
    # Use the higher of the two scores
    score = max(token_sort_ratio, token_set_ratio)
    
    logger.debug(f"Fuzzy match score: {score} for '{title1}' and '{title2}'")
    return score >= threshold, score

def find_best_title_match(search_title: str, candidates: List[Dict], threshold: int = 85, 
                       search_year=None, search_authors=None) -> Optional[Dict]:
    """Find best matching title in a list of candidates with improved matching using n-gram matching,
    fuzzy matching, year and author data.
    
    Args:
        search_title: Title to search for
        candidates: List of dictionaries containing candidate titles
        threshold: Minimum score to consider a match (0-100)
        search_year: Optional publication year to use for filtering
        search_authors: Optional list of authors to boost confidence in matches
        
    Returns:
        Optional[Dict]: Best matching candidate or None if no match found.
    """
    if not candidates:
        return None
        
    # Normalize the search title
    norm_search_title = normalize_title(search_title)
    
    # Find the best match
    best_match = None
    best_score = 0
    best_match_info = {}
    
    for candidate in candidates:
        if 'title' in candidate and candidate['title']:
            # Extract years if present to filter out obviously wrong matches
            candidate_year = candidate.get('year') or candidate.get('publication_year') or None
            if search_year and candidate_year:
                try:
                    # Convert years to integers for comparison
                    search_year_int = int(re.search(r'\d{4}', str(search_year)).group(0)) if re.search(r'\d{4}', str(search_year)) else None
                    candidate_year_int = int(re.search(r'\d{4}', str(candidate_year)).group(0)) if re.search(r'\d{4}', str(candidate_year)) else None
                    
                    # Filter out candidates with year difference > 1
                    if search_year_int and candidate_year_int and abs(search_year_int - candidate_year_int) > 1:
                        continue
                except (ValueError, AttributeError):
                    # If there's an error parsing the years, don't apply the filter
                    pass
            
            # --- STANDARD FUZZY MATCHING ---
            # Calculate fuzzy match scores 
            token_sort_ratio = fuzz.token_sort_ratio(norm_search_title, normalize_title(candidate['title']))
            token_set_ratio = fuzz.token_set_ratio(norm_search_title, normalize_title(candidate['title']))
            
            # Take the best scoring method
            base_score = max(token_sort_ratio, token_set_ratio)
            match_type = "token_set_ratio" if token_set_ratio >= token_sort_ratio else "token_sort_ratio"
            
            # --- N-GRAM MATCHING (Additional technique) ---
            # Try enhanced n-gram matching if the fuzzy score is below a good threshold
            # This can help with differently abbreviated titles
            if base_score < 90:  # Only use n-gram if fuzzy matching isn't already strong
                ngram_score, ngram_technique = enhanced_title_similarity(search_title, candidate['title'])
                # Convert the 0-1 score to 0-100 scale
                ngram_score_100 = int(ngram_score * 100)
                
                # If the n-gram matching produces a better score, use it instead
                if ngram_score_100 > base_score + 5:  # Only replace if significantly better
                    base_score = ngram_score_100
                    match_type = ngram_technique
            
            # Start with the base score from the best matching technique
            adjusted_score = base_score
            
            # --- AUTHOR MATCHING ---
            # Bonus points for author match
            author_bonus = 0
            if search_authors and (candidate.get('authors') or candidate.get('author')):
                # Extract and normalize search authors
                search_last_names = set()
                for author in search_authors:
                    if isinstance(author, str):
                        parts = author.split()
                        if parts:
                            search_last_names.add(parts[-1].lower())
                
                # Extract last names from candidate authors
                candidate_authors = candidate.get('authors', [])
                if not candidate_authors and candidate.get('author'):
                    # Handle different author formats
                    if isinstance(candidate['author'], str):
                        candidate_authors = [a.strip() for a in candidate['author'].split(',')] 
                    elif isinstance(candidate['author'], list):
                        candidate_authors = candidate['author']
                
                candidate_last_names = set()
                for author in candidate_authors:
                    if isinstance(author, str):
                        parts = author.split()
                        if parts:
                            candidate_last_names.add(parts[-1].lower())
                
                if search_last_names and candidate_last_names:
                    overlap = len(search_last_names.intersection(candidate_last_names))
                    if overlap > 0:
                        # Add bonus for author match - more for first author match
                        author_bonus = min(8, overlap * 2)  # Increased author bonus 
                        
                        # Extra bonus if first author matches
                        if len(search_authors) > 0 and len(candidate_authors) > 0:
                            try:
                                search_first = search_authors[0].split()[-1].lower() if isinstance(search_authors[0], str) else ""
                                candidate_first = candidate_authors[0].split()[-1].lower() if isinstance(candidate_authors[0], str) else ""
                                if search_first and candidate_first and search_first == candidate_first:
                                    author_bonus += 5  # Increased first author bonus
                            except (IndexError, AttributeError):
                                pass
            
            # Apply author bonus to adjusted score
            adjusted_score += author_bonus
            
            # Check if this is our best match so far
            if adjusted_score > best_score and adjusted_score >= threshold:
                best_score = adjusted_score
                best_match = candidate
                best_match_info = {
                    'fuzzy_matched': True,
                    'query_title': search_title,
                    'matched_title': candidate['title'],
                    'match_score': adjusted_score,
                    'base_score': base_score,
                    'author_bonus': author_bonus,
                    'match_type': match_type,
                    'threshold': threshold
                }
    
    # Prepare the final best match with metadata
    if best_match:
        best_match['match_info'] = best_match_info
    
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
                 cache_ttl_days: int = 30,
                 disable_pubmed: bool = False,
                 disable_semantic_scholar: bool = False):
        
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
            disable_pubmed: Set to True to disable PubMed API lookups
            disable_semantic_scholar: Set to True to disable Semantic Scholar API lookups
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
        
        # API disable flags
        self.pubmed_disabled = disable_pubmed
        self.semantic_scholar_disabled = disable_semantic_scholar
        
        # Match statistics tracking
        self.match_stats = {
            'elsevier': {
                'exact_match_attempts': 0,
                'exact_match_successes': 0,
                'fuzzy_match_attempts': 0,
                'fuzzy_match_successes': 0
            },
            'pubmed': {
                'exact_match_attempts': 0,
                'exact_match_successes': 0,
                'fuzzy_match_attempts': 0,
                'fuzzy_match_successes': 0
            },
            'crossref': {
                'exact_match_attempts': 0,
                'exact_match_successes': 0,
                'fuzzy_match_attempts': 0,
                'fuzzy_match_successes': 0
            },
            'semantic_scholar': {
                'exact_match_attempts': 0,
                'exact_match_successes': 0,
                'fuzzy_match_attempts': 0,
                'fuzzy_match_successes': 0
            }
        }
        
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
        
    def get_match_statistics(self) -> Dict:
        """Get comprehensive statistics about matching attempts and successes.
        
        Returns:
            Dict containing detailed matching statistics for all APIs
        """
        # Calculate overall stats
        total_exact_attempts = sum(stats['exact_match_attempts'] for source, stats in self.match_stats.items())
        total_exact_successes = sum(stats['exact_match_successes'] for source, stats in self.match_stats.items())
        total_fuzzy_attempts = sum(stats['fuzzy_match_attempts'] for source, stats in self.match_stats.items())
        total_fuzzy_successes = sum(stats['fuzzy_match_successes'] for source, stats in self.match_stats.items())
        
        # Calculate success rates
        exact_success_rate = (total_exact_successes / total_exact_attempts * 100) if total_exact_attempts > 0 else 0
        fuzzy_success_rate = (total_fuzzy_successes / total_fuzzy_attempts * 100) if total_fuzzy_attempts > 0 else 0
        
        return {
            'by_source': self.match_stats,
            'totals': {
                'exact_match_attempts': total_exact_attempts,
                'exact_match_successes': total_exact_successes,
                'fuzzy_match_attempts': total_fuzzy_attempts,
                'fuzzy_match_successes': total_fuzzy_successes
            },
            'rates': {
                'exact_success_rate': round(exact_success_rate, 1),
                'fuzzy_success_rate': round(fuzzy_success_rate, 1)
            }
        }
    
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
        """Get publication metadata from Elsevier API by DOI."""
        cache_key = f"doi:{doi}"
        
        # Count DOI lookup as an exact match attempt
        self.match_stats['elsevier']['exact_match_attempts'] += 1
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # If cache hit contains match_info, count it for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                source = cached.get('source', 'elsevier')
                if source in self.match_stats:
                    if cached['match_info'].get('fuzzy_matched', False):
                        self.match_stats[source]['fuzzy_match_successes'] += 1
                    else:
                        self.match_stats[source]['exact_match_successes'] += 1
            else:
                # Count cache hit as an exact match success if it doesn't have match_info
                self.match_stats['elsevier']['exact_match_successes'] += 1
            return cached
        
        try:
            url = f"https://api.elsevier.com/content/article/doi/{doi}"
            data = await self._make_request(session, url, source='elsevier')
            
            if 'full-text-retrieval-response' in data:
                result = {
                    'title': data['full-text-retrieval-response'].get('coredata', {}).get('dc:title'),
                    'doi': doi,
                    'abstract': data['full-text-retrieval-response'].get('coredata', {}).get('dc:description'),
                    'source': 'elsevier',
                    'match_info': {
                        'fuzzy_matched': False,
                        'query_doi': doi,
                        'match_type': 'doi'
                    }
                }
                await self.save_to_cache(cache_key, result)
                # Count as exact match success
                self.match_stats['elsevier']['exact_match_successes'] += 1
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None

    async def get_abstract_from_crossref_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from Crossref by DOI."""
        if not self.has_crossref:
            return None
            
        # Count DOI lookup as an exact match attempt
        self.match_stats['crossref']['exact_match_attempts'] += 1
            
        cache_key = f"crossref_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # Count cache hit as an exact match success
            self.match_stats['crossref']['exact_match_successes'] += 1
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
                    'source': 'crossref',
                    'match_info': {
                        'fuzzy_matched': False,
                        'query_doi': doi,
                        'match_type': 'doi'
                    }
                }
                await self.save_to_cache(cache_key, result)
                # Count as exact match success
                self.match_stats['crossref']['exact_match_successes'] += 1
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None
        
    async def get_abstract_from_crossref_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title using Crossref API with exact matching first, then fuzzy matching as fallback."""
        if not self.has_crossref:
            return None
            
        cache_key = f"crossref_title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # Count cache hits for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                if cached['match_info'].get('fuzzy_matched', False):
                    self.match_stats['crossref']['fuzzy_match_successes'] += 1
                else:
                    self.match_stats['crossref']['exact_match_successes'] += 1
            return cached
        
        try:
            # STEP 1: Try exact match first
            self.match_stats['crossref']['exact_match_attempts'] += 1
            
            # Prepare exact search URL with appropriate quoting for title
            exact_url = "https://api.crossref.org/works"
            exact_params = {
                'query.bibliographic': f'"{title}"',  # Quote the title for exact phrase matching
                'rows': 2,  # Limit results for exact match
                'sort': 'score',  # Sort by relevance score
                'order': 'desc'   # Most relevant first
            }
            
            # Add polite parameter if email is provided
            if self.crossref_email:
                exact_params['mailto'] = self.crossref_email
                
            exact_data = await self._make_request(session, exact_url, exact_params, source='crossref')
            
            # Check if we have an exact match
            if exact_data and exact_data.get('message') and exact_data['message'].get('items'):
                items = exact_data['message']['items']
                if items and len(items) > 0:
                    # Check the top result for an exact title match (case-insensitive)
                    top_item = items[0]
                    if top_item.get('title'):
                        item_title = top_item['title'][0] if isinstance(top_item['title'], list) else top_item['title']
                        
                        # Use normalized titles for more robust exact matching
                        normalized_item_title = normalize_title(item_title)
                        normalized_query_title = normalize_title(title)
                        
                        if normalized_item_title == normalized_query_title:
                            # Exact match found
                            self.match_stats['crossref']['exact_match_successes'] += 1
                            
                            # Extract abstract if available (rare in Crossref)
                            abstract = top_item.get('abstract')
                            
                            # Get DOI
                            doi = top_item.get('DOI')
                            
                            result = {
                                'title': item_title,
                                'doi': doi,
                                'abstract': abstract,  # Often None from Crossref
                                'source': 'crossref',
                                'match_info': {
                                    'fuzzy_matched': False,
                                    'query_title': title,
                                    'matched_title': item_title,
                                    'match_type': 'exact'
                                }
                            }
                            await self.save_to_cache(cache_key, result)
                            return result
            
            # STEP 2: If exact match failed, try fuzzy matching with a broader search
            self.match_stats['crossref']['fuzzy_match_attempts'] += 1
            
            # For fuzzy search, remove quotes and use normalized title
            search_title = normalize_title(title)
            # Get first few words for a broader search
            search_words = search_title.split()[:5]  # First 5 words
            search_query = ' '.join(search_words)
            
            fuzzy_params = {
                'query.bibliographic': search_query,
                'rows': 10,  # Get more results for fuzzy matching
                'sort': 'score',
                'order': 'desc'
            }
            
            # Add polite parameter if email is provided
            if self.crossref_email:
                fuzzy_params['mailto'] = self.crossref_email
                
            fuzzy_data = await self._make_request(session, exact_url, fuzzy_params, source='crossref')
            
            if fuzzy_data and fuzzy_data.get('message') and fuzzy_data['message'].get('items'):
                items = fuzzy_data['message']['items']
                if items and len(items) > 0:
                    # Prepare candidates for fuzzy matching
                    candidates = []
                    for item in items:
                        if item.get('title'):
                            item_title = item['title'][0] if isinstance(item['title'], list) else item['title']
                            candidates.append({
                                'title': item_title,
                                'doi': item.get('DOI'),
                                'abstract': item.get('abstract'),  # Usually None
                                'source': 'crossref'
                            })
                    
                    # Extract year information if available in items
                    for item in items:
                        if item.get('title'):
                            # Look for year in different possible locations
                            year = None
                            if item.get('published-print') and item['published-print'].get('date-parts'):
                                year = item['published-print']['date-parts'][0][0] if item['published-print']['date-parts'][0] else None
                            elif item.get('published-online') and item['published-online'].get('date-parts'):
                                year = item['published-online']['date-parts'][0][0] if item['published-online']['date-parts'][0] else None
                            elif item.get('created') and item['created'].get('date-parts'):
                                year = item['created']['date-parts'][0][0] if item['created']['date-parts'][0] else None
                            
                            # Try to extract authors if available
                            authors = []
                            if item.get('author'):
                                authors = [f"{a.get('given', '')} {a.get('family', '')}" for a in item['author'] if a.get('family')]
                            
                            # Update the candidate with year and author info
                            for candidate in candidates:
                                if candidate.get('doi') == item.get('DOI'):
                                    candidate['year'] = year
                                    candidate['authors'] = authors
                    
                    # Find best match using enhanced fuzzy matching with year and author data
                    # Extract year from the query title's publication if available
                    search_year = None
                    search_authors = None
                    
                    # Use a high threshold (90) for fuzzy matching to ensure quality matches
                    best_match = find_best_title_match(title, candidates, threshold=85, 
                                                      search_year=search_year, search_authors=search_authors)
                    
                    if best_match:
                        self.match_stats['crossref']['fuzzy_match_successes'] += 1
                        await self.save_to_cache(cache_key, best_match)
                        return best_match
                        
        except Exception as e:
            # Silent fail for individual title lookups
            logger.debug(f"Crossref title search error: {str(e)}")
            pass
        
        return None
    
    async def get_abstract_from_semantic_scholar_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from Semantic Scholar by DOI."""
        # Return None if Semantic Scholar API is disabled
        if self.semantic_scholar_disabled:
            return None
            
        # Count DOI lookup as an exact match attempt
        self.match_stats['semantic_scholar']['exact_match_attempts'] += 1
        
        cache_key = f"semantic_scholar_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # Count cache hits for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                if cached['match_info'].get('fuzzy_matched', False):
                    self.match_stats['semantic_scholar']['fuzzy_match_successes'] += 1
                else:
                    self.match_stats['semantic_scholar']['exact_match_successes'] += 1
            else:
                # Count cache hit as an exact match success if it doesn't have match_info
                self.match_stats['semantic_scholar']['exact_match_successes'] += 1
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
                    'source': 'semantic_scholar',
                    'match_info': {
                        'fuzzy_matched': False,
                        'query_doi': doi,
                        'match_type': 'doi'
                    }
                }
                await self.save_to_cache(cache_key, result)
                # Count as exact match success
                self.match_stats['semantic_scholar']['exact_match_successes'] += 1
                return result
                
        except Exception as e:
            # Silent fail for individual DOI lookups
            pass
        
        return None

    async def get_abstract_from_semantic_scholar_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title using Semantic Scholar API with exact matching first, then fuzzy matching as fallback."""
        # Return None if Semantic Scholar API is disabled
        if self.semantic_scholar_disabled:
            return None
            
        cache_key = f"semantic_scholar_title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # Count cache hits for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                if cached['match_info'].get('fuzzy_matched', False):
                    self.match_stats['semantic_scholar']['fuzzy_match_successes'] += 1
                else:
                    self.match_stats['semantic_scholar']['exact_match_successes'] += 1
            return cached
        
        try:
            # STEP 1: Try exact match first
            self.match_stats['semantic_scholar']['exact_match_attempts'] += 1
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            exact_params = {
                'query': f'"{title}"',  # Use quotes for exact phrase matching
                'fields': 'title,abstract,doi',
                'limit': 1
            }
            
            exact_data = await self._make_request(session, url, exact_params, source='semantic_scholar')
            
            # Check if we got an exact match
            if (exact_data and exact_data.get('data') and len(exact_data['data']) > 0 and 
                exact_data['data'][0].get('abstract') and exact_data['data'][0].get('title')):
                
                paper = exact_data['data'][0]
                # Check if titles match exactly using normalized titles
                if paper.get('title'):
                    normalized_paper_title = normalize_title(paper.get('title'))
                    normalized_query_title = normalize_title(title)
                    
                    if normalized_paper_title == normalized_query_title:
                        self.match_stats['semantic_scholar']['exact_match_successes'] += 1
                        result = {
                        'title': paper.get('title'),
                        'doi': paper.get('doi'),
                        'abstract': paper.get('abstract'),
                        'source': 'semantic_scholar',
                        'match_info': {
                            'fuzzy_matched': False,
                            'query_title': title,
                            'matched_title': paper.get('title'),
                            'match_type': 'exact'
                        }
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
            
            # STEP 2: If exact match failed, try fuzzy matching
            self.match_stats['semantic_scholar']['fuzzy_match_attempts'] += 1
            # Remove special characters that might interfere with search
            search_title = normalize_title(title)
            search_words = search_title.split()[:5]  # First 5 words for broader search
            search_query = ' '.join(search_words)
            
            fuzzy_params = {
                'query': search_query,
                'fields': 'title,abstract,doi',
                'limit': 10  # Get multiple results for fuzzy matching
            }
            
            fuzzy_data = await self._make_request(session, url, fuzzy_params, source='semantic_scholar')
            
            if fuzzy_data and fuzzy_data.get('data') and len(fuzzy_data['data']) > 0:
                # Prepare candidates for fuzzy matching with additional metadata
                candidates = []
                for paper in fuzzy_data['data']:
                    if paper.get('abstract'):
                        # Create base candidate
                        candidate = {
                            'title': paper.get('title'),
                            'doi': paper.get('doi'),
                            'abstract': paper.get('abstract'),
                            'source': 'semantic_scholar'
                        }
                        
                        # Add year information if available
                        if paper.get('year'):
                            candidate['year'] = paper.get('year')
                            
                        # Add author information if available
                        if paper.get('authors'):
                            candidate['authors'] = [author.get('name') for author in paper.get('authors', []) if author.get('name')]
                            
                        candidates.append(candidate)
                
                # Extract search metadata if available
                search_year = None
                search_authors = None
                
                # Find best match using enhanced fuzzy matching with high threshold (90%)
                best_match = find_best_title_match(title, candidates, threshold=85, 
                                                search_year=search_year, search_authors=search_authors)
                
                if best_match:
                    self.match_stats['semantic_scholar']['fuzzy_match_successes'] += 1
                    await self.save_to_cache(cache_key, best_match)
                    return best_match
                
        except Exception as e:
            # Silent fail for individual title lookups
            logger.debug(f"Semantic Scholar title search error: {str(e)}")
            pass
        
        return None
    
    async def get_abstract_by_title(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict]:
        """Search for publication by title using Elsevier API with exact matching first, then fuzzy matching as fallback."""
        cache_key = f"title:{title}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # If cache hit contains match_info, count it for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                source = cached.get('source', 'elsevier')
                if source in self.match_stats:
                    if cached['match_info'].get('fuzzy_matched', False):
                        self.match_stats[source]['fuzzy_match_successes'] += 1
                    else:
                        self.match_stats[source]['exact_match_successes'] += 1
            return cached
        
        try:
            # STEP 1: Try exact title match first (faster)
            self.match_stats['elsevier']['exact_match_attempts'] += 1
            url = "https://api.elsevier.com/content/search/scopus"
            exact_params = {
                'query': f'title("{title}")',
                'field': 'dc:title,dc:identifier,abstract'
            }
            
            exact_data = await self._make_request(session, url, exact_params, source='elsevier')
            
            # Check if we got an exact match
            if ('search-results' in exact_data and 
                exact_data['search-results'].get('entry') and 
                len(exact_data['search-results']['entry']) > 0):
                
                entry = exact_data['search-results']['entry'][0]
                # Check if titles match exactly using normalized titles
                if entry.get('dc:title'):
                    normalized_entry_title = normalize_title(entry.get('dc:title'))
                    normalized_query_title = normalize_title(title)
                    
                    # Use normalized title comparison for more robust exact matching
                    if normalized_entry_title == normalized_query_title:
                        self.match_stats['elsevier']['exact_match_successes'] += 1
                        result = {
                        'title': entry.get('dc:title'),
                        'doi': entry.get('prism:doi'),
                        'abstract': entry.get('dc:description'),
                        'source': 'elsevier',
                        'match_info': {
                            'fuzzy_matched': False,
                            'query_title': title,
                            'matched_title': entry.get('dc:title'),
                            'match_type': 'exact'
                        }
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
            
            # STEP 2: If exact match failed, try fuzzy matching
            self.match_stats['elsevier']['fuzzy_match_attempts'] += 1
            
            # Remove quotes and special characters that might interfere with search
            search_title = re.sub(r'["\[\]\(\)\{\}]', '', title)
            
            # Get first few words of the title for a broader search
            search_words = search_title.split()[:5]  # First 5 words
            search_query = ' '.join(search_words)
            
            fuzzy_params = {
                'query': f'title({search_query})',
                'field': 'dc:title,dc:identifier,abstract',
                'count': 10  # Get multiple results for fuzzy matching
            }
            
            fuzzy_data = await self._make_request(session, url, fuzzy_params, source='elsevier')
            
            if 'search-results' in fuzzy_data and fuzzy_data['search-results'].get('entry'):
                entries = fuzzy_data['search-results']['entry']
                
                # Prepare candidate list for fuzzy matching with enhanced metadata
                candidates = []
                for entry in entries:
                    # Create base candidate
                    candidate = {
                        'title': entry.get('dc:title'),
                        'doi': entry.get('prism:doi'),
                        'abstract': entry.get('dc:description'),
                        'source': 'elsevier'
                    }
                    
                    # Add publication year if available
                    if entry.get('prism:coverDate'):
                        # Extract year from date format YYYY-MM-DD
                        date_match = re.search(r'(\d{4})-\d{2}-\d{2}', entry.get('prism:coverDate'))
                        if date_match:
                            candidate['year'] = date_match.group(1)
                    
                    # Add author information if available
                    if entry.get('dc:creator'):
                        candidate['authors'] = [entry.get('dc:creator')]
                    elif entry.get('author'):
                        if isinstance(entry.get('author'), list):
                            candidate['authors'] = [author.get('authname') for author in entry.get('author', []) if author.get('authname')]
                        else:
                            candidate['authors'] = [entry.get('author')]
                    
                    candidates.append(candidate)
                
                # Find best match using enhanced fuzzy matching with year and author data
                search_year = None  # We would need to extract this from the source publication if available
                search_authors = None  # We would need to extract this from the source publication if available
                
                # Find best match using fuzzy matching with moderate threshold (85%)
                best_match = find_best_title_match(title, candidates, threshold=85,
                                                 search_year=search_year, search_authors=search_authors)
                
                if best_match:
                    self.match_stats['elsevier']['fuzzy_match_successes'] += 1
                    await self.save_to_cache(cache_key, best_match)
                    return best_match
                
        except Exception as e:
            # Silent fail for individual title lookups
            logger.debug(f"Elsevier title search error: {str(e)}")
            pass
        
        return None
    
    async def get_abstract_from_pubmed_by_doi(self, session: aiohttp.ClientSession, doi: str) -> Optional[Dict]:
        """Get publication metadata from PubMed by DOI."""
        if not self.has_pubmed or self.pubmed_disabled:
            return None
            
        # Count DOI lookup as an exact match attempt
        self.match_stats['pubmed']['exact_match_attempts'] += 1
            
        # Check if we've hit the error threshold
        if self.pubmed_error_count >= self.pubmed_error_threshold:
            logger.warning(f"PubMed API disabled due to {self.pubmed_error_count} consecutive errors")
            self.pubmed_disabled = True
            return None
            
        cache_key = f"pubmed_doi:{doi}"
        
        # Check cache first
        cached = await self.get_from_cache(cache_key)
        if cached:
            # Count cache hit as an exact match success
            self.match_stats['pubmed']['exact_match_successes'] += 1
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
                        'source': 'pubmed',
                        'match_info': {
                            'fuzzy_matched': False,
                            'query_doi': doi,
                            'match_type': 'doi'
                        }
                    }
                    await self.save_to_cache(cache_key, result)
                    # Count as exact match success
                    self.match_stats['pubmed']['exact_match_successes'] += 1
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
        """Get publication metadata from PubMed by title using exact matching first, then fuzzy matching as fallback."""
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
            # Count cache hits for statistics
            if isinstance(cached, dict) and 'match_info' in cached:
                if cached['match_info'].get('fuzzy_matched', False):
                    self.match_stats['pubmed']['fuzzy_match_successes'] += 1
                else:
                    self.match_stats['pubmed']['exact_match_successes'] += 1
            return cached
        
        try:
            # STEP 1: Try exact match first
            self.match_stats['pubmed']['exact_match_attempts'] += 1
            encoded_title = urllib.parse.quote(title)
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            exact_params = {
                'db': 'pubmed',
                'term': f"\"{encoded_title}\"[Title]",  # Exact phrase match
                'retmode': 'json',
                'retmax': 1
            }
            
            if self.pubmed_email:
                exact_params['email'] = self.pubmed_email
                exact_params['tool'] = 'PublicationEnricher'
                
            # Add API key as parameter if available
            if self.pubmed_api_key:
                exact_params['api_key'] = self.pubmed_api_key
                
            exact_data = await self._make_request(session, search_url, exact_params, source='pubmed')
            
            # Check if we got an exact match
            if exact_data.get('esearchresult', {}).get('idlist') and len(exact_data['esearchresult']['idlist']) > 0:
                pmid = exact_data['esearchresult']['idlist'][0]
                
                # Fetch the exact match
                fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': pmid,
                    'retmode': 'xml',
                }
                
                if self.pubmed_email:
                    fetch_params['email'] = self.pubmed_email
                    fetch_params['tool'] = 'PublicationEnricher'
                    
                if self.pubmed_api_key:
                    fetch_params['api_key'] = self.pubmed_api_key
                
                # For XML, we'll use text() instead of json()
                async with self.semaphore:
                    async with session.get(fetch_url, params=fetch_params) as response:
                        response.raise_for_status()
                        xml_text = await response.text()
                
                title_match = re.search(r'<ArticleTitle>([\s\S]*?)</ArticleTitle>', xml_text)
                abstract_match = re.search(r'<AbstractText[^>]*>([\s\S]*?)</AbstractText>', xml_text)
                doi_match = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml_text)
                
                # Check if we got a valid abstract and the title matches using normalized comparison
                if abstract_match and title_match:
                    normalized_result_title = normalize_title(title_match.group(1))
                    normalized_query_title = normalize_title(title)
                    
                    if normalized_result_title == normalized_query_title:
                        self.match_stats['pubmed']['exact_match_successes'] += 1
                        result = {
                        'title': title_match.group(1),
                        'doi': doi_match.group(1) if doi_match else None,
                        'abstract': abstract_match.group(1),
                        'pmid': pmid,
                        'source': 'pubmed',
                        'match_info': {
                            'fuzzy_matched': False,
                            'query_title': title,
                            'matched_title': title_match.group(1),
                            'match_type': 'exact'
                        }
                    }
                    await self.save_to_cache(cache_key, result)
                    return result
            
            # STEP 2: If exact match failed, try fuzzy matching
            self.match_stats['pubmed']['fuzzy_match_attempts'] += 1
            # Use a broader search with just the main keywords
            search_title = normalize_title(title)
            search_words = search_title.split()[:5]  # First 5 words for broader search
            search_query = ' '.join(search_words)
            encoded_search = urllib.parse.quote(search_query)
            
            fuzzy_params = {
                'db': 'pubmed',
                'term': f"{encoded_search}[Title]",  # No quotes for broader search
                'retmode': 'json',
                'retmax': 10  # Get more results for fuzzy matching
            }
            
            if self.pubmed_email:
                fuzzy_params['email'] = self.pubmed_email
                fuzzy_params['tool'] = 'PublicationEnricher'
                
            # Add API key as parameter if available
            if self.pubmed_api_key:
                fuzzy_params['api_key'] = self.pubmed_api_key
                
            fuzzy_data = await self._make_request(session, search_url, fuzzy_params, source='pubmed')
            
            # Get multiple PMIDs for fuzzy matching
            pmids = []
            if fuzzy_data.get('esearchresult', {}).get('idlist'):
                pmids = fuzzy_data['esearchresult']['idlist']
                
                # If we have PMIDs, collect them into candidate results for fuzzy matching
                candidates = []
                
                # Process up to 5 PMIDs to avoid too many requests
                for pmid in pmids[:5]:
                    try:
                        # Fetch the abstract using the PMID
                        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                        fetch_params = {
                            'db': 'pubmed',
                            'id': pmid,
                            'retmode': 'xml',
                        }
                        
                        if self.pubmed_email:
                            fetch_params['email'] = self.pubmed_email
                            fetch_params['tool'] = 'PublicationEnricher'
                            
                        if self.pubmed_api_key:
                            fetch_params['api_key'] = self.pubmed_api_key
                        
                        # For XML, we'll use text() instead of json()
                        async with self.semaphore:
                            async with session.get(fetch_url, params=fetch_params) as response:
                                response.raise_for_status()
                                xml_text = await response.text()
                                
                        # Basic extraction of abstract and other data from XML
                        abstract_match = re.search(r'<AbstractText[^>]*>([\s\S]*?)</AbstractText>', xml_text)
                        title_match = re.search(r'<ArticleTitle>([\s\S]*?)</ArticleTitle>', xml_text)
                        doi_match = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml_text)
                        
                        if abstract_match and title_match:
                            candidates.append({
                                'title': title_match.group(1),
                                'doi': doi_match.group(1) if doi_match else None,
                                'abstract': abstract_match.group(1),
                                'pmid': pmid,
                                'source': 'pubmed'
                            })
                    except Exception as e:
                        # Skip this PMID and try the next one
                        logger.debug(f"Error processing PMID {pmid}: {str(e)}")
                        continue
                
                # Apply fuzzy matching to find the best match
                if candidates:
                    best_match = find_best_title_match(title, candidates, threshold=85)
                    
                    if best_match:
                        self.match_stats['pubmed']['fuzzy_match_successes'] += 1
                        await self.save_to_cache(cache_key, best_match)
                        return best_match
        
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
        Verify a batch of publications in parallel, with sequential API fallback strategy.
        Tries APIs in order: Elsevier, Crossref, PubMed, Semantic Scholar.
        Stops trying additional APIs once a match is found for a publication.
        
        Args:
            publications: List of dictionaries with 'title' and optional 'doi' keys
            
        Returns:
            List of enriched publication dictionaries
        """
        # Create a clean copy of match stats before we start
        # This will be used to reset inflated statistics later
        original_match_stats = copy.deepcopy(self.match_stats)
        
        async with aiohttp.ClientSession() as session:
            # Initialize results for each publication
            results = [None] * len(publications)
            
            # Track publications with missing data
            missing_data_indices = []
            
            # Check for publications with missing data first
            for i, pub in enumerate(publications):
                has_valid_doi = pub.get('doi') and str(pub['doi']).lower() != 'nan'
                has_valid_title = pub.get('title') and str(pub['title']).strip() != ''
                if not has_valid_doi and not has_valid_title:
                    # Mark as having missing data
                    missing_data_indices.append(i)
                    # Add a marker to indicate why it couldn't be searched
                    results[i] = {
                        'source': 'none',
                        'abstract': None,
                        'match_info': {
                            'fuzzy_matched': False,
                            'match_type': 'none',
                            'error': 'missing_data',
                            'error_details': 'Publication missing both DOI and title'
                        }
                    }
            
            # Track which publications still need lookup (exclude ones with missing data)
            need_lookup = [i for i in range(len(publications)) if i not in missing_data_indices]
            
            #-----------------------------------------------------
            # PHASE 1: Try Elsevier first (DOI lookup, then title search)
            #-----------------------------------------------------
            # STEP 1a: Try Elsevier DOI lookup for publications with DOIs
            elsevier_doi_tasks = []
            elsevier_doi_indices = []
            
            for i in need_lookup:
                pub = publications[i]
                if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                    task = self.get_abstract_by_doi(session, pub['doi'])
                    elsevier_doi_tasks.append(task)
                    elsevier_doi_indices.append(i)
            
            if elsevier_doi_tasks:
                elsevier_doi_results = await asyncio.gather(*elsevier_doi_tasks, return_exceptions=True)
                
                # Process results and update who still needs lookup
                still_need_lookup = []
                for task_idx, pub_idx in enumerate(elsevier_doi_indices):
                    result = elsevier_doi_results[task_idx]
                    if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                        results[pub_idx] = result
                    else:
                        still_need_lookup.append(pub_idx)
                
                # Update need_lookup list to remove successful lookups
                need_lookup = [idx for idx in need_lookup if idx in elsevier_doi_indices and idx in still_need_lookup] + \
                            [idx for idx in need_lookup if idx not in elsevier_doi_indices]
            
            # STEP 1b: For remaining publications, try Elsevier title search
            if need_lookup:  # Only continue if we still have publications that need lookup
                elsevier_title_tasks = []
                elsevier_title_indices = []
                
                for i in need_lookup:
                    pub = publications[i]
                    if pub.get('title'):
                        task = self.get_abstract_by_title(session, pub['title'])
                        elsevier_title_tasks.append(task)
                        elsevier_title_indices.append(i)
                
                if elsevier_title_tasks:
                    elsevier_title_results = await asyncio.gather(*elsevier_title_tasks, return_exceptions=True)
                    
                    # Process results
                    still_need_lookup = []
                    for task_idx, pub_idx in enumerate(elsevier_title_indices):
                        result = elsevier_title_results[task_idx]
                        if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                            results[pub_idx] = result
                        else:
                            still_need_lookup.append(pub_idx)
                    
                    # Update need_lookup list to remove successful lookups
                    need_lookup = [idx for idx in need_lookup if idx in elsevier_title_indices and idx in still_need_lookup] + \
                                [idx for idx in need_lookup if idx not in elsevier_title_indices]
            
            #-----------------------------------------------------
            # PHASE 2: Try Crossref for publications that still need it
            #-----------------------------------------------------
            if self.has_crossref and need_lookup:
                # STEP 2a: Try Crossref DOI lookup for publications with DOIs
                crossref_doi_tasks = []
                crossref_doi_indices = []
                
                for i in need_lookup:
                    pub = publications[i]
                    if pub.get('doi') and str(pub['doi']).lower() != 'nan':
                        task = self.get_abstract_from_crossref_by_doi(session, pub['doi'])
                        crossref_doi_tasks.append(task)
                        crossref_doi_indices.append(i)
                
                if crossref_doi_tasks:
                    crossref_doi_results = await asyncio.gather(*crossref_doi_tasks, return_exceptions=True)
                    
                    # Process results
                    still_need_lookup = []
                    for task_idx, pub_idx in enumerate(crossref_doi_indices):
                        result = crossref_doi_results[task_idx]
                        if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                            results[pub_idx] = result
                        else:
                            still_need_lookup.append(pub_idx)
                    
                    # Update need_lookup list to remove successful lookups
                    need_lookup = [idx for idx in need_lookup if idx in crossref_doi_indices and idx in still_need_lookup] + \
                                [idx for idx in need_lookup if idx not in crossref_doi_indices]
                
                # STEP 2b: For remaining publications, try Crossref title lookup
                if need_lookup:  # Only continue if we still have publications that need lookup
                    crossref_title_tasks = []
                    crossref_title_indices = []
                    
                    for i in need_lookup:
                        pub = publications[i]
                        if pub.get('title'):
                            task = self.get_abstract_from_crossref_by_title(session, pub['title'])
                            crossref_title_tasks.append(task)
                            crossref_title_indices.append(i)
                    
                    if crossref_title_tasks:
                        crossref_title_results = await asyncio.gather(*crossref_title_tasks, return_exceptions=True)
                        
                        # Process results
                        still_need_lookup = []
                        for task_idx, pub_idx in enumerate(crossref_title_indices):
                            result = crossref_title_results[task_idx]
                            if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                                results[pub_idx] = result
                            else:
                                still_need_lookup.append(pub_idx)
                        
                        # Update need_lookup list to remove successful lookups
                        need_lookup = [idx for idx in need_lookup if idx in crossref_title_indices and idx in still_need_lookup] + \
                                    [idx for idx in need_lookup if idx not in crossref_title_indices]
            
            #-----------------------------------------------------
            # PHASE 3: Try PubMed for publications that still need it
            #-----------------------------------------------------
            if self.has_pubmed and need_lookup:
                # Only process publications that still need lookup
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
                        if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                            results[pub_idx] = result
                        else:
                            still_need_lookup.append(pub_idx)
                    
                    # Update need_lookup list
                    need_lookup = still_need_lookup
            
            #-----------------------------------------------------
            # PHASE 4: Try Semantic Scholar as last resort
            #-----------------------------------------------------
            if self.has_semantic_scholar and need_lookup:
                # Only try this for publications that still need lookup
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
                    
                    # Process results (no need to track still_need_lookup since this is the last API)
                    for task_idx, pub_idx in enumerate(semantic_indices):
                        result = semantic_results[task_idx]
                        if not isinstance(result, Exception) and result is not None and result.get('abstract'):
                            results[pub_idx] = result
            
            #-----------------------------------------------------
            # MERGE RESULTS back into original publications
            #-----------------------------------------------------
            enriched = []
            
            # We'll store the actual statistics from the processing
            # These statistics reflect what actually happened during API calls
            original_match_attempts = {}
            for source, stats in self.match_stats.items():
                original_match_attempts[source] = {
                    'exact_match_attempts': stats['exact_match_attempts'],
                    'fuzzy_match_attempts': stats['fuzzy_match_attempts']
                }
            
            # Reset match stats to avoid double counting successes
            # But preserve the attempt counts to maintain accurate success rates
            self.match_stats = copy.deepcopy(original_match_stats)
            
            # Restore the actual attempt counts
            for source, stats in original_match_attempts.items():
                if source in self.match_stats:
                    self.match_stats[source]['exact_match_attempts'] = stats['exact_match_attempts']
                    self.match_stats[source]['fuzzy_match_attempts'] = stats['fuzzy_match_attempts']
            
            # Create a record of which match type was used for each source
            used_matches = {
                'elsevier': {'exact': 0, 'fuzzy': 0},
                'crossref': {'exact': 0, 'fuzzy': 0},
                'pubmed': {'exact': 0, 'fuzzy': 0},
                'semantic_scholar': {'exact': 0, 'fuzzy': 0}
            }
            
            for i, pub in enumerate(publications):
                if results[i] is not None:
                    # Count this match in the statistics
                    source = results[i].get('source', 'unknown')
                    if source in self.match_stats:
                        if results[i].get('match_info', {}).get('fuzzy_matched', False):
                            # Count as a fuzzy match success
                            self.match_stats[source]['fuzzy_match_successes'] += 1
                            used_matches[source]['fuzzy'] += 1
                        else:
                            # Count as an exact match success
                            self.match_stats[source]['exact_match_successes'] += 1
                            used_matches[source]['exact'] += 1
                            
                    # Update the publication with enrichment information
                    pub.update(results[i])
                    
                enriched.append(pub)
            
            # Log which matches were actually used
            logger.debug(f"Used matches (counted in statistics): {used_matches}")
            
            return enriched