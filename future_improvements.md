# Future Improvements for Publication Enricher

This document outlines potential strategies to further improve the match rates and functionality of the publication enricher tool.

## Match Rate Enhancement Strategies

### 1. Cascading Threshold Approach

Implement a dynamic fuzzy matching threshold strategy:

- Start with a high threshold (90+) for highest quality matches
- If no match is found, progressively lower the threshold while increasing scrutiny
- For lower thresholds (below 80), only accept matches with supporting evidence (author match, year match)
- Example implementation:

```python
def cascading_threshold_match(title, candidates, authors=None, year=None):
    # Try with high threshold first
    match = find_best_title_match(title, candidates, threshold=90, 
                                 search_authors=authors, search_year=year)
    if match:
        return match, "high_confidence"
        
    # Try with medium threshold + stricter validation
    match = find_best_title_match(title, candidates, threshold=80, 
                                 search_authors=authors, search_year=year) 
    if match and (author_match_score(authors, match.get('authors', [])) > 0.5):
        return match, "medium_confidence"
        
    # Try with lower threshold + very strict validation
    match = find_best_title_match(title, candidates, threshold=70, 
                                 search_authors=authors, search_year=year)
    if match and author_match_score(authors, match.get('authors', [])) > 0.7 and year == match.get('year'):
        return match, "low_confidence"
        
    return None, None
```

### 2. Leverage Journal Information

Use journal names as an additional matching criterion:

- Compare normalized journal names between query and candidates
- Add confidence bonus for journals that match or are similar
- Account for journal abbreviations and variations
- Example implementation:

```python
# Add journal name matching to boost confidence
if publication.get('journal') and candidate.get('journal'):
    norm_pub_journal = normalize_journal_name(publication['journal'])
    norm_cand_journal = normalize_journal_name(candidate['journal'])
    
    if norm_pub_journal == norm_cand_journal:
        # Same journal - significant boost
        adjusted_score += 10
    elif fuzz.partial_ratio(norm_pub_journal, norm_cand_journal) > 80:
        # Similar journal - moderate boost
        adjusted_score += 5
```

### 3. Parse Publications with Missing DOIs

For publications with no DOI but other identifying information:

- Implement parsers for common citation formats
- Extract potential identifiers like PMIDs, ISBNs, etc.
- Use regular expressions to find embedded DOIs in title or abstract text
- Example implementation:

```python
def extract_identifiers_from_citation(citation_text):
    """Extract potential DOIs, PMIDs, and other identifiers from citation text"""
    # Look for DOI patterns
    doi_match = re.search(r'10\.\d{4,}[\w\d\.\-\/]+', citation_text)
    if doi_match:
        return {'doi': doi_match.group(0)}
    
    # Look for PMID patterns
    pmid_match = re.search(r'PMID:\s*(\d+)', citation_text)
    if pmid_match:
        return {'pmid': pmid_match.group(1)}
        
    # Extract potential journal, volume, page information
    journal_match = re.search(r'([A-Za-z\s&]+)[.,]\s*(\d{4})\s*[;:]\s*(\d+)\s*[(:].*?(\d+)[-–](\d+)', citation_text)
    if journal_match:
        return {
            'journal': journal_match.group(1).strip(),
            'year': journal_match.group(2),
            'volume': journal_match.group(3),
            'start_page': journal_match.group(4),
            'end_page': journal_match.group(5)
        }
```

### 4. Add Google Scholar as a Fallback API

Implement a Google Scholar search capability:

- Use as a last resort when other APIs fail
- Incorporate appropriate rate limiting and proxy rotation
- Handle anti-scraping measures with stealth techniques
- Example implementation:

```python
async def search_google_scholar(title, session, proxy=None, delay=5.0):
    """Search Google Scholar as a last resort (with appropriate rate limiting)"""
    # Use rotating proxies and appropriate delays
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://scholar.google.com/'
    }
    
    # Add random delay to avoid detection
    await asyncio.sleep(delay + random.uniform(1.0, 3.0))
    
    # Use scholarly library or direct API calls with appropriate parsing
    url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}&hl=en"
    
    proxy_options = {}
    if proxy:
        proxy_options['proxy'] = proxy
    
    async with session.get(url, headers=headers, **proxy_options) as response:
        if response.status == 200:
            html = await response.text()
            # Parse HTML response to extract publication info
            # ...
```

### 5. Title Abbreviation Expansion

Handle academic abbreviations in titles:

- Create dictionary of common academic abbreviations
- Expand abbreviated forms for better matching
- Handle discipline-specific terminology
- Example implementation:

```python
def expand_common_abbreviations(title):
    """Expand common academic abbreviations"""
    replacements = {
        'intl': 'international',
        'natl': 'national',
        'exp': 'experimental',
        'dev': 'development',
        'bio': 'biology',
        'clin': 'clinical',
        'psych': 'psychology',
        'J': 'Journal',
        'Am': 'American',
        'Assoc': 'Association',
        'Soc': 'Society',
        'Acad': 'Academy',
        'Sci': 'Science',
        'Med': 'Medicine',
        'Res': 'Research',
        'Eng': 'Engineering'
    }
    
    # Apply the replacements
    for abbr, full in replacements.items():
        title = re.sub(r'\b' + abbr + r'\b', full, title, flags=re.IGNORECASE)
    
    return title
```

### 6. Alternative Matching Algorithms

Implement additional string similarity algorithms:

- Cosine similarity with TF-IDF weights for term importance
- Jaccard similarity for word set overlap
- Word embeddings similarity using NLP models
- Example implementation:

```python
def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def calculate_jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between word sets"""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0
```

### 7. Pre-processing Failed Matches for Special Handling

Create a system to analyze failed matches and identify specific patterns:

- Categorize failed matches by common characteristics
- Develop specialized handling for each category
- Generate insights for further improvements
- Example implementation:

```python
def analyze_failed_matches(failed_df):
    """Analyze failed matches for common patterns"""
    patterns = {
        'contains_special_chars': r'[^\x00-\x7F]+',
        'very_short_title': r'^.{1,15}$',
        'all_uppercase': r'^[A-Z\s]+$',
        'contains_formula': r'[A-Za-z][0-9]|[0-9][A-Za-z]',
        'title_with_question': r'\?',
        'contains_numbers': r'\d+'
    }
    
    for pattern_name, regex in patterns.items():
        failed_df[pattern_name] = failed_df['title'].apply(
            lambda x: bool(re.search(regex, x)) if isinstance(x, str) else False
        )
    
    return failed_df.groupby(list(patterns.keys())).size().reset_index()
```

## Performance and Efficiency Improvements

### 1. Cache Optimization

- Implement more sophisticated caching with prioritized eviction
- Add cross-request cache sharing for similar titles
- Pre-cache commonly searched terms

### 2. Parallel Processing Enhancement

- Optimize worker distribution for multi-processing
- Implement adaptive batch sizing based on system capabilities
- Add progress estimation based on title complexity

### 3. API-Specific Optimizations

- Customize queries for each API's strengths
- Implement API-specific fallback strategies
- Develop specialized parsers for each API's response format

## User Experience Improvements

### 1. Interactive Mode

- Add an interactive mode to resolve difficult matches
- Allow users to choose between multiple potential matches
- Implement a simple web interface for reviewing ambiguous cases

### 2. Enhanced Reporting

- Generate detailed match quality reports
- Provide visualizations of match score distributions
- Add confidence metrics for each matched publication

### 3. Configuration System

- Add a configuration file for customizing matching parameters
- Allow users to prioritize APIs based on their needs
- Implement domain-specific settings for different academic fields

## Conclusion

These improvements would further enhance the publication enricher's ability to match publications across various data sources. Implementation priority should be based on the specific use cases and current performance bottlenecks.
