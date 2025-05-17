"""
Title matching using character n-grams.

This module provides additional matching techniques beyond standard fuzzy matching
to improve publication matching rates.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)

def generate_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from a text string.
    
    Args:
        text: Input text
        n: Size of n-grams (default: 3)
        
    Returns:
        Set of character n-grams
    """
    if not text or len(text) < n:
        return set()
    
    # Normalize text: lowercase, remove punctuation, collapse spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Generate n-grams
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Calculate similarity between two texts using character n-grams.
    
    Args:
        text1: First text
        text2: Second text
        n: Size of n-grams (default: 3)
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Generate n-grams for both texts
    ngrams1 = generate_character_ngrams(text1, n)
    ngrams2 = generate_character_ngrams(text2, n)
    
    # Calculate Jaccard similarity: intersection / union
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union > 0 else 0.0

def prefix_match_score(text1: str, text2: str, prefix_length: int = 10) -> float:
    """Calculate similarity based on shared prefix.
    
    This is particularly useful for titles that start the same but 
    might have different suffixes due to abbreviations or different formulations.
    
    Args:
        text1: First text
        text2: Second text
        prefix_length: Length of prefix to compare (default: 10)
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = re.sub(r'[^\w\s]', ' ', text1.lower())
    text1 = re.sub(r'\s+', ' ', text1).strip()
    
    text2 = re.sub(r'[^\w\s]', ' ', text2.lower())
    text2 = re.sub(r'\s+', ' ', text2).strip()
    
    # Get prefixes
    prefix1 = text1[:min(prefix_length, len(text1))]
    prefix2 = text2[:min(prefix_length, len(text2))]
    
    # Calculate similarity (Levenshtein distance would be better but using simple matching for efficiency)
    matching_chars = sum(1 for a, b in zip(prefix1, prefix2) if a == b)
    max_possible = max(len(prefix1), len(prefix2))
    
    return matching_chars / max_possible if max_possible > 0 else 0.0

def enhanced_title_similarity(title1: str, title2: str) -> Tuple[float, str]:
    """Calculate enhanced title similarity using multiple techniques.
    
    This combines n-gram similarity, prefix matching, and other techniques
    to provide a more robust similarity score for academic publication titles.
    
    Args:
        title1: First title
        title2: Second title
        
    Returns:
        Tuple of (similarity_score, matching_technique)
    """
    # Calculate similarities using different techniques
    trigram_sim = ngram_similarity(title1, title2, n=3)
    prefix_sim = prefix_match_score(title1, title2, prefix_length=15)
    
    # Return the best score and technique
    if trigram_sim > 0.75:
        return trigram_sim, "character_trigram"
    elif prefix_sim > 0.9:
        return prefix_sim, "prefix_match"
    else:
        # Return the higher of the two scores
        if trigram_sim >= prefix_sim:
            return trigram_sim, "character_trigram"
        else:
            return prefix_sim, "prefix_match"
