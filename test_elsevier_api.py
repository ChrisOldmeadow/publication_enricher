#!/usr/bin/env python3
"""
Simple script to test Elsevier API key and permissions.
"""
import os
import requests
from dotenv import load_dotenv

def test_elsevier_api():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.environ.get('ELSEVIER_API_KEY')
    if not api_key:
        print("ERROR: ELSEVIER_API_KEY not found in environment variables")
        return
    
    print(f"Testing Elsevier API key: {api_key[:4]}...{api_key[-4:]} (middle part hidden)")
    
    # Test simple article search - this is a basic endpoint that should work with any valid key
    url = "https://api.elsevier.com/content/search/scopus"
    params = {
        'query': 'cancer',
        'count': 1
    }
    headers = {
        'X-ELS-APIKey': api_key,
        'Accept': 'application/json'
    }
    
    scopus_ok = False
    print("\nMaking test request to Scopus search API...")
    try:
        response = requests.get(url, params=params, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("SUCCESS: API key is valid and has permission to access this endpoint")
            data = response.json()
            if 'search-results' in data:
                total = data['search-results'].get('opensearch:totalResults', '0')
                print(f"Found {total} results")
            scopus_ok = True
        elif response.status_code == 401:
            print("ERROR: Unauthorized - API key may be invalid or expired")
            print(response.text)
        elif response.status_code == 403:
            print("ERROR: Forbidden - API key doesn't have permission to access this endpoint")
            print(response.text)
        elif response.status_code == 429:
            print("ERROR: Rate limit exceeded - too many requests")
            print(response.text)
        else:
            print(f"ERROR: Unexpected response code: {response.status_code}")
            print(response.text)
        
    except Exception as e:
        print(f"ERROR: Exception occurred during API test: {str(e)}")
    
    # Test article retrieval by DOI
    print("\nTesting article retrieval by DOI...")
    doi = "10.1016/j.cell.2016.01.027"  # Example DOI for a well-known paper
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    
    doi_ok = False
    try:
        response = requests.get(url, headers=headers)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("SUCCESS: API key can retrieve full article metadata")
            doi_ok = True
        elif response.status_code == 401:
            print("ERROR: Unauthorized - API key may be invalid or expired")
            print(response.text)
        elif response.status_code == 403:
            print("ERROR: Forbidden - Your API key doesn't have permission to access full article content")
            print("This usually means your institution doesn't have a subscription or your API key needs additional entitlements")
            print(response.text)
        else:
            print(f"ERROR: Unexpected response code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"ERROR: Exception occurred during DOI test: {str(e)}")
    
    # Return overall status
    print("\nSummary:")
    print(f"- Scopus Search API: {'OK' if scopus_ok else 'FAILED'}")
    print(f"- Article Retrieval by DOI: {'OK' if doi_ok else 'FAILED'}")
    
    return scopus_ok and doi_ok

if __name__ == "__main__":
    test_elsevier_api()
