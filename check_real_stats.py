#!/usr/bin/env python3
"""
Check the real enrichment statistics
"""
import pandas as pd

def analyze_enrichment_results():
    """Analyze the true enrichment statistics"""
    
    # Read the files
    print("Reading output files...")
    
    try:
        enriched = pd.read_csv('tests/De-identified pubs May 2025_enriched_mp.csv')
        print(f"✓ Enriched file: {len(enriched)} publications")
    except Exception as e:
        print(f"✗ Error reading enriched file: {e}")
        return
    
    try:
        failed = pd.read_csv('tests/De-identified pubs May 2025_enriched_mp_failed.csv')
        print(f"✓ Failed file: {len(failed)} publications")
    except Exception as e:
        print(f"✗ Error reading failed file: {e}")
        return
    
    try:
        fuzzy = pd.read_csv('tests/De-identified pubs May 2025_enriched_mp_fuzzy.csv')
        print(f"✓ Fuzzy file: {len(fuzzy)} publications")
    except Exception as e:
        print(f"✗ Error reading fuzzy file: {e}")
        return
    
    # Analyze enriched file
    print(f"\n=== ANALYZING ENRICHMENT QUALITY ===")
    
    # Check for abstracts
    enriched_with_abstracts = enriched[enriched['abstract'].notna() & (enriched['abstract'] != '')].shape[0]
    enriched_without_abstracts = enriched[enriched['abstract'].isna() | (enriched['abstract'] == '')].shape[0]
    
    print(f"Enriched with abstracts: {enriched_with_abstracts}")
    print(f"Enriched without abstracts: {enriched_without_abstracts}")
    
    # Check for other enrichment indicators
    enriched_with_doi = enriched[enriched['doi'].notna() & (enriched['doi'] != '')].shape[0]
    enriched_with_pmid = enriched[enriched['pmid'].notna() & (enriched['pmid'] != '')].shape[0] if 'pmid' in enriched.columns else 0
    enriched_with_authors = enriched[enriched['authors'].notna() & (enriched['authors'] != '')].shape[0] if 'authors' in enriched.columns else 0
    enriched_with_year = enriched[enriched['year'].notna()].shape[0] if 'year' in enriched.columns else 0
    
    print(f"Enriched with DOI: {enriched_with_doi}")
    print(f"Enriched with PMID: {enriched_with_pmid}")
    print(f"Enriched with authors: {enriched_with_authors}")
    print(f"Enriched with year: {enriched_with_year}")
    
    # Analyze fuzzy matches
    fuzzy_with_abstracts = fuzzy[fuzzy['abstract'].notna() & (fuzzy['abstract'] != '')].shape[0] if len(fuzzy) > 0 else 0
    
    # Calculate true enrichment rate
    total_pubs = len(enriched) + len(failed)
    
    # Count publications that have ANY enrichment (abstract, DOI, PMID, etc.)
    truly_enriched = 0
    for idx, row in enriched.iterrows():
        has_abstract = pd.notna(row['abstract']) and row['abstract'] != ''
        has_doi = pd.notna(row['doi']) and row['doi'] != ''
        has_pmid = pd.notna(row.get('pmid', '')) and row.get('pmid', '') != ''
        has_authors = pd.notna(row.get('authors', '')) and row.get('authors', '') != ''
        has_year = pd.notna(row.get('year', ''))
        
        if has_abstract or has_doi or has_pmid or has_authors or has_year:
            truly_enriched += 1
    
    # Add fuzzy matches that have enrichment
    for idx, row in fuzzy.iterrows():
        has_abstract = pd.notna(row['abstract']) and row['abstract'] != ''
        has_doi = pd.notna(row['doi']) and row['doi'] != ''
        has_pmid = pd.notna(row.get('pmid', '')) and row.get('pmid', '') != ''
        has_authors = pd.notna(row.get('authors', '')) and row.get('authors', '') != ''
        has_year = pd.notna(row.get('year', ''))
        
        if has_abstract or has_doi or has_pmid or has_authors or has_year:
            truly_enriched += 1
    
    enrichment_rate = (truly_enriched / total_pubs) * 100
    
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total publications processed: {total_pubs}")
    print(f"Publications with ANY enrichment: {truly_enriched}")
    print(f"Publications with abstracts only: {enriched_with_abstracts + fuzzy_with_abstracts}")
    print(f"Failed to enrich: {len(failed)} ({len(failed)/total_pubs*100:.1f}%)")
    print(f"True enrichment rate: {enrichment_rate:.1f}%")
    
    # Sample some failed entries to understand why they failed
    print(f"\n=== SAMPLE FAILED ENTRIES ===")
    if len(failed) > 0:
        print("Sample of failed publications:")
        for i in range(min(5, len(failed))):
            row = failed.iloc[i]
            print(f"  {i+1}. ID: {row.get('Publication_ID', 'N/A')}, Title: '{row.get('title', 'N/A')[:60]}...', DOI: {row.get('doi', 'N/A')}")
    
    # Sample some enriched entries
    print(f"\n=== SAMPLE ENRICHED ENTRIES ===")
    enriched_sample = enriched[enriched['abstract'].notna() & (enriched['abstract'] != '')].head(3)
    for i, (idx, row) in enumerate(enriched_sample.iterrows()):
        abstract_preview = row['abstract'][:100] + "..." if len(row['abstract']) > 100 else row['abstract']
        print(f"  {i+1}. ID: {row.get('Publication_ID', 'N/A')}, Abstract: '{abstract_preview}'")
    
    return {
        'total': total_pubs,
        'enriched': truly_enriched,
        'failed': len(failed),
        'enrichment_rate': enrichment_rate
    }

if __name__ == '__main__':
    analyze_enrichment_results()