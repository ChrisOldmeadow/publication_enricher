#!/usr/bin/env python3
"""
Script to manually aggregate checkpoint files into final output
"""
import json
import pandas as pd
import glob
from pathlib import Path

def aggregate_checkpoints():
    """Aggregate all checkpoint files into final CSV files"""
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob('checkpoint_chunk_*.json'))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    all_publications = []
    enriched_pubs = []
    failed_pubs = []
    fuzzy_pubs = []
    
    for checkpoint_file in checkpoint_files:
        print(f"Processing {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                
            processed = data.get('processed', [])
            print(f"  Found {len(processed)} publications")
            
            for pub in processed:
                all_publications.append(pub)
                
                # Categorize publications
                if pub.get('abstract') or pub.get('doi') or pub.get('title') != pub.get('original_query_title', pub.get('title')):
                    # This publication was enriched
                    match_info = pub.get('match_info', {})
                    if match_info.get('fuzzy_matched', False):
                        fuzzy_pubs.append(pub)
                    else:
                        enriched_pubs.append(pub)
                else:
                    # This publication failed to enrich
                    failed_pubs.append(pub)
                    
        except Exception as e:
            print(f"  Error processing {checkpoint_file}: {e}")
    
    print(f"\nTotal publications processed: {len(all_publications)}")
    print(f"Successfully enriched: {len(enriched_pubs)}")
    print(f"Fuzzy matches: {len(fuzzy_pubs)}")
    print(f"Failed to enrich: {len(failed_pubs)}")
    
    # Create output files
    base_name = "tests/De-identified pubs May 2025_enriched_mp"
    
    # Main enriched file (enriched + fuzzy)
    if enriched_pubs or fuzzy_pubs:
        all_enriched = enriched_pubs + fuzzy_pubs
        df_enriched = pd.DataFrame(all_enriched)
        enriched_file = f"{base_name}.csv"
        df_enriched.to_csv(enriched_file, index=False)
        print(f"Created {enriched_file} with {len(all_enriched)} publications")
    
    # Failed file
    if failed_pubs:
        df_failed = pd.DataFrame(failed_pubs)
        failed_file = f"{base_name}_failed.csv"
        df_failed.to_csv(failed_file, index=False)
        print(f"Updated {failed_file} with {len(failed_pubs)} publications")
    
    # Fuzzy file
    if fuzzy_pubs:
        df_fuzzy = pd.DataFrame(fuzzy_pubs)
        fuzzy_file = f"{base_name}_fuzzy.csv"
        df_fuzzy.to_csv(fuzzy_file, index=False)
        print(f"Updated {fuzzy_file} with {len(fuzzy_pubs)} publications")
    
    # Summary statistics
    print(f"\n=== FINAL STATISTICS ===")
    print(f"Total publications: {len(all_publications)}")
    print(f"Successfully enriched: {len(enriched_pubs)} ({len(enriched_pubs)/len(all_publications)*100:.1f}%)")
    print(f"Fuzzy matches: {len(fuzzy_pubs)} ({len(fuzzy_pubs)/len(all_publications)*100:.1f}%)")
    print(f"Failed to enrich: {len(failed_pubs)} ({len(failed_pubs)/len(all_publications)*100:.1f}%)")
    
    # Check abstracts with HTML
    html_count = 0
    clean_count = 0
    for pub in enriched_pubs + fuzzy_pubs:
        abstract = pub.get('abstract', '')
        if abstract:
            if '<' in abstract and '>' in abstract:
                html_count += 1
            else:
                clean_count += 1
    
    print(f"\n=== HTML CLEANING CHECK ===")
    print(f"Abstracts with HTML tags: {html_count}")
    print(f"Clean abstracts: {clean_count}")
    
    return len(all_publications), len(enriched_pubs), len(fuzzy_pubs), len(failed_pubs)

if __name__ == '__main__':
    aggregate_checkpoints()