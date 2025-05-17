#!/usr/bin/env python3
"""
Script to create a subset of the large publication dataset for testing.
"""
import pandas as pd
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a subset of publications for testing")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output-file", 
        help="Path to the output CSV file (default: input_file_subset.csv)",
        default=None
    )
    parser.add_argument(
        "--percentage", 
        type=float, 
        default=50.0,
        help="Percentage of data to include (default: 50.0)"
    )
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set output filename if not provided
    if not args.output_file:
        input_base = args.input_file.rsplit('.', 1)[0]
        args.output_file = f"{input_base}_subset_{int(args.percentage)}pct.csv"
    
    # Read the full dataset
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file)
    total_pubs = len(df)
    print(f"Total publications: {total_pubs}")
    
    # Calculate sample size
    sample_size = int(total_pubs * args.percentage / 100)
    print(f"Creating subset with {sample_size} publications ({args.percentage}%)")
    
    # Sample the data
    subset = df.sample(n=sample_size, random_state=args.random_seed)
    
    # Save the subset
    subset.to_csv(args.output_file, index=False)
    print(f"Subset saved to: {args.output_file}")
    
if __name__ == "__main__":
    main()
