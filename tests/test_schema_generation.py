#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator - Schema Test
--------------------------------------
This script tests the synthetic data generator with a schema file to ensure it's
generating standardized field formats.
"""

import json
import pandas as pd
from synthetic_data_generator import (
    load_schema_from_json,
    generate_synthetic_data_from_schema,
    post_process_synthetic_data,
    save_to_csv
)

def main():
    """Main function to test schema-based generation"""
    print("=== Testing Schema-Based Generation ===")
    
    # Load schema
    schema_path = "test_schema.json"
    print(f"Loading schema from {schema_path}...")
    
    with open(schema_path, 'r') as f:
        schema_data = json.load(f)
    
    schema = schema_data["schema"]
    
    # Generate data
    num_rows = 100
    print(f"Generating {num_rows} rows with {len(schema)} columns...")
    
    synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
    
    # Apply post-processing
    print("Applying post-processing to make data more realistic...")
    processed_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)
    
    # Display sample rows
    print("\nSample data (first 5 rows):")
    print(processed_df.head())
    
    # Save to CSV
    output_path = "test_schema_output.csv"
    save_to_csv(processed_df, output_path)
    print(f"\nTest data saved to {output_path}")

if __name__ == "__main__":
    main()
