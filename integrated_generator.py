#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Synthetic Data Generator
--------------------------------------------------
This script provides a simple interface to generate synthetic data
using different modes and applying field parsing.
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
import random
from faker import Faker

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from synthetic data generator
from synthetic_data_generator import (
    generate_synthetic_data_from_schema,
    generate_synthetic_data_model_based,
    load_csv,
    save_to_csv,
    post_process_synthetic_data,
    infer_schema
)

# Import field parser for special field handling
try:
    from field_parser import apply_field_parsing
    FIELD_PARSER_AVAILABLE = True
except ImportError:
    FIELD_PARSER_AVAILABLE = False

def generate_diverse_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate diverse names for name columns with high repetition.
    
    Args:
        df: DataFrame with potentially repetitive names
        
    Returns:
        DataFrame with more diverse names
    """
    result_df = df.copy()
    
    # Check for name columns with high repetition
    for col in df.columns:
        col_lower = col.lower()
        if 'name' in col_lower and 'id' not in col_lower:
            unique_names_count = df[col].nunique()
            total_rows = len(df)
            
            # If there's high repetition (less than 10% unique values), replace with more diverse names
            if unique_names_count < total_rows * 0.1 and total_rows > 100:
                print(f"Detected high name repetition in column {col}. Generating more diverse names...")
                
                # Create a Faker instance with different locales
                fake_multi = Faker(['en_US', 'en_GB', 'en_CA', 'en_AU'])
                
                # Generate diverse names based on context
                if 'first' in col_lower:
                    result_df[col] = [fake_multi.first_name() for _ in range(total_rows)]
                elif 'last' in col_lower:
                    result_df[col] = [fake_multi.last_name() for _ in range(total_rows)]
                else:
                    # Generate diverse full names
                    diverse_names = []
                    for _ in range(total_rows):
                        name_type = random.choices([1, 2, 3], weights=[70, 15, 15], k=1)[0]
                        
                        if name_type == 1:  # 70% standard names
                            name = fake_multi.name()
                        elif name_type == 2:  # 15% names with middle initial
                            first = fake_multi.first_name()
                            middle = fake_multi.first_name()[0] + "."
                            last = fake_multi.last_name()
                            name = f"{first} {middle} {last}"
                        else:  # 15% names with hyphenated components
                            if random.random() < 0.5:
                                first = f"{fake_multi.first_name()}-{fake_multi.first_name()}"
                                last = fake_multi.last_name()
                            else:
                                first = fake_multi.first_name()
                                last = f"{fake_multi.last_name()}-{fake_multi.last_name()}"
                            name = f"{first} {last}"
                        
                        diverse_names.append(name)
                    
                    result_df[col] = diverse_names
    
    return result_df

def generate_and_process(
    input_path: str = None, 
    schema_path: str = None,
    output_path: str = None,
    num_rows: int = 100,
    model_type: str = 'gaussian'
) -> pd.DataFrame:
    """
    Generate synthetic data and apply field parsing.
    
    Args:
        input_path: Path to input CSV (optional)
        schema_path: Path to schema JSON (optional)
        output_path: Path to save the output CSV
        num_rows: Number of rows to generate
        model_type: Model type for model-based generation
        
    Returns:
        DataFrame with synthetic data
    """
    synthetic_df = None
    schema = None
    
    # Determine generation mode
    if input_path and os.path.exists(input_path):
        # Mode 1: Generate from CSV sample
        print(f"Loading CSV from {input_path}...")
        original_df = load_csv(input_path)
        
        # Infer schema if not provided
        if not schema_path:
            schema_dict = infer_schema(original_df)
            schema = [
                dict(info, name=col) for col, info in schema_dict.items()
            ]
        
        # Generate synthetic data
        print(f"Generating {num_rows} rows of synthetic data from CSV sample...")
        synthetic_df = generate_synthetic_data_model_based(original_df, num_rows, model_type)
        
    elif schema_path and os.path.exists(schema_path):
        # Mode 3: Generate from schema file
        print(f"Loading schema from {schema_path}...")
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Generate synthetic data
        print(f"Generating {num_rows} rows of synthetic data from schema...")
        synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
        synthetic_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)
    else:
        raise ValueError("Either input_path or schema_path must be provided")
    
    # Apply field parsing for special cases
    if FIELD_PARSER_AVAILABLE:
        print("Applying field parsing rules...")
        synthetic_df = apply_field_parsing(synthetic_df)
    
    # Generate diverse names if applicable
    synthetic_df = generate_diverse_names(synthetic_df)
    
    # Save to file if output path provided
    if output_path:
        print(f"Saving synthetic data to {output_path}...")
        save_to_csv(synthetic_df, output_path)
    
    return synthetic_df

def main():
    """Main function to parse arguments and run the generator"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data"
    )
    
    parser.add_argument(
        "--input", "-i", type=str, help="Input CSV file (for model-based generation)"
    )
    parser.add_argument(
        "--schema", "-s", type=str, help="Schema JSON file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="synthetic_data.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--rows", "-r", type=int, default=100,
        help="Number of rows to generate"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gaussian",
        choices=["gaussian", "ctgan"],
        help="Model type for model-based generation"
    )
    
    args = parser.parse_args()
    
    # Check if we have the required input
    if not args.input and not args.schema:
        parser.error("Either --input or --schema must be provided")
    
    # Generate synthetic data
    try:
        synthetic_df = generate_and_process(
            input_path=args.input,
            schema_path=args.schema,
            output_path=args.output,
            num_rows=args.rows,
            model_type=args.model
        )
        
        print(f"\nGenerated {len(synthetic_df)} rows of synthetic data")
        print(f"Data preview:")
        print(synthetic_df.head())
        
        print(f"\nSynthetic data generation completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
