#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Synthetic Data Generator with Validation
--------------------------------------------------
This script combines the synthetic data generator with the data validation module
to produce high-quality, validated synthetic data.
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Dict, Any, List, Tuple

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

# Import from validation module
from experimental.validation.data_validator import validate_and_fix_data

# Import field parser for special field handling
try:
    from field_parser import apply_field_parsing
    FIELD_PARSER_AVAILABLE = True
except ImportError:
    FIELD_PARSER_AVAILABLE = False

def generate_and_validate(
    input_path: str = None, 
    schema_path: str = None,
    output_path: str = None,
    num_rows: int = 100,
    model_type: str = 'gaussian',
    validate: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic data and validate it using the data validator.
    
    Args:
        input_path: Path to input CSV (optional)
        schema_path: Path to schema JSON (optional)
        output_path: Path to save the output CSV
        num_rows: Number of rows to generate
        model_type: Model type for model-based generation
        validate: Whether to validate and fix the generated data
        
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
    
    # Validate and fix data if requested
    if validate:
        print("Validating and fixing data...")
        corrected_df, corrections = validate_and_fix_data(synthetic_df, schema)
        
        # Print validation results
        print(f"\n=== Validation Results ===")
        print(f"Total corrections: {len(corrections)}")
        
        # Group corrections by type
        correction_types = {}
        for correction in corrections:
            if ' - ' in correction:
                reason = correction.split(' - ')[-1].strip()
                if reason not in correction_types:
                    correction_types[reason] = 0
                correction_types[reason] += 1
        
        # Print corrections by type
        if correction_types:
            print("\nCorrections by type:")
            for reason, count in sorted(correction_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")
        
        # Use the corrected data
        synthetic_df = corrected_df
    
    # Apply field parsing for special cases
    if FIELD_PARSER_AVAILABLE:
        synthetic_df = apply_field_parsing(synthetic_df)
    
    # Save to file if output path provided
    if output_path:
        print(f"Saving synthetic data to {output_path}...")
        save_to_csv(synthetic_df, output_path)
    
    return synthetic_df

def main():
    """Main function to parse arguments and run the generation+validation"""
    parser = argparse.ArgumentParser(
        description="Generate and validate synthetic data"
    )
    
    parser.add_argument(
        "--input", "-i", type=str, help="Input CSV file (for model-based generation)"
    )
    parser.add_argument(
        "--schema", "-s", type=str, help="Schema JSON file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="synthetic_data_validated.csv",
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
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation step"
    )
    
    args = parser.parse_args()
    
    # Check if we have the required input
    if not args.input and not args.schema:
        parser.error("Either --input or --schema must be provided")
    
    # Generate and validate
    try:
        synthetic_df = generate_and_validate(
            input_path=args.input,
            schema_path=args.schema,
            output_path=args.output,
            num_rows=args.rows,
            model_type=args.model,
            validate=not args.no_validate
        )
        
        print(f"\nGenerated {len(synthetic_df)} rows of synthetic data")
        print(f"Data preview:")
        print(synthetic_df.head())
        
        print(f"\nSynthetic data generation and validation completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
