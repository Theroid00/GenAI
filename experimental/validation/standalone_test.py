#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Test for Data Validator
----------------------------------
This script tests the data validator in isolation, without dependencies on
the rest of the synthetic data generator system.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the data validator directly
from experimental.validation.data_validator import validate_and_fix_data

def create_test_data_with_errors():
    """
    Create a test dataset with intentional errors for validation testing.
    
    Returns:
        DataFrame with test data containing various errors
    """
    print("Creating test dataset with intentional errors...")
    
    # Sample data with various common errors
    data = {
        'id': ['ID-00001', 'not_an_id', 'ID-003', 'ID-00004', 'ID-00001'],  # Duplicate and malformed IDs
        'name': ['john smith', 'Jane Doe', 'robert JOHNSON', 'Maria Garcia-Lopez', 'Test User'],  # Capitalization issues
        'email': ['notmatchingname@example.com', 'jane.doe@gmail.com', 'rj@example.com', 'maria@gmail.com', 'jane.doe@gmail.com'],  # Non-matching and duplicate emails
        'phone': ['5551234', '(555) 123-4567', '12345', '555.123.4567', '123-456-7890'],  # Invalid formats
        'salary': [5000, 2000000, 'invalid', 75000, 85000],  # Out of range and invalid type
        'age': [17, 35, 'old', 70, 45],  # Out of range and invalid type
        'hire_date': ['2025-01-01', '2020-02-15', 'invalid', '2021-05-20', '2019-04-10'],  # Future date and invalid
        'end_date': ['2024-01-01', '2019-02-15', '2025-01-01', '2023-05-20', '2022-04-10'],  # End before start date
        'city': ['New York', 'Boston', 'New York', 'Chicago', 'Dallas'],  # Valid
        'state': ['NY', 'MA', 'CA', 'IL', 'FL']  # Geographic inconsistency (New York is in NY, not CA)
    }
    
    return pd.DataFrame(data)

def create_test_schema():
    """
    Create a schema for the test data.
    
    Returns:
        List of column schemas
    """
    return [
        {
            'name': 'id',
            'type': 'string',
            'id_type': 'generic'
        },
        {
            'name': 'name',
            'type': 'string',
            'subtype': 'full_name'
        },
        {
            'name': 'email',
            'type': 'string',
            'subtype': 'email'
        },
        {
            'name': 'phone',
            'type': 'string',
            'subtype': 'phone'
        },
        {
            'name': 'salary',
            'type': 'float',
            'min': 30000,
            'max': 150000,
            'mean': 75000
        },
        {
            'name': 'age',
            'type': 'int',
            'min': 18,
            'max': 65,
            'mean': 40
        },
        {
            'name': 'hire_date',
            'type': 'date',
            'min': '2015-01-01',
            'max': datetime.now().strftime('%Y-%m-%d')
        },
        {
            'name': 'end_date',
            'type': 'date'
        },
        {
            'name': 'city',
            'type': 'category',
            'categories': ['New York', 'Boston', 'Chicago', 'San Francisco', 'Seattle', 'Dallas']
        },
        {
            'name': 'state',
            'type': 'category',
            'categories': ['NY', 'MA', 'IL', 'CA', 'WA', 'TX']
        }
    ]

def analyze_corrections(original_df, corrected_df, corrections):
    """
    Analyze corrections made to the dataset.
    
    Args:
        original_df: Original DataFrame before validation
        corrected_df: Corrected DataFrame after validation
        corrections: List of correction messages
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_rows': len(original_df),
        'total_corrections': len(corrections),
        'changed_rows': sum(1 for i in range(len(original_df)) 
                           if not original_df.iloc[i].equals(corrected_df.iloc[i])),
        'corrections_by_column': {},
        'correction_types': {}
    }
    
    # Count corrections by column
    for col in original_df.columns:
        diff_count = sum(1 for i in range(len(original_df))
                        if original_df.at[i, col] != corrected_df.at[i, col])
        if diff_count > 0:
            analysis['corrections_by_column'][col] = diff_count
    
    # Extract correction types from messages
    for correction in corrections:
        if ' - ' in correction:
            reason = correction.split(' - ')[-1].strip()
            if reason not in analysis['correction_types']:
                analysis['correction_types'][reason] = 0
            analysis['correction_types'][reason] += 1
    
    return analysis

def main():
    """Main function to run the standalone test."""
    # Create output directory
    output_dir = 'validation_standalone_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data with errors
    df = create_test_data_with_errors()
    
    # Create schema
    schema = create_test_schema()
    
    # Save original data
    original_path = os.path.join(output_dir, 'original_data.csv')
    df.to_csv(original_path, index=False)
    print(f"Saved original test data to {original_path}")
    
    # Save schema
    schema_path = os.path.join(output_dir, 'schema.json')
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"Saved schema to {schema_path}")
    
    # Run validation
    print("\nValidating and fixing data...")
    corrected_df, corrections = validate_and_fix_data(df, schema)
    
    # Save corrected data
    corrected_path = os.path.join(output_dir, 'corrected_data.csv')
    corrected_df.to_csv(corrected_path, index=False)
    print(f"Saved corrected data to {corrected_path}")
    
    # Save corrections
    corrections_path = os.path.join(output_dir, 'corrections.json')
    with open(corrections_path, 'w') as f:
        json.dump(corrections, f, indent=2, default=str)
    print(f"Saved corrections to {corrections_path}")
    
    # Analyze corrections
    analysis = analyze_corrections(df, corrected_df, corrections)
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Print analysis
    print("\n====== Validation Results ======")
    print(f"Total rows: {analysis['total_rows']}")
    print(f"Total corrections: {analysis['total_corrections']}")
    print(f"Changed rows: {analysis['changed_rows']} ({analysis['changed_rows']/analysis['total_rows']*100:.1f}%)")
    
    print("\nCorrections by column:")
    for col, count in sorted(analysis['corrections_by_column'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {count}")
    
    print("\nCorrection types:")
    for reason, count in sorted(analysis['correction_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    # Print comparison of a few rows before and after
    print("\n====== Sample Corrections ======")
    for i in range(min(5, len(df))):
        changes = []
        for col in df.columns:
            if df.at[i, col] != corrected_df.at[i, col]:
                changes.append(f"{col}: '{df.at[i, col]}' → '{corrected_df.at[i, col]}'")
        
        if changes:
            print(f"Row {i+1}:")
            for change in changes:
                print(f"  {change}")
    
    print("\nStandalone validation test completed successfully!")

if __name__ == "__main__":
    main()
