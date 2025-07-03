#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the data validator module.
This script loads synthetic data from a file, adds intentional errors,
and then shows how the validator fixes those errors.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import validator
from data_validator import validate_and_fix_data, DataValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation_test')

def create_test_data(num_rows=100):
    """Create test data with intentional errors."""
    logger.info(f"Creating test dataset with {num_rows} rows...")
    
    # Create base data
    data = {
        'emp_id': [f"EMP-{10001 + i}" for i in range(num_rows)],
        'name': [f"Test User {i}" for i in range(num_rows)],
        'email': [f"test.user{i}@example.com" for i in range(num_rows)],
        'phone': [f"(555) 123-{1000 + i}" for i in range(num_rows)],
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], num_rows),
        'salary': np.random.uniform(50000, 120000, num_rows),
        'join_date': [datetime(2020, 1, 1) + pd.Timedelta(days=i*10) for i in range(num_rows)],
        'city': np.random.choice(['New York', 'Boston', 'Chicago', 'San Francisco', 'Seattle'], num_rows),
        'state': np.random.choice(['NY', 'MA', 'IL', 'CA', 'WA'], num_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce errors
    logger.info("Introducing errors into the dataset...")
    
    # 1. Invalid ID formats
    error_indices = np.random.choice(num_rows, size=10, replace=False)
    for idx in error_indices:
        df.at[idx, 'emp_id'] = f"BAD-ID-{idx}"
    
    # 2. Name capitalization errors
    error_indices = np.random.choice(num_rows, size=10, replace=False)
    for idx in error_indices:
        df.at[idx, 'name'] = df.at[idx, 'name'].lower()
    
    # 3. Email-name mismatches
    error_indices = np.random.choice(num_rows, size=10, replace=False)
    for idx in error_indices:
        df.at[idx, 'email'] = f"wrong{idx}@example.com"
    
    # 4. Invalid phone formats
    error_indices = np.random.choice(num_rows, size=10, replace=False)
    for idx in error_indices:
        df.at[idx, 'phone'] = f"BAD-PHONE-{idx}"
    
    # 5. Salary outliers
    error_indices = np.random.choice(num_rows, size=5, replace=False)
    for idx in error_indices:
        df.at[idx, 'salary'] = 500000  # Unrealistically high
    
    error_indices = np.random.choice(num_rows, size=5, replace=False)
    for idx in error_indices:
        df.at[idx, 'salary'] = 10000  # Unrealistically low
    
    # 6. Geographic inconsistencies
    error_indices = np.random.choice(num_rows, size=10, replace=False)
    for idx in error_indices:
        if df.at[idx, 'city'] == 'New York':
            df.at[idx, 'state'] = 'CA'
        elif df.at[idx, 'city'] == 'Boston':
            df.at[idx, 'state'] = 'TX'
        elif df.at[idx, 'city'] == 'Chicago':
            df.at[idx, 'state'] = 'FL'
        elif df.at[idx, 'city'] == 'San Francisco':
            df.at[idx, 'state'] = 'NY'
        elif df.at[idx, 'city'] == 'Seattle':
            df.at[idx, 'state'] = 'MA'
    
    # 7. Duplicate values in unique columns
    error_indices = np.random.choice(num_rows, size=5, replace=False)
    for i, idx in enumerate(error_indices):
        if i > 0:
            df.at[idx, 'emp_id'] = df.at[error_indices[0], 'emp_id']
    
    error_indices = np.random.choice(num_rows, size=5, replace=False)
    for i, idx in enumerate(error_indices):
        if i > 0:
            df.at[idx, 'email'] = df.at[error_indices[0], 'email']
    
    # 8. Non-numeric values in numeric fields
    error_indices = np.random.choice(num_rows, size=5, replace=False)
    for idx in error_indices:
        df.at[idx, 'salary'] = 'NOT A NUMBER'
    
    logger.info(f"Created test dataset with intentional errors")
    return df

def create_test_schema():
    """Create a test schema for validation."""
    schema = [
        {
            'name': 'emp_id',
            'type': 'uuid'
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
            'name': 'department',
            'type': 'category',
            'categories': ['Engineering', 'Marketing', 'Sales', 'HR']
        },
        {
            'name': 'salary',
            'type': 'float',
            'min': 30000,
            'max': 200000
        },
        {
            'name': 'join_date',
            'type': 'date',
            'start_date': '2020-01-01',
            'end_date': '2023-12-31'
        },
        {
            'name': 'city',
            'type': 'category',
            'categories': ['New York', 'Boston', 'Chicago', 'San Francisco', 'Seattle']
        },
        {
            'name': 'state',
            'type': 'category',
            'categories': ['NY', 'MA', 'IL', 'CA', 'WA']
        }
    ]
    
    return schema

def main():
    """Main function to test the data validator."""
    # Create directory for test outputs
    output_dir = 'test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data
    df = create_test_data(100)
    
    # Save original data
    original_path = os.path.join(output_dir, 'original_data.csv')
    df.to_csv(original_path, index=False)
    logger.info(f"Saved original data with errors to {original_path}")
    
    # Create test schema
    schema = create_test_schema()
    
    # Save schema
    schema_path = os.path.join(output_dir, 'test_schema.json')
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    logger.info(f"Saved test schema to {schema_path}")
    
    # Run validation
    logger.info("Running data validation...")
    validator = DataValidator(df, schema)
    corrected_df = validator.validate_and_fix()
    corrections = validator.get_correction_log()
    
    # Save corrected data
    corrected_path = os.path.join(output_dir, 'corrected_data.csv')
    corrected_df.to_csv(corrected_path, index=False)
    logger.info(f"Saved corrected data to {corrected_path}")
    
    # Save correction log
    log_path = os.path.join(output_dir, 'correction_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Validation Corrections ({len(corrections)} total):\n")
        f.write("="*50 + "\n\n")
        for correction in corrections:
            f.write(f"{correction}\n")
    logger.info(f"Saved correction log to {log_path}")
    
    # Generate validation report
    report = validator.get_validation_report()
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved validation report to {report_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("DATA VALIDATION TEST SUMMARY")
    print("="*50)
    print(f"Total corrections made: {len(corrections)}")
    
    # Print corrections by type
    print("\nCorrections by type:")
    for type_, count in report['corrections_by_type'].items():
        print(f"  {type_}: {count}")
    
    # Print corrections by column
    print("\nCorrections by column:")
    for col, count in report['corrections_by_column'].items():
        print(f"  {col}: {count}")
    
    print("\nTest completed successfully!")
    print(f"All test outputs saved to the '{output_dir}' directory")

if __name__ == "__main__":
    main()
