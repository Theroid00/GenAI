#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration Test for Field Standards and Data Validator
------------------------------------------------------
This script demonstrates how to use the field standards module together 
with the data validator to generate and validate synthetic data.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import json
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import field standards and validator
from field_standards import (
    detect_id_field_type,
    generate_id,
    format_email,
    format_phone_number,
    get_weighted_email_domain,
    EMAIL_DOMAINS
)

# Import the validation module from the same directory
from experimental.validation.data_validator import validate_and_fix_data

def generate_synthetic_data(num_rows=100, error_rate=0.0):
    """
    Generate synthetic data with optional intentional errors.
    
    Args:
        num_rows: Number of rows to generate
        error_rate: Percentage of data that should contain errors (0.0 to 1.0)
        
    Returns:
        DataFrame with synthetic data
    """
    print(f"Generating {num_rows} rows of synthetic data...")
    
    # Generate employee IDs
    employee_ids = [generate_id('employee', i) for i in range(num_rows)]
    
    # Generate names
    first_names = [f"First{i}" for i in range(num_rows)]
    last_names = [f"Last{i}" for i in range(num_rows)]
    full_names = [f"{first} {last}" for first, last in zip(first_names, last_names)]
    
    # Generate emails based on names
    emails = [format_email(first, last) for first, last in zip(first_names, last_names)]
    
    # Generate phone numbers
    phones = [format_phone_number() for _ in range(num_rows)]
    
    # Generate departments and job titles
    departments = np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], num_rows)
    
    job_titles = []
    for dept in departments:
        if dept == 'Engineering':
            job_titles.append(np.random.choice(['Software Engineer', 'QA Engineer', 'DevOps Engineer', 'Engineering Manager']))
        elif dept == 'Marketing':
            job_titles.append(np.random.choice(['Marketing Specialist', 'Content Writer', 'SEO Analyst', 'Marketing Manager']))
        elif dept == 'Sales':
            job_titles.append(np.random.choice(['Sales Representative', 'Account Executive', 'Sales Manager', 'Business Developer']))
        elif dept == 'HR':
            job_titles.append(np.random.choice(['HR Specialist', 'Recruiter', 'HR Manager', 'Talent Acquisition']))
        elif dept == 'Finance':
            job_titles.append(np.random.choice(['Accountant', 'Financial Analyst', 'Finance Manager', 'Payroll Specialist']))
    
    # Generate salaries based on department and job title
    salaries = []
    for dept, title in zip(departments, job_titles):
        base = 0
        if 'Manager' in title:
            base = 90000
        elif 'Engineer' in title or 'Developer' in title or 'Analyst' in title:
            base = 80000
        elif 'Specialist' in title or 'Representative' in title or 'Executive' in title:
            base = 70000
        else:
            base = 60000
            
        # Add department adjustment
        if dept == 'Engineering':
            base += 10000
        elif dept == 'Finance':
            base += 5000
        
        # Add random variation
        salary = int(base + np.random.normal(0, 5000))
        salaries.append(salary)
    
    # Generate hire dates
    start_date = datetime(2018, 1, 1)
    hire_dates = [start_date + timedelta(days=random.randint(0, 365*4)) for _ in range(num_rows)]
    
    # Generate manager IDs (every 5th employee is a manager)
    manager_indices = list(range(0, num_rows, 5))
    manager_ids = []
    for i in range(num_rows):
        if i in manager_indices:
            manager_ids.append(None)  # Managers don't have managers
        else:
            # Assign to a random manager
            manager_idx = random.choice(manager_indices)
            manager_ids.append(employee_ids[manager_idx])
    
    # Create the DataFrame
    data = {
        'employee_id': employee_ids,
        'first_name': first_names,
        'last_name': last_names,
        'full_name': full_names,
        'email': emails,
        'phone': phones,
        'department': departments,
        'job_title': job_titles,
        'salary': salaries,
        'hire_date': hire_dates,
        'manager_id': manager_ids
    }
    
    df = pd.DataFrame(data)
    
    # Introduce errors if requested
    if error_rate > 0:
        df = introduce_errors(df, error_rate)
    
    return df

def introduce_errors(df, error_rate):
    """
    Introduce various types of errors into the dataset.
    
    Args:
        df: DataFrame to modify
        error_rate: Percentage of data to corrupt (0.0 to 1.0)
        
    Returns:
        DataFrame with introduced errors
    """
    print(f"Introducing errors at a rate of {error_rate*100}%...")
    num_rows = len(df)
    errors_per_type = int(num_rows * error_rate / 6)  # 6 error types
    
    # 1. Invalid ID formats
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        df.at[idx, 'employee_id'] = f"INVALID-ID-{idx}"
    
    # 2. Email-name mismatches
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        df.at[idx, 'email'] = f"unrelated{idx}@{random.choice(list(EMAIL_DOMAINS.keys()))}"
    
    # 3. Invalid phone formats
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        df.at[idx, 'phone'] = f"{random.randint(10000, 9999999)}"
    
    # 4. Salary outliers
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        multiplier = random.choice([0.1, 10])  # Either too low or too high
        df.at[idx, 'salary'] = int(df.at[idx, 'salary'] * multiplier)
    
    # 5. Invalid manager IDs (references to non-existent employees)
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        df.at[idx, 'manager_id'] = f"EMP-{random.randint(100000, 999999)}"
    
    # 6. Job title - department mismatches
    error_indices = np.random.choice(num_rows, errors_per_type, replace=False)
    for idx in error_indices:
        current_dept = df.at[idx, 'department']
        # Assign a job title from a different department
        other_depts = [d for d in ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'] if d != current_dept]
        wrong_dept = random.choice(other_depts)
        
        if wrong_dept == 'Engineering':
            df.at[idx, 'job_title'] = random.choice(['Software Engineer', 'QA Engineer', 'DevOps Engineer'])
        elif wrong_dept == 'Marketing':
            df.at[idx, 'job_title'] = random.choice(['Marketing Specialist', 'Content Writer', 'SEO Analyst'])
        elif wrong_dept == 'Sales':
            df.at[idx, 'job_title'] = random.choice(['Sales Representative', 'Account Executive', 'Business Developer'])
        elif wrong_dept == 'HR':
            df.at[idx, 'job_title'] = random.choice(['HR Specialist', 'Recruiter', 'Talent Acquisition'])
        elif wrong_dept == 'Finance':
            df.at[idx, 'job_title'] = random.choice(['Accountant', 'Financial Analyst', 'Payroll Specialist'])
    
    return df

def create_schema():
    """
    Create a schema for the employee data.
    
    Returns:
        Dictionary with schema information
    """
    schema = {
        'columns': [
            {
                'name': 'employee_id',
                'type': 'id',
                'id_type': 'employee'
            },
            {
                'name': 'first_name',
                'type': 'string',
                'subtype': 'first_name'
            },
            {
                'name': 'last_name',
                'type': 'string',
                'subtype': 'last_name'
            },
            {
                'name': 'full_name',
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
                'categories': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
            },
            {
                'name': 'job_title',
                'type': 'category',
                'categories': [
                    'Software Engineer', 'QA Engineer', 'DevOps Engineer', 'Engineering Manager',
                    'Marketing Specialist', 'Content Writer', 'SEO Analyst', 'Marketing Manager',
                    'Sales Representative', 'Account Executive', 'Sales Manager', 'Business Developer',
                    'HR Specialist', 'Recruiter', 'HR Manager', 'Talent Acquisition',
                    'Accountant', 'Financial Analyst', 'Finance Manager', 'Payroll Specialist'
                ]
            },
            {
                'name': 'salary',
                'type': 'float',
                'min': 40000,
                'max': 150000
            },
            {
                'name': 'hire_date',
                'type': 'date',
                'start_date': '2018-01-01',
                'end_date': '2023-12-31'
            },
            {
                'name': 'manager_id',
                'type': 'id',
                'id_type': 'employee'
            }
        ],
        'relationships': [
            {
                'source': 'employee_id',
                'target': 'manager_id',
                'type': 'foreign_key'
            },
            {
                'source': 'department',
                'target': 'job_title',
                'type': 'logical',
                'expression': (
                    '(department == "Engineering" and job_title.str.contains("Engineer|Engineering")) or '
                    '(department == "Marketing" and job_title.str.contains("Marketing|Content|SEO")) or '
                    '(department == "Sales" and job_title.str.contains("Sales|Account|Business")) or '
                    '(department == "HR" and job_title.str.contains("HR|Recruiter|Talent")) or '
                    '(department == "Finance" and job_title.str.contains("Finance|Accountant|Payroll"))'
                )
            }
        ],
        'check_duplicates': True
    }
    
    # Convert to the format expected by the validator (list of column objects)
    return schema.get('columns', [])

def main():
    """Main function to run the integration test."""
    parser = argparse.ArgumentParser(description='Integration test for field standards and data validator')
    parser.add_argument('--rows', type=int, default=100, help='Number of rows to generate')
    parser.add_argument('--error-rate', type=float, default=0.2, help='Percentage of rows with errors')
    parser.add_argument('--output-dir', type=str, default='validation_test', help='Output directory')
    parser.add_argument('--no-schema', action='store_true', help='Run validation without schema')
    parser.add_argument('--strict', action='store_true', help='Use strict validation mode')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic data
    df = generate_synthetic_data(args.rows, args.error_rate)
    
    # Create schema (unless --no-schema is specified)
    schema = None if args.no_schema else create_schema()
    
    # Save generated data
    generated_path = os.path.join(args.output_dir, 'generated_data.csv')
    df.to_csv(generated_path, index=False)
    print(f"Saved generated data to {generated_path}")
    
    # Save schema
    if schema:
        schema_path = os.path.join(args.output_dir, 'schema.json')
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2, default=str)
        print(f"Saved schema to {schema_path}")
    
    # Run validation
    print("\nValidating and fixing data...")
    corrected_df, corrections = validate_and_fix_data(df, schema)
    
    # Save corrected data
    corrected_path = os.path.join(args.output_dir, 'corrected_data.csv')
    corrected_df.to_csv(corrected_path, index=False)
    print(f"Saved corrected data to {corrected_path}")
    
    # Save corrections
    corrections_path = os.path.join(args.output_dir, 'corrections.json')
    with open(corrections_path, 'w') as f:
        json.dump(corrections, f, indent=2, default=str)
    print(f"Saved corrections to {corrections_path}")
    
    # Print statistics
    print("\n====== Validation Results ======")
    print(f"Total corrections: {len(corrections)}")
    
    # Group corrections by type
    correction_types = {}
    for correction in corrections:
        # Try to extract the reason from the correction message
        parts = correction.split('-')
        if len(parts) > 1:
            reason = parts[-1].strip()
            if reason not in correction_types:
                correction_types[reason] = 0
            correction_types[reason] += 1
    
    # Print corrections by type
    print("\nCorrections by type:")
    for reason, count in sorted(correction_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {reason}: {count}")
    
    print("\nIntegration test completed successfully!")

if __name__ == "__main__":
    main()
