#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator Test Script
------------------------------------
This script tests the synthetic data generator with various data types to verify
that it's generating realistic and correctly formatted data in all modes.
"""

import os
import pandas as pd
import re
from field_standards import (
    ID_PATTERNS, 
    EMAIL_DOMAINS,
    detect_id_field_type,
    generate_id,
    format_email, 
    format_phone_number
)

# Import from main package
from synthetic_data_gen import SyntheticDataGenerator

def test_id_generation():
    """Test generation of various ID types"""
    print("\n=== Testing ID Generation ===")
    
    # Test detection
    test_columns = [
        "employee_id", "emp_id", "staff_id", "worker_number",
        "customer_id", "client_number", "buyer_id",
        "product_code", "prod_id", "item_sku",
        "order_number", "transaction_id", "purchase_id",
        "invoice_id", "bill_number", "receipt_id",
        "ticket_id", "issue_number", "case_id",
        "student_id", "learner_number", "pupil_id",
        "generic_id", "id", "uuid"
    ]
    
    print("Testing ID type detection:")
    for column in test_columns:
        id_type = detect_id_field_type(column)
        print(f"  Column '{column}' detected as: {id_type or 'Not an ID'}")
    
    # Test generation
    print("\nTesting ID generation:")
    for id_type in ID_PATTERNS.keys():
        ids = [generate_id(id_type, i) for i in range(5)]
        print(f"  {id_type.capitalize()} IDs: {', '.join(ids)}")

def test_email_generation():
    """Test generation of realistic email addresses"""
    print("\n=== Testing Email Generation ===")
    
    test_names = [
        ("John", "Doe"),
        ("Alice", "Smith"),
        ("Maria", "Garcia-Rodriguez"),
        ("James", "O'Connor"),
        ("Åsa", "Larsson"),
        ("Wei", "Zhang")
    ]
    
    print("Testing email generation with various names:")
    for first, last in test_names:
        email = format_email(first, last)
        print(f"  {first} {last} -> {email}")
    
    # Check format validity
    print("\nChecking email format validity...")
    valid_count = 0
    total_count = 100
    
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    
    for i in range(total_count):
        email = format_email("Test", f"User{i}")
        if re.match(email_pattern, email):
            valid_count += 1
    
    print(f"  {valid_count}/{total_count} emails are valid ({valid_count/total_count*100:.1f}%)")
    
    # Check domain distribution
    print("\nChecking email domain distribution...")
    domains = {}
    total_count = 1000
    
    for i in range(total_count):
        email = format_email("Test", f"User{i}")
        domain = email.split('@')[1]
        domains[domain] = domains.get(domain, 0) + 1
    
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        expected = EMAIL_DOMAINS.get(domain, 0) * 100
        actual = count / total_count * 100
        print(f"  {domain}: {count} ({actual:.1f}% vs expected {expected:.1f}%)")

def test_phone_generation():
    """Test generation of realistic phone numbers"""
    print("\n=== Testing Phone Number Generation ===")
    
    # Generate sample phone numbers
    phones = [format_phone_number() for _ in range(10)]
    
    print("Sample phone numbers:")
    for phone in phones:
        print(f"  {phone}")
    
    # Check format validity
    print("\nChecking phone format validity...")
    valid_count = 0
    total_count = 100
    
    # Multiple patterns to match various phone formats
    patterns = [
        r'^\(\d{3}\) \d{3}-\d{4}$',              # (555) 123-4567
        r'^\d{3}-\d{3}-\d{4}$',                  # 555-123-4567
        r'^\d{3}\.\d{3}\.\d{4}$',                # 555.123.4567
        r'^\d{1}-\d{3}-\d{3}-\d{4}$',            # 1-555-123-4567
        r'^\d{3}-\d{4}$',                        # 123-4567
        r'^\d{10}$',                             # 5551234567
        r'^\d{1} \(\d{3}\) \d{3}-\d{4}$'         # 1 (555) 123-4567
    ]
    
    for i in range(total_count):
        phone = format_phone_number()
        if any(re.match(pattern, phone) for pattern in patterns):
            valid_count += 1
    
    print(f"  {valid_count}/{total_count} phone numbers are valid ({valid_count/total_count*100:.1f}%)")
    
    # Check format distribution
    formats = {}
    total_count = 500
    
    for i in range(total_count):
        phone = format_phone_number()
        
        # Determine format type
        if re.match(r'^\(\d{3}\) \d{3}-\d{4}$', phone):
            format_type = "(XXX) XXX-XXXX"
        elif re.match(r'^\d{3}-\d{3}-\d{4}$', phone):
            format_type = "XXX-XXX-XXXX"
        elif re.match(r'^\d{3}\.\d{3}\.\d{4}$', phone):
            format_type = "XXX.XXX.XXXX"
        elif re.match(r'^\d{1}-\d{3}-\d{3}-\d{4}$', phone):
            format_type = "X-XXX-XXX-XXXX"
        elif re.match(r'^\d{3}-\d{4}$', phone):
            format_type = "XXX-XXXX"
        elif re.match(r'^\d{10}$', phone):
            format_type = "XXXXXXXXXX"
        elif re.match(r'^\d{1} \(\d{3}\) \d{3}-\d{4}$', phone):
            format_type = "X (XXX) XXX-XXXX"
        else:
            format_type = "Other"
        
        formats[format_type] = formats.get(format_type, 0) + 1
    
    print("\nPhone format distribution:")
    for format_type, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {format_type}: {count} ({count/total_count*100:.1f}%)")

def test_synthetic_data_generation():
    """Test generation of full synthetic datasets with various column types"""
    print("\n=== Testing Full Synthetic Data Generation ===")
    
    # Define a test schema with various column types
    schema = [
        {
            'name': 'employee_id',
            'type': 'uuid'
        },
        {
            'name': 'customer_id',
            'type': 'uuid'
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
            'name': 'age',
            'type': 'int',
            'min': 18,
            'max': 65
        },
        {
            'name': 'salary',
            'type': 'float',
            'min': 30000,
            'max': 150000,
            'distribution': 'normal',
            'mean': 75000,
            'std': 25000,
            'decimals': 2
        },
        {
            'name': 'department',
            'type': 'category',
            'categories': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
        },
        {
            'name': 'join_date',
            'type': 'date',
            'start_date': datetime.strptime('2015-01-01', '%Y-%m-%d'),
            'end_date': datetime.strptime('2023-12-31', '%Y-%m-%d')
        }
    ]
    
    # Generate data
    num_rows = 100
    print(f"Generating {num_rows} rows with {len(schema)} columns...")
    
    generator = SyntheticDataGenerator()
    processed_df = generator.generate_from_schema(schema, num_rows)
    
    # Display sample rows
    print("\nSample data (first 5 rows):")
    print(processed_df.head().to_string())
    
    # Validate ID formats
    print("\nValidating ID column formats...")
    valid_emp_id = all(str(id).startswith('EMP-') for id in processed_df['employee_id'])
    valid_cust_id = all(str(id).startswith('CUST-') for id in processed_df['customer_id'])
    
    print(f"  Employee IDs valid: {'Yes' if valid_emp_id else 'No'}")
    print(f"  Customer IDs valid: {'Yes' if valid_cust_id else 'No'}")
    
    # Validate email formats
    print("\nValidating email formats...")
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    valid_emails = sum(1 for email in processed_df['email'] if re.match(email_pattern, str(email)))
    
    print(f"  Valid emails: {valid_emails}/{num_rows} ({valid_emails/num_rows*100:.1f}%)")
    
    # Check if emails match names
    print("\nChecking if emails match names...")
    name_in_email_count = 0
    
    for idx, row in processed_df.iterrows():
        name = str(row['full_name']).lower()
        email = str(row['email']).lower()
        
        # Extract first and last name
        parts = name.split()
        if len(parts) >= 2:
            first = parts[0].lower().replace('-', '').replace('.', '')
            last = parts[-1].lower().replace('-', '').replace('.', '')
            
            # Check if either first or last name appears in email
            email_parts = email.split('@')[0]
            if first in email_parts or last in email_parts:
                name_in_email_count += 1
    
    print(f"  Emails containing name parts: {name_in_email_count}/{num_rows} ({name_in_email_count/num_rows*100:.1f}%)")
    
    # Save test data to CSV
    output_path = "test_synthetic_data.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"\nTest data saved to {output_path}")

if __name__ == "__main__":
    from datetime import datetime
    
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR TEST SUITE")
    print("=" * 80)
    print(f"Running tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    test_id_generation()
    test_email_generation()
    test_phone_generation()
    test_synthetic_data_generation()
    
    print("\nAll tests completed!")
