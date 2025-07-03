#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Start for Synthetic Data Generator
----------------------------------------
This script provides a simplified command-line interface for generating synthetic data.
"""

import os
import sys
import argparse
from integrated_generator import generate_and_validate

def main():
    """Main function with simplified argument parsing"""
    parser = argparse.ArgumentParser(
        description="Quick start for synthetic data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define three simple modes with separate help text
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample", "-s", metavar="CSV_FILE",
        help="Generate data based on a sample CSV file"
    )
    group.add_argument(
        "--schema", "-j", metavar="JSON_FILE",
        help="Generate data based on a schema JSON file"
    )
    group.add_argument(
        "--template", "-t", metavar="TEMPLATE_NAME",
        help="Use a built-in template (options: customer, employee, sales, student)"
    )
    
    # Common arguments
    parser.add_argument(
        "--output", "-o", default="generated_data.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--rows", "-r", type=int, default=100,
        help="Number of rows to generate"
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation step"
    )
    
    args = parser.parse_args()
    
    # Handle built-in templates
    schema_path = None
    input_path = None
    
    if args.template:
        template_name = args.template.lower()
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
            
        # Define paths for built-in templates
        template_options = {
            "customer": "customer_template.json",
            "employee": "employee_template.json",
            "sales": "sales_template.json",
            "student": "student_template.json"
        }
        
        if template_name in template_options:
            schema_path = os.path.join(templates_dir, template_options[template_name])
            
            # If template file doesn't exist, copy from test schema
            if not os.path.exists(schema_path) and template_name == "customer":
                import shutil
                test_schema = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          "test_customer_schema.json")
                if os.path.exists(test_schema):
                    shutil.copyfile(test_schema, schema_path)
                    print(f"Created template from test schema: {schema_path}")
                else:
                    print(f"Error: Template {template_name} not found and couldn't create it")
                    sys.exit(1)
            elif not os.path.exists(schema_path):
                print(f"Error: Template {template_name} not found")
                print(f"Available templates: {', '.join(template_options.keys())}")
                sys.exit(1)
        else:
            print(f"Error: Unknown template '{template_name}'")
            print(f"Available templates: {', '.join(template_options.keys())}")
            sys.exit(1)
    elif args.schema:
        schema_path = args.schema
    elif args.sample:
        input_path = args.sample
    
    # Run the generator
    try:
        df = generate_and_validate(
            input_path=input_path,
            schema_path=schema_path,
            output_path=args.output,
            num_rows=args.rows,
            validate=not args.no_validate
        )
        
        print(f"\nSuccessfully generated {len(df)} rows of synthetic data!")
        print(f"Output saved to: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
