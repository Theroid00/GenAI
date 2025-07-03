#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator CLI
----------------------------
A comprehensive command-line interface for the synthetic data generator.
This script combines all functionality into an easy-to-use CLI tool.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
import pandas as pd

# Import from the project
from integrated_generator import generate_and_validate
from synthetic_data_generator import load_schema_from_json

def list_templates() -> List[str]:
    """
    List all available templates in the templates directory.
    
    Returns:
        List of template names (without .json extension)
    """
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if not os.path.exists(templates_dir):
        return []
        
    template_files = [f for f in os.listdir(templates_dir) if f.endswith('.json')]
    return [os.path.splitext(f)[0] for f in template_files]

def print_template_info(template_name: str) -> None:
    """
    Print detailed information about a specific template.
    
    Args:
        template_name: Name of the template (without .json extension)
    """
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    template_path = os.path.join(templates_dir, f"{template_name}.json")
    
    if not os.path.exists(template_path):
        print(f"Template '{template_name}' not found.")
        return
    
    try:
        # Load the template
        with open(template_path, 'r') as f:
            schema = json.load(f)
        
        print(f"\nTemplate: {template_name}")
        print("=" * (len(template_name) + 10))
        print(f"Fields ({len(schema)}):")
        
        for i, field in enumerate(schema, 1):
            field_type = field.get('type', 'unknown')
            field_name = field.get('name', f'field_{i}')
            
            print(f"\n{i}. {field_name} ({field_type})")
            
            # Print additional details based on field type
            if field_type == 'int' or field_type == 'float':
                print(f"   Range: {field.get('min', 'N/A')} to {field.get('max', 'N/A')}")
                if 'distribution' in field:
                    print(f"   Distribution: {field['distribution']}")
                    if field['distribution'] == 'normal':
                        print(f"   Mean: {field.get('mean', 'N/A')}, Std Dev: {field.get('std', 'N/A')}")
            
            elif field_type == 'category':
                categories = field.get('categories', [])
                if len(categories) <= 5:
                    print(f"   Categories: {', '.join(str(c) for c in categories)}")
                else:
                    print(f"   Categories: {', '.join(str(c) for c in categories[:5])}... ({len(categories)} total)")
                
                if 'weights' in field:
                    print(f"   Custom weights: Yes")
            
            elif field_type == 'string':
                print(f"   Subtype: {field.get('subtype', 'text')}")
        
        print("\nUse this template with:")
        print(f"./synthetic_data_cli.py --template {template_name}")
        
    except Exception as e:
        print(f"Error reading template: {str(e)}")

def create_new_template(template_name: str, interactive: bool = True) -> None:
    """
    Create a new template file.
    
    Args:
        template_name: Name for the new template (without .json extension)
        interactive: Whether to prompt the user for field definitions
    """
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    template_path = os.path.join(templates_dir, f"{template_name}.json")
    
    # Check if template already exists
    if os.path.exists(template_path):
        overwrite = input(f"Template '{template_name}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Template creation cancelled.")
            return
    
    # Create the schema
    schema = []
    
    if interactive:
        print(f"\nCreating new template: {template_name}")
        print("Define the fields for your template. Enter blank field name to finish.")
        
        while True:
            field_name = input("\nField name (or press Enter to finish): ").strip()
            if not field_name:
                break
            
            # Get field type
            while True:
                field_type = input("Field type (int/float/string/category/date/uuid): ").strip().lower()
                if field_type in ['int', 'float', 'string', 'category', 'date', 'uuid']:
                    break
                print("Invalid type. Please use one of the specified types.")
            
            field = {
                "name": field_name,
                "type": field_type
            }
            
            # Get additional properties based on type
            if field_type == 'int' or field_type == 'float':
                field['min'] = float(input(f"Minimum value: "))
                field['max'] = float(input(f"Maximum value: "))
                
                dist_type = input("Distribution (uniform/normal): ").strip().lower()
                if dist_type == 'normal':
                    field['distribution'] = 'normal'
                    field['mean'] = float(input(f"Mean value: "))
                    field['std'] = float(input(f"Standard deviation: "))
                else:
                    field['distribution'] = 'uniform'
                
                if field_type == 'float':
                    field['decimals'] = int(input("Decimal places: "))
                
                # Convert min/max to int for int type
                if field_type == 'int':
                    field['min'] = int(field['min'])
                    field['max'] = int(field['max'])
                    if 'mean' in field:
                        field['mean'] = int(field['mean'])
            
            elif field_type == 'string':
                subtypes = ['text', 'name', 'first_name', 'last_name', 'full_name', 
                           'email', 'phone', 'address', 'city', 'country', 'company', 'job']
                
                print(f"Available subtypes: {', '.join(subtypes)}")
                field['subtype'] = input(f"String subtype ({subtypes[0]}): ").strip().lower() or subtypes[0]
            
            elif field_type == 'category':
                categories_input = input("Categories (comma separated): ")
                field['categories'] = [c.strip() for c in categories_input.split(',')]
                
                use_weights = input("Define custom probabilities? (y/n): ").lower() == 'y'
                if use_weights:
                    weights_input = input("Probabilities (comma separated, should sum to 1): ")
                    field['weights'] = [float(w.strip()) for w in weights_input.split(',')]
            
            elif field_type == 'date':
                field['start_date'] = input("Start date (YYYY-MM-DD): ").strip()
                field['end_date'] = input("End date (YYYY-MM-DD): ").strip()
            
            schema.append(field)
    else:
        # Create a basic template
        schema = [
            {"name": f"{template_name}_id", "type": "uuid"},
            {"name": "name", "type": "string", "subtype": "full_name"},
            {"name": "category", "type": "category", "categories": ["Category A", "Category B", "Category C"]},
            {"name": "value", "type": "float", "min": 0, "max": 100, "decimals": 2}
        ]
    
    # Save the schema
    try:
        with open(template_path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"\nTemplate '{template_name}' created successfully!")
        print(f"Template saved to: {template_path}")
    except Exception as e:
        print(f"Error creating template: {str(e)}")

def main():
    """Main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data using a template
  ./synthetic_data_cli.py --template customer --rows 100 --output customer_data.csv

  # Generate data from a schema file
  ./synthetic_data_cli.py --schema my_schema.json --rows 500 --output my_data.csv

  # Generate data from a sample CSV file
  ./synthetic_data_cli.py --sample sample_data.csv --rows 1000 --output synthetic_data.csv

  # List all available templates
  ./synthetic_data_cli.py --list-templates

  # Show details about a specific template
  ./synthetic_data_cli.py --template-info customer

  # Create a new template interactively
  ./synthetic_data_cli.py --create-template my_template

Available templates:
  customer - Customer profiles with demographic information
  employee - Employee records with department and salary information
  sales - Sales transaction data with products and prices
  student - Student records with academic information
        """
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--sample", "-s", metavar="CSV_FILE",
        help="Generate data based on a sample CSV file"
    )
    input_group.add_argument(
        "--schema", "-j", metavar="JSON_FILE",
        help="Generate data based on a schema JSON file"
    )
    input_group.add_argument(
        "--template", "-t", metavar="TEMPLATE_NAME",
        help="Use a built-in template"
    )
    
    # Template management
    template_group = parser.add_argument_group("Template Management")
    template_group.add_argument(
        "--list-templates", action="store_true",
        help="List all available templates"
    )
    template_group.add_argument(
        "--template-info", metavar="TEMPLATE_NAME",
        help="Show details about a specific template"
    )
    template_group.add_argument(
        "--create-template", metavar="TEMPLATE_NAME",
        help="Create a new template"
    )
    
    # Common arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o", default="generated_data.csv",
        help="Output CSV filename"
    )
    output_group.add_argument(
        "--rows", "-r", type=int, default=100,
        help="Number of rows to generate"
    )
    output_group.add_argument(
        "--no-validate", action="store_true",
        help="Skip validation step"
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--model", "-m", choices=["gaussian", "ctgan"], default="gaussian",
        help="Model type for sample-based generation"
    )
    advanced_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Template management options
    if args.list_templates:
        templates = list_templates()
        print("\nAvailable templates:")
        if templates:
            for template in templates:
                print(f"  - {template}")
        else:
            print("  No templates found.")
        print("\nUse --template-info TEMPLATE_NAME to see details about a specific template")
        return
    
    if args.template_info:
        print_template_info(args.template_info)
        return
    
    if args.create_template:
        create_new_template(args.create_template)
        return
    
    # Generation mode
    schema_path = None
    input_path = None
    
    if args.template:
        template_name = args.template.lower()
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        template_path = os.path.join(templates_dir, f"{template_name}.json")
        
        if not os.path.exists(template_path):
            available_templates = list_templates()
            print(f"Error: Template '{template_name}' not found")
            if available_templates:
                print(f"Available templates: {', '.join(available_templates)}")
            else:
                print("No templates found. Create one with --create-template")
            sys.exit(1)
        
        schema_path = template_path
    elif args.schema:
        schema_path = args.schema
    elif args.sample:
        input_path = args.sample
    else:
        # No input source specified, show help and exit
        parser.print_help()
        print("\nError: You must specify an input source (--sample, --schema, or --template)")
        sys.exit(1)
    
    # Run the generator
    try:
        if args.verbose:
            print(f"\nSynthetic Data Generator")
            print(f"=======================")
            
            if input_path:
                print(f"Mode: Generate from CSV sample")
                print(f"Input: {input_path}")
            elif schema_path:
                print(f"Mode: Generate from {'template' if args.template else 'schema'}")
                print(f"Schema: {schema_path}")
            
            print(f"Rows: {args.rows}")
            print(f"Output: {args.output}")
            print(f"Validation: {'Disabled' if args.no_validate else 'Enabled'}")
            if input_path:
                print(f"Model: {args.model}")
            print()
        
        df = generate_and_validate(
            input_path=input_path,
            schema_path=schema_path,
            output_path=args.output,
            num_rows=args.rows,
            model_type=args.model,
            validate=not args.no_validate
        )
        
        print(f"\nSuccessfully generated {len(df)} rows of synthetic data!")
        print(f"Output saved to: {os.path.abspath(args.output)}")
        
        # Show a preview of the data
        print("\nData Preview:")
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        print(df.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
