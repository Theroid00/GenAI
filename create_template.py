#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template Creator for Synthetic Data Generator
--------------------------------------------
This script provides a guided interface for creating custom templates
for the synthetic data generator.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

def prompt_for_input(prompt: str, default: Any = None) -> str:
    """
    Prompt the user for input with an optional default value.
    
    Args:
        prompt: The prompt message
        default: Default value if user enters nothing
        
    Returns:
        User input or default value
    """
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        return user_input
    else:
        while True:
            user_input = input(f"{prompt}: ").strip()
            if user_input:
                return user_input
            print("This field is required. Please enter a value.")

def create_template_interactively() -> List[Dict[str, Any]]:
    """
    Create a template interactively by prompting the user for field definitions.
    
    Returns:
        List of field definitions for the template
    """
    print("\n=== Interactive Template Creation ===")
    
    # Get template context
    context = prompt_for_input("Template context/theme (e.g., 'Customer profiles')")
    
    # Define columns
    columns = []
    print("\nLet's define the fields for your template:")
    print("Enter a blank field name when finished.")
    
    while True:
        field_name = prompt_for_input("\nField name (or blank to finish)", "").strip()
        if not field_name:
            if columns:  # Only break if we have at least one column
                break
            else:
                print("You must define at least one field.")
                continue
        
        # Get data type
        data_type_options = ['int', 'float', 'string', 'category', 'date', 'uuid']
        data_type = prompt_for_input(
            f"Data type ({', '.join(data_type_options)})"
        ).lower()
        
        if data_type not in data_type_options:
            print(f"Invalid data type. Using 'string' as default.")
            data_type = 'string'
        
        field_info = {
            'name': field_name,
            'type': data_type
        }
        
        # Get additional details based on data type
        if data_type == 'int':
            field_info['min'] = int(prompt_for_input("Minimum value", 0))
            field_info['max'] = int(prompt_for_input("Maximum value", 100))
            
            # Ask about distribution
            dist_type = prompt_for_input("Distribution type (uniform/normal)", "uniform").lower()
            if dist_type == 'normal':
                field_info['distribution'] = 'normal'
                field_info['mean'] = float(prompt_for_input("Mean value", (field_info['min'] + field_info['max']) / 2))
                field_info['std'] = float(prompt_for_input("Standard deviation", (field_info['max'] - field_info['min']) / 6))
            else:
                field_info['distribution'] = 'uniform'
                
        elif data_type == 'float':
            field_info['min'] = float(prompt_for_input("Minimum value", 0.0))
            field_info['max'] = float(prompt_for_input("Maximum value", 1.0))
            
            # Ask about distribution
            dist_type = prompt_for_input("Distribution type (uniform/normal)", "uniform").lower()
            if dist_type == 'normal':
                field_info['distribution'] = 'normal'
                field_info['mean'] = float(prompt_for_input("Mean value", (field_info['min'] + field_info['max']) / 2))
                field_info['std'] = float(prompt_for_input("Standard deviation", (field_info['max'] - field_info['min']) / 6))
            else:
                field_info['distribution'] = 'uniform'
                
            # Ask about decimal places
            field_info['decimals'] = int(prompt_for_input("Decimal places", 2))
                
        elif data_type == 'string':
            string_types = ['name', 'first_name', 'last_name', 'full_name', 'city', 'country', 
                           'email', 'phone', 'address', 'company', 'job', 'text', 'custom']
            
            string_type = prompt_for_input(
                f"String type ({', '.join(string_types)})", 
                "text"
            ).lower()
            
            if string_type not in string_types:
                print(f"Invalid string type. Using 'text' as default.")
                string_type = 'text'
                
            field_info['subtype'] = string_type
            
            if string_type == 'custom':
                field_info['pattern'] = prompt_for_input("Custom pattern (or leave empty for random text)")
                if not field_info['pattern']:
                    field_info['min_length'] = int(prompt_for_input("Minimum length", 5))
                    field_info['max_length'] = int(prompt_for_input("Maximum length", 20))
                    
        elif data_type == 'category':
            categories_input = prompt_for_input("Categories (comma separated)")
            field_info['categories'] = [c.strip() for c in categories_input.split(',')]
            
            # Ask for weights (probabilities)
            use_weights = prompt_for_input("Use custom probabilities? (yes/no)", "no").lower()
            if use_weights.startswith('y'):
                weights_input = prompt_for_input("Probabilities (comma separated, must sum to 1)")
                weights = [float(w.strip()) for w in weights_input.split(',')]
                
                if len(weights) != len(field_info['categories']):
                    print("Warning: Number of weights doesn't match number of categories. Using uniform probabilities.")
                elif abs(sum(weights) - 1.0) > 0.01:
                    print("Warning: Weights don't sum to 1. Using uniform probabilities.")
                else:
                    field_info['weights'] = weights
                
        elif data_type == 'date':
            start_date = prompt_for_input("Start date (YYYY-MM-DD)", "2020-01-01")
            end_date = prompt_for_input("End date (YYYY-MM-DD)", "2023-12-31")
            field_info['start_date'] = start_date
            field_info['end_date'] = end_date
        
        columns.append(field_info)
    
    print(f"\nTemplate created with {len(columns)} fields.")
    return columns

def create_example_template(template_type: str) -> List[Dict[str, Any]]:
    """
    Create an example template based on the specified type.
    
    Args:
        template_type: Type of example template to create
        
    Returns:
        List of field definitions for the template
    """
    if template_type == 'customer':
        return [
            {"name": "customer_id", "type": "uuid"},
            {"name": "full_name", "type": "string", "subtype": "full_name"},
            {"name": "email", "type": "string", "subtype": "email"},
            {"name": "phone", "type": "string", "subtype": "phone"},
            {"name": "age", "type": "int", "min": 18, "max": 85, "distribution": "normal", "mean": 42, "std": 12},
            {"name": "membership_level", "type": "category", "categories": ["Silver", "Gold", "Platinum", "Diamond"], "weights": [0.5, 0.3, 0.15, 0.05]},
            {"name": "subscription_fee", "type": "float", "min": 9.99, "max": 199.99, "decimals": 2},
            {"name": "join_date", "type": "date", "start_date": "2020-01-01", "end_date": "2023-12-31"}
        ]
    elif template_type == 'employee':
        return [
            {"name": "employee_id", "type": "uuid"},
            {"name": "full_name", "type": "string", "subtype": "full_name"},
            {"name": "email", "type": "string", "subtype": "email"},
            {"name": "department", "type": "category", "categories": ["Engineering", "Marketing", "Sales", "HR", "Finance"]},
            {"name": "position", "type": "string", "subtype": "job"},
            {"name": "salary", "type": "float", "min": 35000, "max": 150000, "distribution": "normal", "mean": 75000, "std": 20000, "decimals": 2},
            {"name": "years_of_service", "type": "int", "min": 0, "max": 30},
            {"name": "hire_date", "type": "date", "start_date": "2000-01-01", "end_date": "2023-12-31"}
        ]
    elif template_type == 'sales':
        return [
            {"name": "transaction_id", "type": "uuid"},
            {"name": "product_name", "type": "string", "subtype": "custom", "min_length": 10, "max_length": 30},
            {"name": "category", "type": "category", "categories": ["Electronics", "Clothing", "Home", "Food", "Other"]},
            {"name": "price", "type": "float", "min": 4.99, "max": 999.99, "decimals": 2},
            {"name": "quantity", "type": "int", "min": 1, "max": 20},
            {"name": "date", "type": "date", "start_date": "2023-01-01", "end_date": "2023-12-31"},
            {"name": "customer_name", "type": "string", "subtype": "full_name"},
            {"name": "payment_method", "type": "category", "categories": ["Credit Card", "PayPal", "Cash", "Bank Transfer"]}
        ]
    elif template_type == 'student':
        return [
            {"name": "student_id", "type": "uuid"},
            {"name": "full_name", "type": "string", "subtype": "full_name"},
            {"name": "age", "type": "int", "min": 18, "max": 30},
            {"name": "major", "type": "category", "categories": ["Computer Science", "Engineering", "Business", "Arts", "Science", "Medicine"]},
            {"name": "gpa", "type": "float", "min": 0.0, "max": 4.0, "decimals": 2},
            {"name": "enrollment_date", "type": "date", "start_date": "2018-01-01", "end_date": "2023-12-31"},
            {"name": "graduation_date", "type": "date", "start_date": "2022-01-01", "end_date": "2027-12-31"},
            {"name": "scholarship", "type": "category", "categories": ["None", "Partial", "Full"], "weights": [0.7, 0.2, 0.1]}
        ]
    else:
        return [
            {"name": "id", "type": "uuid"},
            {"name": "name", "type": "string", "subtype": "full_name"},
            {"name": "category", "type": "category", "categories": ["Category A", "Category B", "Category C"]},
            {"name": "value", "type": "float", "min": 0, "max": 100, "decimals": 2}
        ]

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Template Creator for Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a template interactively
  ./create_template.py --name my_template
  
  # Create a template based on an example
  ./create_template.py --name customer_template --example customer
        """
    )
    
    parser.add_argument(
        "--name", "-n", required=True,
        help="Name for the template file (without .json extension)"
    )
    parser.add_argument(
        "--example", "-e", choices=["customer", "employee", "sales", "student", "basic"],
        help="Use a predefined example template instead of interactive creation"
    )
    parser.add_argument(
        "--output-dir", "-o", default="templates",
        help="Directory to save the template file (defaults to 'templates')"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {str(e)}")
            sys.exit(1)
    
    # Create template
    if args.example:
        print(f"Creating template from '{args.example}' example...")
        template = create_example_template(args.example)
    else:
        print("Creating template interactively...")
        template = create_template_interactively()
    
    # Save template to file
    template_path = os.path.join(output_dir, f"{args.name}.json")
    
    # Check if file already exists
    if os.path.exists(template_path):
        overwrite = prompt_for_input(f"File '{template_path}' already exists. Overwrite? (yes/no)", "no").lower()
        if not overwrite.startswith('y'):
            print("Template creation cancelled.")
            sys.exit(0)
    
    try:
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"\nTemplate saved to: {template_path}")
        
        # Show example usage
        print("\nUse this template with the synthetic data generator:")
        print(f"python synthetic_data_cli.py --template {args.name} --rows 100 --output {args.name}_data.csv")
    except Exception as e:
        print(f"Error saving template: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
