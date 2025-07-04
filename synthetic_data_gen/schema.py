"""
Schema-related functions for synthetic data generation.
This module contains functions for schema inference, definition, and manipulation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Union
from faker import Faker
import json
from datetime import datetime

# Initialize faker
fake = Faker()

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

def infer_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Infer the schema from a DataFrame including data types and distributions.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing schema information for each column
    """
    schema = {}
    
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype)
        }
        
        # Determine more specific data type
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                col_info['type'] = 'int'
                col_info['min'] = df[col].min()
                col_info['max'] = df[col].max()
                col_info['mean'] = df[col].mean()
                col_info['std'] = df[col].std()
            else:
                col_info['type'] = 'float'
                col_info['min'] = df[col].min()
                col_info['max'] = df[col].max()
                col_info['mean'] = df[col].mean()
                col_info['std'] = df[col].std()
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info['type'] = 'date'
            col_info['min'] = df[col].min()
            col_info['max'] = df[col].max()
        elif df[col].nunique() < len(df) * 0.2:  # Heuristic for categorical
            col_info['type'] = 'category'
            col_info['categories'] = df[col].dropna().unique().tolist()
        else:
            col_info['type'] = 'string'
            # Check if it could be names, cities, etc.
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if any(name.lower() in sample.lower() for name in fake.first_name().lower().split()):
                col_info['subtype'] = 'name'
            elif any(city.lower() in sample.lower() for city in [fake.city().lower()]):
                col_info['subtype'] = 'city'
            else:
                col_info['subtype'] = 'generic'
                
        schema[col] = col_info
    
    return schema

def display_schema_info(schema: Dict[str, Dict[str, Any]]) -> None:
    """
    Display inferred schema information.
    
    Args:
        schema: Dictionary containing schema information
    """
    print("\n=== Inferred Schema ===")
    for col, info in schema.items():
        print(f"Column: {col}")
        print(f"  Type: {info['type']}")
        
        if info['type'] in ['int', 'float']:
            print(f"  Range: {info['min']} to {info['max']}")
            print(f"  Mean: {info['mean']:.2f}, Std Dev: {info['std']:.2f}")
        elif info['type'] == 'category':
            print(f"  Categories: {', '.join(str(c) for c in info['categories'][:5])}" + 
                  ("..." if len(info['categories']) > 5 else ""))
        elif info['type'] == 'date':
            print(f"  Range: {info['min']} to {info['max']}")
        
        print()

def define_column_interactively() -> Dict[str, Any]:
    """
    Define a column schema interactively.
    
    Returns:
        Dictionary with column schema information
    """
    from synthetic_data_gen.utils import prompt_for_input
    
    col_info = {}
    
    # Get column name
    col_info['name'] = prompt_for_input("Column name")
    
    # Get data type
    data_type_options = ['int', 'float', 'string', 'category', 'date', 'uuid']
    while True:
        data_type = prompt_for_input(
            f"Data type ({', '.join(data_type_options)})"
        ).lower()
        
        if data_type in data_type_options:
            col_info['type'] = data_type
            break
        else:
            print(f"Invalid data type. Please choose from: {', '.join(data_type_options)}")
    
    # Get additional details based on data type
    if data_type == 'int':
        col_info['min'] = int(prompt_for_input("Minimum value", 0))
        col_info['max'] = int(prompt_for_input("Maximum value", 100))
        
        # Ask about distribution
        dist_type = prompt_for_input("Distribution type (uniform/normal)", "uniform").lower()
        if dist_type == 'normal':
            col_info['distribution'] = 'normal'
            col_info['mean'] = float(prompt_for_input("Mean value", (col_info['min'] + col_info['max']) / 2))
            col_info['std'] = float(prompt_for_input("Standard deviation", (col_info['max'] - col_info['min']) / 6))
        else:
            col_info['distribution'] = 'uniform'
            
    elif data_type == 'float':
        col_info['min'] = float(prompt_for_input("Minimum value", 0.0))
        col_info['max'] = float(prompt_for_input("Maximum value", 1.0))
        
        # Ask about distribution
        dist_type = prompt_for_input("Distribution type (uniform/normal)", "uniform").lower()
        if dist_type == 'normal':
            col_info['distribution'] = 'normal'
            col_info['mean'] = float(prompt_for_input("Mean value", (col_info['min'] + col_info['max']) / 2))
            col_info['std'] = float(prompt_for_input("Standard deviation", (col_info['max'] - col_info['min']) / 6))
        else:
            col_info['distribution'] = 'uniform'
            
        # Ask about decimal places
        col_info['decimals'] = int(prompt_for_input("Decimal places", 2))
            
    elif data_type == 'string':
        string_types = ['name', 'first_name', 'last_name', 'full_name', 'city', 'country', 
                       'email', 'phone', 'address', 'company', 'job', 'text', 'custom']
        
        string_type = prompt_for_input(
            f"String type ({', '.join(string_types)})", 
            "custom"
        ).lower()
        
        col_info['subtype'] = string_type
        
        if string_type == 'custom':
            col_info['pattern'] = prompt_for_input("Custom pattern (or leave empty for random text)")
            if not col_info['pattern']:
                col_info['min_length'] = int(prompt_for_input("Minimum length", 5))
                col_info['max_length'] = int(prompt_for_input("Maximum length", 20))
                
    elif data_type == 'category':
        categories_input = prompt_for_input("Categories (comma separated)")
        col_info['categories'] = [c.strip() for c in categories_input.split(',')]
        
        # Ask for weights (probabilities)
        use_weights = prompt_for_input("Use custom probabilities? (yes/no)", "no").lower()
        if use_weights == 'yes':
            weights_input = prompt_for_input("Probabilities (comma separated, must sum to 1)")
            weights = [float(w.strip()) for w in weights_input.split(',')]
            
            if len(weights) != len(col_info['categories']):
                print("Warning: Number of weights doesn't match number of categories. Using uniform probabilities.")
                col_info['weights'] = None
            elif abs(sum(weights) - 1.0) > 0.01:
                print("Warning: Weights don't sum to 1. Using uniform probabilities.")
                col_info['weights'] = None
            else:
                col_info['weights'] = weights
        else:
            col_info['weights'] = None
            
    elif data_type == 'date':
        start_date = prompt_for_input("Start date (YYYY-MM-DD)", "2020-01-01")
        end_date = prompt_for_input("End date (YYYY-MM-DD)", "2023-12-31")
        
        try:
            col_info['start_date'] = datetime.strptime(start_date, "%Y-%m-%d")
            col_info['end_date'] = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Warning: Invalid date format. Using default range (2020-01-01 to 2023-12-31).")
            col_info['start_date'] = datetime.strptime("2020-01-01", "%Y-%m-%d")
            col_info['end_date'] = datetime.strptime("2023-12-31", "%Y-%m-%d")
            
    elif data_type == 'uuid':
        # No additional parameters needed for UUID
        pass
    
    return col_info

def interactive_schema_prompt() -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Interactively build a schema definition for synthetic data.
    
    Returns:
        Tuple containing:
        - Dataset title
        - Number of rows to generate
        - List of column definitions
    """
    from synthetic_data_gen.utils import prompt_for_input
    
    print("\n=== Interactive Schema Definition ===")
    
    # Ask for dataset context
    context = prompt_for_input("Dataset context/theme (e.g., 'Heights of students in schools')")
    
    # Ask for dataset title and row count
    title = prompt_for_input("Dataset title", context)
    num_rows = int(prompt_for_input("Number of rows to generate", "100"))
    
    # Warn about memory usage for very large datasets
    if num_rows > 500000:
        print(f"\nWarning: Generating {num_rows:,} rows will require significant memory and processing time.")
        print("Consider generating in smaller batches if you encounter memory issues.")
        confirm = prompt_for_input("Continue? (yes/no)", "yes").lower()
        if confirm != 'yes':
            print("Operation cancelled.")
            return title, 0, []
    
    # Define columns
    columns = []
    print("\nLet's define the columns for your dataset:")
    
    while True:
        columns.append(define_column_interactively())
        
        add_another = prompt_for_input("Add another column? (yes/no)", "yes").lower()
        if add_another != 'yes':
            break
    
    return title, num_rows, columns

def add_relationships_to_schema(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add relationships between columns based on user input.
    
    Args:
        schema: List of column definitions
        
    Returns:
        Updated schema with relationship information
    """
    from synthetic_data_gen.utils import prompt_for_input
    
    print("\n=== Define Relationships Between Columns ===")
    print("This will help generate more realistic data where columns are related.")
    
    # Show available columns
    print("\nAvailable columns:")
    for i, col_info in enumerate(schema):
        print(f"{i+1}. {col_info['name']} ({col_info['type']})")
    
    # Ask if user wants to define relationships
    add_rel = prompt_for_input("Do you want to define relationships between columns? (yes/no)", "no").lower()
    
    if add_rel != 'yes':
        return schema
    
    # Define relationships
    while True:
        # Get source column
        src_idx = int(prompt_for_input("Select source column number", "1")) - 1
        if src_idx < 0 or src_idx >= len(schema):
            print("Invalid column number. Please try again.")
            continue
        
        # Get target column
        tgt_idx = int(prompt_for_input("Select target column number", "2")) - 1
        if tgt_idx < 0 or tgt_idx >= len(schema) or tgt_idx == src_idx:
            print("Invalid column number. Please try again.")
            continue
        
        # Get relationship type
        rel_type = prompt_for_input(
            "Relationship type (correlation, dependency, transformation)", 
            "correlation"
        ).lower()
        
        # Add relationship to schema
        src_col = schema[src_idx]
        tgt_col = schema[tgt_idx]
        
        if 'relationships' not in src_col:
            src_col['relationships'] = []
        
        # Add the relationship details
        relationship = {
            'target_column': tgt_col['name'],
            'type': rel_type
        }
        
        # Add specific parameters based on relationship type
        if rel_type == 'correlation':
            # For numeric columns, get correlation coefficient
            if src_col['type'] in ['int', 'float'] and tgt_col['type'] in ['int', 'float']:
                coef = float(prompt_for_input("Correlation coefficient (-1 to 1)", "0.7"))
                relationship['coefficient'] = max(-1.0, min(1.0, coef))  # Clamp to valid range
        
        elif rel_type == 'dependency':
            # Categorical dependency - target depends on source
            print("\nDefine how target values depend on source values:")
            mapping = {}
            
            if src_col['type'] == 'category':
                for category in src_col.get('categories', []):
                    if tgt_col['type'] == 'category':
                        # For category -> category mapping
                        tgt_options = ", ".join(tgt_col.get('categories', []))
                        value = prompt_for_input(f"When {src_col['name']} is '{category}', {tgt_col['name']} should be ({tgt_options})")
                        mapping[category] = value
                    elif tgt_col['type'] in ['int', 'float']:
                        # For category -> numeric mapping
                        value = float(prompt_for_input(f"When {src_col['name']} is '{category}', {tgt_col['name']} range center"))
                        variance = float(prompt_for_input(f"Variance around center", "10"))
                        mapping[category] = {'center': value, 'variance': variance}
            
            relationship['mapping'] = mapping
            
        elif rel_type == 'transformation':
            # Define a transformation formula
            formula = prompt_for_input(f"Formula to calculate {tgt_col['name']} from {src_col['name']} (e.g., 'x * 2 + 5')")
            relationship['formula'] = formula
        
        src_col['relationships'].append(relationship)
        
        # Ask if user wants to add more relationships
        add_more = prompt_for_input("Add another relationship? (yes/no)", "no").lower()
        if add_more != 'yes':
            break
    
    return schema

def save_schema_to_json(schema: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save schema to a JSON file.
    
    Args:
        schema: Schema definition
        file_path: Path to save the JSON file
    """
    try:
        # Convert schema to serializable format
        serializable_schema = []
        for col_info in schema:
            col_info_copy = dict(col_info)
            
            # Convert numpy types to native Python types
            for key, value in col_info_copy.items():
                if isinstance(value, np.generic):
                    col_info_copy[key] = value.item()
                elif isinstance(value, np.ndarray):
                    col_info_copy[key] = value.tolist()
            
            serializable_schema.append(col_info_copy)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_schema, f, indent=2, default=str)
            
        logger.info(f"Schema saved to {file_path}")
        print(f"Schema saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving schema: {str(e)}")
        print(f"Error saving schema: {str(e)}")
