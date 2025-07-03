"""
Schema-related functions for synthetic data generation.
This module contains functions for schema inference, definition, and manipulation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from faker import Faker
import json
import random
import uuid
from datetime import datetime, timedelta

# Import region manager for region-specific data generation
try:
    from synthetic_data_gen.providers import region_manager
    REGION_MANAGER_AVAILABLE = True
except ImportError:
    REGION_MANAGER_AVAILABLE = False
    logger = logging.getLogger('synthetic_data_generator')
    logger.warning("Region manager not available. Some region-specific features may be limited.")

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

def generate_field_data(field_name: str, field_type: str, constraints: Dict[str, Any], 
                      num_samples: int) -> List[Any]:
    """
    Generate data for a field based on its type and constraints.
    
    Args:
        field_name (str): Name of the field
        field_type (str): Type of the field (int, float, string, category, date, uuid)
        constraints (Dict[str, Any]): Constraints for data generation
        num_samples (int): Number of samples to generate
        
    Returns:
        List[Any]: Generated data for the field
    """
    # Extract region and domain information if available
    region = constraints.get('region')
    domain = constraints.get('domain')
    
    # For numeric types (int, float)
    if field_type in ['int', 'float']:
        return generate_numeric_data(field_type, constraints, num_samples)
    
    # For string types
    elif field_type == 'string':
        return generate_string_data(field_name, constraints, num_samples, region, domain)
    
    # For categorical types
    elif field_type == 'category':
        return generate_categorical_data(constraints, num_samples)
    
    # For date types
    elif field_type == 'date':
        return generate_date_data(constraints, num_samples)
    
    # For UUID types
    elif field_type == 'uuid':
        return [str(uuid.uuid4()) for _ in range(num_samples)]
    
    # For other types
    else:
        logger.warning(f"Unsupported field type: {field_type}. Using string fallback.")
        return [f"Unsupported type: {field_type}" for _ in range(num_samples)]

def generate_numeric_data(field_type: str, constraints: Dict[str, Any], num_samples: int) -> List[Union[int, float]]:
    """
    Generate numeric data based on constraints.
    
    Args:
        field_type (str): 'int' or 'float'
        constraints (Dict[str, Any]): Constraints for data generation
        num_samples (int): Number of samples to generate
        
    Returns:
        List[Union[int, float]]: Generated numeric data
    """
    # Extract constraints
    min_val = constraints.get('min', 0)
    max_val = constraints.get('max', 100)
    distribution = constraints.get('distribution', 'uniform')
    
    if distribution == 'normal':
        # For normal distribution
        mean = constraints.get('mean', (min_val + max_val) / 2)
        std = constraints.get('std', (max_val - min_val) / 6)
        
        # Generate values
        values = np.random.normal(mean, std, num_samples)
        
        # Apply bounds
        values = np.clip(values, min_val, max_val)
    else:
        # For uniform distribution
        if field_type == 'int':
            values = np.random.randint(min_val, max_val + 1, num_samples)
        else:
            values = np.random.uniform(min_val, max_val, num_samples)
    
    # Apply rounding for float
    if field_type == 'float':
        decimals = constraints.get('decimals', 2)
        values = np.round(values, decimals)
    else:
        values = values.astype(int)
    
    return values.tolist()

def generate_string_data(field_name: str, constraints: Dict[str, Any], num_samples: int, 
                      region: Optional[str] = None, domain: Optional[str] = None) -> List[str]:
    """
    Generate string data based on constraints, with region-specific customization.
    
    Args:
        field_name (str): Name of the field
        constraints (Dict[str, Any]): Constraints for data generation
        num_samples (int): Number of samples to generate
        region (Optional[str]): Region code for region-specific data (e.g., 'india', 'usa')
        domain (Optional[str]): Domain code for domain-specific data (e.g., 'healthcare')
        
    Returns:
        List[str]: Generated string data
    """
    subtype = constraints.get('subtype', 'text')
    
    # Check if we should use region-specific data
    if region and REGION_MANAGER_AVAILABLE:
        # Try to use region-specific data provider
        # First, determine the data type based on field name and subtype
        data_type = subtype
        
        # If subtype is generic, try to infer data type from field name
        if subtype == 'generic':
            field_lower = field_name.lower()
            if 'name' in field_lower:
                data_type = 'name'
            elif 'city' in field_lower:
                data_type = 'city'
            elif 'address' in field_lower:
                data_type = 'address'
            elif 'phone' in field_lower:
                data_type = 'phone_number'
            elif 'email' in field_lower:
                data_type = 'email'
            elif 'company' in field_lower:
                data_type = 'company'
            elif 'job' in field_lower or 'occupation' in field_lower:
                data_type = 'job'
        
        # Try to use the region-specific generator
        try:
            return [region_manager.region_manager.generate_data(region, data_type) for _ in range(num_samples)]
        except Exception as e:
            logger.warning(f"Failed to use region-specific data for {data_type} in {region}: {str(e)}")
            # Fall back to generic generation
    
    # Check for domain-specific data
    if domain and REGION_MANAGER_AVAILABLE:
        # Try to use domain-specific data provider
        try:
            return [region_manager.region_manager.generate_data(domain, subtype) for _ in range(num_samples)]
        except Exception as e:
            logger.warning(f"Failed to use domain-specific data for {subtype} in {domain}: {str(e)}")
            # Fall back to generic generation
    
    # Generic Faker-based generation
    if subtype == 'name':
        return [fake.name() for _ in range(num_samples)]
    elif subtype == 'first_name':
        return [fake.first_name() for _ in range(num_samples)]
    elif subtype == 'last_name':
        return [fake.last_name() for _ in range(num_samples)]
    elif subtype == 'full_name':
        return [fake.name() for _ in range(num_samples)]
    elif subtype == 'city':
        return [fake.city() for _ in range(num_samples)]
    elif subtype == 'country':
        return [fake.country() for _ in range(num_samples)]
    elif subtype == 'email':
        return [fake.email() for _ in range(num_samples)]
    elif subtype == 'phone':
        return [fake.phone_number() for _ in range(num_samples)]
    elif subtype == 'address':
        return [fake.address().replace('\n', ', ') for _ in range(num_samples)]
    elif subtype == 'company':
        return [fake.company() for _ in range(num_samples)]
    elif subtype == 'job':
        return [fake.job() for _ in range(num_samples)]
    elif subtype == 'text':
        return [fake.text(max_nb_chars=100) for _ in range(num_samples)]
    elif subtype == 'custom':
        pattern = constraints.get('pattern', '')
        
        if pattern:
            # Use pattern with Faker if possible
            return [fake.bothify(pattern) for _ in range(num_samples)]
        else:
            min_length = constraints.get('min_length', 5)
            max_length = constraints.get('max_length', 20)
            
            return [
                ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', 
                                      k=random.randint(min_length, max_length)))
                for _ in range(num_samples)
            ]
    else:
        return [fake.text(max_nb_chars=50) for _ in range(num_samples)]

def generate_categorical_data(constraints: Dict[str, Any], num_samples: int) -> List[Any]:
    """
    Generate categorical data based on constraints.
    
    Args:
        constraints (Dict[str, Any]): Constraints for data generation
        num_samples (int): Number of samples to generate
        
    Returns:
        List[Any]: Generated categorical data
    """
    categories = constraints.get('categories', ['Category A', 'Category B', 'Category C'])
    weights = constraints.get('weights')
    
    if weights and len(weights) == len(categories):
        return np.random.choice(categories, num_samples, p=weights).tolist()
    else:
        return np.random.choice(categories, num_samples).tolist()

def generate_date_data(constraints: Dict[str, Any], num_samples: int) -> List[datetime]:
    """
    Generate date data based on constraints.
    
    Args:
        constraints (Dict[str, Any]): Constraints for data generation
        num_samples (int): Number of samples to generate
        
    Returns:
        List[datetime]: Generated date data
    """
    # Extract constraints
    start_date = constraints.get('start_date', datetime(2020, 1, 1))
    end_date = constraints.get('end_date', datetime(2023, 12, 31))
    
    # Calculate the range in days
    date_range = (end_date - start_date).days
    
    # Generate random dates
    random_days = np.random.randint(0, date_range + 1, num_samples)
    dates = [start_date + timedelta(days=int(days)) for days in random_days]
    
    return dates
