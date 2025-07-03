#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator
------------------------
A dual-mode tool for generating synthetic data either from existing CSV samples
or from scratch via interactive user input.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from faker import Faker
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
from tqdm import tqdm
import json
import csv  # For CSV quoting constants
import re   # For regex pattern matching

# Import field standards
from field_standards import (
    detect_id_field_type, 
    generate_id, 
    format_email, 
    format_phone_number,
    get_weighted_email_domain,
    EMAIL_DOMAINS
)

# For model-based generation using SDV
try:
    import sdv
    from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
    from sdv.single_table import CTGANSynthesizer as CTGAN
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality as evaluate
    SDV_AVAILABLE = True
    print(f"SDV package detected successfully. Version: {sdv.__version__}")
except ImportError as e:
    SDV_AVAILABLE = False
    print(f"Warning: SDV package not installed or error importing: {str(e)}")
    print("Install with: pip install sdv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('synthetic_data_generator')

# Initialize Faker
fake = Faker()

def check_dependencies():
    """
    Check if all required dependencies are installed and provide instructions if not.
    
    Returns:
        Tuple of (all_dependencies_met, message)
    """
    dependencies = {
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy',
        'matplotlib': 'pip install matplotlib',
        'faker': 'pip install faker',
        'tqdm': 'pip install tqdm'
    }
    
    optional_dependencies = {
        'sdv': 'pip install sdv'
    }
    
    missing = []
    missing_optional = []
    
    # Check required dependencies
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, install_cmd))
    
    # Check optional dependencies
    for package, install_cmd in optional_dependencies.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, install_cmd))
    
    # Build message
    message = ""
    if missing:
        message += "The following required dependencies are missing:\n"
        for package, cmd in missing:
            message += f"- {package}: {cmd}\n"
        message += "\n"
    
    if missing_optional:
        message += "The following optional dependencies are missing:\n"
        for package, cmd in missing_optional:
            message += f"- {package}: {cmd}\n"
        message += "\n"
        
        if 'sdv' in [p for p, _ in missing_optional]:
            message += "Note: SDV is required for model-based generation (Mode 1).\n"
    
    return len(missing) == 0, message

# ====== CSV HANDLING FUNCTIONS ======

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise

def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Path where the CSV will be saved
    """
    try:
        # Check if the path is a directory
        if os.path.isdir(output_path):
            # Generate a default filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"synthetic_data_{timestamp}.csv")
            print(f"Directory provided. Saving as: {output_path}")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # For large datasets, use chunked saving to manage memory
        if len(df) > 50000:
            logger.info(f"Large dataset detected ({len(df)} rows). Using chunked saving...")
            chunk_size = 10000
            
            # Save the header first
            df.iloc[:1].to_csv(output_path, index=False, sep=',', quoting=csv.QUOTE_MINIMAL, 
                             quotechar='"', encoding='utf-8')
            
            # Append remaining data in chunks
            for i in range(1, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk.to_csv(output_path, mode='a', header=False, index=False, sep=',', 
                           quoting=csv.QUOTE_MINIMAL, quotechar='"', encoding='utf-8')
                
                # Show progress for very large datasets
                if len(df) > 100000 and i % (chunk_size * 5) == 0:
                    progress = min(100, (i / len(df)) * 100)
                    logger.info(f"Saving progress: {progress:.1f}%")
        else:
            # For smaller datasets, save normally
            df.to_csv(output_path, index=False, sep=',', quoting=csv.QUOTE_MINIMAL, 
                     quotechar='"', encoding='utf-8')
        
        # Verify and fix the CSV if needed (skip for very large files to save time)
        if len(df) <= 50000:
            if not check_and_fix_csv(output_path):
                # If check_and_fix failed, try one more approach
                logger.warning("Initial CSV save had formatting issues. Trying alternative format...")
                df.to_csv(output_path, index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
                
                if not check_and_fix_csv(output_path):
                    logger.warning("Still having CSV formatting issues. Using tab delimiter as fallback...")
                    df.to_csv(output_path, index=False, sep='\t', encoding='utf-8')
        
        logger.info(f"Successfully saved {df.shape[0]} rows to {output_path}")
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        raise

def check_and_fix_csv(file_path: str) -> bool:
    """
    Check if a CSV file has proper formatting (columns are properly separated)
    and attempt to fix it if not.
    
    This function attempts multiple strategies to detect and fix CSV formatting issues:
    1. First checks if all columns are merged into one
    2. Tries different quoting settings if quoting appears to be the issue
    3. Checks for alternative delimiters like semicolons
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if the file is valid or was fixed, False otherwise
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if there's only one column and it might contain delimiter characters
        # This is the most common issue - all columns are merged into one
        if df.shape[1] == 1 and ',' in str(df.iloc[0, 0]):
            logger.warning(f"CSV file appears to have formatting issues. Attempting to fix: {file_path}")
            
            # Read the file content to analyze the issue
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # If both quotes and commas exist in the content, it might be a quoting issue
            # where the CSV writer didn't properly escape quoted fields containing commas
            if '"' in content and ',' in content:
                # Try reading with different quoting settings
                df_fixed = pd.read_csv(file_path, sep=',', quoting=csv.QUOTE_NONE, escapechar='\\')
                
                if df_fixed.shape[1] > 1:
                    # Successfully parsed with multiple columns - save with proper formatting
                    df_fixed.to_csv(file_path, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
                    logger.info(f"Fixed CSV formatting for {file_path}")
                    return True
            
            # If we still have issues, try a more aggressive approach
            try:
                # Check for alternative delimiters - some systems use semicolons instead of commas
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Analyze the header to detect potential alternative delimiters
                header = lines[0].strip()
                if header.count(',') == 0 and header.count(';') > 0:
                    # Semicolon delimiter detected
                    df_fixed = pd.read_csv(file_path, sep=';')
                    # Save with standard comma delimiter
                    df_fixed.to_csv(file_path, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
                    logger.info(f"Fixed CSV formatting (converted from semicolon to comma) for {file_path}")
                    return True
            except Exception as inner_error:
                logger.error(f"Could not fix CSV formatting: {str(inner_error)}")
        
        # If we reached here with no errors, the file is already valid
        # This means either the original file was correctly formatted,
        # or our initial read with default settings was successful
        return True
    except Exception as e:
        logger.error(f"Error checking CSV format: {str(e)}")
        return False

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

def post_process_synthetic_data(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process synthetic data to make it more realistic.
    
    Args:
        original_df: Original DataFrame (for reference)
        synthetic_df: Synthetic DataFrame to process
        
    Returns:
        Processed synthetic DataFrame
    """
    result_df = synthetic_df.copy()
    
    # Identify columns that might need processing
    for col in synthetic_df.columns:
        # Special handling for ID columns
        col_lower = col.lower()
        if ('id' in col_lower or '_id' in col_lower or 'id_' in col_lower) and synthetic_df[col].dtype == 'object':
            # Check if this appears to be a UUID format or placeholder
            sample_values = synthetic_df[col].head(5).astype(str)
            has_uuids = any('-' in str(val) and len(str(val)) > 30 for val in sample_values)
            has_placeholders = any(str(val).startswith('sdv-id-') or str(val).startswith('placeholder-') for val in sample_values)
            
            if has_uuids or has_placeholders:
                # Detect ID type and generate proper IDs
                id_type = detect_id_field_type(col)
                if id_type:
                    logger.info(f"Converting column {col} to {id_type} ID format")
                    result_df[col] = [generate_id(id_type, i) for i in range(len(synthetic_df))]
                else:
                    # If not detected as a specific ID type but still has UUIDs, use generic format
                    logger.info(f"Converting UUIDs to generic IDs for column: {col}")
                    result_df[col] = [f"ID-{10001 + i}" for i in range(len(synthetic_df))]
                continue
                
        # Check if the column contains placeholder emails
        if synthetic_df[col].dtype == 'object' and 'email' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('placeholder-email-') for val in sample_values)
            
            if has_placeholders:
                # Create more realistic emails that match names if available
                name_columns = [c for c in result_df.columns if any(term in c.lower() for term in 
                               ['name', 'firstname', 'first_name', 'lastname', 'last_name', 'fullname'])]
                
                if name_columns and not all(result_df[name_columns[0]].isnull()):
                    # Use the first available name column to generate matching emails
                    name_col = name_columns[0]
                    logger.info(f"Generating emails based on name column: {name_col}")
                    
                    emails = []
                    for idx, row in result_df.iterrows():
                        name = str(row[name_col])
                        
                        # Extract first and last name
                        parts = name.split()
                        if len(parts) >= 2:
                            first = parts[0]
                            last = parts[-1]
                            emails.append(format_email(first, last))
                        else:
                            # Fallback if we can't parse the name properly
                            domain = get_weighted_email_domain()
                            emails.append(f"{name.lower().replace(' ', '')}@{domain}")
                    
                    result_df[col] = emails
                else:
                    # Fallback to generic but realistic emails if no name column is available
                    logger.info(f"Generating generic emails for column: {col} (no matching name column found)")
                    domains = list(EMAIL_DOMAINS.keys())
                    weights = list(EMAIL_DOMAINS.values())
                    
                    result_df[col] = [
                        f"{fake.user_name()}{random.choice(['', '.', '_'])}{random.choice(['', random.randint(1, 999)])}@{random.choices(domains, weights=weights, k=1)[0]}" 
                        for _ in range(len(synthetic_df))
                    ]
        
        # Handle phone numbers
        elif synthetic_df[col].dtype == 'object' and any(phone_term in col_lower for phone_term in ['phone', 'mobile', 'cell']):
            sample_values = synthetic_df[col].head(5).astype(str)
            needs_formatting = any(not re.match(r'^\(\d{3}\) \d{3}-\d{4}$', str(val)) for val in sample_values)
            
            if needs_formatting:
                logger.info(f"Formatting phone numbers for column: {col}")
                result_df[col] = [format_phone_number() for _ in range(len(synthetic_df))]
                
        # Handle other common field types that could use better formatting
        elif any(name_term in col_lower for name_term in ['name', 'person', 'student', 'user']) and 'id' not in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                logger.info(f"Replacing synthetic IDs with realistic names for column: {col}")
                # Generate realistic names
                result_df[col] = [fake.name() for _ in range(len(synthetic_df))]
        elif 'city' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                result_df[col] = [fake.city() for _ in range(len(synthetic_df))]
        elif 'address' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                result_df[col] = [fake.address().replace('\n', ', ') for _ in range(len(synthetic_df))]
        elif 'company' in col_lower or 'school' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                result_df[col] = [fake.company() for _ in range(len(synthetic_df))]
        elif 'job' in col_lower or 'occupation' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                result_df[col] = [fake.job() for _ in range(len(synthetic_df))]
        elif 'country' in col_lower:
            sample_values = synthetic_df[col].head(5).astype(str)
            has_placeholders = any(str(val).startswith('sdv-id-') for val in sample_values)
            
            if has_placeholders:
                result_df[col] = [fake.country() for _ in range(len(synthetic_df))]
    
    return result_df

# ====== MODEL-BASED GENERATION (MODE 1) ======

def generate_synthetic_data_model_based(df: pd.DataFrame, num_rows: int, model_type: str = 'gaussian') -> pd.DataFrame:
    """
    Generate synthetic data using SDV models.
    
    Args:
        df: Sample DataFrame
        num_rows: Number of synthetic rows to generate
        model_type: Type of model to use ('gaussian' or 'ctgan')
        
    Returns:
        DataFrame with synthetic data
    """
    if not SDV_AVAILABLE:
        raise ImportError("SDV package is required for model-based generation")
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Select and train model
    if model_type.lower() == 'gaussian':
        model = GaussianCopula(metadata)
    elif model_type.lower() == 'ctgan':
        model = CTGAN(metadata)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gaussian' or 'ctgan'")
    
    logger.info(f"Training {model_type} model on sample data...")
    model.fit(df)
    
    # Generate synthetic data
    logger.info(f"Generating {num_rows} synthetic rows...")
    synthetic_data = model.sample(num_rows)
    
    # Post-process generated data to make it more realistic
    synthetic_data = post_process_synthetic_data(df, synthetic_data)
    
    return synthetic_data

def compare_distributions(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                         num_cols: int = 3) -> None:
    """
    Generate histograms comparing original and synthetic data distributions.
    
    Args:
        original_df: Original DataFrame
        synthetic_df: Synthetic DataFrame
        num_cols: Number of columns to visualize (defaults to 3)
    """
    # Select numeric columns for comparison
    numeric_cols = [col for col in original_df.columns if 
                   pd.api.types.is_numeric_dtype(original_df[col])]
    
    if not numeric_cols:
        logger.warning("No numeric columns available for distribution comparison")
        return
    
    # Limit to specified number of columns
    cols_to_plot = numeric_cols[:min(num_cols, len(numeric_cols))]
    
    # Create subplots
    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3*len(cols_to_plot)))
    
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    # Plot histograms
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(original_df[col], alpha=0.5, label='Original', bins=15)
        axes[i].hist(synthetic_df[col], alpha=0.5, label='Synthetic', bins=15)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# ====== INTERACTIVE SCHEMA DEFINITION (MODE 2) ======

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

def define_column_interactively() -> Dict[str, Any]:
    """
    Define a column schema interactively.
    
    Returns:
        Dictionary with column schema information
    """
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

# ====== SYNTHETIC DATA GENERATION (MODE 2) ======

def generate_synthetic_data_from_schema(schema: List[Dict[str, Any]], num_rows: int) -> pd.DataFrame:
    """
    Generate synthetic data based on the provided schema.
    
    Args:
        schema: List of column definitions
        num_rows: Number of rows to generate
        
    Returns:
        DataFrame with synthetic data
    """
    data = {}
    
    # For large datasets, use batch processing to manage memory
    batch_size = min(10000, num_rows) if num_rows > 50000 else num_rows
    
    logger.info(f"Generating {num_rows} rows with batch size of {batch_size}")
    
    for col_info in tqdm(schema, desc="Generating data", unit="column"):
        col_name = col_info['name']
        col_type = col_info['type']
        
        # Generate data based on column type
        if col_type == 'int':
            if col_info.get('distribution') == 'normal':
                mean = col_info['mean']
                std = col_info['std']
                min_val = col_info['min']
                max_val = col_info['max']
                
                # Generate values from normal distribution with truncation
                # Use memory-efficient approach for large datasets
                if num_rows > 50000:
                    values = []
                    for i in range(0, num_rows, batch_size):
                        current_batch = min(batch_size, num_rows - i)
                        batch_values = np.random.normal(mean, std, current_batch)
                        batch_values = np.clip(batch_values, min_val, max_val)
                        values.extend(batch_values.astype(int))
                    values = np.array(values)
                else:
                    values = np.random.normal(mean, std, num_rows)
                    values = np.clip(values, min_val, max_val)
                    values = values.astype(int)
            else:
                min_val = col_info['min']
                max_val = col_info['max']
                values = np.random.randint(min_val, max_val + 1, num_rows)
                
            data[col_name] = values
            
        elif col_type == 'float':
            decimals = col_info.get('decimals', 2)
            
            if col_info.get('distribution') == 'normal':
                mean = col_info['mean']
                std = col_info['std']
                min_val = col_info['min']
                max_val = col_info['max']
                
                # Generate values from normal distribution with truncation
                # Use memory-efficient approach for large datasets
                if num_rows > 50000:
                    values = []
                    for i in range(0, num_rows, batch_size):
                        current_batch = min(batch_size, num_rows - i)
                        batch_values = np.random.normal(mean, std, current_batch)
                        batch_values = np.clip(batch_values, min_val, max_val)
                        values.extend(batch_values)
                    values = np.array(values)
                else:
                    values = np.random.normal(mean, std, num_rows)
                    values = np.clip(values, min_val, max_val)
            else:
                min_val = col_info['min']
                max_val = col_info['max']
                values = np.random.uniform(min_val, max_val, num_rows)
                
            # Round to specified decimal places
            values = np.round(values, decimals)
            data[col_name] = values
            
        elif col_type == 'string':
            subtype = col_info.get('subtype', 'text')
            
            # For large datasets, use batch generation to improve performance
            if num_rows > 50000:
                values = []
                for i in range(0, num_rows, batch_size):
                    current_batch = min(batch_size, num_rows - i)
                    if subtype == 'name':
                        batch_values = [fake.name() for _ in range(current_batch)]
                    elif subtype == 'first_name':
                        batch_values = [fake.first_name() for _ in range(current_batch)]
                    elif subtype == 'last_name':
                        batch_values = [fake.last_name() for _ in range(current_batch)]
                    elif subtype == 'full_name':
                        batch_values = [fake.name() for _ in range(current_batch)]
                    elif subtype == 'city':
                        batch_values = [fake.city() for _ in range(current_batch)]
                    elif subtype == 'country':
                        batch_values = [fake.country() for _ in range(current_batch)]
                    elif subtype == 'email':
                        batch_values = [fake.email() for _ in range(current_batch)]
                    elif subtype == 'phone':
                        batch_values = [fake.phone_number() for _ in range(current_batch)]
                    elif subtype == 'address':
                        batch_values = [fake.address().replace('\n', ', ') for _ in range(current_batch)]
                    elif subtype == 'company':
                        batch_values = [fake.company() for _ in range(current_batch)]
                    elif subtype == 'job':
                        batch_values = [fake.job() for _ in range(current_batch)]
                    elif subtype == 'text':
                        batch_values = [fake.text(max_nb_chars=100) for _ in range(current_batch)]
                    elif subtype == 'custom':
                        pattern = col_info.get('pattern', '')
                        if pattern:
                            batch_values = [fake.bothify(pattern) for _ in range(current_batch)]
                        else:
                            min_length = col_info.get('min_length', 5)
                            max_length = col_info.get('max_length', 20)
                            batch_values = [
                                ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', 
                                                      k=random.randint(min_length, max_length)))
                                for _ in range(current_batch)
                            ]
                    else:
                        batch_values = [fake.text(max_nb_chars=50) for _ in range(current_batch)]
                    
                    values.extend(batch_values)
                data[col_name] = values
            else:
                # For smaller datasets, generate all at once
                if subtype == 'name':
                    data[col_name] = [fake.name() for _ in range(num_rows)]
                elif subtype == 'first_name':
                    data[col_name] = [fake.first_name() for _ in range(num_rows)]
                elif subtype == 'last_name':
                    data[col_name] = [fake.last_name() for _ in range(num_rows)]
                elif subtype == 'full_name':
                    data[col_name] = [fake.name() for _ in range(num_rows)]
                elif subtype == 'city':
                    data[col_name] = [fake.city() for _ in range(num_rows)]
                elif subtype == 'country':
                    data[col_name] = [fake.country() for _ in range(num_rows)]
                elif subtype == 'email':
                    # Don't generate emails here, we'll handle them in post-processing
                    # This placeholder will be replaced with proper emails that match names
                    data[col_name] = [f"placeholder-email-{i}" for i in range(num_rows)]
                elif subtype == 'phone':
                    data[col_name] = [fake.phone_number() for _ in range(num_rows)]
                elif subtype == 'address':
                    data[col_name] = [fake.address().replace('\n', ', ') for _ in range(num_rows)]
                elif subtype == 'company':
                    data[col_name] = [fake.company() for _ in range(num_rows)]
                elif subtype == 'job':
                    data[col_name] = [fake.job() for _ in range(num_rows)]
                elif subtype == 'text':
                    data[col_name] = [fake.text(max_nb_chars=100) for _ in range(num_rows)]
                elif subtype == 'custom':
                    pattern = col_info.get('pattern', '')
                    
                    if pattern:
                        # Use pattern with Faker if possible
                        # This is a simplistic approach - for complex patterns, more logic would be needed
                        data[col_name] = [fake.bothify(pattern) for _ in range(num_rows)]
                    else:
                        min_length = col_info.get('min_length', 5)
                        max_length = col_info.get('max_length', 20)
                        
                        data[col_name] = [
                            ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', 
                                                  k=random.randint(min_length, max_length)))
                            for _ in range(num_rows)
                        ]
            
        elif col_type == 'category':
            categories = col_info['categories']
            weights = col_info.get('weights')
            
            if weights:
                data[col_name] = np.random.choice(categories, num_rows, p=weights)
            else:
                data[col_name] = np.random.choice(categories, num_rows)
                
        elif col_type == 'date':
            start_date = col_info['start_date']
            end_date = col_info['end_date']
            
            # Calculate the range in days
            date_range = (end_date - start_date).days
            
            # Generate random dates efficiently
            random_days = np.random.randint(0, date_range + 1, num_rows)
            
            # For large datasets, use batch processing for date generation
            if num_rows > 50000:
                dates = []
                for i in range(0, num_rows, batch_size):
                    current_batch = min(batch_size, num_rows - i)
                    batch_days = random_days[i:i+current_batch]
                    batch_dates = [start_date + timedelta(days=int(days)) for days in batch_days]
                    dates.extend(batch_dates)
                data[col_name] = dates
            else:
                dates = [start_date + timedelta(days=int(days)) for days in random_days]
                data[col_name] = dates
            
        elif col_type == 'uuid':
            # Check if the column appears to be an ID field
            id_type = detect_id_field_type(col_name)
            
            if id_type:
                # Generate IDs with proper patterns based on the detected type
                logger.info(f"Generated {id_type} IDs for column: {col_name}")
                data[col_name] = [generate_id(id_type, i) for i in range(num_rows)]
            else:
                # Generate regular UUIDs for non-ID columns
                # For large datasets, generate UUIDs in batches to improve performance
                if num_rows > 50000:
                    uuids = []
                    for i in range(0, num_rows, batch_size):
                        current_batch = min(batch_size, num_rows - i)
                        batch_uuids = [str(uuid.uuid4()) for _ in range(current_batch)]
                        uuids.extend(batch_uuids)
                    data[col_name] = uuids
                else:
                    data[col_name] = [str(uuid.uuid4()) for _ in range(num_rows)]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# ====== MAIN FUNCTIONALITY ======

def mode_1_generate_from_csv():
    """Function to handle Mode 1: Generate from CSV sample"""
    print("\n=== Mode 1: Generate Synthetic Data From a Sample CSV ===")
    
    # Get CSV path
    while True:
        file_path = input("Enter path to CSV file: ").strip()
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            break
        print("File not found or not a CSV. Please enter a valid path.")
    
    # Load CSV and infer schema
    df = load_csv(file_path)
    print(f"\nLoaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nSample data:")
    print(df.head())
    
    # Infer and display schema
    schema = infer_schema(df)
    display_schema_info(schema)
    
    if not SDV_AVAILABLE:
        print("Warning: SDV package not installed. Please install it to use this mode.")
        print("Install with: pip install sdv")
        return
    
    # Get number of rows to generate
    default_rows = "100" if not any("large" in input().lower() for _ in [""]) else "100"
    num_rows = int(prompt_for_input("Number of synthetic rows to generate", default_rows))
    
    # Warn about memory usage for very large datasets
    if num_rows > 500000:
        print(f"\nWarning: Generating {num_rows:,} rows will require significant memory and processing time.")
        print("Consider generating in smaller batches if you encounter memory issues.")
        confirm = prompt_for_input("Continue? (yes/no)", "yes").lower()
        if confirm != 'yes':
            print("Operation cancelled.")
            return
    
    # Get model type
    model_type = prompt_for_input("Model type (gaussian/ctgan)", "gaussian").lower()
    
    # Option to save schema
    save_schema = prompt_for_input("Save inferred schema to JSON file? (yes/no)", "no").lower()
    if save_schema == 'yes':
        schema_path = prompt_for_input("Schema JSON path", "inferred_schema.json")
        try:
            # Convert schema to serializable format
            serializable_schema = []
            for col, info in schema.items():
                col_info = dict(info)
                col_info['name'] = col
                
                # Convert numpy types to native Python types
                for key, value in col_info.items():
                    if isinstance(value, np.generic):
                        col_info[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        col_info[key] = value.tolist()
                
                serializable_schema.append(col_info)
            
            with open(schema_path, 'w') as f:
                json.dump(serializable_schema, f, indent=2, default=str)
            print(f"Schema saved to {schema_path}")
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            print(f"Error saving schema: {str(e)}")
    
    # Generate synthetic data with progress bar
    try:
        print("\nGenerating synthetic data...")
        with tqdm(total=100) as pbar:
            # Update pbar in callback
            def progress_callback(progress):
                pbar.update(int(progress * 100) - pbar.n)
            
            synthetic_df = generate_synthetic_data_model_based(df, num_rows, model_type)
            pbar.update(100 - pbar.n)  # Ensure we reach 100%
        
        print("\nGenerated synthetic data:")
        print(synthetic_df.head())
        
        # Validate the synthetic data
        validate_data = prompt_for_input("Validate synthetic data quality? (yes/no)", "yes").lower()
        if validate_data == 'yes':
            metrics = validate_synthetic_data(df, synthetic_df)
            display_validation_results(metrics)
        
        # Save to CSV
        output_path = prompt_for_input("Output CSV path (include filename)", "synthetic_data.csv")
        save_to_csv(synthetic_df, output_path)
        
        # Optionally compare distributions
        compare_dist = prompt_for_input("Compare distributions? (yes/no)", "yes").lower()
        if compare_dist == 'yes':
            num_cols = int(prompt_for_input("Number of columns to compare", "3"))
            compare_distributions(df, synthetic_df, num_cols)
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")

def mode_2_generate_interactive():
    """Function to handle Mode 2: Generate from interactive input"""
    print("\n=== Mode 2: Generate Synthetic Data From Scratch ===")
    
    # Get schema definition
    title, num_rows, schema = interactive_schema_prompt()
    
    # Define relationships between columns
    schema = add_relationships_to_schema(schema)
    
    # Option to save schema
    save_schema = prompt_for_input("Save schema to JSON file? (yes/no)", "no").lower()
    if save_schema == 'yes':
        schema_path = prompt_for_input("Schema JSON path", f"{title.lower().replace(' ', '_')}_schema.json")
        try:
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
            print(f"Schema saved to {schema_path}")
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            print(f"Error saving schema: {str(e)}")        # Generate synthetic data with progress bar
    try:
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        
        with tqdm(total=num_rows) as pbar:
            # First generate the basic synthetic data
            batch_size = min(1000, num_rows)  # Process in batches for large datasets
            synthetic_df = pd.DataFrame()
            
            for i in range(0, num_rows, batch_size):
                current_batch_size = min(batch_size, num_rows - i)
                batch_df = generate_synthetic_data_from_schema(schema, current_batch_size)
                synthetic_df = pd.concat([synthetic_df, batch_df], ignore_index=True)
                pbar.update(current_batch_size)
            
            # Apply relationships if defined
            if any('relationships' in col_info for col_info in schema):
                print("\nApplying relationships between columns...")
                synthetic_df = apply_relationships(synthetic_df, schema)
            
            # Post-process the data to ensure realistic emails, etc.
            print("\nPost-processing data to make it more realistic...")
            synthetic_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)  # Empty DataFrame as original
        
        print(f"\nGenerated {num_rows} rows of synthetic data for '{title}':")
        print(synthetic_df.head())
        
        # Save to CSV
        default_output = f"{title.lower().replace(' ', '_')}.csv"
        output_path = prompt_for_input("Output CSV path (include filename)", default_output)
        save_to_csv(synthetic_df, output_path)
        
        # Option to visualize distributions
        visualize = prompt_for_input("Visualize data distributions? (yes/no)", "no").lower()
        if visualize == 'yes':
            numeric_cols = [col for col in synthetic_df.columns if pd.api.types.is_numeric_dtype(synthetic_df[col])]
            if numeric_cols:
                num_cols_to_plot = min(3, len(numeric_cols))
                cols_to_plot = numeric_cols[:num_cols_to_plot]
                
                fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3*len(cols_to_plot)))
                if len(cols_to_plot) == 1:
                    axes = [axes]
                
                for i, col in enumerate(cols_to_plot):
                    axes[i].hist(synthetic_df[col], bins=15)
                    axes[i].set_title(f'Distribution of {col}')
                
                plt.tight_layout()
                plt.show()
            else:
                print("No numeric columns available for visualization.")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")

def load_schema_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a schema definition from a JSON file.
    
    Args:
        file_path: Path to the JSON schema file
        
    Returns:
        List of column definitions
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Schema file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            schema = json.load(f)
        
        logger.info(f"Successfully loaded schema with {len(schema)} columns")
        
        # Convert date strings back to datetime objects if needed
        for col_info in schema:
            if col_info.get('type') == 'date':
                if 'start_date' in col_info and isinstance(col_info['start_date'], str):
                    try:
                        col_info['start_date'] = datetime.strptime(col_info['start_date'], "%Y-%m-%d")
                    except ValueError:
                        # Try ISO format
                        col_info['start_date'] = datetime.fromisoformat(col_info['start_date'].replace('Z', '+00:00'))
                
                if 'end_date' in col_info and isinstance(col_info['end_date'], str):
                    try:
                        col_info['end_date'] = datetime.strptime(col_info['end_date'], "%Y-%m-%d")
                    except ValueError:
                        # Try ISO format
                        col_info['end_date'] = datetime.fromisoformat(col_info['end_date'].replace('Z', '+00:00'))
        
        return schema
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        raise

def mode_3_generate_from_schema():
    """Function to handle Mode 3: Generate from saved schema"""
    print("\n=== Mode 3: Generate Synthetic Data From Saved Schema ===")
    
    # Get schema file path
    while True:
        file_path = input("Enter path to schema JSON file: ").strip()
        if os.path.exists(file_path) and file_path.endswith('.json'):
            break
        print("File not found or not a JSON. Please enter a valid path.")
    
    # Load schema
    try:
        schema = load_schema_from_json(file_path)
        
        # Display schema info
        print(f"\nLoaded schema with {len(schema)} columns:")
        for col_info in schema:
            print(f"- {col_info['name']} ({col_info['type']})")
        
        # Get number of rows to generate
        num_rows = int(prompt_for_input("Number of synthetic rows to generate", "100"))
        
        # Generate synthetic data with progress bar
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        
        with tqdm(total=num_rows) as pbar:
            # First generate the basic synthetic data
            batch_size = min(1000, num_rows)  # Process in batches for large datasets
            synthetic_df = pd.DataFrame()
            
            for i in range(0, num_rows, batch_size):
                current_batch_size = min(batch_size, num_rows - i)
                batch_df = generate_synthetic_data_from_schema(schema, current_batch_size)
                synthetic_df = pd.concat([synthetic_df, batch_df], ignore_index=True)
                pbar.update(current_batch_size)
            
            # Apply relationships if defined
            if any('relationships' in col_info for col_info in schema):
                print("\nApplying relationships between columns...")
                synthetic_df = apply_relationships(synthetic_df, schema)
            
            # Post-process the data to ensure realistic emails, etc.
            print("\nPost-processing data to make it more realistic...")
            synthetic_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)  # Empty DataFrame as original
        
        # Get title from schema file
        title = os.path.basename(file_path).replace('_schema.json', '').replace('.json', '')
        
        print(f"\nGenerated {num_rows} rows of synthetic data:")
        print(synthetic_df.head())
        
        # Save to CSV
        default_output = f"{title}_synthetic.csv"
        output_path = prompt_for_input("Output CSV path (include filename)", default_output)
        save_to_csv(synthetic_df, output_path)
        
        # Option to visualize distributions
        visualize = prompt_for_input("Visualize data distributions? (yes/no)", "no").lower()
        if visualize == 'yes':
            numeric_cols = [col for col in synthetic_df.columns if pd.api.types.is_numeric_dtype(synthetic_df[col])]
            if numeric_cols:
                num_cols_to_plot = min(3, len(numeric_cols))
                cols_to_plot = numeric_cols[:num_cols_to_plot]
                
                fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3*len(cols_to_plot)))
                if len(cols_to_plot) == 1:
                    axes = [axes]
                
                for i, col in enumerate(cols_to_plot):
                    axes[i].hist(synthetic_df[col], bins=15)
                    axes[i].set_title(f'Distribution of {col}')
                
                plt.tight_layout()
                plt.show()
            else:
                print("No numeric columns available for visualization.")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")

def validate_synthetic_data(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the quality of synthetic data compared to the original data.
    
    Args:
        original_df: Original DataFrame
        synthetic_df: Synthetic DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    if not SDV_AVAILABLE:
        logger.warning("SDV package is required for data validation")
        return {}
    
    try:
        # Create metadata for evaluation
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_df)
        
        # Use SDV's evaluate function to get quality score
        logger.info("Evaluating synthetic data quality...")
        quality_report = evaluate(synthetic_df, original_df, metadata)
        
        # Calculate additional metrics
        metrics = {
            'sdv_quality_score': quality_report,
        }
        
        # Add column-specific metrics for numeric columns
        for col in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                orig_mean = original_df[col].mean()
                synth_mean = synthetic_df[col].mean()
                orig_std = original_df[col].std()
                synth_std = synthetic_df[col].std()
                
                # Mean difference as percentage
                if orig_mean != 0:
                    mean_diff_pct = abs((synth_mean - orig_mean) / orig_mean) * 100
                    metrics[f'{col}_mean_diff_pct'] = mean_diff_pct
                
                # Std deviation difference as percentage
                if orig_std != 0:
                    std_diff_pct = abs((synth_std - orig_std) / orig_std) * 100
                    metrics[f'{col}_std_diff_pct'] = std_diff_pct
        
        return metrics
    except Exception as e:
        logger.error(f"Error validating synthetic data: {str(e)}")
        return {}

def display_validation_results(metrics: Dict[str, Any]) -> None:
    """
    Display the validation results in a readable format.
    
    Args:
        metrics: Dictionary with quality metrics
    """
    if not metrics:
        print("No validation metrics available.")
        return
    
    print("\n=== Synthetic Data Validation Results ===")
    
    # Display overall quality score
    if 'sdv_quality_score' in metrics:
        # Handle the case where sdv_quality_score is a QualityReport object
        if hasattr(metrics['sdv_quality_score'], 'get_score'):
            score = metrics['sdv_quality_score'].get_score()
            print(f"Overall Quality Score: {score:.4f}")
        else:
            # Try to convert to string first to avoid formatting issues
            print(f"Overall Quality Score: {metrics['sdv_quality_score']}")
        print("(Score ranges from 0 to 1, where 1 means perfect similarity)")
    
    # Display column-specific metrics
    col_metrics = {k: v for k, v in metrics.items() if k != 'sdv_quality_score'}
    
    if col_metrics:
        print("\nColumn-specific Metrics:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.2f}%")

def add_relationships_to_schema(schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add relationships between columns based on user input.
    
    Args:
        schema: List of column definitions
        
    Returns:
        Updated schema with relationship information
    """
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

def apply_relationships(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply defined relationships between columns in the synthetic data.
    
    Args:
        df: DataFrame with synthetic data
        schema: Schema with relationship definitions
        
    Returns:
        Updated DataFrame with relationships applied
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Process each column that has relationships defined
    for col_info in schema:
        if 'relationships' not in col_info:
            continue
        
        src_col_name = col_info['name']
        
        for relationship in col_info['relationships']:
            tgt_col_name = relationship['target_column']
            rel_type = relationship['type']
            
            # Apply the relationship based on type
            if rel_type == 'correlation':
                if 'coefficient' in relationship and src_col_name in df.columns and tgt_col_name in df.columns:
                    # Only applicable for numeric columns
                    if (pd.api.types.is_numeric_dtype(df[src_col_name]) and 
                        pd.api.types.is_numeric_dtype(df[tgt_col_name])):
                        
                        coef = relationship['coefficient']
                        
                        # Find target column info in schema
                        tgt_col_info = next((c for c in schema if c['name'] == tgt_col_name), None)
                        
                        if tgt_col_info:
                            # Generate correlated data
                            src_data = df[src_col_name].values
                            
                            # Normalize source data
                            src_norm = (src_data - np.mean(src_data)) / np.std(src_data)
                            
                            # Generate random data for target
                            tgt_random = np.random.normal(0, 1, len(df))
                            
                            # Combine based on correlation coefficient
                            tgt_correlated = coef * src_norm + np.sqrt(1 - coef**2) * tgt_random
                            
                            # Scale back to target range
                            tgt_min = tgt_col_info.get('min', 0)
                            tgt_max = tgt_col_info.get('max', 100)
                            tgt_mean = tgt_col_info.get('mean', (tgt_min + tgt_max) / 2)
                            tgt_std = tgt_col_info.get('std', (tgt_max - tgt_min) / 6)
                            
                            tgt_data = tgt_mean + tgt_std * tgt_correlated
                            
                            # Apply constraints
                            tgt_data = np.clip(tgt_data, tgt_min, tgt_max)
                            
                            # Convert to correct data type
                            if tgt_col_info['type'] == 'int':
                                tgt_data = np.round(tgt_data).astype(int)
                            else:
                                decimals = tgt_col_info.get('decimals', 2)
                                tgt_data = np.round(tgt_data, decimals)
                            
                            # Update the dataframe
                            result_df[tgt_col_name] = tgt_data
            
            elif rel_type == 'dependency':
                if 'mapping' in relationship and src_col_name in df.columns:
                    mapping = relationship['mapping']
                    
                    # Find all source values that have mappings
                    mapped_src_values = list(mapping.keys())
                    
                    # Create a new target column
                    new_tgt_values = result_df[tgt_col_name].copy()
                    
                    # Find target column info in schema
                    tgt_col_info = next((c for c in schema if c['name'] == tgt_col_name), None)
                    
                    if tgt_col_info:
                        # Apply mappings for each source value
                        for src_value in mapped_src_values:
                            # Find rows with this source value
                            mask = (result_df[src_col_name] == src_value)
                            
                            if tgt_col_info['type'] in ['int', 'float']:
                                # For numeric target, apply center and variance
                                if isinstance(mapping[src_value], dict):
                                    center = mapping[src_value].get('center', 0)
                                    variance = mapping[src_value].get('variance', 10)
                                    
                                    # Generate values around center with variance
                                    n_values = mask.sum()
                                    if n_values > 0:
                                        values = np.random.normal(center, variance, n_values)
                                        
                                        # Apply constraints
                                        if 'min' in tgt_col_info:
                                            values = np.maximum(values, tgt_col_info['min'])
                                        if 'max' in tgt_col_info:
                                            values = np.minimum(values, tgt_col_info['max'])
                                        
                                        # Convert to correct data type
                                        if tgt_col_info['type'] == 'int':
                                            values = np.round(values).astype(int)
                                        else:
                                            decimals = tgt_col_info.get('decimals', 2)
                                            values = np.round(values, decimals)
                                        
                                        # Update values
                                        new_tgt_values[mask] = values
                            else:
                                # For non-numeric target, directly apply the mapping
                                new_tgt_values[mask] = mapping[src_value]
                        
                        # Update the dataframe
                        result_df[tgt_col_name] = new_tgt_values
            
            elif rel_type == 'transformation':
                if 'formula' in relationship and src_col_name in df.columns:
                    formula = relationship['formula']
                    
                    try:
                        # Apply the formula (x represents the source column)
                        x = df[src_col_name].values
                        new_values = eval(formula)
                        
                        # Find target column info in schema
                        tgt_col_info = next((c for c in schema if c['name'] == tgt_col_name), None)
                        
                        if tgt_col_info:
                            # Apply constraints if needed
                            if 'min' in tgt_col_info:
                                new_values = np.maximum(new_values, tgt_col_info['min'])
                            if 'max' in tgt_col_info:
                                new_values = np.minimum(new_values, tgt_col_info['max'])
                            
                            # Convert to correct data type
                            if tgt_col_info['type'] == 'int':
                                new_values = np.round(new_values).astype(int)
                            elif tgt_col_info['type'] == 'float':
                                decimals = tgt_col_info.get('decimals', 2)
                                new_values = np.round(new_values, decimals)
                            
                            # Update the dataframe
                            result_df[tgt_col_name] = new_values
                    except Exception as e:
                        logger.error(f"Error applying transformation: {str(e)}")
    
    return result_df

# ====== SYNTHETIC DATA GENERATOR CLASS ======

class SyntheticDataGenerator:
    """
    Main class for the Synthetic Data Generator.
    This class provides an object-oriented interface to the generator functions.
    """
    
    def __init__(self):
        """Initialize the generator"""
        self.last_original_df = None
        self.last_synthetic_df = None
        self.last_schema = None
    
    def generate_from_csv(self, csv_path: str, num_rows: int = 100, model_type: str = 'gaussian') -> pd.DataFrame:
        """
        Generate synthetic data based on a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            num_rows: Number of rows to generate
            model_type: Type of model to use ('gaussian' or 'ctgan')
            
        Returns:
            DataFrame with synthetic data
        """
        # Load CSV
        original_df = load_csv(csv_path)
        self.last_original_df = original_df
        
        # Generate synthetic data
        synthetic_df = generate_synthetic_data_model_based(original_df, num_rows, model_type)
        self.last_synthetic_df = synthetic_df
        
        return synthetic_df
    
    def generate_from_schema(self, schema, num_rows: int = 100) -> pd.DataFrame:
        """
        Generate synthetic data based on a schema.
        
        Args:
            schema: Schema definition (either list or dict with 'schema' key)
            num_rows: Number of rows to generate
            
        Returns:
            DataFrame with synthetic data
        """
        # Extract schema if needed
        if isinstance(schema, dict) and 'schema' in schema:
            schema = schema['schema']
        
        self.last_schema = schema
        
        # Generate synthetic data
        synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
        self.last_synthetic_df = synthetic_df
        
        return synthetic_df
    
    def visualize_data(self, num_cols: int = 3) -> None:
        """
        Visualize comparison between original and synthetic data.
        
        Args:
            num_cols: Number of columns to visualize
        """
        if self.last_original_df is None or self.last_synthetic_df is None:
            logger.warning("No data available for visualization")
            print("No data available for visualization. Generate data first.")
            return
        
        compare_distributions(self.last_original_df, self.last_synthetic_df, num_cols)
    
    def interactive_mode(self) -> None:
        """Start the interactive mode for schema definition and data generation"""
        title, num_rows, schema = interactive_schema_prompt()
        
        # Apply relationships if requested
        schema = add_relationships_to_schema(schema)
        
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
        
        # Post-process the data to ensure realistic emails, etc.
        print("\nPost-processing data to make it more realistic...")
        synthetic_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)  # Empty DataFrame as original
        
        self.last_synthetic_df = synthetic_df
        self.last_schema = schema
        
        print("\nSample of generated data:")
        print(synthetic_df.head())
        
        # Ask for output file
        output_path = prompt_for_input(
            "Output CSV path (include filename)", 
            f"{title.lower().replace(' ', '_')}.csv"
        )
        
        save_to_csv(synthetic_df, output_path)
        print(f"Saved {num_rows} rows to {output_path}")

def show_example_usage():
    """Display example usage of the synthetic data generator"""
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    
    print("""
Example 1: Student Heights Dataset (Mode 2)
-------------------------------------------
This example creates a dataset with student information including heights that follow
a normal distribution and have a correlation with age.

When prompted:
- Dataset context: "Heights of students in schools"
- Dataset title: "Student Heights in Bangalore"
- Number of rows: 100

Columns:
1. student_id (UUID)
   - Type: uuid

2. name (Full Name)
   - Type: string
   - Subtype: full_name

3. age (Integer between 14-17)
   - Type: int
   - Min: 14
   - Max: 17
   - Distribution: uniform

4. school (School Name)
   - Type: string
   - Subtype: company (used for school names)

5. height_cm (Normal distribution)
   - Type: float
   - Min: 145
   - Max: 185
   - Distribution: normal
   - Mean: 160
   - Std: 10
   - Decimal places: 1

Relationships:
- Source: age, Target: height_cm
  - Type: correlation
  - Coefficient: 0.7
  (This creates a positive correlation where older students tend to be taller)

Example 2: Sales Transactions (Mode 2)
--------------------------------------
This example creates a dataset with sales transaction information.

When prompted:
- Dataset context: "Retail sales transactions"
- Dataset title: "Electronics Store Sales"
- Number of rows: 500

Columns:
1. transaction_id (UUID)
   - Type: uuid

2. date (Date range)
   - Type: date
   - Start date: 2023-01-01
   - End date: 2023-12-31

3. product_category (Category)
   - Type: category
   - Categories: Laptop, Smartphone, Tablet, Headphones, Camera
   - Custom probabilities: 0.3, 0.4, 0.1, 0.15, 0.05

4. price (Float)
   - Type: float
   - Min: 50
   - Max: 2000
   - Distribution: uniform

5. quantity (Integer)
   - Type: int
   - Min: 1
   - Max: 5
   - Distribution: uniform

6. customer_name (Name)
   - Type: string
   - Subtype: full_name

Relationships:
- Source: product_category, Target: price
  - Type: dependency
  - Mapping:
    - Laptop: center=1200, variance=300
    - Smartphone: center=800, variance=200
    - Tablet: center=500, variance=100
    - Headphones: center=150, variance=50
    - Camera: center=600, variance=150

- Source: price, Target: quantity
  - Type: correlation
  - Coefficient: -0.5
  (This creates a negative correlation where higher priced items are purchased in smaller quantities)
    """)
    
    input("\nPress Enter to return to the main menu...")

def main():
    """Main function to run the synthetic data generator"""
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR")
    print("=" * 80)
    print("\nThis tool allows you to generate synthetic data in three modes:")
    print("  1. Generate from a sample CSV file using model-based approach")
    print("  2. Generate from scratch via interactive input")
    print("  3. Generate from saved schema file")
    
    all_dependencies_met, message = check_dependencies()
    if message:
        print("\n=== Dependency Check ===")
        print(message)
    
    if not all_dependencies_met:
        print("Please install the required dependencies before continuing.")
        return
    
    while True:
        print("\nSelect mode:")
        print("  1. Generate from CSV sample")
        print("  2. Generate from scratch (interactive)")
        print("  3. Generate from saved schema")
        print("  e. View example usage")
        print("  q. Quit")
        
        choice = input("\nEnter your choice (1/2/3/e/q): ").strip().lower()
        
        if choice == '1':
            mode_1_generate_from_csv()
        elif choice == '2':
            mode_2_generate_interactive()
        elif choice == '3':
            mode_3_generate_from_schema()
        elif choice == 'e':
            show_example_usage()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, e, or q.")

if __name__ == "__main__":
    main()
