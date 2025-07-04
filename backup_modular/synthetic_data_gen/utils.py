"""
Utility functions for the synthetic data generator.
"""

import os
import pandas as pd
import csv
import logging
import json
from typing import Any, Dict, Tuple

# Configure logging
logger = logging.getLogger('synthetic_data_generator')

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
            from datetime import datetime
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

def load_schema_from_json(file_path: str) -> list:
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
                        from datetime import datetime
                        col_info['start_date'] = datetime.strptime(col_info['start_date'], "%Y-%m-%d")
                    except ValueError:
                        # Try ISO format
                        col_info['start_date'] = datetime.fromisoformat(col_info['start_date'].replace('Z', '+00:00'))
                
                if 'end_date' in col_info and isinstance(col_info['end_date'], str):
                    try:
                        from datetime import datetime
                        col_info['end_date'] = datetime.strptime(col_info['end_date'], "%Y-%m-%d")
                    except ValueError:
                        # Try ISO format
                        col_info['end_date'] = datetime.fromisoformat(col_info['end_date'].replace('Z', '+00:00'))
        
        return schema
    except Exception as e:
        logger.error(f"Error loading schema: {str(e)}")
        raise
