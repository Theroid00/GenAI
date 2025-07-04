"""
Module for schema handling in synthetic data generation.
"""

import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple
from faker import Faker
from datetime import datetime

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

# Initialize faker
fake = Faker()

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

def load_schema_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a schema definition from a JSON file.
    
    Args:
        file_path: Path to the JSON schema file
        
    Returns:
        List of column definitions
    """
    try:
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
