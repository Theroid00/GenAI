"""
Module for managing relationships between columns in synthetic data.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger('synthetic_data_generator')

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
