#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core generator functionality for synthetic data generation.
"""

import os
import sys
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
from tqdm import tqdm
import json
import csv

# Try importing SDV for model-based generation
try:
    import sdv
    from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
    from sdv.single_table import CTGANSynthesizer as CTGAN
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality as evaluate
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

# Import from other modules
from ..utils.standards import (
    detect_id_field_type, 
    generate_id, 
    format_email, 
    format_phone_number,
    get_weighted_email_domain,
    EMAIL_DOMAINS
)
from ..utils.file_io import load_csv, save_to_csv
from ..validation.validator import post_process_synthetic_data
from .relationships import apply_relationships

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('synthetic_data_generator')

# Initialize Faker
from faker import Faker
fake = Faker()

class SyntheticDataGenerator:
    """
    Main class for synthetic data generation.
    """
    
    def __init__(self):
        """Initialize the generator"""
        self.last_original_df = None
        self.last_synthetic_df = None
        self.last_schema = None
        
    def generate_from_csv(self, csv_path: str, num_rows: int = 100, 
                          model_type: str = 'gaussian', output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic data based on a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            num_rows: Number of rows to generate
            model_type: Type of model to use ('gaussian' or 'ctgan')
            output_path: Optional path to save the generated data
            
        Returns:
            DataFrame with synthetic data
        """
        # Check if SDV is available
        if not SDV_AVAILABLE:
            raise ImportError("SDV package is required for model-based generation. Install with: pip install sdv")
        
        # Load CSV
        original_df = load_csv(csv_path)
        self.last_original_df = original_df
        
        # Generate synthetic data
        synthetic_df = self._generate_synthetic_data_model_based(original_df, num_rows, model_type)
        self.last_synthetic_df = synthetic_df
        
        # Save to file if output path is provided
        if output_path:
            save_to_csv(synthetic_df, output_path)
            logger.info(f"Saved {len(synthetic_df)} rows to {output_path}")
        
        return synthetic_df
    
    def generate_from_schema(self, schema: List[Dict[str, Any]], num_rows: int = 100, 
                            output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic data based on a schema.
        
        Args:
            schema: Schema definition (list of column definitions)
            num_rows: Number of rows to generate
            output_path: Optional path to save the generated data
            
        Returns:
            DataFrame with synthetic data
        """
        self.last_schema = schema
        
        # Generate synthetic data
        synthetic_df = self._generate_synthetic_data_from_schema(schema, num_rows)
        
        # Apply relationships if defined
        if any('relationships' in col_info for col_info in schema):
            logger.info("Applying relationships between columns...")
            synthetic_df = apply_relationships(synthetic_df, schema)
        
        # Post-process the data to ensure realistic values
        logger.info("Post-processing data to make it more realistic...")
        synthetic_df = post_process_synthetic_data(pd.DataFrame(), synthetic_df)  # Empty DataFrame as original
        
        self.last_synthetic_df = synthetic_df
        
        # Save to file if output path is provided
        if output_path:
            save_to_csv(synthetic_df, output_path)
            logger.info(f"Saved {len(synthetic_df)} rows to {output_path}")
        
        return synthetic_df
    
    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
    
    def _generate_synthetic_data_model_based(self, df: pd.DataFrame, num_rows: int, 
                                           model_type: str = 'gaussian') -> pd.DataFrame:
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
    
    def _generate_synthetic_data_from_schema(self, schema: List[Dict[str, Any]], num_rows: int) -> pd.DataFrame:
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
                            batch_values = [f"placeholder-email-{i}" for i in range(current_batch)]
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
                            data[col_name] = [fake.bothify(pattern) for _ in range(num_rows)]
                        else:
                            min_length = col_info.get('min_length', 5)
                            max_length = col_info.get('max_length', 20)
                            
                            data[col_name] = [
                                ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', 
                                                      k=random.randint(min_length, max_length)))
                                for _ in range(num_rows)
                            ]
                    else:
                        data[col_name] = [fake.text(max_nb_chars=50) for _ in range(num_rows)]
                
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
    
    def evaluate_synthetic_data(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the quality of synthetic data compared to the original data.
        
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
