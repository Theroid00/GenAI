"""
Functions for generating synthetic data based on schema or existing data.
This module contains all functions related to synthetic data generation.
"""

import os
import numpy as np
import pandas as pd
import random
import logging
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import uuid
from faker import Faker

# Import the field standards module
from synthetic_data_gen.field_standards import (
    detect_id_field_type, 
    generate_id, 
    format_email, 
    format_phone_number,
    get_weighted_email_domain
)

# Initialize faker for generating realistic data
fake = Faker()

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

# Check if SDV is available
try:
    import sdv
    from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
    from sdv.single_table import CTGANSynthesizer as CTGAN
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

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
    
    # Pre-process name columns immediately to ensure uniqueness
    for col in synthetic_df.columns:
        col_lower = col.lower()
        if any(name_term in col_lower for name_term in ['name', 'person', 'student', 'user']) and 'id' not in col_lower:
            logger.info(f"Generating unique names for column: {col}")
            
            # Set a seed based on the column name for reproducibility but different across columns
            col_seed = hash(col) % 10000
            random.seed(col_seed)
            
            # Use multiple Faker instances with different locales to increase variety
            all_names = []
            locales = ['en_US', 'en_GB', 'en_CA', 'en_AU']
            
            # Generate more names than needed to ensure uniqueness after deduplication
            needed_names = len(synthetic_df)
            target_count = needed_names * 3  # Generate 3x as many as needed
            
            # Generate names using different patterns
            for i in range(target_count):
                locale = random.choice(locales)
                temp_faker = Faker(locale)
                
                # Add variety with different name patterns
                if i % 3 == 0:
                    # Standard full name
                    name = temp_faker.name()
                elif i % 3 == 1:
                    # Name with middle initial
                    first = temp_faker.first_name()
                    middle = temp_faker.first_name()[0] + "."
                    last = temp_faker.last_name()
                    name = f"{first} {middle} {last}"
                else:
                    # Double first or last name
                    if random.choice([True, False]):
                        first = f"{temp_faker.first_name()}-{temp_faker.first_name()}"
                        last = temp_faker.last_name()
                    else:
                        first = temp_faker.first_name()
                        last = f"{temp_faker.last_name()}-{temp_faker.last_name()}"
                    name = f"{first} {last}"
                
                all_names.append(name)
            
            # Ensure uniqueness by removing duplicates while preserving order
            unique_names = list(dict.fromkeys(all_names))
            
            # If we don't have enough unique names, generate more with even more variation
            while len(unique_names) < needed_names:
                # Generate very specific names with multiple components
                temp_faker = Faker()
                first = temp_faker.first_name()
                middle1 = temp_faker.first_name()[0] + "."
                middle2 = temp_faker.first_name()[0] + "."
                last = temp_faker.last_name()
                name = f"{first} {middle1} {middle2} {last}"
                unique_names.append(name)
            
            # Use only as many names as we need
            final_names = unique_names[:needed_names]
            # Shuffle once more for good measure
            random.shuffle(final_names)
            
            # Assign unique names to the dataframe
            result_df[col] = final_names
    
    # Continue with regular processing for other columns
    for col in synthetic_df.columns:
        # Skip name columns as we've already processed them
        col_lower = col.lower()
        if any(name_term in col_lower for name_term in ['name', 'person', 'student', 'user']) and 'id' not in col_lower:
            continue
            
        # Special handling for ID columns with UUIDs or placeholders
        id_type = detect_id_field_type(col)
        if id_type and synthetic_df[col].dtype == 'object':
            # Check if this appears to be a UUID format or placeholder
            sample_values = synthetic_df[col].head(5).astype(str)
            has_uuids = any('-' in str(val) and len(str(val)) > 30 for val in sample_values)
            has_placeholders = any('placeholder' in str(val) or 'sdv-id' in str(val) for val in sample_values)
            
            if has_uuids or has_placeholders:
                logger.info(f"Converting to proper {id_type} IDs for column: {col}")
                # Generate proper IDs based on detected type
                result_df[col] = [generate_id(id_type, i) for i in range(len(synthetic_df))]
                continue
            
        # Check if the column contains auto-generated IDs (SDV often creates "sdv-id-*" values)
        if synthetic_df[col].dtype == 'object':  # String columns
            # Check first few values to see if they match the SDV ID pattern
            sample_values = synthetic_df[col].head(5).astype(str)
            if any(str(val).startswith('sdv-id-') for val in sample_values):
                # Check if this column should maintain its original pattern
                original_sample = original_df[col].head(5).astype(str) if col in original_df.columns else []
                
                # Detect what type of data this should be
                col_lower = col.lower()
                
                # Special handling for ID columns - preserve the original pattern
                if 'id' in col_lower and len(original_sample) > 0:
                    # Try to detect the pattern from original data
                    original_pattern = str(original_sample.iloc[0]) if len(original_sample) > 0 else ""
                    
                    if original_pattern.startswith('EMP-') or 'emp' in original_pattern.lower():
                        # Generate employee ID pattern
                        logger.info(f"Generating employee ID pattern for column: {col}")
                        # Find the highest number in original data to continue the sequence
                        max_num = 0
                        for val in original_df[col].astype(str):
                            if 'EMP-' in val:
                                try:
                                    num = int(val.split('EMP-')[1])
                                    max_num = max(max_num, num)
                                except (ValueError, IndexError):
                                    pass
                        
                        # Generate new IDs starting from max_num + 1
                        new_ids = [f"EMP-{max_num + i + 1}" for i in range(len(synthetic_df))]
                        result_df[col] = new_ids
                    elif original_pattern.isdigit():
                        # Generate numeric IDs
                        logger.info(f"Generating numeric ID pattern for column: {col}")
                        start_id = max([int(x) for x in original_df[col].astype(str) if x.isdigit()], default=1000)
                        result_df[col] = [str(start_id + i + 1) for i in range(len(synthetic_df))]
                    else:
                        # Default ID generation
                        logger.info(f"Generating generic ID pattern for column: {col}")
                        result_df[col] = [f"ID-{i+1:04d}" for i in range(len(synthetic_df))]
                elif 'city' in col_lower:
                    result_df[col] = [fake.city() for _ in range(len(synthetic_df))]
                elif 'address' in col_lower:
                    result_df[col] = [fake.address().replace('\n', ', ') for _ in range(len(synthetic_df))]
                elif 'company' in col_lower or 'school' in col_lower:
                    result_df[col] = [fake.company() for _ in range(len(synthetic_df))]
                elif 'email' in col_lower:
                    # Check if this is a placeholder or auto-generated email
                    sample_values = synthetic_df[col].head(5).astype(str)
                    needs_generation = any('placeholder' in str(val) or 'sdv-id' in str(val) for val in sample_values)
                    
                    if needs_generation or 'email' in col_lower:
                        # Create realistic emails that match names if available
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
                                    
                                    # Use our standardized email formatter
                                    email = format_email(first, last)
                                    emails.append(email)
                                else:
                                    # Fallback if we can't parse the name properly
                                    domain = get_weighted_email_domain()
                                    emails.append(f"{name.lower().replace(' ', '')}@{domain}")
                            
                            result_df[col] = emails
                        else:
                            # Fallback to more realistic emails if no name column is available
                            emails = []
                            for _ in range(len(synthetic_df)):
                                username = fake.user_name()
                                domain = get_weighted_email_domain()
                                
                                # Maybe add random number for uniqueness
                                if random.random() < 0.3:
                                    username = f"{username}{random.randint(1, 999)}"
                                    
                                emails.append(f"{username}@{domain}")
                                
                            result_df[col] = emails
                elif 'phone' in col_lower:
                    # Generate properly formatted phone numbers using our standardized formatter
                    result_df[col] = [format_phone_number() for _ in range(len(synthetic_df))]
                elif 'job' in col_lower or 'occupation' in col_lower:
                    result_df[col] = [fake.job() for _ in range(len(synthetic_df))]
                elif 'country' in col_lower:
                    result_df[col] = [fake.country() for _ in range(len(synthetic_df))]
    
    return result_df

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
                logger.info(f"Generating {id_type} IDs for column: {col_name}")
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
