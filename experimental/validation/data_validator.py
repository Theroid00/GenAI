#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Validation Module for Synthetic Data Generator
--------------------------------------------------
This module provides functionality to detect and fix issues in generated synthetic data,
ensuring consistency, realism, and adherence to business rules.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import sys
import os

# Add parent directory to path so we can import from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import field standards if available
try:
    from field_standards import (
        detect_id_field_type, 
        generate_id, 
        format_email, 
        format_phone_number,
        EMAIL_DOMAINS
    )
    FIELD_STANDARDS_AVAILABLE = True
except ImportError:
    FIELD_STANDARDS_AVAILABLE = False
    print("Warning: field_standards module not found. Some validation features will be limited.")

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_validator')

class DataValidator:
    """Validates and corrects synthetic data for consistency and realism."""
    
    def __init__(self, df: pd.DataFrame, schema: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize with a dataframe and optional schema.
        
        Args:
            df: The synthetic dataframe to validate
            schema: Optional schema providing column metadata
        """
        self.df = df.copy()
        self.schema = schema
        self.corrections_made = 0
        self.correction_log = []
        self.field_standards_available = FIELD_STANDARDS_AVAILABLE
        
    def validate_and_fix(self) -> pd.DataFrame:
        """
        Run all validation and correction methods on the dataframe.
        
        Returns:
            Corrected dataframe
        """
        # Run basic validations first
        self._remove_null_rows()
        self._fix_data_types()
        
        # Field-specific validations
        self._validate_id_formats()
        self._validate_name_formats()
        self._validate_name_email_consistency()
        self._validate_phone_formats()
        self._validate_numeric_ranges()
        self._validate_date_consistency()
        self._validate_geographic_consistency()
        self._validate_categorical_consistency()
        
        # Inter-field validations
        self._validate_relational_consistency()
        self._validate_unique_constraints()
        
        logger.info(f"Data validation complete: {self.corrections_made} corrections made")
        return self.df
    
    def get_correction_log(self) -> List[str]:
        """Returns the log of all corrections made."""
        return self.correction_log
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a report of validation results.
        
        Returns:
            Dictionary with validation statistics
        """
        report = {
            "total_corrections": self.corrections_made,
            "corrections_by_type": {},
            "corrections_by_column": {},
        }
        
        # Analyze correction log
        for correction in self.correction_log:
            # Extract column and reason from correction log entry
            parts = correction.split(':')
            if len(parts) >= 2:
                column_info = parts[0].strip()
                reason = parts[1].strip().split('-')[-1].strip()
                
                # Extract column name
                col_match = re.search(r"Column '([^']+)'", column_info)
                if col_match:
                    column = col_match.group(1)
                    
                    # Update column stats
                    if column not in report["corrections_by_column"]:
                        report["corrections_by_column"][column] = 0
                    report["corrections_by_column"][column] += 1
                    
                    # Update reason stats
                    if reason not in report["corrections_by_type"]:
                        report["corrections_by_type"][reason] = 0
                    report["corrections_by_type"][reason] += 1
        
        return report
    
    def _log_correction(self, column: str, row_idx: int, original: Any, corrected: Any, reason: str):
        """Log a correction that was made."""
        self.corrections_made += 1
        message = f"Row {row_idx}, Column '{column}': Changed '{original}' to '{corrected}' - {reason}"
        self.correction_log.append(message)
        if self.corrections_made <= 10:  # Only log the first 10 corrections to avoid spam
            logger.debug(message)
    
    def _remove_null_rows(self):
        """Remove rows that are entirely null."""
        initial_count = len(self.df)
        self.df = self.df.dropna(how='all')
        removed_count = initial_count - len(self.df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} completely empty rows")
    
    def _fix_data_types(self):
        """Fix data types based on schema or inferred types."""
        if not self.schema:
            return
            
        for col_info in self.schema:
            col_name = col_info.get('name')
            col_type = col_info.get('type')
            
            if col_name not in self.df.columns:
                continue
                
            # Fix data types based on schema
            if col_type == 'int':
                for idx, value in enumerate(self.df[col_name]):
                    try:
                        # Try to convert to int
                        self.df.at[idx, col_name] = int(float(value))
                    except (ValueError, TypeError):
                        # If conversion fails, use a default or mean value
                        original = value
                        if 'mean' in col_info:
                            corrected = int(col_info['mean'])
                        elif 'min' in col_info and 'max' in col_info:
                            corrected = int((col_info['min'] + col_info['max']) / 2)
                        else:
                            # Try to use the mean of the column
                            try:
                                corrected = int(self.df[col_name].mean())
                            except:
                                corrected = 0
                                
                        self.df.at[idx, col_name] = corrected
                        self._log_correction(col_name, idx, original, corrected, 
                                           "Invalid integer value")
            
            elif col_type == 'float':
                for idx, value in enumerate(self.df[col_name]):
                    try:
                        # Try to convert to float
                        self.df.at[idx, col_name] = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, use a default or mean value
                        original = value
                        if 'mean' in col_info:
                            corrected = float(col_info['mean'])
                        elif 'min' in col_info and 'max' in col_info:
                            corrected = (col_info['min'] + col_info['max']) / 2
                        else:
                            # Try to use the mean of the column
                            try:
                                corrected = float(self.df[col_name].mean())
                            except:
                                corrected = 0.0
                                
                        self.df.at[idx, col_name] = corrected
                        self._log_correction(col_name, idx, original, corrected, 
                                           "Invalid float value")
    
    def _validate_id_formats(self):
        """Validate and correct ID formats."""
        if not self.field_standards_available:
            return
            
        # Find columns that are likely IDs
        for col in self.df.columns:
            id_type = detect_id_field_type(col)
            if id_type:
                # Check each value in the column
                for idx, value in enumerate(self.df[col]):
                    # Get the pattern for this ID type
                    prefix = f"{id_type.upper()}-" if id_type != 'generic' else "ID-"
                    
                    # Check if the value matches the expected pattern
                    if not str(value).startswith(prefix) or not re.match(r'^[A-Z]+-\d{5}$', str(value)):
                        original = value
                        # Generate a proper ID with the sequence number based on row index
                        corrected = generate_id(id_type, idx)
                        self.df.at[idx, col] = corrected
                        self._log_correction(col, idx, original, corrected, 
                                           f"Incorrect ID format for {id_type}")
    
    def _validate_name_formats(self):
        """Validate and fix name formats (capitalization, spacing, etc.)"""
        # Find name columns
        name_cols = [col for col in self.df.columns if any(term in col.lower() for term in 
                     ['name', 'firstname', 'first_name', 'lastname', 'last_name', 'fullname'])]
        
        for col in name_cols:
            for idx, value in enumerate(self.df[col]):
                value_str = str(value)
                
                # Skip likely IDs or non-name values
                if re.match(r'^[A-Z]+-\d+$', value_str) or len(value_str) < 3:
                    continue
                
                # Fix capitalization and spacing issues
                name_parts = value_str.split()
                if len(name_parts) >= 2:
                    # Proper capitalization for each name part
                    corrected_parts = []
                    for part in name_parts:
                        # Handle hyphenated names
                        if '-' in part:
                            subparts = part.split('-')
                            corrected_subparts = [subpart.capitalize() for subpart in subparts]
                            corrected_parts.append('-'.join(corrected_subparts))
                        else:
                            corrected_parts.append(part.capitalize())
                    
                    corrected = ' '.join(corrected_parts)
                    
                    # Only log if actually changed
                    if corrected != value_str:
                        self.df.at[idx, col] = corrected
                        self._log_correction(col, idx, value, corrected, 
                                           "Improper name capitalization or format")
    
    def _validate_name_email_consistency(self):
        """Ensure emails match the names in the same row."""
        if not self.field_standards_available:
            return
            
        # Find name and email columns
        name_cols = [col for col in self.df.columns if any(term in col.lower() for term in 
                     ['name', 'firstname', 'first_name', 'lastname', 'last_name', 'fullname'])]
        email_cols = [col for col in self.df.columns if 'email' in col.lower()]
        
        if name_cols and email_cols:
            name_col = name_cols[0]  # Use the first name column found
            email_col = email_cols[0]  # Use the first email column found
            
            for idx, row in self.df.iterrows():
                name = str(row[name_col])
                email = str(row[email_col])
                
                # Skip if email is missing or name is too short
                if len(name) < 3 or '@' not in email:
                    continue
                
                # Get the first and last names
                name_parts = name.strip().split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0].lower()
                    last_name = name_parts[-1].lower()
                    
                    # Check if email contains the name
                    email_username = email.split('@')[0].lower() if '@' in email else ''
                    
                    # If the email doesn't contain either first or last name
                    if (first_name not in email_username and 
                        last_name not in email_username and
                        first_name[0] + last_name not in email_username and
                        last_name + first_name[0] not in email_username and
                        len(email_username) > 3):  # Ignore very short usernames that might be initials
                        
                        original = email
                        corrected = format_email(first_name, last_name)
                        self.df.at[idx, email_col] = corrected
                        self._log_correction(email_col, idx, original, corrected, 
                                           "Email doesn't match name")
    
    def _validate_phone_formats(self):
        """Validate and fix phone number formats."""
        if not self.field_standards_available:
            return
            
        # Find phone columns
        phone_cols = [col for col in self.df.columns if any(term in col.lower() for term in 
                     ['phone', 'mobile', 'cell', 'telephone'])]
        
        for col in phone_cols:
            # Common phone formats to check against
            phone_patterns = [
                r'^\(\d{3}\) \d{3}-\d{4}$',              # (555) 123-4567
                r'^\d{3}-\d{3}-\d{4}$',                  # 555-123-4567
                r'^\d{3}\.\d{3}\.\d{4}$',                # 555.123.4567
                r'^\d{1}-\d{3}-\d{3}-\d{4}$',            # 1-555-123-4567
                r'^\d{3}-\d{4}$',                        # 123-4567
                r'^\d{10}$',                             # 5551234567
                r'^\d{1} \(\d{3}\) \d{3}-\d{4}$'         # 1 (555) 123-4567
            ]
            
            for idx, value in enumerate(self.df[col]):
                value_str = str(value)
                
                # Skip empty values
                if not value_str:
                    continue
                
                # Check if value matches any valid phone pattern
                if not any(re.match(pattern, value_str) for pattern in phone_patterns):
                    original = value
                    corrected = format_phone_number()
                    self.df.at[idx, col] = corrected
                    self._log_correction(col, idx, original, corrected, 
                                       "Invalid phone number format")
    
    def _validate_numeric_ranges(self):
        """Validate numeric columns are within reasonable ranges."""
        if not self.schema:
            # Try to infer reasonable ranges for numeric columns
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Calculate statistics
                    try:
                        mean = self.df[col].mean()
                        std = self.df[col].std()
                        min_val = mean - 3 * std  # 3 std deviations below mean
                        max_val = mean + 3 * std  # 3 std deviations above mean
                        
                        # Check for outliers
                        for idx, value in enumerate(self.df[col]):
                            try:
                                num_val = float(value)
                                if num_val < min_val:
                                    original = value
                                    # Set to min value
                                    corrected = min_val
                                    self.df.at[idx, col] = corrected
                                    self._log_correction(col, idx, original, corrected, 
                                                      "Outlier below 3 std deviations")
                                elif num_val > max_val:
                                    original = value
                                    # Set to max value
                                    corrected = max_val
                                    self.df.at[idx, col] = corrected
                                    self._log_correction(col, idx, original, corrected, 
                                                      "Outlier above 3 std deviations")
                            except (ValueError, TypeError):
                                pass
                    except:
                        pass
            
            return
            
        # If schema is available, use it to validate ranges
        for col_info in self.schema:
            col = col_info.get('name')
            if col not in self.df.columns:
                continue
                
            if col_info.get('type') in ('int', 'float', 'number'):
                min_val = col_info.get('min')
                max_val = col_info.get('max')
                
                if min_val is not None and max_val is not None:
                    # Find values outside the range
                    for idx, value in enumerate(self.df[col]):
                        try:
                            num_val = float(value)
                            if num_val < min_val:
                                original = value
                                # Set to min value
                                corrected = min_val
                                self.df.at[idx, col] = corrected
                                self._log_correction(col, idx, original, corrected, 
                                                  f"Value below minimum ({min_val})")
                            elif num_val > max_val:
                                original = value
                                # Set to max value
                                corrected = max_val
                                self.df.at[idx, col] = corrected
                                self._log_correction(col, idx, original, corrected, 
                                                  f"Value above maximum ({max_val})")
                        except (ValueError, TypeError):
                            # If value can't be converted to float, replace with mean
                            try:
                                original = value
                                mean_val = self.df[col].mean()
                                corrected = mean_val
                                self.df.at[idx, col] = corrected
                                self._log_correction(col, idx, original, corrected, 
                                                  "Non-numeric value replaced with mean")
                            except:
                                pass
    
    def _validate_date_consistency(self):
        """Validate dates are consistent (e.g., start dates before end dates)."""
        # Find date columns
        date_cols = []
        for col in self.df.columns:
            # Check if column contains dates
            try:
                if pd.to_datetime(self.df[col], errors='coerce').notna().all():
                    date_cols.append(col)
            except:
                continue
        
        # Look for start/end date pairs
        start_end_pairs = []
        for col in date_cols:
            if 'start' in col.lower() or 'begin' in col.lower():
                # Look for corresponding end date
                base_name = col.lower().replace('start', '').replace('begin', '').strip('_')
                for end_col in date_cols:
                    if ('end' in end_col.lower() or 'finish' in end_col.lower()) and base_name in end_col.lower():
                        start_end_pairs.append((col, end_col))
                        break
        
        # Validate each start/end pair
        for start_col, end_col in start_end_pairs:
            for idx in range(len(self.df)):
                try:
                    start_date = pd.to_datetime(self.df.at[idx, start_col])
                    end_date = pd.to_datetime(self.df.at[idx, end_col])
                    
                    if end_date < start_date:
                        original = self.df.at[idx, end_col]
                        # Set end date to start date + random days (1-365)
                        days_to_add = np.random.randint(1, 365)
                        corrected = (start_date + pd.Timedelta(days=days_to_add)).strftime('%Y-%m-%d')
                        self.df.at[idx, end_col] = corrected
                        self._log_correction(end_col, idx, original, corrected, 
                                           f"End date before start date")
                except:
                    continue
    
    def _validate_geographic_consistency(self):
        """Ensure geographic data is consistent (e.g., cities match states)."""
        # Common geographic column combinations
        geo_pairs = [
            ('city', 'state'),
            ('city', 'country'),
            ('state', 'country'),
            ('postal_code', 'city'),
            ('zip', 'city'),
            ('zip', 'state'),
            ('zipcode', 'city'),
            ('zipcode', 'state')
        ]
        
        # Check if any of these pairs exist in the dataframe
        for col1, col2 in geo_pairs:
            col1_match = next((c for c in self.df.columns if c.lower() == col1 or 
                             (col1 + '_') in c.lower() or ('_' + col1) in c.lower()), None)
            col2_match = next((c for c in self.df.columns if c.lower() == col2 or 
                             (col2 + '_') in c.lower() or ('_' + col2) in c.lower()), None)
            
            if col1_match and col2_match:
                # Create a frequency table of valid combinations
                valid_combinations = self.df.groupby([col1_match, col2_match]).size().reset_index().rename(columns={0: 'count'})
                valid_combinations = valid_combinations.sort_values('count', ascending=False)
                
                # Create a mapping of col1 to most common col2
                mapping = {}
                for _, row in valid_combinations.iterrows():
                    if row[col1_match] not in mapping:
                        mapping[row[col1_match]] = row[col2_match]
                
                # Check each row for consistency
                for idx, row in self.df.iterrows():
                    val1 = row[col1_match]
                    val2 = row[col2_match]
                    
                    # If this val1 has a most common val2 that's different from the current val2
                    if val1 in mapping and mapping[val1] != val2:
                        original = val2
                        corrected = mapping[val1]
                        self.df.at[idx, col2_match] = corrected
                        self._log_correction(col2_match, idx, original, corrected, 
                                           f"Geographic inconsistency: {col1}={val1} usually has {col2}={corrected}")
    
    def _validate_categorical_consistency(self):
        """Ensure categorical variables are consistent with related columns."""
        # Look for categorical columns
        cat_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].nunique() < 20:  # Likely categorical
                cat_cols.append(col)
        
        # Check relationships between categorical columns
        for i, col1 in enumerate(cat_cols):
            for col2 in cat_cols[i+1:]:
                # Check if there's a strong relationship between the columns
                try:
                    cross_tab = pd.crosstab(self.df[col1], self.df[col2])
                    
                    # For each value in col1, find the most common value in col2
                    for val1 in cross_tab.index:
                        if cross_tab.loc[val1].max() > 0.8 * cross_tab.loc[val1].sum():  # Strong relationship
                            most_common_val2 = cross_tab.loc[val1].idxmax()
                            
                            # Find rows with val1 but not the most common val2
                            for idx, row in self.df[self.df[col1] == val1].iterrows():
                                if row[col2] != most_common_val2:
                                    original = row[col2]
                                    corrected = most_common_val2
                                    self.df.at[idx, col2] = corrected
                                    self._log_correction(col2, idx, original, corrected, 
                                                       f"Categorical inconsistency: {col1}={val1} usually has {col2}={corrected}")
                except Exception as e:
                    # Skip if crosstab fails
                    continue
    
    def _validate_relational_consistency(self):
        """Validate consistency between related fields, e.g., job title and salary."""
        # Map of common related fields and rules
        related_fields = [
            {
                'fields': ['job_title', 'salary'],
                'rule': lambda row: self._validate_job_salary(row)
            },
            {
                'fields': ['age', 'join_date'],
                'rule': lambda row: self._validate_age_joindate(row)
            },
            {
                'fields': ['department', 'job_title'],
                'rule': lambda row: self._validate_dept_jobtitle(row)
            }
        ]
        
        # Check each set of related fields
        for field_set in related_fields:
            # Check if all required fields exist
            if all(field in self.df.columns for field in field_set['fields']):
                # Apply rule to each row
                for idx, row in self.df.iterrows():
                    field_set['rule'](row)
    
    def _validate_job_salary(self, row):
        """Validate that job title and salary are consistent."""
        # Placeholder for implementation
        pass
    
    def _validate_age_joindate(self, row):
        """Validate that age and join date are consistent."""
        # Placeholder for implementation
        pass
    
    def _validate_dept_jobtitle(self, row):
        """Validate that department and job title are consistent."""
        # Placeholder for implementation
        pass
    
    def _validate_unique_constraints(self):
        """Ensure uniqueness constraints are met (e.g., employee IDs, emails)."""
        # Columns that should be unique
        unique_cols = []
        
        # Detect ID columns
        if self.field_standards_available:
            for col in self.df.columns:
                if detect_id_field_type(col):
                    unique_cols.append(col)
        
        # Email columns should be unique
        email_cols = [col for col in self.df.columns if 'email' in col.lower()]
        unique_cols.extend(email_cols)
        
        # Check each unique column
        for col in unique_cols:
            # Find duplicate values
            duplicates = self.df[col].duplicated()
            if duplicates.any():
                # Get indices of duplicated rows
                dup_indices = self.df.index[duplicates].tolist()
                
                # Fix duplicates
                for idx in dup_indices:
                    original = self.df.at[idx, col]
                    
                    if 'email' in col.lower():
                        # For emails, find the name column and regenerate email
                        if self.field_standards_available:
                            name_cols = [c for c in self.df.columns if any(term in c.lower() for term in 
                                        ['name', 'firstname', 'first_name', 'lastname', 'last_name', 'fullname'])]
                            
                            if name_cols:
                                name = str(self.df.at[idx, name_cols[0]])
                                name_parts = name.strip().split()
                                
                                if len(name_parts) >= 2:
                                    first_name = name_parts[0]
                                    last_name = name_parts[-1]
                                    # Add a unique number to make the email unique
                                    corrected = format_email(first_name, last_name).replace('@', f"{idx}@")
                                    self.df.at[idx, col] = corrected
                                    self._log_correction(col, idx, original, corrected, 
                                                       "Duplicate email address")
                        else:
                            # Simple fix without field_standards
                            username, domain = original.split('@')
                            corrected = f"{username}{idx}@{domain}"
                            self.df.at[idx, col] = corrected
                            self._log_correction(col, idx, original, corrected, 
                                               "Duplicate email address")
                    else:
                        # For other columns, append a unique ID
                        if self.field_standards_available and detect_id_field_type(col):
                            # For ID columns, generate a new unique ID
                            id_type = detect_id_field_type(col)
                            corrected = generate_id(id_type, idx + 10000)  # Ensure it's different
                            self.df.at[idx, col] = corrected
                            self._log_correction(col, idx, original, corrected, 
                                               f"Duplicate {id_type} ID")
                        else:
                            # Generic handling for other columns
                            corrected = f"{original}_DUP{idx}"
                            self.df.at[idx, col] = corrected
                            self._log_correction(col, idx, original, corrected, 
                                               "Duplicate value in unique column")


def validate_and_fix_data(df: pd.DataFrame, schema: Optional[List[Dict[str, Any]]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and fix issues in synthetic data.
    
    Args:
        df: The synthetic dataframe to validate
        schema: Optional schema providing column metadata
        
    Returns:
        Tuple of (corrected_dataframe, correction_log)
    """
    validator = DataValidator(df, schema)
    corrected_df = validator.validate_and_fix()
    return corrected_df, validator.get_correction_log()


if __name__ == "__main__":
    # Simple self-test if run directly
    print("Data Validator Module")
    print("====================")
    
    # Check if field standards are available
    print(f"Field standards module: {'Available' if FIELD_STANDARDS_AVAILABLE else 'Not available'}")
    
    # Generate a simple test dataframe
    test_data = {
        'emp_id': ['INVALID-1', 'EMP-10002', 'notanid', 'EMP-10004', 'EMP-10002'],
        'name': ['john smith', 'Jane Doe', 'robert JOHNSON', 'Maria Garcia-Lopez', 'Test User'],
        'email': ['wrong@example.com', 'jane.doe@gmail.com', 'rj@example.com', 'maria@gmail.com', 'jane.doe@gmail.com'],
        'salary': [50000, 2000000, 'invalid', 75000, 85000],
        'join_date': ['2022-01-01', '2022-02-15', '2022-03-01', '2020-01-01', '2023-05-01'],
        'city': ['New York', 'Boston', 'New York', 'Chicago', 'Boston'],
        'state': ['NY', 'MA', 'CA', 'IL', 'FL']
    }
    
    test_df = pd.DataFrame(test_data)
    print(f"\nTest dataframe created with {len(test_df)} rows and {len(test_df.columns)} columns")
    
    # Create a simple schema
    test_schema = [
        {'name': 'salary', 'type': 'float', 'min': 30000, 'max': 150000}
    ]
    
    # Run validation
    print("\nValidating and fixing data...")
    fixed_df, corrections = validate_and_fix_data(test_df, test_schema)
    
    # Print results
    print(f"\nMade {len(corrections)} corrections:")
    for correction in corrections:
        print(f"  {correction}")
    
    print("\nFixed dataframe:")
    print(fixed_df)
