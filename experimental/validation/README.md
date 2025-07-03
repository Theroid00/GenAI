# Data Validation Module for Synthetic Data

This module provides a robust framework for validating and fixing issues in synthetic data. The goal is to ensure that generated data is realistic, consistent, and adheres to business rules.

## Overview

The data validation module analyzes synthetic datasets and automatically corrects common issues such as:

- Invalid ID formats
- Inconsistent email addresses that don't match names
- Improper phone number formatting
- Statistical outliers in numeric fields
- Date inconsistencies (e.g., end dates before start dates)
- Geographic inconsistencies (e.g., cities not matching states)
- Violations of uniqueness constraints
- Relational integrity issues
- Categorical inconsistencies

## Installation

This module is part of the GenAI project. No additional installation is required beyond the project dependencies.

## Usage

There are two ways to use the validation module:

### 1. Simple Function Call

For basic validation, use the convenience function:

```python
from validation.data_validator import validate_and_fix_data

# Validate data
corrected_df, corrections = validate_and_fix_data(df, schema=None, strict_mode=False, fix_automatically=True)

# Print corrections made
for correction in corrections:
    print(correction)
```

### 2. Using the DataValidator Class

For more control and access to detailed validation reports:

```python
from validation.data_validator import DataValidator

# Create validator instance
validator = DataValidator(strict_mode=False, fix_automatically=True)

# Run validation
corrected_df, corrections = validator.validate_and_fix(df, schema=None, auto_detect=True)

# Get validation report
report = validator.validation_report

# Print statistics
print(f"Total issues found: {report['total_issues']}")
print(f"Issues fixed: {report['issues_fixed']}")
print(f"Fields with issues: {report['fields_with_issues']}")
```

## Schema Definition

Providing a schema helps the validator make more informed corrections. A schema should be a list of column definitions, each with properties that define validation rules:

```python
schema = [
    {
        'name': 'employee_id',
        'type': 'id',
        'id_type': 'employee_id'
    },
    {
        'name': 'salary',
        'type': 'float',
        'min': 30000,
        'max': 200000
    },
    {
        'name': 'department',
        'type': 'category',
        'categories': ['Engineering', 'Marketing', 'Sales', 'HR']
    },
    {
        'name': 'hire_date',
        'type': 'date',
        'start_date': '2018-01-01',
        'end_date': '2023-12-31'
    }
]
```

You can also define relationships between fields:

```python
schema_with_rel = {
    'columns': [...],  # Column definitions as above
    'relationships': [
        {
            'source': 'employee_id',
            'target': 'manager_id',
            'type': 'foreign_key'
        },
        {
            'source': 'start_date',
            'target': 'end_date',
            'type': 'temporal'
        }
    ],
    'check_duplicates': True
}
```

## Running the Test Script

A test script is included to demonstrate the validator's capabilities:

```bash
cd experimental/validation
python test_validator.py
```

This will:
1. Create a test dataset with intentional errors
2. Run the validator to fix those errors
3. Save the original data, corrected data, and validation report
4. Print a summary of corrections made

## Extending the Validator

To add custom validation rules:

1. Create a subclass of `DataValidator`
2. Add new validation methods
3. Call those methods in your overridden `validate_and_fix` method

Example:

```python
class CustomValidator(DataValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def validate_custom_rule(self, df):
        # Implement custom validation logic
        pass
    
    def validate_and_fix(self, df, schema=None, auto_detect=True):
        # Call parent method first
        df, corrections = super().validate_and_fix(df, schema, auto_detect)
        
        # Add custom validation
        self.validate_custom_rule(df)
        
        return df, self.corrections
```

## Integration with Field Standards

The validator works best when the `field_standards` module is available, as it can leverage standardized formatting functions for emails, phone numbers, and IDs. If the module is not found, the validator will fall back to simpler validation logic.

## Fine-tuning Data

For information on what kind of data is needed for proper fine-tuning of synthetic data models, refer to the `FINE_TUNING_GUIDE.md` document in this directory.
