"""
Synthetic Data Generator

A comprehensive toolkit for generating high-quality synthetic data with
built-in validation and customizable templates.
"""

__version__ = "1.1.0"

# Import core functionality
from synthetic_data_gen.core import (
    SyntheticDataGenerator,
    apply_relationships,
    infer_schema,
    load_schema_from_json,
    save_schema_to_json
)

# Import utils
from synthetic_data_gen.utils import (
    load_csv,
    save_to_csv,
    prompt_for_input,
    check_dependencies
)

# Import validation
from synthetic_data_gen.validation import (
    validate_synthetic_data,
    compare_distributions,
    visualize_data_distributions
)

# Check if SDV is available
try:
    import sdv
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

__all__ = [
    'SyntheticDataGenerator',
    'apply_relationships',
    'infer_schema',
    'load_schema_from_json',
    'save_schema_to_json',
    'load_csv',
    'save_to_csv',
    'validate_synthetic_data',
    'compare_distributions',
    'visualize_data_distributions',
    'SDV_AVAILABLE'
]
