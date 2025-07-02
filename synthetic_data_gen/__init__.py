"""
Synthetic Data Generator Package

A versatile tool for generating synthetic data based on real samples or custom schemas.
This package provides both a programmatic API and a command-line interface.
"""

__version__ = "1.0.0"

# Make key components available at package level
from .generators import (
    generate_synthetic_data_model_based,
    generate_synthetic_data_from_schema,
    apply_relationships
)

from .schema import (
    infer_schema,
    add_relationships_to_schema,
    interactive_schema_prompt,
    save_schema_to_json
)

from .utils import (
    load_csv,
    save_to_csv,
    load_schema_from_json,
    prompt_for_input,
    check_dependencies
)

from .validation import (
    validate_synthetic_data,
    compare_distributions,
    visualize_data_distributions
)

from .main import SyntheticDataGenerator

# Check if SDV is available
try:
    import sdv
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
