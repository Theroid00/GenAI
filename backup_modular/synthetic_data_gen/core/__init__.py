"""
Core functionality for synthetic data generation.

This package contains the core components for synthetic data generation:
- generator.py: Main synthetic data generation logic
- schema.py: Schema definition and handling
- relationships.py: Handling relationships between fields
"""

from .generator import SyntheticDataGenerator
from .relationships import apply_relationships
from .schema import (
    infer_schema,
    load_schema_from_json,
    save_schema_to_json
)

__all__ = [
    'SyntheticDataGenerator',
    'apply_relationships',
    'infer_schema',
    'load_schema_from_json',
    'save_schema_to_json'
]
