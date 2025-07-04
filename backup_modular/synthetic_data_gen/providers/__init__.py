"""
Custom data providers for region-specific and domain-specific synthetic data.
This module provides a framework for extending the synthetic data generator with
specialized data providers for different regions, industries, or data domains.
"""

from synthetic_data_gen.providers.region_manager import RegionManager
from synthetic_data_gen.providers.flexible_provider import FlexibleDataProvider, generate_data as flexible_generate_data

# Initialize the region manager
region_manager = RegionManager()

# Export a generate_data function that uses the region manager
def generate_data(provider: str, data_type: str, **kwargs):
    """
    Generate data using a specific provider and data type.
    
    Args:
        provider: Provider code (e.g., 'india', 'usa', 'healthcare')
        data_type: Type of data to generate (e.g., 'name', 'city', 'diagnosis')
        **kwargs: Additional arguments to pass to the generator
        
    Returns:
        Generated data
    """
    return region_manager.generate_data(provider, data_type, **kwargs)