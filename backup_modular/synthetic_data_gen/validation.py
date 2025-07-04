"""
Validation functions for synthetic data generation.
This module contains functions for validating and comparing synthetic data with original data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

# Check if SDV is available
try:
    import sdv
    from sdv.evaluation.single_table import evaluate_quality as evaluate
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False

def validate_synthetic_data(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the quality of synthetic data compared to the original data.
    
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

def display_validation_results(metrics: Dict[str, Any]) -> None:
    """
    Display the validation results in a readable format.
    
    Args:
        metrics: Dictionary with quality metrics
    """
    if not metrics:
        print("No validation metrics available.")
        return
    
    print("\n=== Synthetic Data Validation Results ===")
    
    # Display overall quality score
    if 'sdv_quality_score' in metrics:
        # Handle the case where sdv_quality_score is a QualityReport object
        if hasattr(metrics['sdv_quality_score'], 'get_score'):
            score = metrics['sdv_quality_score'].get_score()
            print(f"Overall Quality Score: {score:.4f}")
        else:
            # Try to convert to string first to avoid formatting issues
            print(f"Overall Quality Score: {metrics['sdv_quality_score']}")
        print("(Score ranges from 0 to 1, where 1 means perfect similarity)")
    
    # Display column-specific metrics
    col_metrics = {k: v for k, v in metrics.items() if k != 'sdv_quality_score'}
    
    if col_metrics:
        print("\nColumn-specific Metrics:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.2f}%")

def compare_distributions(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                         num_cols: int = 3) -> None:
    """
    Generate histograms comparing original and synthetic data distributions.
    
    Args:
        original_df: Original DataFrame
        synthetic_df: Synthetic DataFrame
        num_cols: Number of columns to visualize (defaults to 3)
    """
    # Select numeric columns for comparison
    numeric_cols = [col for col in original_df.columns if 
                   pd.api.types.is_numeric_dtype(original_df[col])]
    
    if not numeric_cols:
        logger.warning("No numeric columns available for distribution comparison")
        return
    
    # Limit to specified number of columns
    cols_to_plot = numeric_cols[:min(num_cols, len(numeric_cols))]
    
    # Create subplots
    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3*len(cols_to_plot)))
    
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    # Plot histograms
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(original_df[col], alpha=0.5, label='Original', bins=15)
        axes[i].hist(synthetic_df[col], alpha=0.5, label='Synthetic', bins=15)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def visualize_data_distributions(df: pd.DataFrame, num_cols: int = 3) -> None:
    """
    Visualize distributions of numeric columns in a dataframe.
    
    Args:
        df: DataFrame to visualize
        num_cols: Number of columns to visualize
    """
    # Select numeric columns for visualization
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        logger.warning("No numeric columns available for visualization")
        print("No numeric columns available for visualization.")
        return
    
    # Limit to specified number of columns
    cols_to_plot = numeric_cols[:min(num_cols, len(numeric_cols))]
    
    # Create subplots
    fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(10, 3*len(cols_to_plot)))
    
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    # Plot histograms
    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col], bins=15)
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
