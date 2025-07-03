#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Field Parser for Synthetic Data Generator
-----------------------------------------
Provides specialized parsing and processing for template fields with
custom formats or relationships.
"""

import re
import pandas as pd
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('field_parser')

def fix_product_category_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that product names match their categories.
    
    Args:
        df: DataFrame with product_name and category columns
        
    Returns:
        DataFrame with corrected product_name and category pairs
    """
    if 'product_name' not in df.columns or 'category' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Define the mapping of categories to product names
    category_products_map = {
        "Electronics": ["Premium 4K Smart TV", "Bluetooth Wireless Headphones", "Smartphone Pro Max", "Laptop Ultra Slim", "Tablet Air Plus"],
        "Clothing": ["Designer Jeans", "Cotton T-Shirt Pack", "Leather Jacket", "Running Shoes", "Winter Coat Collection"],
        "Home & Kitchen": ["Non-Stick Cookware Set", "Coffee Maker Deluxe", "Memory Foam Mattress", "Kitchen Knife Set", "Robot Vacuum Cleaner"],
        "Books": ["Bestselling Novel", "Children's Illustrated Series", "Self-Help Guide", "Historical Biography", "Cooking Encyclopedia"],
        "Toys": ["Building Blocks Set", "Remote Control Car", "Interactive Plush Animal", "Educational Board Game", "Art and Craft Kit"],
        "Sports": ["Tennis Racket Pro", "Yoga Mat Premium", "Mountain Bike", "Fitness Tracker Watch", "Camping Tent 4-Person"],
        "Beauty": ["Facial Cleanser", "Anti-Aging Serum", "Makeup Palette", "Perfume Collection", "Hair Care Set"]
    }
    
    # Create the reverse mapping for lookup (normalize case for reliable matching)
    product_to_category = {}
    for category, products in category_products_map.items():
        for product in products:
            product_to_category[product.lower()] = category
    
    mismatch_count = 0
    
    # Fix mismatches - iterate through all rows
    for idx, row in result_df.iterrows():
        category = row['category']
        product_name = row['product_name']
        
        # Normalize product name for comparison
        product_name_lower = product_name.lower()
        
        # Check if product exists in our mapping
        product_expected_category = None
        for prod, cat in product_to_category.items():
            if prod == product_name_lower:
                product_expected_category = cat
                break
        
        # If we found a category and it doesn't match
        if product_expected_category and product_expected_category != category:
            mismatch_count += 1
            # Change the product to match the category
            if category in category_products_map:
                result_df.at[idx, 'product_name'] = random.choice(category_products_map[category])
        
        # Handle case where product isn't in any category mapping
        elif product_expected_category is None:
            # Assign a product from the right category
            if category in category_products_map:
                result_df.at[idx, 'product_name'] = random.choice(category_products_map[category])
                mismatch_count += 1
    
    if mismatch_count > 0:
        logger.info(f"Fixed {mismatch_count} product-category mismatches")
    
    return result_df

def normalize_case_in_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes case in text columns (proper case for product names, title case for categories).
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with normalized text
    """
    result_df = df.copy()
    
    # Fix product names (each word capitalized)
    if 'product_name' in result_df.columns:
        result_df['product_name'] = result_df['product_name'].apply(
            lambda x: ' '.join(word.capitalize() for word in str(x).split())
        )
    
    # Fix categories (title case)
    if 'category' in result_df.columns:
        result_df['category'] = result_df['category'].apply(
            lambda x: str(x).title() if x != "Home & Kitchen" else "Home & Kitchen"
        )
    
    return result_df

def fix_price_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that total_amount is correctly calculated as quantity * unit_price.
    Also ensures unit prices are reasonable for each product category.
    
    Args:
        df: DataFrame with quantity, unit_price, and total_amount columns
        
    Returns:
        DataFrame with corrected price calculations
    """
    if 'quantity' not in df.columns or 'unit_price' not in df.columns or 'total_amount' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Define reasonable price ranges for different categories
    category_price_ranges = {
        "Electronics": (199.99, 1299.99),
        "Clothing": (19.99, 299.99),
        "Home & Kitchen": (29.99, 499.99),
        "Books": (9.99, 49.99),
        "Toys": (14.99, 99.99),
        "Sports": (24.99, 399.99),
        "Beauty": (7.99, 89.99)
    }
    
    # Define reasonable price ranges for specific products
    product_price_ranges = {
        "Premium 4K Smart TV": (499.99, 1299.99),
        "Smartphone Pro Max": (699.99, 1199.99),
        "Laptop Ultra Slim": (599.99, 1499.99),
        "Designer Jeans": (59.99, 199.99),
        "Leather Jacket": (99.99, 299.99),
        "Memory Foam Mattress": (299.99, 899.99),
        "Coffee Maker Deluxe": (79.99, 199.99),
        "Mountain Bike": (199.99, 899.99)
    }
    
    price_fixes = 0
    
    # Fix prices and calculate correct totals
    for idx, row in result_df.iterrows():
        category = row['category'] if 'category' in result_df.columns else None
        product_name = row['product_name'] if 'product_name' in result_df.columns else None
        quantity = row['quantity']
        unit_price = row['unit_price']
        
        # Check if unit price needs adjustment
        price_adjusted = False
        
        # First check product-specific price range
        if product_name in product_price_ranges:
            min_price, max_price = product_price_ranges[product_name]
            if unit_price < min_price or unit_price > max_price:
                result_df.at[idx, 'unit_price'] = round(random.uniform(min_price, max_price), 2)
                price_adjusted = True
        
        # Then check category price range if we haven't adjusted yet
        elif not price_adjusted and category in category_price_ranges:
            min_price, max_price = category_price_ranges[category]
            if unit_price < min_price or unit_price > max_price:
                result_df.at[idx, 'unit_price'] = round(random.uniform(min_price, max_price), 2)
                price_adjusted = True
        
        # Recalculate the total amount based on quantity and (possibly adjusted) unit price
        result_df.at[idx, 'total_amount'] = round(result_df.at[idx, 'quantity'] * result_df.at[idx, 'unit_price'], 2)
        
        if price_adjusted:
            price_fixes += 1
    
    if price_fixes > 0:
        logger.info(f"Adjusted {price_fixes} unit prices to realistic values")
    
    logger.info(f"Recalculated all total_amount values based on quantity * unit_price")
    
    return result_df

def apply_field_parsing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all field parsing functions to a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame with all parsing applied
    """
    # Apply each parser in sequence
    df = normalize_case_in_columns(df)
    df = fix_product_category_mismatch(df)
    df = fix_price_calculations(df)
    
    logger.info(f"Field parsing complete. Processed {len(df)} rows.")
    return df
