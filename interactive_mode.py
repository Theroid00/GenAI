#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Mode for Synthetic Data Generator
---------------------------------------------
This script provides an interactive interface for generating synthetic data.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import from the synthetic_data_gen package
from synthetic_data_gen import (
    SyntheticDataGenerator,
    infer_schema,
    load_schema_from_json,
    save_schema_to_json,
    load_csv,
    save_to_csv,
    prompt_for_input
)

def interactive_schema_prompt():
    """
    Interactive prompt to create a schema from scratch.
    
    Returns:
        tuple: (title, num_rows, schema)
    """
    print("\n=== Create Schema Interactively ===")
    
    # Get title and number of rows
    title = input("Dataset title (e.g., 'Customer Data'): ").strip()
    num_rows = int(input("Number of rows to generate: ").strip() or "100")
    
    # Initialize schema
    schema = []
    
    print("\nDefine your columns. Enter a blank name to finish.")
    col_num = 1
    
    while True:
        # Get column name
        col_name = input(f"\nColumn {col_num} name (or press Enter to finish): ").strip()
        if not col_name:
            break
        
        # Get column type
        col_type = prompt_for_input(
            f"Column type for '{col_name}'",
            options=["int", "float", "string", "category", "date", "uuid"],
            default="string"
        )
        
        # Create column definition
        col_def = {
            "name": col_name,
            "type": col_type
        }
        
        # Get additional properties based on type
        if col_type == "int" or col_type == "float":
            # Get min/max values
            col_def["min"] = float(input(f"Minimum value: ").strip() or "0")
            col_def["max"] = float(input(f"Maximum value: ").strip() or "100")
            
            # Get distribution
            dist_type = prompt_for_input(
                "Distribution type",
                options=["uniform", "normal"],
                default="uniform"
            )
            
            col_def["distribution"] = dist_type
            
            if dist_type == "normal":
                col_def["mean"] = float(input(f"Mean value: ").strip() or str((col_def["min"] + col_def["max"]) / 2))
                col_def["std"] = float(input(f"Standard deviation: ").strip() or str((col_def["max"] - col_def["min"]) / 6))
            
            # For integers, convert values to int
            if col_type == "int":
                col_def["min"] = int(col_def["min"])
                col_def["max"] = int(col_def["max"])
                if "mean" in col_def:
                    col_def["mean"] = int(col_def["mean"])
            
            # For floats, get decimal places
            if col_type == "float":
                col_def["decimals"] = int(input(f"Decimal places: ").strip() or "2")
        
        elif col_type == "string":
            # Get string subtype
            subtypes = [
                "text", "name", "first_name", "last_name", "full_name",
                "email", "phone", "address", "city", "country", "company", "job"
            ]
            
            col_def["subtype"] = prompt_for_input(
                f"String subtype for '{col_name}'",
                options=subtypes,
                default="text"
            )
            
            # For custom patterns
            if col_def["subtype"] == "text":
                use_pattern = prompt_for_input(
                    "Do you want to specify a custom pattern?",
                    options=["yes", "no"],
                    default="no"
                ) == "yes"
                
                if use_pattern:
                    col_def["subtype"] = "custom"
                    col_def["pattern"] = input("Pattern (use # for digits, ? for letters): ").strip()
        
        elif col_type == "category":
            # Get categories
            cats_input = input("Categories (comma separated): ").strip()
            if cats_input:
                col_def["categories"] = [c.strip() for c in cats_input.split(",")]
            else:
                col_def["categories"] = ["Category A", "Category B", "Category C"]
            
            # Ask about weights
            use_weights = prompt_for_input(
                "Do you want to specify custom weights?",
                options=["yes", "no"],
                default="no"
            ) == "yes"
            
            if use_weights:
                weights_input = input("Weights (comma separated, should sum to 1): ").strip()
                if weights_input:
                    col_def["weights"] = [float(w.strip()) for w in weights_input.split(",")]
        
        elif col_type == "date":
            # Get date range
            col_def["start_date"] = input("Start date (YYYY-MM-DD): ").strip() or "2020-01-01"
            col_def["end_date"] = input("End date (YYYY-MM-DD): ").strip() or "2023-12-31"
        
        # Add column to schema
        schema.append(col_def)
        col_num += 1
    
    if not schema:
        print("No columns defined. Using a simple default schema.")
        schema = [
            {"name": "id", "type": "uuid"},
            {"name": "name", "type": "string", "subtype": "full_name"},
            {"name": "age", "type": "int", "min": 18, "max": 90, "distribution": "uniform"},
            {"name": "category", "type": "category", "categories": ["A", "B", "C"]},
            {"name": "created_at", "type": "date", "start_date": "2020-01-01", "end_date": "2023-12-31"}
        ]
    
    return title, num_rows, schema

def add_relationships_to_schema(schema):
    """
    Interactively add relationships between columns in the schema.
    
    Args:
        schema: The schema to add relationships to
        
    Returns:
        The updated schema
    """
    print("\n=== Add Relationships Between Columns ===")
    print("Relationships make your data more realistic by creating dependencies between columns.")
    
    add_rels = prompt_for_input(
        "Would you like to add relationships between columns?",
        options=["yes", "no"],
        default="no"
    ) == "yes"
    
    if not add_rels:
        return schema
    
    # Show available columns
    print("\nAvailable columns:")
    for i, col in enumerate(schema, 1):
        print(f"{i}. {col['name']} ({col['type']})")
    
    # Let user create relationships
    while True:
        print("\nRelationship types:")
        print("1. Correlation (numeric columns)")
        print("2. Conditional values (any column types)")
        print("3. Finish adding relationships")
        
        rel_type = prompt_for_input(
            "Select relationship type",
            options=["1", "2", "3"],
            default="3"
        )
        
        if rel_type == "3":
            break
        
        # Get source column
        source_idx = int(input("Select source column (number): ").strip()) - 1
        if source_idx < 0 or source_idx >= len(schema):
            print("Invalid column index.")
            continue
        
        source_col = schema[source_idx]
        
        # Get target column
        target_idx = int(input("Select target column (number): ").strip()) - 1
        if target_idx < 0 or target_idx >= len(schema) or target_idx == source_idx:
            print("Invalid column index.")
            continue
        
        target_col = schema[target_idx]
        
        # Initialize relationships if not present
        if "relationships" not in target_col:
            target_col["relationships"] = []
        
        # Handle correlation
        if rel_type == "1":
            if source_col["type"] not in ["int", "float"] or target_col["type"] not in ["int", "float"]:
                print("Correlation requires numeric columns.")
                continue
            
            corr = float(input("Correlation coefficient (-1 to 1): ").strip() or "0.8")
            corr = max(-1.0, min(1.0, corr))  # Clamp to valid range
            
            # Add relationship
            target_col["relationships"].append({
                "type": "correlation",
                "source": source_col["name"],
                "correlation": corr
            })
            
            print(f"Added correlation between {source_col['name']} and {target_col['name']}.")
        
        # Handle conditional values
        elif rel_type == "2":
            # Get condition
            if source_col["type"] == "category":
                print(f"Available categories for {source_col['name']}: {source_col.get('categories', [])}")
                condition_value = input("When source equals: ").strip()
            elif source_col["type"] in ["int", "float"]:
                condition_type = prompt_for_input(
                    "Condition type",
                    options=["equals", "greater_than", "less_than"],
                    default="equals"
                )
                condition_value = input(f"When source {condition_type}: ").strip()
            else:
                condition_type = "equals"
                condition_value = input("When source equals: ").strip()
            
            # Get result value
            result_value = input(f"Then {target_col['name']} should be: ").strip()
            
            # Add relationship
            target_col["relationships"].append({
                "type": "conditional",
                "source": source_col["name"],
                "condition": condition_type if source_col["type"] in ["int", "float"] else "equals",
                "value": condition_value,
                "result": result_value
            })
            
            print(f"Added conditional relationship from {source_col['name']} to {target_col['name']}.")
    
    return schema

def main():
    """Main function for interactive mode"""
    print("\nWelcome to Interactive Mode!")
    print("This mode allows you to create a schema and generate data interactively.\n")
    
    # Create generator instance
    generator = SyntheticDataGenerator()
    
    try:
        # Get schema definition
        title, num_rows, schema = interactive_schema_prompt()
        
        # Apply relationships
        schema = add_relationships_to_schema(schema)
        
        # Option to save schema
        save_schema_opt = prompt_for_input(
            "Do you want to save this schema for future use?",
            options=["yes", "no"],
            default="yes"
        ) == "yes"
        
        if save_schema_opt:
            schema_path = f"{title.lower().replace(' ', '_')}_schema.json"
            save_schema_to_json(schema, schema_path)
            print(f"Schema saved to {schema_path}")
        
        # Generate data
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        df = generator.generate_from_schema(schema, num_rows)
        
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"synthetic_data_{timestamp}.csv"
        save_to_csv(df, output_path)
        
        print(f"\nGenerated {len(df)} rows of synthetic data.")
        print(f"Data saved to {output_path}")
        
        # Show data preview
        print("\nData Preview:")
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        print(df.head())
        
        # Visualization option
        visualize_opt = prompt_for_input(
            "Do you want to visualize the data distributions?",
            options=["yes", "no"],
            default="no"
        ) == "yes"
        
        if visualize_opt:
            from synthetic_data_gen.validation import visualize_data_distributions
            print("\nVisualizing data distributions...")
            visualize_data_distributions(df)
        
        print("\nInteractive mode completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInteractive mode cancelled.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
