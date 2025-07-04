"""
Mode-specific functions for synthetic data generation.
This module implements the three main operational modes of the synthetic data generator.
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

from synthetic_data_gen.utils import prompt_for_input, load_csv, save_to_csv, load_schema_from_json
from synthetic_data_gen.schema import infer_schema, display_schema_info, interactive_schema_prompt, add_relationships_to_schema, save_schema_to_json
from synthetic_data_gen.generators import generate_synthetic_data_model_based, generate_synthetic_data_from_schema, apply_relationships
from synthetic_data_gen.validation import validate_synthetic_data, display_validation_results, compare_distributions, visualize_data_distributions

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

# Check if SDV is available
try:
    import sdv
    SDV_AVAILABLE = True
    logger.info("SDV package detected. Model-based generation is available.")
except ImportError:
    SDV_AVAILABLE = False
    logger.warning("SDV package not found. Model-based generation will be unavailable.")

def mode_1_generate_from_csv():
    """Function to handle Mode 1: Generate from CSV sample"""
    print("\n=== Mode 1: Generate Synthetic Data From a Sample CSV ===")
    
    # Get CSV path
    while True:
        file_path = input("Enter path to CSV file: ").strip()
        if os.path.exists(file_path) and file_path.endswith('.csv'):
            break
        print("File not found or not a CSV. Please enter a valid path.")
    
    # Load CSV and infer schema
    try:
        df = load_csv(file_path)
        print(f"\nLoaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")
        print("\nSample data:")
        print(df.head())
        
        # Infer and display schema
        schema = infer_schema(df)
        display_schema_info(schema)
    except Exception as e:
        logger.error(f"Error loading or processing CSV file: {str(e)}")
        print(f"Error: {str(e)}")
        return
        
    if not SDV_AVAILABLE:
        print("\nWarning: SDV package not installed. Please install it to use this mode.")
        print("Install with: pip install sdv")
        return
    
    # Get number of rows to generate
    default_rows = "100"
    num_rows_input = prompt_for_input("Number of synthetic rows to generate", default_rows)
    try:
        num_rows = int(num_rows_input)
    except ValueError:
        print(f"Invalid input. Using default value of {default_rows} rows.")
        num_rows = int(default_rows)
    
    # Warn about memory usage for very large datasets
    if num_rows > 500000:
        print(f"\nWarning: Generating {num_rows:,} rows will require significant memory and processing time.")
        print("Consider generating in smaller batches if you encounter memory issues.")
        confirm = prompt_for_input("Continue? (yes/no)", "yes").lower()
        if confirm != 'yes':
            print("Operation cancelled.")
            return
    
    # Get model type
    model_type = prompt_for_input("Model type (gaussian/ctgan)", "gaussian").lower()
    if model_type not in ['gaussian', 'ctgan']:
        print(f"Invalid model type. Using default (gaussian).")
        model_type = 'gaussian'
    
    # Option to save schema
    save_schema = prompt_for_input("Save inferred schema to JSON file? (yes/no)", "no").lower()
    if save_schema == 'yes':
        schema_path = prompt_for_input("Schema JSON path", "inferred_schema.json")
        try:
            # Convert schema to serializable format
            serializable_schema = []
            for col, info in schema.items():
                col_info = dict(info)
                col_info['name'] = col
                
                # Convert numpy types to native Python types
                for key, value in col_info.items():
                    if isinstance(value, np.generic):
                        col_info[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        col_info[key] = value.tolist()
                
                serializable_schema.append(col_info)
            
            save_schema_to_json(serializable_schema, schema_path)
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            print(f"Error saving schema: {str(e)}")
    
    # Generate synthetic data with progress bar
    try:
        print("\nGenerating synthetic data...")
        with tqdm(total=100) as pbar:
            # Update pbar in callback
            def progress_callback(progress):
                pbar.update(int(progress * 100) - pbar.n)
            
            synthetic_df = generate_synthetic_data_model_based(df, num_rows, model_type)
            pbar.update(100 - pbar.n)  # Ensure we reach 100%
        
        print("\nGenerated synthetic data:")
        print(synthetic_df.head())
        
        # Validate the synthetic data
        validate_data = prompt_for_input("Validate synthetic data quality? (yes/no)", "yes").lower()
        if validate_data == 'yes':
            metrics = validate_synthetic_data(df, synthetic_df)
            display_validation_results(metrics)
        
        # Save to CSV
        output_path = prompt_for_input("Output CSV path (include filename)", "synthetic_data.csv")
        save_to_csv(synthetic_df, output_path)
        
        # Optionally compare distributions
        compare_dist = prompt_for_input("Compare distributions? (yes/no)", "yes").lower()
        if compare_dist == 'yes':
            try:
                num_cols = int(prompt_for_input("Number of columns to compare", "3"))
                compare_distributions(df, synthetic_df, num_cols)
            except Exception as e:
                logger.error(f"Error comparing distributions: {str(e)}")
                print(f"Error visualizing distributions: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")
        
def mode_2_generate_interactive():
    """Function to handle Mode 2: Generate from interactive input"""
    print("\n=== Mode 2: Generate Synthetic Data From Scratch ===")
    
    # Get schema definition
    try:
        title, num_rows, schema = interactive_schema_prompt()
        
        # Check if user cancelled
        if num_rows == 0:
            return
        
        # Define relationships between columns
        schema = add_relationships_to_schema(schema)
    except Exception as e:
        logger.error(f"Error defining schema: {str(e)}")
        print(f"Error defining schema: {str(e)}")
        return
    
    # Option to save schema
    save_schema = prompt_for_input("Save schema to JSON file? (yes/no)", "no").lower()
    if save_schema == 'yes':
        default_schema_path = f"{title.lower().replace(' ', '_')}_schema.json"
        schema_path = prompt_for_input("Schema JSON path", default_schema_path)
        try:
            save_schema_to_json(schema, schema_path)
        except Exception as e:
            logger.error(f"Error saving schema: {str(e)}")
            print(f"Error saving schema: {str(e)}")
    
    # Generate synthetic data with progress bar
    try:
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        
        with tqdm(total=num_rows) as pbar:
            # First generate the basic synthetic data
            batch_size = min(1000, num_rows)  # Process in batches for large datasets
            synthetic_df = pd.DataFrame()
            
            for i in range(0, num_rows, batch_size):
                current_batch_size = min(batch_size, num_rows - i)
                batch_df = generate_synthetic_data_from_schema(schema, current_batch_size)
                synthetic_df = pd.concat([synthetic_df, batch_df], ignore_index=True)
                pbar.update(current_batch_size)
            
            # Apply relationships if defined
            if any('relationships' in col_info for col_info in schema):
                print("\nApplying relationships between columns...")
                synthetic_df = apply_relationships(synthetic_df, schema)
        
        print(f"\nGenerated {num_rows} rows of synthetic data for '{title}':")
        print(synthetic_df.head())
        
        # Save to CSV
        default_output = f"{title.lower().replace(' ', '_')}.csv"
        output_path = prompt_for_input("Output CSV path (include filename)", default_output)
        save_to_csv(synthetic_df, output_path)
        
        # Option to visualize distributions
        visualize = prompt_for_input("Visualize data distributions? (yes/no)", "no").lower()
        if visualize == 'yes':
            numeric_cols = [col for col in synthetic_df.columns if pd.api.types.is_numeric_dtype(synthetic_df[col])]
            if numeric_cols:
                num_cols_to_plot = min(3, len(numeric_cols))
                try:
                    visualize_data_distributions(synthetic_df, num_cols_to_plot)
                except Exception as e:
                    logger.error(f"Error visualizing distributions: {str(e)}")
                    print(f"Error visualizing distributions: {str(e)}")
            else:
                print("No numeric columns available for visualization.")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")

def mode_3_generate_from_schema():
    """Function to handle Mode 3: Generate from saved schema"""
    print("\n=== Mode 3: Generate Synthetic Data From Saved Schema ===")
    
    # Get schema file path
    while True:
        file_path = input("Enter path to schema JSON file: ").strip()
        if os.path.exists(file_path) and file_path.endswith('.json'):
            break
        print("File not found or not a JSON. Please enter a valid path.")
    
    # Load schema
    try:
        schema = load_schema_from_json(file_path)
        
        # Display schema info
        print(f"\nLoaded schema with {len(schema)} columns:")
        for col_info in schema:
            print(f"- {col_info['name']} ({col_info['type']})")
        
        # Get number of rows to generate
        default_rows = "100"
        num_rows_input = prompt_for_input("Number of synthetic rows to generate", default_rows)
        try:
            num_rows = int(num_rows_input)
        except ValueError:
            print(f"Invalid input. Using default value of {default_rows} rows.")
            num_rows = int(default_rows)
        
        # Generate synthetic data with progress bar
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        
        with tqdm(total=num_rows) as pbar:
            # First generate the basic synthetic data
            batch_size = min(1000, num_rows)  # Process in batches for large datasets
            synthetic_df = pd.DataFrame()
            
            for i in range(0, num_rows, batch_size):
                current_batch_size = min(batch_size, num_rows - i)
                batch_df = generate_synthetic_data_from_schema(schema, current_batch_size)
                synthetic_df = pd.concat([synthetic_df, batch_df], ignore_index=True)
                pbar.update(current_batch_size)
            
            # Apply relationships if defined
            if any('relationships' in col_info for col_info in schema):
                print("\nApplying relationships between columns...")
                synthetic_df = apply_relationships(synthetic_df, schema)
        
        # Get title from schema file
        title = os.path.basename(file_path).replace('_schema.json', '').replace('.json', '')
        
        print(f"\nGenerated {num_rows} rows of synthetic data:")
        print(synthetic_df.head())
        
        # Save to CSV
        default_output = f"{title}_synthetic.csv"
        output_path = prompt_for_input("Output CSV path (include filename)", default_output)
        save_to_csv(synthetic_df, output_path)
        
        # Option to visualize distributions
        visualize = prompt_for_input("Visualize data distributions? (yes/no)", "no").lower()
        if visualize == 'yes':
            numeric_cols = [col for col in synthetic_df.columns if pd.api.types.is_numeric_dtype(synthetic_df[col])]
            if numeric_cols:
                num_cols_to_plot = min(3, len(numeric_cols))
                try:
                    visualize_data_distributions(synthetic_df, num_cols_to_plot)
                except Exception as e:
                    logger.error(f"Error visualizing distributions: {str(e)}")
                    print(f"Error visualizing distributions: {str(e)}")
            else:
                print("No numeric columns available for visualization.")
            
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        print(f"Error: {str(e)}")

def show_example_usage():
    """Display example usage of the synthetic data generator"""
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    
    print("""
Example 1: Student Heights Dataset (Mode 2)
-------------------------------------------
This example creates a dataset with student information including heights that follow
a normal distribution and have a correlation with age.

When prompted:
- Dataset context: "Heights of students in schools"
- Dataset title: "Student Heights in Bangalore"
- Number of rows: 100

Columns:
1. student_id (UUID)
   - Type: uuid

2. name (Full Name)
   - Type: string
   - Subtype: full_name

3. age (Integer between 14-17)
   - Type: int
   - Min: 14
   - Max: 17
   - Distribution: uniform

4. school (School Name)
   - Type: string
   - Subtype: company (used for school names)

5. height_cm (Normal distribution)
   - Type: float
   - Min: 145
   - Max: 185
   - Distribution: normal
   - Mean: 160
   - Std: 10
   - Decimal places: 1

Relationships:
- Source: age, Target: height_cm
  - Type: correlation
  - Coefficient: 0.7
  (This creates a positive correlation where older students tend to be taller)

Example 2: Sales Transactions (Mode 2)
--------------------------------------
This example creates a dataset with sales transaction information.

When prompted:
- Dataset context: "Retail sales transactions"
- Dataset title: "Electronics Store Sales"
- Number of rows: 500

Columns:
1. transaction_id (UUID)
   - Type: uuid

2. date (Date range)
   - Type: date
   - Start date: 2023-01-01
   - End date: 2023-12-31

3. product_category (Category)
   - Type: category
   - Categories: Laptop, Smartphone, Tablet, Headphones, Camera
   - Custom probabilities: 0.3, 0.4, 0.1, 0.15, 0.05

4. price (Float)
   - Type: float
   - Min: 50
   - Max: 2000
   - Distribution: uniform

5. quantity (Integer)
   - Type: int
   - Min: 1
   - Max: 5
   - Distribution: uniform

6. customer_name (Name)
   - Type: string
   - Subtype: full_name

Relationships:
- Source: product_category, Target: price
  - Type: dependency
  - Mapping:
    - Laptop: center=1200, variance=300
    - Smartphone: center=800, variance=200
    - Tablet: center=500, variance=100
    - Headphones: center=150, variance=50
    - Camera: center=600, variance=150

- Source: price, Target: quantity
  - Type: correlation
  - Coefficient: -0.5
  (This creates a negative correlation where higher priced items are purchased in smaller quantities)
    """)
    
    input("\nPress Enter to return to the main menu...")
