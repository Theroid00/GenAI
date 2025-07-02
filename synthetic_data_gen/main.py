"""
Main entry point for the synthetic data generator.
This module provides the command-line interface for the synthetic data generator.
"""

import logging
from synthetic_data_gen.utils import check_dependencies
from synthetic_data_gen.modes import (
    mode_1_generate_from_csv,
    mode_2_generate_interactive,
    mode_3_generate_from_schema,
    show_example_usage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='synthetic_data_generator.log'
)
logger = logging.getLogger('synthetic_data_generator')

class SyntheticDataGenerator:
    """
    Main class for the Synthetic Data Generator.
    This class provides an object-oriented interface to the generator functions.
    """
    
    def __init__(self):
        """Initialize the generator"""
        self.last_original_df = None
        self.last_synthetic_df = None
        self.last_schema = None
    
    def generate_from_csv(self, csv_path: str, num_rows: int = 100, model_type: str = 'gaussian'):
        """
        Generate synthetic data based on a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            num_rows: Number of rows to generate
            model_type: Type of model to use ('gaussian' or 'ctgan')
            
        Returns:
            DataFrame with synthetic data
        """
        from synthetic_data_gen.utils import load_csv
        from synthetic_data_gen.generators import generate_synthetic_data_model_based
        
        # Load CSV
        original_df = load_csv(csv_path)
        self.last_original_df = original_df
        
        # Generate synthetic data
        synthetic_df = generate_synthetic_data_model_based(original_df, num_rows, model_type)
        self.last_synthetic_df = synthetic_df
        
        return synthetic_df
    
    def generate_from_schema(self, schema, num_rows: int = 100):
        """
        Generate synthetic data based on a schema.
        
        Args:
            schema: Schema definition (either list or dict with 'schema' key)
            num_rows: Number of rows to generate
            
        Returns:
            DataFrame with synthetic data
        """
        from synthetic_data_gen.generators import generate_synthetic_data_from_schema
        
        # Extract schema if needed
        if isinstance(schema, dict) and 'schema' in schema:
            schema = schema['schema']
        
        self.last_schema = schema
        
        # Generate synthetic data
        synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
        self.last_synthetic_df = synthetic_df
        
        return synthetic_df
    
    def visualize_data(self, num_cols: int = 3):
        """
        Visualize comparison between original and synthetic data.
        
        Args:
            num_cols: Number of columns to visualize
        """
        from synthetic_data_gen.validation import compare_distributions
        
        if self.last_original_df is None or self.last_synthetic_df is None:
            logger.warning("No data available for visualization")
            print("No data available for visualization. Generate data first.")
            return
        
        compare_distributions(self.last_original_df, self.last_synthetic_df, num_cols)
    
    def interactive_mode(self):
        """Start the interactive mode for schema definition and data generation"""
        from synthetic_data_gen.schema import interactive_schema_prompt, add_relationships_to_schema
        from synthetic_data_gen.generators import generate_synthetic_data_from_schema
        from synthetic_data_gen.utils import prompt_for_input, save_to_csv
        
        title, num_rows, schema = interactive_schema_prompt()
        
        # Apply relationships if requested
        schema = add_relationships_to_schema(schema)
        
        print(f"\nGenerating {num_rows} rows of synthetic data...")
        synthetic_df = generate_synthetic_data_from_schema(schema, num_rows)
        self.last_synthetic_df = synthetic_df
        self.last_schema = schema
        
        print("\nSample of generated data:")
        print(synthetic_df.head())
        
        # Ask for output file
        output_path = prompt_for_input(
            "Output CSV path (include filename)", 
            f"{title.lower().replace(' ', '_')}.csv"
        )
        
        save_to_csv(synthetic_df, output_path)
        print(f"Saved {num_rows} rows to {output_path}")

def main():
    """Main function to run the synthetic data generator"""
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR")
    print("=" * 80)
    print("\nThis tool allows you to generate synthetic data in three modes:")
    print("  1. Generate from a sample CSV file using model-based approach")
    print("  2. Generate from scratch via interactive input")
    print("  3. Generate from saved schema file")
    
    all_dependencies_met, message = check_dependencies()
    if message:
        print("\n=== Dependency Check ===")
        print(message)
    
    if not all_dependencies_met:
        print("Please install the required dependencies before continuing.")
        return
    
    while True:
        print("\nSelect mode:")
        print("  1. Generate from CSV sample")
        print("  2. Generate from scratch (interactive)")
        print("  3. Generate from saved schema")
        print("  e. View example usage")
        print("  q. Quit")
        
        choice = input("\nEnter your choice (1/2/3/e/q): ").strip().lower()
        
        if choice == '1':
            mode_1_generate_from_csv()
        elif choice == '2':
            mode_2_generate_interactive()
        elif choice == '3':
            mode_3_generate_from_schema()
        elif choice == 'e':
            show_example_usage()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, e, or q.")

if __name__ == "__main__":
    main()
