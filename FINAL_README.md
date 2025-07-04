# Synthetic Data Generator

A comprehensive tool for generating high-quality synthetic data with built-in validation and customizable templates.

## Features

- **Multiple Generation Modes**:
  - Generate from CSV sample (model-based approach)
  - Generate from JSON schema definitions
  - Generate interactively through prompts
  - Use built-in templates for common data structures
- **Automatic Validation**: Detect and fix data quality issues
- **Standardized Field Formats**: Generate realistic IDs, emails, phone numbers, and more
- **Rich Data Types**: Support for various data types and distributions
- **Relationship Modeling**: Create dependencies between columns

## Quick Start

### Installation

No installation required, just clone the repository and ensure you have the required dependencies:

```bash
pip install pandas numpy faker tqdm matplotlib
# Optional for model-based generation
pip install sdv
```

### Usage

The easiest way to generate data is using the comprehensive CLI interface:

```bash
# Generate data using a built-in template
python synthetic_data_cli.py --template customer_template --rows 100 --output customer_data.csv

# Generate data from a schema file
python synthetic_data_cli.py --schema my_schema.json --rows 500 --output my_data.csv

# Generate data from a sample CSV file
python synthetic_data_cli.py --sample sample_data.csv --rows 1000 --output synthetic_data.csv
```

### Managing Templates

```bash
# List all available templates
python synthetic_data_cli.py --list-templates

# Show details about a specific template
python synthetic_data_cli.py --template-info customer_template

# Create a new template interactively
python synthetic_data_cli.py --create-template my_template
```

The dedicated template creator provides a more guided experience:

```bash
# Create a template interactively
python create_template.py --name my_template

# Create a template based on an example
python create_template.py --name retail_template --example sales
```

Available templates:
- `customer_template` - Customer profiles with demographic information
- `employee_template` - Employee records with department and salary information
- `sales_template` - Sales transaction data with products and prices
- `student_template` - Student records with academic information

## Data Validation

The integrated validation system automatically fixes various data quality issues:

- ID format inconsistencies
- Name capitalization issues
- Email consistency with names
- Phone number formatting
- Numeric value range enforcement
- Geographic consistency
- Uniqueness constraints

## Creating Custom Schemas

Create a JSON file with an array of column definitions:

```json
[
  {
    "name": "customer_id",
    "type": "uuid"
  },
  {
    "name": "full_name",
    "type": "string",
    "subtype": "full_name"
  },
  {
    "name": "age",
    "type": "int",
    "min": 18,
    "max": 85,
    "distribution": "normal",
    "mean": 42,
    "std": 12
  }
]
```

See the `templates` directory for more examples.

## Field Types and Properties

| Type | Required Properties | Optional Properties |
|------|---------------------|---------------------|
| `int` | `name`, `min`, `max` | `distribution`, `mean`, `std` |
| `float` | `name`, `min`, `max` | `distribution`, `mean`, `std`, `decimals` |
| `string` | `name` | `subtype`, `pattern`, `min_length`, `max_length` |
| `category` | `name`, `categories` | `weights` |
| `date` | `name` | `start_date`, `end_date` |
| `uuid` | `name` | - |

## Programmatic API

You can use the generator from your Python code:

```python
from synthetic_data_gen import SyntheticDataGenerator, load_schema_from_json, save_to_csv

# Create generator instance
generator = SyntheticDataGenerator()

# Generate data from a schema file
schema = load_schema_from_json("my_schema.json")
df = generator.generate_from_schema(
    schema=schema,
    num_rows=500,
    output_path="output.csv"
)

# Generate data from a CSV sample
df = generator.generate_from_csv(
    csv_path="sample.csv",
    num_rows=1000,
    model_type="gaussian",
    output_path="synthetic.csv"
)

# Visualize data distributions
from synthetic_data_gen.validation import visualize_data_distributions
visualize_data_distributions(df)
```

## Generation Modes

### Model-based Generation

This mode uses an existing CSV dataset to learn patterns and generate similar synthetic data. It:
1. Loads a sample CSV file
2. Infers the schema and data distributions
3. Trains a model (Gaussian Copula or CTGAN)
4. Generates synthetic data that preserves the statistical properties of the original
5. Post-processes and validates the data to ensure quality

### Schema-based Generation

This mode uses a JSON schema to define the structure and properties of the data to generate. It:
1. Loads a schema definition
2. Generates data according to the specified types and constraints
3. Applies relationships between fields if defined
4. Validates and fixes any quality issues

### Template-based Generation

This mode uses pre-defined templates for common data structures:
1. Select a template (customer, employee, sales, student)
2. Customize parameters like row count and output file
3. Generate data with consistent structure and realistic values

## Project Structure

The project has been modularized for better maintainability and extensibility:

- `synthetic_data_gen/` - Main package with modular components
  - `__init__.py` - Package exports
  - `core/` - Core functionality
    - `generator.py` - SyntheticDataGenerator class
    - `schema.py` - Schema handling
    - `relationships.py` - Column relationship handling
  - `utils/` - Utility functions
    - `file_io.py` - File I/O operations
    - `standards.py` - Field standards
  - `validation/` - Data validation
    - `validator.py` - Data validation
  - `providers/` - Custom data providers
    - `region_manager.py` - Region-specific data
    - Various region modules (usa.py, germany.py, etc.)

### Interface Scripts

- `launcher.py` - Main entry point with interactive menu
- `synthetic_data_cli.py` - Comprehensive command-line interface
- `interactive_mode.py` - Interactive data generation
- `run_generator.sh` - Shell script to run the generator

### Supporting Files

- `templates/` - Pre-defined templates
- `tests/` - Test files and test data
- `example_data/` - Example datasets
- `experimental/` - Experimental features

## Dependencies

- Required: pandas, numpy, matplotlib, faker, tqdm
- Optional: sdv (for model-based generation)

## Documentation

For more detailed information about the project structure and implementation, refer to:

- `CHANGELOG.md` - History of changes and version information
- `FIELD_STANDARDS.md` - Documentation of standardized field formats for realistic data
- `experimental/validation/README.md` - Documentation for the validation module

## License

This project is licensed under the MIT License - see the LICENSE file for details.
