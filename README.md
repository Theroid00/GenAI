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
- **Field Parsing**: Ensure logical consistency between related fields

## Quick Start

### Installation

No installation required, just clone the repository and ensure you have the required dependencies:

```bash
# Install all required dependencies
pip install -r requirements.txt

# Or install core dependencies manually
pip install pandas numpy faker tqdm matplotlib
# Optional for model-based generation
pip install sdv
```

### Running the Generator

You can use the launcher script for an interactive experience:

```bash
# Start the launcher interface
python launcher.py

# Or use the shell script
./run_generator.sh
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

## Field Parsing

The field parser ensures logical consistency between related fields that might not be fully captured by template relationships:

- **Product-Category Matching**: Ensures products belong to the appropriate categories
- **Case Normalization**: Standardizes text formatting for consistent display
- **Custom Parsing Rules**: Extensible system for adding domain-specific rules

See `FIELD_PARSER.md` for more details on how to extend the field parser with custom rules.

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

You can also use the generator from your Python code:

```python
from integrated_generator import generate_and_validate

# Generate data from a schema file
df = generate_and_validate(
    schema_path="my_schema.json",
    num_rows=500,
    output_path="output.csv"
)

# Generate data from a CSV sample
df = generate_and_validate(
    input_path="sample.csv",
    num_rows=1000,
    model_type="gaussian",
    output_path="synthetic.csv"
)
```

## Generation Modes

### Interactive Mode

This mode allows you to create datasets from scratch through an interactive command-line interface:
1. Launch the tool with `python launcher.py` and select option 6
2. Follow the prompts to define fields, types, and constraints
3. Generate and save your custom dataset

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

- `launcher.py` - Main entry point with interactive menu
- `synthetic_data_cli.py` - Comprehensive command-line interface
- `create_template.py` - Interactive template creation tool
- `quick_generate.py` - Simplified command-line interface
- `integrated_generator.py` - Combined generator with validation
- `synthetic_data_generator.py` - Core generator functionality
- `interactive_mode.py` - Direct launcher for interactive data generation
- `field_standards.py` - Standards for field formatting
- `field_parser.py` - Logical consistency between related fields
- `run_generator.sh` - Shell script to run the generator
- `synthetic_data_gen/` - Package with modular components
- `experimental/validation/` - Validation module
- `templates/` - Pre-defined templates
- `tests/` - Test files and test data

## Dependencies

- Required: pandas, numpy, matplotlib, faker, tqdm
- Optional: sdv (for model-based generation)

## Documentation

For more detailed information about the project structure and implementation, refer to:

- `CHANGELOG.md` - History of changes and version information
- `FIELD_STANDARDS.md` - Documentation of standardized field formats for realistic data
- `FIELD_PARSER.md` - Documentation of field parsing rules for logical consistency
- `experimental/validation/README.md` - Documentation for the validation module

## License

This project is licensed under the MIT License - see the LICENSE file for details.
