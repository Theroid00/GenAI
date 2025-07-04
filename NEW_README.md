# Synthetic Data Generator

A comprehensive tool for generating high-quality synthetic data with built-in validation and customizable templates.

## Features

- **Multiple Generation Modes**
  - **Sample-based**: Learn patterns from existing CSV data using models
  - **Schema-based**: Define precise data structures with JSON schemas
  - **Template-based**: Use built-in templates for common data structures
  - **Interactive**: Create data schemas through guided prompts

- **Comprehensive Validation**
  - Automatic detection and correction of data quality issues
  - Format validation for IDs, emails, phone numbers, and more
  - Range validation for numeric fields
  - Consistency validation between related fields
  - Uniqueness enforcement for identifiers

- **Standardized Field Formats**
  - Realistic ID patterns (employee, customer, product, etc.)
  - Properly formatted email addresses with configurable domains
  - Varied phone number formats with proper distribution
  - Name capitalization and formatting
  - Geographic consistency between related fields

- **Rich Data Types**
  - Numeric (int/float) with uniform or normal distributions
  - Categorical with weighted probabilities
  - Strings with various subtypes (name, address, etc.)
  - Dates with configurable ranges
  - IDs and UUIDs with format control

- **Relationship Modeling**
  - Define dependencies between columns
  - Create correlations between numeric fields
  - Ensure consistency in related categorical values

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-data-generator.git
cd synthetic-data-generator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

The easiest way to generate data is using the `synthetic_data_cli.py` script:

```bash
# Generate data using a built-in template
./synthetic_data_cli.py --template customer --rows 100 --output customer_data.csv

# Generate data from a schema file
./synthetic_data_cli.py --schema my_schema.json --rows 500 --output my_data.csv

# Generate data from a sample CSV file
./synthetic_data_cli.py --sample sample_data.csv --rows 1000 --output synthetic_data.csv
```

### Managing Templates

```bash
# List all available templates
./synthetic_data_cli.py --list-templates

# Show details about a specific template
./synthetic_data_cli.py --template-info customer

# Create a new template interactively
./synthetic_data_cli.py --create-template my_template
```

## Available Templates

- `customer` - Customer profiles with demographic information
- `employee` - Employee records with department and salary information
- `sales` - Sales transaction data with products and prices
- `student` - Student records with academic information

## Creating Custom Schemas

You can create a JSON schema file with an array of column definitions:

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
  },
  {
    "name": "membership_level",
    "type": "category",
    "categories": ["Silver", "Gold", "Platinum", "Diamond"],
    "weights": [0.5, 0.3, 0.15, 0.05]
  }
]
```

### Field Types and Properties

| Type | Required Properties | Optional Properties |
|------|---------------------|---------------------|
| `int` | `name`, `min`, `max` | `distribution`, `mean`, `std` |
| `float` | `name`, `min`, `max` | `distribution`, `mean`, `std`, `decimals` |
| `string` | `name` | `subtype`, `pattern`, `min_length`, `max_length` |
| `category` | `name`, `categories` | `weights` |
| `date` | `name` | `start_date`, `end_date` |
| `uuid` | `name` | - |

#### String Subtypes

- `full_name` - Complete person names
- `first_name` - First names only
- `last_name` - Last names only
- `email` - Email addresses
- `phone` - Phone numbers
- `address` - Physical addresses
- `city` - City names
- `country` - Country names
- `company` - Company names
- `job` - Job titles
- `text` - Generic text

## Advanced Usage

### Programmatic API

You can use the generator from your Python code:

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

### Using the Core Generator

For more control, you can use the original generator directly:

```python
from synthetic_data_generator import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator()

# Generate from schema
df = generator.generate_from_schema(schema, num_rows=500)

# Generate from CSV
df = generator.generate_from_csv("sample.csv", num_rows=1000, model_type="gaussian")

# Interactive mode
generator.interactive_mode()
```

## Generation Modes

### Model-based Generation

This mode uses an existing CSV dataset to learn patterns and generate similar synthetic data:

1. Loads a sample CSV file
2. Infers the schema and data distributions
3. Trains a model (Gaussian Copula or CTGAN)
4. Generates synthetic data preserving statistical properties
5. Post-processes and validates the data

Requirements: The optional `sdv` package for model training.

### Schema-based Generation

This mode uses a JSON schema to define the structure and properties of the data:

1. Loads a schema definition
2. Generates data according to specified types and constraints
3. Applies relationships between fields if defined
4. Validates and fixes any quality issues

### Template-based Generation

This mode uses pre-defined templates for common data structures:

1. Select a template (customer, employee, sales, student)
2. Customize parameters like row count and output file
3. Generate data with consistent structure and realistic values

## Data Validation

The integrated validation system automatically fixes various data quality issues:

- ID format inconsistencies
- Name capitalization issues
- Email consistency with names
- Phone number formatting
- Numeric value range enforcement
- Geographic consistency
- Uniqueness constraints

## Dependencies

- Required: pandas, numpy, matplotlib, faker, tqdm
- Optional: sdv (for model-based generation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
