# Synthetic Data Generator

A versatile Python tool for generating synthetic data with multiple operational modes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)

## Features

- **Mode 1: Generate from CSV sample** - Use a sample CSV file to train a model and generate similar synthetic data
- **Mode 2: Generate from scratch** - Interactively define a schema and generate data based on that schema
- **Mode 3: Generate from saved schema** - Load a previously saved schema and generate data

## Installation

### Quick Setup (Recommended)

The easiest way to get started is to use the provided setup script, which creates a virtual environment and installs all dependencies:

```bash
# Clone the repository
git clone https://github.com/theroid/synthetic-data-generator.git
cd synthetic-data-generator

# Make the setup script executable (Linux/Mac)
chmod +x setup_env.py

# Run the setup script
./setup_env.py  # On Linux/Mac if executable
# OR
python setup_env.py  # On any platform

# Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### Manual Installation

Alternatively, you can set up the environment manually:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### With Model-Based Generation Support

For model-based generation (Mode 1), you'll need to install the SDV package:

```bash
pip install sdv
```

## Usage

After installation, you can run the generator using the following command:

```bash
# If installed with setup_env.py or manually
python run_generator.py

# OR (if the script is executable on Linux/Mac)
./run_generator.py
```

### Operational Modes

1. **Mode 1: Generate from CSV Sample**
   - Provide a sample CSV file
   - The generator will analyze it and create similar synthetic data
   - Model-based generation using the SDV package (if installed)

2. **Mode 2: Generate Interactively**
   - Define your data schema through interactive prompts
   - Specify column types, distributions, and relationships
   - Generate data based on your specifications

3. **Mode 3: Generate from Saved Schema**
   - Load a previously saved schema file
   - Generate data according to the pre-defined schema

### Programmatic API

You can also use the generator from your own Python code:

```python
from synthetic_data_gen import SyntheticDataGenerator

# Create a generator instance
generator = SyntheticDataGenerator()

# Generate from CSV
synthetic_data = generator.generate_from_csv('sample.csv', num_rows=100)

# Save to CSV
synthetic_data.to_csv('synthetic_output.csv', index=False)
```

## Documentation

For more detailed information about the project structure and implementation, refer to:

- `MODULARIZATION.md` - Details about the project's module structure
- `CHANGELOG.md` - History of changes and version information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Usage

### Command-Line Interface

```bash
# Run the generator with the interactive interface
python run_generator.py

# Or use the console script
synthetic-data
```

### Python API

```python
from synthetic_data_gen import SyntheticDataGenerator

# Create a generator instance
generator = SyntheticDataGenerator()

# Mode 1: Generate from CSV
df = generator.generate_from_csv(
    csv_path="sample_data.csv",
    num_rows=1000,
    model_type='gaussian'
)

# Mode 2: Interactive schema definition
generator.interactive_mode()

# Mode 3: Generate from saved schema
from synthetic_data_gen import load_schema_from_json
schema = load_schema_from_json("schema.json")
df = generator.generate_from_schema(schema, num_rows=500)

# Visualization
generator.visualize_data(num_cols=3)
```

## Operational Modes

### Mode 1: Generate from CSV Sample

This mode uses a model-based approach to learn the patterns in an existing CSV dataset and generate similar synthetic data. It requires the optional SDV package:

```bash
pip install sdv
```

This mode:
1. Loads a sample CSV file
2. Infers the schema and data distributions
3. Trains a model (Gaussian Copula or CTGAN)
4. Generates synthetic data that preserves the statistical properties of the original

### Mode 2: Generate from Scratch

This mode allows you to interactively define a schema from scratch:

1. Define columns, their data types, and constraints
2. Set up relationships between columns (correlations, dependencies, transformations)
3. Generate synthetic data based on the defined schema

### Mode 3: Generate from Saved Schema

This mode loads a previously saved schema definition from a JSON file and generates data accordingly. This is useful for repeatable data generation with consistent properties.

## Dependencies

- Required: pandas, numpy, matplotlib, faker, tqdm
- Optional: sdv (for model-based generation)

## License

MIT
