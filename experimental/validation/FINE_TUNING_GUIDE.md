# Guide to Fine-tuning Synthetic Data Generation Models

This guide outlines the types of data needed for proper fine-tuning of synthetic data generation models and best practices for improving generation quality.

## Types of Data Needed for Fine-tuning

### 1. Representative Training Data

The most important component for fine-tuning is high-quality, representative training data that matches your target domain:

- **Quantity**: At minimum 1,000-5,000 rows for simple datasets, 10,000+ rows for complex datasets with many columns and relationships
- **Completeness**: Data should have minimal missing values (ideally <5% per column)
- **Quality**: Data should be clean, consistent, and follow the patterns you want to generate
- **Distribution**: Data should follow the real statistical distributions you want to replicate
- **Coverage**: Data should cover the full range of values and edge cases

### 2. Validation Data

A separate validation dataset (not used for training) to evaluate the quality of synthetic data:

- **Holdout Set**: Typically 10-20% of your original data
- **Benchmark Metrics**: Data to compare statistical properties (means, standard deviations, correlations)
- **Domain Tests**: Data that can be used to test domain-specific rules and constraints

### 3. Reference Data for Patterns

Reference data for specific patterns that need to be replicated:

- **ID Formats**: Examples of different ID formats used in your organization
- **Name Patterns**: Cultural and regional name patterns relevant to your data
- **Address Formats**: Address formats for relevant countries/regions
- **Email Conventions**: Company or organization-specific email conventions
- **Field Relationships**: Examples showing relationships between fields (e.g., job titles and salaries)

### 4. Constraint Data

Data that defines constraints and rules:

- **Valid Value Lists**: Lists of valid values for categorical fields
- **Range Constraints**: Min/max boundaries for numeric fields
- **Regex Patterns**: Regular expressions for formatted fields
- **Relational Rules**: Rules defining relationships between fields

## Data Preparation for Fine-tuning

### Clean and Normalize Input Data

Before fine-tuning, prepare your training data:

1. **Fix Inconsistencies**: Standardize formats, capitalization, spacing
2. **Remove Outliers**: Identify and remove or correct statistical outliers
3. **Handle Missing Values**: Impute or remove missing values
4. **Normalize Dates**: Ensure consistent date formats
5. **Validate Relationships**: Ensure relationships between fields are correct

### Augment with Domain Knowledge

Enhance training data with domain-specific information:

1. **Add Derived Features**: Calculate features that capture important relationships
2. **Enrich with Metadata**: Add categorical markers or groupings
3. **Include Business Rules**: Encode business rules as additional features
4. **Annotate Edge Cases**: Mark special cases in the data

## Model Selection and Configuration

Different data characteristics require different models:

### Tabular Data with Strong Correlations

- **GaussianCopula**: Best for data with linear and monotonic relationships
- **Required Data**: Data with clear statistical correlations between columns
- **Configuration Example**:
  ```python
  model = GaussianCopula(
      field_distributions={
          'age': 'beta',
          'income': 'gamma',
          'tenure': 'beta'
      },
      default_distribution='gaussian_kde'
  )
  ```

### Complex Non-linear Relationships

- **CTGAN/TVAE**: Better for capturing complex, non-linear relationships
- **Required Data**: Larger datasets (5,000+ rows) with complex patterns
- **Configuration Example**:
  ```python
  model = CTGAN(
      epochs=500,
      batch_size=200,
      discriminator_steps=3
  )
  ```

### Sequence or Time-series Data

- **PAR**: For sequential or time-series data
- **Required Data**: Sequential records with temporal patterns
- **Configuration Example**:
  ```python
  model = PAR(
      sequence_length=10,
      context_columns=['id']
  )
  ```

## Field-specific Fine-tuning Approaches

### ID Fields

- **Data Needed**: Examples of all ID formats used in your organization
- **Approach**: Separate prefix/sequence pattern from the numeric part
- **Example**:
  ```python
  # Extract patterns from existing IDs
  df['id_prefix'] = df['employee_id'].str.extract(r'([A-Z]+-)')
  df['id_number'] = df['employee_id'].str.extract(r'-(\d+)').astype(int)
  # Train on these components separately
  ```

### Names

- **Data Needed**: Examples of culturally diverse names relevant to your data
- **Approach**: Consider training on first/last name components separately
- **Example**:
  ```python
  # Split names into components
  df[['first_name', 'last_name']] = df['full_name'].str.split(' ', n=1, expand=True)
  # Train on these components
  ```

### Addresses

- **Data Needed**: Address examples from all relevant geographic regions
- **Approach**: Decompose into components (street, city, state, etc.)
- **Example**:
  ```python
  # Define regex patterns for address components
  street_pattern = r'^(.*?),'
  city_pattern = r', (.*?),'
  # Extract components
  df['street'] = df['address'].str.extract(street_pattern)
  df['city'] = df['address'].str.extract(city_pattern)
  # Train on these components
  ```

### Email Addresses

- **Data Needed**: Examples of email patterns used in your organization
- **Approach**: Model username and domain components separately
- **Example**:
  ```python
  # Split emails into components
  df[['username', 'domain']] = df['email'].str.split('@', expand=True)
  # Train on these components
  ```

## Fine-tuning Process

1. **Start with a Baseline Model**:
   ```python
   from sdv.single_table import GaussianCopula
   
   model = GaussianCopula()
   model.fit(training_data)
   synthetic_data = model.sample(num_rows=1000)
   ```

2. **Evaluate Quality**:
   ```python
   from sdv.evaluation.single_table import evaluate_quality
   
   # Compare synthetic data to original
   quality_report = evaluate_quality(
       synthetic_data,
       training_data
   )
   ```

3. **Analyze Issues**:
   - Identify columns with poor quality
   - Examine relationships that aren't preserved
   - Look for unrealistic values or patterns

4. **Adjust Model Configuration**:
   ```python
   # Define specific distributions for problematic columns
   model = GaussianCopula(
       field_distributions={
           'problematic_column': 'beta',
       }
   )
   ```

5. **Add Constraints**:
   ```python
   from sdv.constraints import Unique, GreaterThan
   
   # Define constraints
   constraints = [
       Unique('id'),
       GreaterThan(
           high='end_date',
           low='start_date'
       )
   ]
   
   # Apply constraints to model
   model = GaussianCopula(constraints=constraints)
   ```

6. **Re-train and Re-evaluate**:
   ```python
   model.fit(training_data)
   new_synthetic_data = model.sample(num_rows=1000)
   new_quality_report = evaluate_quality(
       new_synthetic_data,
       training_data
   )
   ```

7. **Apply Post-processing**:
   ```python
   # Use the validation module to fix remaining issues
   from validation.data_validator import validate_and_fix_data
   
   final_data, corrections = validate_and_fix_data(new_synthetic_data, schema)
   ```

## Continuous Improvement

### Monitor and Track Quality

- **Track Quality Metrics**: Keep a record of quality metrics over time
- **Document Changes**: Record which configuration changes improved which aspects
- **Maintain Test Suite**: Develop tests for common issues and edge cases

### Expanding Training Data

- **Incremental Addition**: Add new examples that cover edge cases
- **Focused Augmentation**: Generate additional examples for underrepresented categories
- **Synthetic Boosting**: Use high-quality synthetic data to augment real data in areas with privacy concerns

## Additional Resources

For more advanced fine-tuning techniques, refer to:

1. The [SDV documentation](https://sdv.dev/SDV/) for model-specific parameters
2. Academic papers on synthetic data generation for domain-specific applications
3. Our experimental validation module for analyzing and fixing common issues

## Example Minimal Training Dataset

Here's an example of the minimum data needed for training a realistic employee dataset:

```
# employee_minimal_training.csv (recommended: 1000+ rows)
employee_id,full_name,email,department,job_title,salary,start_date,manager_id
EMP-10001,John Smith,john.smith@company.com,Engineering,Senior Developer,95000,2020-01-15,EMP-10005
EMP-10002,Jane Doe,jane.doe@company.com,Marketing,Marketing Manager,85000,2019-05-20,EMP-10006
EMP-10003,Robert Johnson,robert.johnson@company.com,Finance,Financial Analyst,75000,2021-03-10,EMP-10007
...
```

The key is having enough examples to capture:
1. The format patterns (IDs, emails, etc.)
2. The statistical distributions (salary ranges, etc.)
3. The relationships between fields (departments and job titles, etc.)

With this data and proper fine-tuning, you can generate much larger synthetic datasets that maintain the patterns and relationships in your original data.
