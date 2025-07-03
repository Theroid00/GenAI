# Synthetic Data Generator - Field Standards Documentation

## Overview

This document outlines the standardized field formats implemented in the Synthetic Data Generator to ensure consistency and realism across all generated datasets.

## Field Types and Formats

### 1. ID Fields

The system automatically detects and formats various ID field types based on column names:

| ID Type   | Format        | Detection Terms                           | Example      |
|-----------|---------------|------------------------------------------|--------------|
| Employee  | EMP-{number}  | emp, employee, staff, worker             | EMP-10001    |
| Customer  | CUST-{number} | cust, customer, client, buyer            | CUST-20001   |
| Product   | PROD-{number} | prod, product, item, good, sku           | PROD-30001   |
| Order     | ORD-{number}  | ord, order, transaction, purchase        | ORD-40001    |
| Invoice   | INV-{number}  | inv, invoice, bill, receipt              | INV-50001    |
| Ticket    | TKT-{number}  | tkt, ticket, issue, case                 | TKT-60001    |
| Student   | STU-{number}  | stu, student, learner, pupil             | STU-70001    |
| Generic   | ID-{number}   | id, uuid (fallback for any ID field)     | ID-90001     |

### 2. Email Addresses

Email addresses are generated with realistic formats based on the person's name and weighted domain distribution:

#### Email Format Patterns:
- {first}.{last}@{domain} (30%)
- {first}{last}@{domain} (15%)
- {first}_{last}@{domain} (10%)
- {first}{last_initial}@{domain} (10%)
- {first_initial}{last}@{domain} (10%)
- {last}.{first}@{domain} (5%)
- {first}-{last}@{domain} (5%)
- {first}@{domain} (5%)
- {last}@{domain} (5%)
- {first}.{last_initial}@{domain} (5%)

#### Domain Distribution:
- gmail.com (35%)
- yahoo.com (15%)
- outlook.com (12%)
- hotmail.com (10%)
- icloud.com (8%)
- aol.com (5%)
- protonmail.com (4%)
- mail.com (3%)
- zoho.com (2%)
- gmx.com (2%)
- yandex.com (1%)
- tutanota.com (1%)
- fastmail.com (1%)
- comcast.net (1%)

There's also a 25% chance to add a random number to the email for uniqueness.

### 3. Phone Numbers

Phone numbers are generated in various realistic formats with weighted distribution:

- (XXX) XXX-XXXX (30%) - Example: (555) 123-4567
- XXX-XXX-XXXX (20%) - Example: 555-123-4567
- XXX.XXX.XXXX (15%) - Example: 555.123.4567
- X-XXX-XXX-XXXX (10%) - Example: 1-555-123-4567
- XXX-XXXX (10%) - Example: 123-4567
- XXXXXXXXXX (10%) - Example: 5551234567
- X (XXX) XXX-XXXX (5%) - Example: 1 (555) 123-4567

## Implementation Details

The field standards are implemented in the following files:

1. `field_standards.py` - Core standards module with all patterns and utility functions
2. `synthetic_data_generator.py` - Main script with updated post-processing function

The key functions in `field_standards.py` are:

- `detect_id_field_type(column_name)` - Detects what type of ID a column represents
- `generate_id(id_type, index)` - Generates properly formatted IDs
- `format_email(first_name, last_name)` - Generates realistic emails based on names
- `format_phone_number()` - Generates realistic phone number formats

## Usage Examples

### Generating IDs

```python
from field_standards import detect_id_field_type, generate_id

# Detect ID type from column name
id_type = detect_id_field_type("employee_id")  # Returns "employee"

# Generate IDs
emp_id = generate_id("employee", 0)  # Returns "EMP-10001"
cust_id = generate_id("customer", 0)  # Returns "CUST-20001"
```

### Generating Emails

```python
from field_standards import format_email

# Generate an email based on a name
email = format_email("John", "Doe")  # Returns something like "john.doe@gmail.com"
```

### Generating Phone Numbers

```python
from field_standards import format_phone_number

# Generate a phone number
phone = format_phone_number()  # Returns something like "(555) 123-4567"
```

## Testing

A comprehensive test suite is provided in `test_synthetic_data.py` to validate the field standards implementation. The tests cover:

1. ID detection and generation
2. Email format validity and distribution
3. Phone number format validity and distribution
4. End-to-end testing with a full synthetic dataset

An additional schema-based test is available in `test_schema_generation.py` to validate the integration with the schema-based generation mode.

## Benefits

The standardized field formats provide several benefits:

1. **Realism** - Data looks more realistic and professional
2. **Consistency** - Formats are consistent across all generated datasets
3. **Customizability** - Patterns and distributions can be easily adjusted
4. **Maintainability** - Field format logic is centralized and well-documented
