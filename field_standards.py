"""
Field standards module for synthetic data generation.

This module defines standards and formats for different types of synthetic data fields
to ensure consistency and realism across generated datasets.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Pattern
import random
from datetime import datetime, timedelta

# Standard patterns for ID fields
ID_PATTERNS = {
    'employee': {
        'prefix': 'EMP-',
        'start': 10001,
        'format': '{prefix}{id}',
        'detection': ['emp', 'employee', 'staff', 'worker'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    },
    'customer': {
        'prefix': 'CUST-',
        'start': 20001,
        'format': '{prefix}{id}',
        'detection': ['cust', 'customer', 'client', 'buyer'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    },
    'product': {
        'prefix': 'PROD-',
        'start': 30001,
        'format': '{prefix}{id}',
        'detection': ['prod', 'product', 'item', 'good'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no', 'code', 'sku']
    },
    'order': {
        'prefix': 'ORD-',
        'start': 40001,
        'format': '{prefix}{id}',
        'detection': ['ord', 'order', 'transaction', 'purchase'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    },
    'invoice': {
        'prefix': 'INV-',
        'start': 50001,
        'format': '{prefix}{id}',
        'detection': ['inv', 'invoice', 'bill', 'receipt'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    },
    'ticket': {
        'prefix': 'TKT-',
        'start': 60001,
        'format': '{prefix}{id}',
        'detection': ['tkt', 'ticket', 'issue', 'case'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    },
    'student': {
        'prefix': 'STU-',
        'start': 70001,
        'format': '{prefix}{id}',
        'detection': ['stu', 'student', 'learner', 'pupil'],
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no', 'roll']
    },
    'generic': {
        'prefix': 'ID-',
        'start': 90001,
        'format': '{prefix}{id}',
        'detection': [],  # Fallback for any ID field not matching other patterns
        'id_markers': ['id', '_id', 'id_', 'number', 'num', 'no']
    }
}

# Email domain distribution for realistic email generation
EMAIL_DOMAINS = {
    'gmail.com': 0.35,        # 35% chance
    'yahoo.com': 0.15,        # 15% chance
    'outlook.com': 0.12,      # 12% chance 
    'hotmail.com': 0.10,      # 10% chance
    'icloud.com': 0.08,       # 8% chance
    'aol.com': 0.05,          # 5% chance
    'protonmail.com': 0.04,   # 4% chance
    'mail.com': 0.03,         # 3% chance
    'zoho.com': 0.02,         # 2% chance
    'gmx.com': 0.02,          # 2% chance
    'yandex.com': 0.01,       # 1% chance
    'tutanota.com': 0.01,     # 1% chance
    'fastmail.com': 0.01,     # 1% chance
    'comcast.net': 0.01       # 1% chance
}

# Email format patterns with relative frequency
EMAIL_FORMATS = [
    ('{first}.{last}@{domain}', 0.30),          # john.doe@domain.com
    ('{first}{last}@{domain}', 0.15),           # johndoe@domain.com
    ('{first}_{last}@{domain}', 0.10),          # john_doe@domain.com
    ('{first}{last_initial}@{domain}', 0.10),   # johnd@domain.com
    ('{first_initial}{last}@{domain}', 0.10),   # jdoe@domain.com
    ('{last}.{first}@{domain}', 0.05),          # doe.john@domain.com
    ('{first}-{last}@{domain}', 0.05),          # john-doe@domain.com
    ('{first}@{domain}', 0.05),                 # john@domain.com
    ('{last}@{domain}', 0.05),                  # doe@domain.com
    ('{first}.{last_initial}@{domain}', 0.05)   # john.d@domain.com
]

# Percentage chance to add a number to an email
EMAIL_NUMBER_CHANCE = 0.25  # 25% chance to add a number

# Phone number formats with relative frequency
PHONE_FORMATS = [
    ('({area_code}) {prefix}-{line}', 0.30),               # (555) 123-4567
    ('{area_code}-{prefix}-{line}', 0.20),                 # 555-123-4567
    ('{area_code}.{prefix}.{line}', 0.15),                 # 555.123.4567
    ('{country_code}-{area_code}-{prefix}-{line}', 0.10),  # 1-555-123-4567
    ('{prefix}-{line}', 0.10),                             # 123-4567
    ('{area_code}{prefix}{line}', 0.10),                   # 5551234567
    ('{country_code} ({area_code}) {prefix}-{line}', 0.05) # 1 (555) 123-4567
]

def detect_id_field_type(column_name: str) -> str:
    """
    Detect what type of ID field a column might be based on its name.
    
    Args:
        column_name: The name of the column to analyze
        
    Returns:
        The detected ID type or 'generic' if no specific type is detected
    """
    column_lower = column_name.lower()
    
    # First check if it's an ID field at all
    is_id_field = False
    for id_type in ID_PATTERNS.values():
        for marker in id_type['id_markers']:
            if marker in column_lower:
                is_id_field = True
                break
        if is_id_field:
            break
    
    if not is_id_field:
        return None
    
    # Then try to determine which specific type of ID
    for id_type, config in ID_PATTERNS.items():
        if id_type == 'generic':
            continue  # Skip generic, it's our fallback
            
        for term in config['detection']:
            if term in column_lower:
                return id_type
    
    return 'generic'  # Fallback to generic ID type

def generate_id(id_type: str, index: int) -> str:
    """
    Generate an ID of the specified type.
    
    Args:
        id_type: The type of ID to generate (must be in ID_PATTERNS)
        index: The index of the ID in the sequence
        
    Returns:
        A formatted ID string
    """
    if id_type not in ID_PATTERNS:
        id_type = 'generic'
        
    pattern = ID_PATTERNS[id_type]
    id_num = pattern['start'] + index
    
    return pattern['format'].format(prefix=pattern['prefix'], id=id_num)

def get_weighted_email_domain() -> str:
    """
    Select an email domain based on weighted probabilities.
    
    Returns:
        A domain string like 'gmail.com'
    """
    domains = list(EMAIL_DOMAINS.keys())
    weights = list(EMAIL_DOMAINS.values())
    
    return random.choices(domains, weights=weights, k=1)[0]

def get_weighted_email_format() -> str:
    """
    Select an email format based on weighted probabilities.
    
    Returns:
        An email format pattern
    """
    formats = [f[0] for f in EMAIL_FORMATS]
    weights = [f[1] for f in EMAIL_FORMATS]
    
    return random.choices(formats, weights=weights, k=1)[0]

def get_weighted_phone_format() -> str:
    """
    Select a phone number format based on weighted probabilities.
    
    Returns:
        A phone format pattern
    """
    formats = [f[0] for f in PHONE_FORMATS]
    weights = [f[1] for f in PHONE_FORMATS]
    
    return random.choices(formats, weights=weights, k=1)[0]

def format_email(first_name: str, last_name: str) -> str:
    """
    Format an email address based on first and last name.
    
    Args:
        first_name: First name
        last_name: Last name
        
    Returns:
        A formatted email address
    """
    # Clean and normalize names
    first = first_name.lower().replace(' ', '').replace('-', '').replace('.', '')
    last = last_name.lower().replace(' ', '').replace('-', '').replace('.', '')
    first_initial = first[0] if first else 'x'
    last_initial = last[0] if last else 'x'
    
    # Get format and domain
    format_pattern = get_weighted_email_format()
    domain = get_weighted_email_domain()
    
    # Format the email
    email = format_pattern.format(
        first=first,
        last=last,
        first_initial=first_initial,
        last_initial=last_initial,
        domain=domain
    )
    
    # Maybe add a number for uniqueness
    if random.random() < EMAIL_NUMBER_CHANCE:
        # Insert number before the @ symbol
        parts = email.split('@')
        parts[0] = f"{parts[0]}{random.randint(1, 999)}"
        email = '@'.join(parts)
    
    return email

def format_phone_number() -> str:
    """
    Generate a formatted phone number.
    
    Returns:
        A formatted phone number string
    """
    # Generate components
    country_code = '1'  # US code
    area_code = f"{random.randint(200, 999)}"
    prefix = f"{random.randint(200, 999)}"
    line = f"{random.randint(1000, 9999)}"
    
    # Get format
    format_pattern = get_weighted_phone_format()
    
    # Format the phone number
    phone = format_pattern.format(
        country_code=country_code,
        area_code=area_code,
        prefix=prefix,
        line=line
    )
    
    return phone
