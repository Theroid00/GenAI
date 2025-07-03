"""
Flexible data provider using model-based generation for any region or domain.
This module uses machine learning models to generate realistic synthetic data
for any requested region or domain without requiring individual provider files.
"""

import logging
import json
import os
import random
from typing import Dict, Any, List, Optional, Union, Callable
import pkg_resources
from faker import Faker
import numpy as np

# Set up logger
logger = logging.getLogger('synthetic_data_generator')

class FlexibleDataProvider:
    """
    A flexible data provider that can generate data for any region or domain
    using machine learning models or embedded data.
    """
    
    def __init__(self):
        """Initialize the provider with data for various regions and domains."""
        self.faker_instances = {}
        self.data_cache = {}
        self.load_embedded_data()
    
    def load_embedded_data(self):
        """
        Load embedded data for various regions and domains.
        This data would be used when specific ML models aren't available.
        """
        # Load region data
        self._load_data_files('regions')
        
        # Load domain data
        self._load_data_files('domains')
    
    def _load_data_files(self, category):
        """
        Load data files for a specific category (regions or domains).
        
        Args:
            category: The category to load ('regions' or 'domains')
        """
        try:
            # Path to data files
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', category)
            
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Load all JSON files in the directory
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_dir, filename)
                    provider_name = filename[:-5]  # Remove .json extension
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.data_cache[f"{category}.{provider_name}"] = json.load(f)
                            logger.info(f"Loaded {category} data for {provider_name}")
                    except Exception as e:
                        logger.error(f"Error loading {category} data for {provider_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading {category} data files: {str(e)}")
    
    def get_faker(self, locale: str = 'en_US') -> Faker:
        """
        Get a Faker instance for a specific locale.
        
        Args:
            locale: The locale to use for the Faker instance
            
        Returns:
            A Faker instance for the specified locale
        """
        if locale not in self.faker_instances:
            self.faker_instances[locale] = Faker(locale)
        return self.faker_instances[locale]
    
    def get_locale_for_region(self, region: str) -> str:
        """
        Get the appropriate locale for a region.
        
        Args:
            region: The region code
            
        Returns:
            The locale code for the region
        """
        # Map of regions to locales
        locale_mapping = {
            'us': 'en_US',
            'usa': 'en_US',
            'uk': 'en_GB',
            'india': 'en_IN',
            'australia': 'en_AU',
            'canada': 'en_CA',
            'germany': 'de_DE',
            'france': 'fr_FR',
            'italy': 'it_IT',
            'spain': 'es_ES',
            'mexico': 'es_MX',
            'brazil': 'pt_BR',
            'portugal': 'pt_PT',
            'netherlands': 'nl_NL',
            'belgium': 'nl_BE',
            'sweden': 'sv_SE',
            'norway': 'no_NO',
            'denmark': 'dk_DK',
            'finland': 'fi_FI',
            'russia': 'ru_RU',
            'japan': 'ja_JP',
            'china': 'zh_CN',
            'taiwan': 'zh_TW',
            'korea': 'ko_KR',
            'thailand': 'th_TH',
            'vietnam': 'vi_VN',
            'indonesia': 'id_ID',
            'malaysia': 'ms_MY',
            'singapore': 'en_SG',
            'philippines': 'fil_PH',
            'poland': 'pl_PL',
            'turkey': 'tr_TR',
            'greece': 'el_GR',
            'israel': 'he_IL',
            'saudi_arabia': 'ar_SA',
            'egypt': 'ar_EG',
            'uae': 'ar_AE',
            'south_africa': 'en_ZA'
        }
        
        return locale_mapping.get(region.lower(), 'en_US')
    
    def get_data(self, provider_type: str, provider_name: str, data_type: str, **kwargs) -> Any:
        """
        Get data for a specific provider and data type.
        
        Args:
            provider_type: The type of provider ('regions' or 'domains')
            provider_name: The name of the provider (e.g., 'india', 'healthcare')
            data_type: The type of data to generate (e.g., 'name', 'city')
            **kwargs: Additional arguments for data generation
            
        Returns:
            Generated data
        """
        cache_key = f"{provider_type}.{provider_name}"
        
        # Check if we have embedded data for this provider
        if cache_key in self.data_cache and data_type in self.data_cache[cache_key]:
            # Use embedded data
            data_values = self.data_cache[cache_key][data_type]
            return random.choice(data_values)
        
        # If no embedded data, fall back to Faker
        if provider_type == 'regions':
            locale = self.get_locale_for_region(provider_name)
            faker = self.get_faker(locale)
            return self._generate_with_faker(faker, data_type, **kwargs)
        elif provider_type == 'domains':
            # For domains, use domain-specific generation
            return self._generate_domain_data(provider_name, data_type, **kwargs)
        
        # Default fallback
        return f"No data available for {provider_type}.{provider_name}.{data_type}"
    
    def _generate_with_faker(self, faker: Faker, data_type: str, **kwargs) -> Any:
        """
        Generate data using Faker.
        
        Args:
            faker: Faker instance
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated data
        """
        # Get the locale from the faker instance
        locale = faker.locales[0] if hasattr(faker, 'locales') and faker.locales else 'en_US'
        
        # Map common data types to Faker methods
        faker_mapping = {
            'name': faker.name,
            'first_name': faker.first_name,
            'last_name': faker.last_name,
            'address': lambda: faker.address().replace('\n', ', '),
            'street_address': faker.street_address,
            'city': faker.city,
            'country': faker.country,
            'postal_code': faker.postcode,
            'phone': faker.phone_number,
            'phone_number': faker.phone_number,
            'email': faker.email,
            'company': faker.company,
            'job': faker.job,
            'job_title': faker.job,
            'text': faker.text,
            'paragraph': faker.paragraph,
            'sentence': faker.sentence,
            'word': faker.word,
            'date': faker.date,
            'time': faker.time,
            'datetime': faker.date_time,
            'url': faker.url,
            'domain': faker.domain_name,
            'username': faker.user_name,
            'password': faker.password
        }
        
        # Add state only for locales that support it
        if locale in ['en_US', 'en_CA', 'en_IN', 'en_AU', 'de_DE']:
            faker_mapping['state'] = faker.state
        
        if data_type in faker_mapping:
            try:
                return faker_mapping[data_type]()
            except Exception as e:
                logger.error(f"Faker generation for {data_type} failed: {str(e)}")
                return f"Error generating {data_type}"
        else:
            return f"Unsupported data type: {data_type}"
    
    def _generate_domain_data(self, domain: str, data_type: str, **kwargs) -> Any:
        """
        Generate domain-specific data.
        
        Args:
            domain: Domain name (e.g., 'healthcare', 'finance')
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated data
        """
        # Simple generation for common domains
        if domain == 'healthcare':
            return self._generate_healthcare_data(data_type, **kwargs)
        elif domain == 'finance':
            return self._generate_finance_data(data_type, **kwargs)
        else:
            return f"Unsupported domain: {domain}"
    
    def _generate_healthcare_data(self, data_type: str, **kwargs) -> Any:
        """
        Generate healthcare-specific data.
        
        Args:
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated healthcare data
        """
        # Load healthcare data if not already loaded
        cache_key = "domains.healthcare"
        if cache_key not in self.data_cache:
            # Some example healthcare data
            self.data_cache[cache_key] = {
                'medical_condition': [
                    "Hypertension", "Type 2 Diabetes", "Asthma", "COPD", "Heart Disease",
                    "Arthritis", "Depression", "Anxiety", "Migraine", "Hypothyroidism"
                ],
                'medication': [
                    "Lisinopril", "Metformin", "Albuterol", "Prednisone", "Atorvastatin",
                    "Levothyroxine", "Amlodipine", "Metoprolol", "Omeprazole", "Sertraline"
                ],
                'procedure': [
                    "Appendectomy", "Colonoscopy", "CT Scan", "MRI", "X-Ray",
                    "Blood Test", "EKG", "Physical Therapy", "Vaccination", "Surgery"
                ],
                'speciality': [
                    "Cardiology", "Dermatology", "Neurology", "Orthopedics", "Pediatrics",
                    "Psychiatry", "Oncology", "Gastroenterology", "Endocrinology", "Rheumatology"
                ]
            }
        
        # Return healthcare data
        if data_type in self.data_cache[cache_key]:
            return random.choice(self.data_cache[cache_key][data_type])
        else:
            return f"Unsupported healthcare data type: {data_type}"
    
    def _generate_finance_data(self, data_type: str, **kwargs) -> Any:
        """
        Generate finance-specific data.
        
        Args:
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated finance data
        """
        # Load finance data if not already loaded
        cache_key = "domains.finance"
        if cache_key not in self.data_cache:
            # Some example finance data
            self.data_cache[cache_key] = {
                'bank': [
                    "Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One",
                    "HSBC", "TD Bank", "PNC Bank", "US Bank", "Barclays"
                ],
                'payment_method': [
                    "Credit Card", "Debit Card", "Cash", "Check", "Bank Transfer",
                    "PayPal", "Venmo", "Apple Pay", "Google Pay", "Cryptocurrency"
                ],
                'transaction_type': [
                    "Purchase", "Refund", "Deposit", "Withdrawal", "Transfer",
                    "Payment", "Fee", "Interest", "Dividend", "Investment"
                ],
                'currency': [
                    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD", "INR"
                ]
            }
        
        # Return finance data
        if data_type in self.data_cache[cache_key]:
            return random.choice(self.data_cache[cache_key][data_type])
        else:
            return f"Unsupported finance data type: {data_type}"
    
    def generate_region_data(self, region: str, data_type: str, **kwargs) -> Any:
        """
        Generate region-specific data.
        
        Args:
            region: Region code (e.g., 'india', 'usa')
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated region-specific data
        """
        return self.get_data('regions', region, data_type, **kwargs)
    
    def generate_domain_data(self, domain: str, data_type: str, **kwargs) -> Any:
        """
        Generate domain-specific data.
        
        Args:
            domain: Domain name (e.g., 'healthcare', 'finance')
            data_type: Type of data to generate
            **kwargs: Additional arguments
            
        Returns:
            Generated domain-specific data
        """
        return self.get_data('domains', domain, data_type, **kwargs)

# Initialize the flexible data provider
flexible_provider = FlexibleDataProvider()

def generate_data(provider_type: str, provider_name: str, data_type: str, **kwargs) -> Any:
    """
    Generate data using the flexible provider.
    
    Args:
        provider_type: Type of provider ('region' or 'domain')
        provider_name: Name of the provider (e.g., 'india', 'healthcare')
        data_type: Type of data to generate
        **kwargs: Additional arguments
        
    Returns:
        Generated data
    """
    if provider_type.lower() == 'region':
        return flexible_provider.generate_region_data(provider_name, data_type, **kwargs)
    elif provider_type.lower() == 'domain':
        return flexible_provider.generate_domain_data(provider_name, data_type, **kwargs)
    else:
        return f"Unsupported provider type: {provider_type}"
