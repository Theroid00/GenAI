"""
Region manager for managing and accessing region-specific data providers.
This module provides a central registry for all region and domain-specific data providers.
"""

from typing import Dict, Any, Callable, Optional, List, Union
import importlib
import logging
import os
import sys
from faker import Faker
from .flexible_provider import FlexibleDataProvider

# Configure logging
logger = logging.getLogger('synthetic_data_generator')

class RegionManager:
    """
    Manager class for region-specific and domain-specific data generators.
    This class loads and provides access to data generators for specific regions or domains.
    Uses a flexible provider approach for generating data without requiring separate provider modules.
    """
    
    def __init__(self):
        """Initialize the region manager"""
        self.providers = {}
        self.faker_instances = {}
        self.flexible_provider = FlexibleDataProvider()
        self._load_legacy_providers() # For backward compatibility
    
    def _load_legacy_providers(self):
        """
        Dynamically discover and load legacy provider modules for backward compatibility.
        This method scans the providers directory to find all potential data providers.
        """
        # Get the directory of the current file
        providers_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List all Python files in the directory (excluding __init__.py, this file, and flexible_provider.py)
        provider_files = [f for f in os.listdir(providers_dir) 
                        if f.endswith('.py') and f != '__init__.py' 
                        and f != os.path.basename(__file__)
                        and f != 'flexible_provider.py']
        
        # Load each provider module
        for provider_file in provider_files:
            provider_name = provider_file[:-3]  # Remove .py extension
            try:
                # Dynamically import the provider module
                module_name = f"synthetic_data_gen.providers.{provider_name}"
                module = importlib.import_module(module_name)
                
                # Check if this is a valid provider module
                if self._is_valid_provider(module, provider_name):
                    self.providers[provider_name] = module
                    logger.info(f"Loaded legacy data provider: {provider_name}")
            except ImportError as e:
                logger.warning(f"Could not load legacy provider {provider_name}: {str(e)}")
    
    def _is_valid_provider(self, module, name: str) -> bool:
        """
        Check if a module is a valid data provider.
        
        Args:
            module: The imported module to check
            name: The name of the provider
            
        Returns:
            True if the module is a valid provider, False otherwise
        """
        # Check for common provider attributes or functions
        if hasattr(module, 'get_generator') or hasattr(module, f'get_{name}_data_generator'):
            return True
            
        # Check for specific provider types we know about
        if name == 'india' and hasattr(module, 'get_indian_data_generator'):
            return True
            
        # Check for provider classes
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if attr_name.endswith('Provider') and hasattr(attr, 'generate_name'):
                return True
                
        # Count the number of generator functions
        generator_count = 0
        for attr_name in dir(module):
            if attr_name.startswith(f'{name}_') or attr_name.startswith('generate_'):
                generator_count += 1
                
        return generator_count > 0
    
    def register_provider(self, name: str, provider) -> None:
        """
        Register a new provider.
        
        Args:
            name: The name of the provider
            provider: The provider module or class
        """
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")
    
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
    
    def get_generator(self, provider: str, data_type: str) -> Optional[Callable]:
        """
        Get a data generator for a specific provider and data type.
        This method first checks if a legacy provider is available and falls back to
        the flexible provider if not.
        
        Args:
            provider: Provider code (e.g., 'india', 'usa', 'healthcare')
            data_type: Type of data to generate (e.g., 'name', 'city', 'diagnosis')
            
        Returns:
            A generator function or None if not available
        """
        provider = provider.lower()
        
        # Determine if this is a region or domain provider
        provider_type = 'region'
        common_domains = ['healthcare', 'finance', 'education', 'technology', 'legal']
        if provider in common_domains:
            provider_type = 'domain'
        
        # Create a callable that uses the flexible provider
        def flexible_generator(**kwargs):
            if provider_type == 'region':
                return self.flexible_provider.generate_region_data(provider, data_type, **kwargs)
            else:
                return self.flexible_provider.generate_domain_data(provider, data_type, **kwargs)
        
        # Check if this is a legacy provider
        if provider in self.providers:
            # Get the module for the provider
            module = self.providers[provider]
            
            # Try different naming conventions for the generator
            try:
                # Check for specific provider get_generator functions
                if hasattr(module, 'get_generator'):
                    return module.get_generator(data_type)
                
                # Check for provider-specific generator functions
                elif hasattr(module, f'get_{provider}_data_generator'):
                    return getattr(module, f'get_{provider}_data_generator')(data_type)
                    
                # Check for India-specific function (backward compatibility)
                elif provider == 'india' and hasattr(module, 'get_indian_data_generator'):
                    return module.get_indian_data_generator(data_type)
                
                # Check for direct generator functions
                elif hasattr(module, f'generate_{data_type}'):
                    return getattr(module, f'generate_{data_type}')
                    
                # Check for provider-prefixed functions
                elif hasattr(module, f'{provider}_{data_type}'):
                    return getattr(module, f'{provider}_{data_type}')
                    
                # Check for provider classes
                else:
                    # Look for a provider class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if attr_name.endswith('Provider') and hasattr(attr, f'generate_{data_type}'):
                            # Get the function from the class
                            return getattr(attr, f'generate_{data_type}')
                    
                    logger.info(f"Legacy provider '{provider}' does not support data type '{data_type}', falling back to flexible provider.")
                    return flexible_generator
            except AttributeError as e:
                logger.info(f"Could not find generator for {data_type} in legacy provider {provider}, falling back to flexible provider: {str(e)}")
                return flexible_generator
        else:
            # If no legacy provider, use the flexible provider
            logger.debug(f"Using flexible provider for {provider_type} '{provider}' data type '{data_type}'")
            return flexible_generator
    
    def generate_data(self, provider: str, data_type: str, **kwargs) -> Any:
        """
        Generate data using a specific provider and data type.
        
        Args:
            provider: Provider code (e.g., 'india', 'usa', 'healthcare')
            data_type: Type of data to generate (e.g., 'name', 'city', 'diagnosis')
            **kwargs: Additional arguments to pass to the generator
            
        Returns:
            Generated data, or a default value if the generator is not available
        """
        generator = self.get_generator(provider, data_type)
        
        if generator is not None:
            try:
                return generator(**kwargs)
            except Exception as e:
                logger.error(f"Error generating {data_type} with {provider} provider: {str(e)}")
                
        # If the provider-specific generator fails or doesn't exist, try to use Faker as fallback
        logger.warning(f"Using Faker fallback for {data_type}")
        
        # Determine the appropriate locale based on provider
        locale_mapping = {
            'india': 'en_IN',
            'usa': 'en_US',
            'uk': 'en_GB',
            'france': 'fr_FR',
            'germany': 'de_DE',
            'italy': 'it_IT',
            'spain': 'es_ES',
            'japan': 'ja_JP',
            'china': 'zh_CN',
            'russia': 'ru_RU',
            'brazil': 'pt_BR',
            'mexico': 'es_MX',
            'canada': 'en_CA',
            'australia': 'en_AU'
        }
        
        locale = locale_mapping.get(provider, 'en_US')
        faker = self.get_faker(locale)
        
        # Map common data types to Faker methods
        faker_mapping = {
            'name': faker.name,
            'first_name': faker.first_name,
            'last_name': faker.last_name,
            'address': faker.address,
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
                logger.error(f"Faker fallback for {data_type} failed: {str(e)}")
                
        return f"Unsupported data type: {data_type}"
    
    def is_provider_supported(self, provider: str) -> bool:
        """
        Check if a provider is supported.
        With the flexible provider approach, all providers are supported.
        
        Args:
            provider: The provider name to check
            
        Returns:
            True if the provider is supported
        """
        # With flexible provider, all providers are supported
        return True
    
    def get_supported_providers(self) -> List[str]:
        """
        Get a list of supported providers.
        This includes both legacy providers and common region/domain providers.
        
        Returns:
            A list of supported providers
        """
        # Legacy providers
        legacy_providers = list(self.providers.keys())
        
        # Common regions
        common_regions = [
            'usa', 'uk', 'india', 'france', 'germany', 'italy', 'spain', 
            'japan', 'china', 'russia', 'brazil', 'mexico', 'canada', 
            'australia', 'netherlands', 'sweden', 'norway'
        ]
        
        # Common domains
        common_domains = ['healthcare', 'finance', 'education', 'technology', 'legal']
        
        # Combine all providers
        all_providers = set(legacy_providers + common_regions + common_domains)
        
        return sorted(list(all_providers))
    
    def get_supported_data_types(self, provider: str) -> List[str]:
        """
        Get a list of supported data types for a provider.
        
        Args:
            provider: The provider code
            
        Returns:
            A list of supported data types
        """
        provider = provider.lower()
        
        # Try to get data types from legacy provider
        if provider in self.providers:
            module = self.providers[provider]
            data_types = []
            
            # Try different ways to get the supported data types
            
            # 1. Check for a specific function that returns supported types
            if hasattr(module, 'get_supported_data_types'):
                try:
                    return module.get_supported_data_types()
                except Exception as e:
                    logger.warning(f"Error getting supported data types from {provider}: {str(e)}")
                    # Continue with other methods
                
            # 2. Check for India-specific function (backward compatibility)
            if provider == 'india' and hasattr(module, 'get_indian_data_generator'):
                # Try to access the generators dictionary
                try:
                    function_globals = module.get_indian_data_generator.__globals__
                    if 'generators' in function_globals:
                        return list(function_globals['generators'].keys())
                except Exception as e:
                    logger.warning(f"Error getting generators from {provider}: {str(e)}")
                    # Continue with other methods
            
            # 3. Look for provider classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if attr_name.endswith('Provider'):
                    # Get all generate_* methods
                    for method_name in dir(attr):
                        if method_name.startswith('generate_'):
                            data_types.append(method_name[9:])  # Remove 'generate_' prefix
            
            # 4. Look for generate_* functions
            for attr_name in dir(module):
                if attr_name.startswith('generate_'):
                    data_types.append(attr_name[9:])  # Remove 'generate_' prefix
                    
            # 5. Look for provider_* functions
            for attr_name in dir(module):
                if attr_name.startswith(f'{provider}_'):
                    data_types.append(attr_name[len(provider)+1:])  # Remove 'provider_' prefix
            
            if data_types:
                return list(set(data_types))  # Remove duplicates
        
        # For any provider (including legacy ones with no explicit data types)
        # return common data types based on whether it's a region or domain
        common_domains = ['healthcare', 'finance', 'education', 'technology', 'legal']
        
        if provider in common_domains:
            # Domain-specific data types
            if provider == 'healthcare':
                return ['medical_condition', 'medication', 'procedure', 'speciality', 
                       'diagnosis', 'treatment', 'hospital', 'doctor', 'patient']
            elif provider == 'finance':
                return ['bank', 'payment_method', 'transaction_type', 'currency', 
                       'account', 'credit_card', 'loan', 'investment']
            elif provider == 'education':
                return ['school', 'university', 'course', 'degree', 'grade', 'student',
                       'teacher', 'subject', 'department']
            elif provider == 'technology':
                return ['programming_language', 'framework', 'database', 'os', 'device',
                       'software', 'hardware', 'cloud_service']
            elif provider == 'legal':
                return ['court', 'law_firm', 'case_type', 'document', 'attorney',
                       'judge', 'client', 'statute']
            else:
                # Generic domain data types
                return ['name', 'category', 'type', 'description', 'id']
        else:
            # Region-specific data types (common for all regions)
            return [
                'name', 'first_name', 'last_name', 'address', 'street_address', 
                'city', 'state', 'country', 'postal_code', 'phone', 'email',
                'company', 'job', 'date', 'time', 'datetime', 'url', 'username'
            ]
