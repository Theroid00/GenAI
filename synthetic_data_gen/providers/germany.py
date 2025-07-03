"""
Germany-specific data provider for synthetic data generation.
This module contains data and functions to generate realistic German names,
locations, phone numbers, and other region-specific information.
"""

import random
from typing import List, Dict, Any, Optional, Union, Callable
from faker import Faker

# Create a Faker instance with German locale
fake_de = Faker('de_DE')

# ===== German Cities =====

# Major German cities with states
GERMAN_CITIES = [
    # City, State
    ("Berlin", "Berlin"),
    ("Hamburg", "Hamburg"),
    ("Munich", "Bavaria"),
    ("Cologne", "North Rhine-Westphalia"),
    ("Frankfurt", "Hesse"),
    ("Stuttgart", "Baden-Württemberg"),
    ("Düsseldorf", "North Rhine-Westphalia"),
    ("Leipzig", "Saxony"),
    ("Dortmund", "North Rhine-Westphalia"),
    ("Essen", "North Rhine-Westphalia"),
    ("Bremen", "Bremen"),
    ("Dresden", "Saxony"),
    ("Hanover", "Lower Saxony"),
    ("Nuremberg", "Bavaria"),
    ("Duisburg", "North Rhine-Westphalia"),
    ("Bochum", "North Rhine-Westphalia"),
    ("Wuppertal", "North Rhine-Westphalia"),
    ("Bielefeld", "North Rhine-Westphalia"),
    ("Bonn", "North Rhine-Westphalia"),
    ("Münster", "North Rhine-Westphalia"),
    ("Karlsruhe", "Baden-Württemberg"),
    ("Mannheim", "Baden-Württemberg"),
    ("Augsburg", "Bavaria"),
    ("Wiesbaden", "Hesse"),
    ("Mönchengladbach", "North Rhine-Westphalia"),
    ("Gelsenkirchen", "North Rhine-Westphalia"),
    ("Aachen", "North Rhine-Westphalia"),
    ("Braunschweig", "Lower Saxony"),
    ("Kiel", "Schleswig-Holstein"),
    ("Chemnitz", "Saxony"),
    ("Halle", "Saxony-Anhalt"),
    ("Magdeburg", "Saxony-Anhalt"),
    ("Freiburg", "Baden-Württemberg"),
    ("Krefeld", "North Rhine-Westphalia"),
    ("Mainz", "Rhineland-Palatinate"),
    ("Lübeck", "Schleswig-Holstein"),
    ("Erfurt", "Thuringia"),
    ("Rostock", "Mecklenburg-Western Pomerania"),
    ("Kassel", "Hesse"),
    ("Hagen", "North Rhine-Westphalia")
]

# German states with abbreviations
GERMAN_STATES = {
    "Baden-Württemberg": "BW",
    "Bavaria": "BY",
    "Berlin": "BE",
    "Brandenburg": "BB",
    "Bremen": "HB",
    "Hamburg": "HH",
    "Hesse": "HE",
    "Lower Saxony": "NI",
    "Mecklenburg-Western Pomerania": "MV",
    "North Rhine-Westphalia": "NW",
    "Rhineland-Palatinate": "RP",
    "Saarland": "SL",
    "Saxony": "SN",
    "Saxony-Anhalt": "ST",
    "Schleswig-Holstein": "SH",
    "Thuringia": "TH"
}

# ===== German Companies =====

# Major German companies
GERMAN_COMPANIES = [
    "Volkswagen", "Daimler", "BMW", "Siemens", "Allianz", "Deutsche Telekom",
    "BASF", "Bayer", "SAP", "Deutsche Post", "Deutsche Bahn", "Bosch",
    "ThyssenKrupp", "Adidas", "Continental", "Lufthansa", "E.ON", "Metro",
    "RWE", "Fresenius", "Henkel", "Deutsche Bank", "Commerzbank", "Munich Re",
    "Aldi", "Lidl", "Kaufland", "MediaMarkt", "Saturn", "Rewe", "Edeka",
    "Tchibo", "Puma", "Hugo Boss", "Beiersdorf", "Merck", "Infineon", "TUI",
    "Porsche", "Audi", "Mercedes-Benz", "Opel", "DHL", "Postbank", "Zeiss"
]

# ===== German Job Titles =====

# Common German job titles
GERMAN_JOB_TITLES = [
    "Geschäftsführer", "Abteilungsleiter", "Teamleiter", "Projektmanager",
    "Softwareentwickler", "Systemadministrator", "Buchhalter", "Vertriebsleiter",
    "Marketingmanager", "Personalreferent", "Kundenbetreuer", "Berater",
    "Ingenieur", "Techniker", "Mechaniker", "Elektriker", "Arzt", "Lehrer",
    "Krankenpfleger", "Apotheker", "Rechtsanwalt", "Steuerberater", "Architekt",
    "Designer", "Journalist", "Übersetzer", "Forscher", "Wissenschaftler",
    "Kaufmann", "Verkäufer", "Einzelhandelskaufmann", "Bankkaufmann", "Bäcker",
    "Koch", "Friseur", "Gärtner", "Landwirt", "Fahrer", "Pilot", "Polizist",
    "Feuerwehrmann", "Soldat", "Förster", "Bibliothekar", "Sozialarbeiter"
]

# ===== German Street Names =====

# Common German street names
GERMAN_STREET_NAMES = [
    "Hauptstraße", "Schulstraße", "Gartenstraße", "Bahnhofstraße", "Dorfstraße",
    "Bergstraße", "Kirchstraße", "Waldstraße", "Lindenstraße", "Schillerstraße",
    "Goethestraße", "Mozartstraße", "Beethovenstraße", "Rosenstraße", "Mühlenweg",
    "Am Bach", "Feldweg", "Industriestraße", "Marktplatz", "Berliner Straße",
    "Münchener Straße", "Frankfurter Straße", "Kölner Straße", "Hamburger Straße"
]

# ===== German Class Generator =====

class GermanyDataProvider:
    """Provider for Germany-specific data generation."""
    
    @staticmethod
    def generate_name() -> str:
        """Generate a random German name."""
        return fake_de.name()
    
    @staticmethod
    def generate_first_name(gender: Optional[str] = None) -> str:
        """
        Generate a random German first name.
        
        Args:
            gender: Optional gender ('male', 'female', or None for random)
            
        Returns:
            A random German first name
        """
        if gender is None:
            return fake_de.first_name()
        elif gender.lower() == 'male':
            return fake_de.first_name_male()
        elif gender.lower() == 'female':
            return fake_de.first_name_female()
        else:
            return fake_de.first_name()
    
    @staticmethod
    def generate_last_name() -> str:
        """Generate a random German last name."""
        return fake_de.last_name()
    
    @staticmethod
    def generate_city() -> str:
        """Generate a random German city name."""
        return random.choice(GERMAN_CITIES)[0]
    
    @staticmethod
    def generate_city_with_state() -> tuple:
        """Generate a random German city with its state."""
        return random.choice(GERMAN_CITIES)
    
    @staticmethod
    def generate_state() -> str:
        """Generate a random German state name."""
        return random.choice(list(GERMAN_STATES.keys()))
    
    @staticmethod
    def generate_state_abbr() -> str:
        """Generate a random German state abbreviation."""
        return random.choice(list(GERMAN_STATES.values()))
    
    @staticmethod
    def generate_phone_number() -> str:
        """Generate a random German phone number."""
        return fake_de.phone_number()
    
    @staticmethod
    def generate_street_address() -> str:
        """Generate a random German street address."""
        # Generate house number
        house_number = random.randint(1, 150)
        
        # Generate street name
        street_name = random.choice(GERMAN_STREET_NAMES)
        
        # Format the street address
        return f"{street_name} {house_number}"
    
    @staticmethod
    def generate_postal_code() -> str:
        """Generate a random German postal code (PLZ)."""
        return fake_de.postcode()
    
    @staticmethod
    def generate_address() -> str:
        """Generate a random German address."""
        street = GermanyDataProvider.generate_street_address()
        postal_code = GermanyDataProvider.generate_postal_code()
        city = random.choice(GERMAN_CITIES)[0]
        
        return f"{street}\n{postal_code} {city}"
    
    @staticmethod
    def generate_company() -> str:
        """Generate a random German company name."""
        if random.random() < 0.5:  # 50% chance to use predefined companies
            return random.choice(GERMAN_COMPANIES)
        else:
            return fake_de.company()
    
    @staticmethod
    def generate_job_title() -> str:
        """Generate a random German job title."""
        if random.random() < 0.5:  # 50% chance to use predefined job titles
            return random.choice(GERMAN_JOB_TITLES)
        else:
            return fake_de.job()
    
    @staticmethod
    def generate_email(name: Optional[str] = None) -> str:
        """
        Generate a random German email based on name or generate a completely random one.
        
        Args:
            name: Optional name to use as email prefix
            
        Returns:
            A random email address
        """
        domains = ["gmail.com", "gmx.de", "web.de", "t-online.de", 
                  "yahoo.de", "hotmail.de", "outlook.de", "freenet.de"]
        
        if name:
            # Clean the name to make it email-friendly
            # Replace German umlauts
            clean_name = name.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            clean_name = clean_name.replace(" ", ".")
            # Add some randomization if desired
            if random.random() < 0.3:  # 30% chance to add numbers
                clean_name += str(random.randint(1, 9999))
        else:
            # Generate a random name for email
            first = GermanyDataProvider.generate_first_name().lower()
            first = first.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            last = GermanyDataProvider.generate_last_name().lower()
            last = last.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            
            if random.random() < 0.5:  # 50% chance for first.last format
                clean_name = f"{first}.{last}"
            else:  # first+random number
                clean_name = f"{first}{random.randint(1, 9999)}"
        
        domain = random.choice(domains)
        return f"{clean_name}@{domain}"
    
    @staticmethod
    def generate_tax_id() -> str:
        """Generate a random German tax ID (Steueridentifikationsnummer)."""
        # Format: 11 digits
        return ''.join([str(random.randint(0, 9)) for _ in range(11)])
    
    @staticmethod
    def generate_iban() -> str:
        """Generate a random German IBAN."""
        # Format: DE + 2 check digits + 18 digits
        country_code = "DE"
        check_digits = f"{random.randint(0, 99):02d}"
        bank_code = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        account_number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        
        return f"{country_code}{check_digits}{bank_code}{account_number}"
    
    @staticmethod
    def generate_passport_number() -> str:
        """Generate a random German passport number."""
        # Format: C + 8 digits or 9 digits
        if random.random() < 0.5:
            return f"C{random.randint(10000000, 99999999)}"
        else:
            return f"{random.randint(100000000, 999999999)}"

# ===== Helper Functions =====

def get_generator(data_type: str) -> Optional[Callable]:
    """
    Get a generator function for specific German data types.
    
    Args:
        data_type: Type of data to generate
        
    Returns:
        A generator function for the specified data type
    """
    # Map data types to provider methods
    provider = GermanyDataProvider()
    
    generators = {
        'name': provider.generate_name,
        'full_name': provider.generate_name,
        'first_name': provider.generate_first_name,
        'last_name': provider.generate_last_name,
        'city': provider.generate_city,
        'city_state': provider.generate_city_with_state,
        'state': provider.generate_state,
        'state_abbr': provider.generate_state_abbr,
        'address': provider.generate_address,
        'street_address': provider.generate_street_address,
        'postal_code': provider.generate_postal_code,
        'phone': provider.generate_phone_number,
        'phone_number': provider.generate_phone_number,
        'email': provider.generate_email,
        'tax_id': provider.generate_tax_id,
        'iban': provider.generate_iban,
        'passport': provider.generate_passport_number,
        'company': provider.generate_company,
        'job': provider.generate_job_title,
        'job_title': provider.generate_job_title,
    }
    
    return generators.get(data_type.lower())

def get_supported_data_types() -> List[str]:
    """
    Get a list of all supported German data types.
    
    Returns:
        A list of supported data types
    """
    return list(get_generator('').__globals__['generators'].keys())
