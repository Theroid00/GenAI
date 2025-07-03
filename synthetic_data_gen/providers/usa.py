"""
USA-specific data provider for synthetic data generation.
This module contains data and functions to generate realistic US names,
locations, phone numbers, and other region-specific information.
"""

import random
from typing import List, Dict, Any, Optional, Union
from faker import Faker

# Create a Faker instance with US locale
fake_usa = Faker('en_US')

# ===== US Cities =====

# Major US cities with states
US_CITIES = [
    # City, State
    ("New York", "New York"),
    ("Los Angeles", "California"),
    ("Chicago", "Illinois"),
    ("Houston", "Texas"),
    ("Phoenix", "Arizona"),
    ("Philadelphia", "Pennsylvania"),
    ("San Antonio", "Texas"),
    ("San Diego", "California"),
    ("Dallas", "Texas"),
    ("San Jose", "California"),
    ("Austin", "Texas"),
    ("Jacksonville", "Florida"),
    ("Fort Worth", "Texas"),
    ("Columbus", "Ohio"),
    ("Indianapolis", "Indiana"),
    ("Charlotte", "North Carolina"),
    ("San Francisco", "California"),
    ("Seattle", "Washington"),
    ("Denver", "Colorado"),
    ("Washington", "DC"),
    ("Boston", "Massachusetts"),
    ("El Paso", "Texas"),
    ("Nashville", "Tennessee"),
    ("Oklahoma City", "Oklahoma"),
    ("Las Vegas", "Nevada"),
    ("Detroit", "Michigan"),
    ("Portland", "Oregon"),
    ("Memphis", "Tennessee"),
    ("Louisville", "Kentucky"),
    ("Milwaukee", "Wisconsin"),
    ("Baltimore", "Maryland"),
    ("Albuquerque", "New Mexico"),
    ("Tucson", "Arizona"),
    ("Mesa", "Arizona"),
    ("Fresno", "California"),
    ("Sacramento", "California"),
    ("Atlanta", "Georgia"),
    ("Kansas City", "Missouri"),
    ("Miami", "Florida"),
    ("Tampa", "Florida"),
    ("Orlando", "Florida"),
    ("Pittsburgh", "Pennsylvania"),
    ("Cincinnati", "Ohio"),
    ("Minneapolis", "Minnesota"),
    ("Cleveland", "Ohio"),
    ("New Orleans", "Louisiana"),
    ("St. Louis", "Missouri"),
    ("Honolulu", "Hawaii"),
    ("Anchorage", "Alaska")
]

# US states abbreviations
US_STATES_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC", "American Samoa": "AS", "Guam": "GU", "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR", "United States Minor Outlying Islands": "UM", "U.S. Virgin Islands": "VI"
}

# ===== US Phone Numbers =====

# US area codes by region
US_AREA_CODES = {
    "Northeast": ["201", "203", "207", "212", "215", "267", "301", "302", "401", "413", "484", "508", "516", "518", "585", "609", "610", "617", "631", "646", "716", "717", "718", "732", "781", "856", "862", "908", "914", "917", "973"],
    "Midwest": ["216", "234", "248", "269", "270", "304", "312", "313", "317", "330", "334", "402", "405", "414", "419", "440", "502", "513", "517", "563", "614", "616", "636", "651", "660", "708", "712", "734", "740", "765", "773", "810", "812", "847", "913", "914", "920", "937", "952"],
    "South": ["205", "229", "251", "256", "281", "305", "321", "325", "334", "352", "386", "404", "407", "409", "423", "469", "478", "504", "561", "601", "615", "678", "706", "713", "727", "731", "754", "757", "762", "772", "813", "817", "843", "850", "863", "865", "870", "904", "910", "919", "931", "940", "954", "972", "985"],
    "West": ["208", "253", "281", "303", "310", "323", "360", "385", "408", "415", "425", "435", "442", "458", "480", "503", "509", "510", "512", "530", "541", "559", "602", "619", "623", "626", "650", "661", "702", "707", "714", "719", "720", "747", "760", "775", "801", "805", "808", "818", "831", "858", "909", "916", "925", "928", "949", "951", "971", "972"]
}

# ===== US Educational Information =====

# Common US degrees
US_DEGREES = [
    "B.S.", "B.A.", "B.F.A.", "B.B.A.", "B.Arch.", "M.S.", "M.A.", "M.B.A.", 
    "M.F.A.", "M.Arch.", "J.D.", "M.D.", "Ph.D.", "Ed.D.", "D.B.A.", 
    "Associate's Degree", "High School Diploma", "GED"
]

# Top US educational institutions
US_INSTITUTIONS = [
    "Harvard University", "Stanford University", "MIT", "California Institute of Technology",
    "Yale University", "University of Chicago", "Johns Hopkins University", "Columbia University",
    "Princeton University", "Cornell University", "University of Pennsylvania", "Duke University",
    "University of California, Berkeley", "University of California, Los Angeles", "University of Michigan",
    "New York University", "University of Washington", "University of Texas at Austin",
    "Georgia Institute of Technology", "University of Illinois at Urbana-Champaign",
    "Ohio State University", "University of Wisconsin-Madison", "University of Minnesota",
    "Boston University", "University of North Carolina at Chapel Hill", "University of Virginia",
    "University of Florida", "Pennsylvania State University", "University of California, San Diego",
    "University of Southern California", "University of California, Davis", "Purdue University",
    "Arizona State University", "University of Arizona", "University of Colorado Boulder",
    "Michigan State University", "University of Alabama", "Florida State University",
    "Texas A&M University", "University of Georgia", "University of Iowa", "University of Maryland",
    "University of Massachusetts Amherst", "Virginia Tech", "Washington University in St. Louis",
    "University of Pittsburgh", "Northwestern University", "Rice University"
]

# Common US fields of study
US_STUDY_FIELDS = [
    "Computer Science", "Information Technology", "Mechanical Engineering",
    "Electrical Engineering", "Civil Engineering", "Chemical Engineering",
    "Business Administration", "Finance", "Marketing", "Accounting",
    "Economics", "Psychology", "Sociology", "Anthropology",
    "Political Science", "International Relations", "History", "English",
    "Communications", "Journalism", "Media Studies", "Education",
    "Nursing", "Medicine", "Biology", "Chemistry", "Physics", "Mathematics",
    "Environmental Science", "Graphic Design", "Fine Arts", "Music",
    "Theater", "Film Studies", "Architecture", "Law", "Criminal Justice",
    "Public Health", "Social Work", "Philosophy", "Religious Studies",
    "Linguistics", "Gender Studies", "African American Studies", "Latino Studies",
    "Asian Studies", "Middle Eastern Studies", "European Studies"
]

# ===== US Companies and Job Titles =====

# Major US companies
US_COMPANIES = [
    "Apple", "Microsoft", "Amazon", "Google", "Facebook", "Walmart", "ExxonMobil",
    "Chevron", "AT&T", "Verizon", "UnitedHealth Group", "CVS Health", "Berkshire Hathaway",
    "McKesson", "AmerisourceBergen", "Costco", "Cigna", "Cardinal Health", "JPMorgan Chase",
    "General Motors", "Ford Motor", "Kroger", "Centene", "Walgreens Boots Alliance",
    "Home Depot", "Bank of America", "Wells Fargo", "Citigroup", "Comcast", "Target",
    "Johnson & Johnson", "Procter & Gamble", "IBM", "State Farm Insurance",
    "Intel", "Coca-Cola", "PepsiCo", "Boeing", "Pfizer", "FedEx", "Netflix",
    "Nike", "Starbucks", "Goldman Sachs", "Morgan Stanley", "Disney", "Adobe",
    "Salesforce", "Tesla", "Uber", "Lyft", "Airbnb", "Twitter", "LinkedIn"
]

# Common US job titles
US_JOB_TITLES = [
    "Software Engineer", "Systems Analyst", "Data Scientist", "Business Analyst",
    "Project Manager", "Product Manager", "Marketing Manager", "Sales Manager",
    "Financial Analyst", "Accountant", "Administrative Assistant", "Customer Service Representative",
    "Operations Manager", "Human Resources Manager", "Research Scientist", "Teacher",
    "Nurse", "Doctor", "Lawyer", "Engineer", "Architect", "Designer", "Writer",
    "Editor", "Journalist", "Social Media Manager", "Digital Marketing Specialist",
    "Web Developer", "Mobile App Developer", "Database Administrator", "Network Engineer",
    "Systems Administrator", "Security Analyst", "DevOps Engineer", "QA Engineer",
    "UX Designer", "UI Designer", "Graphic Designer", "Art Director", "Creative Director",
    "Chief Executive Officer", "Chief Financial Officer", "Chief Technology Officer",
    "Vice President", "Director", "Senior Manager", "Team Lead", "Supervisor",
    "Consultant", "Contractor", "Freelancer", "Intern"
]

# ===== US Address Elements =====

# Common US street suffixes
US_STREET_SUFFIXES = [
    "Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Lane", "Road",
    "Circle", "Trail", "Parkway", "Highway", "Way", "Terrace", "Run", "Plaza"
]

# Common US street name themes
US_STREET_THEMES = [
    # Nature
    ["Oak", "Pine", "Maple", "Cedar", "Birch", "Elm", "Willow", "Aspen", "Spruce", "Sycamore"],
    # Presidents
    ["Washington", "Lincoln", "Jefferson", "Adams", "Madison", "Monroe", "Jackson", "Roosevelt", "Kennedy", "Wilson"],
    # States
    ["California", "Texas", "Florida", "New York", "Pennsylvania", "Ohio", "Michigan", "Georgia", "Virginia", "Illinois"],
    # Landmarks
    ["Park", "River", "Lake", "Mountain", "Valley", "Meadow", "Forest", "Hill", "Ridge", "Canyon"],
    # Cardinal directions
    ["North", "South", "East", "West", "Northeast", "Northwest", "Southeast", "Southwest"],
    # Numbers (as strings)
    ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
]

# Common US residential building types
US_RESIDENTIAL_TYPES = [
    "House", "Apartment", "Condo", "Townhouse", "Duplex", "Studio", "Loft", "Penthouse"
]

# ===== US Class Generator =====

class USADataProvider:
    """Provider for USA-specific data generation."""
    
    @staticmethod
    def generate_name() -> str:
        """Generate a random US name."""
        return fake_usa.name()
    
    @staticmethod
    def generate_first_name(gender: Optional[str] = None) -> str:
        """
        Generate a random US first name.
        
        Args:
            gender: Optional gender ('male', 'female', or None for random)
            
        Returns:
            A random US first name
        """
        if gender is None:
            return fake_usa.first_name()
        elif gender.lower() == 'male':
            return fake_usa.first_name_male()
        elif gender.lower() == 'female':
            return fake_usa.first_name_female()
        else:
            return fake_usa.first_name()
    
    @staticmethod
    def generate_last_name() -> str:
        """Generate a random US last name."""
        return fake_usa.last_name()
    
    @staticmethod
    def generate_city() -> str:
        """Generate a random US city name."""
        return random.choice(US_CITIES)[0]
    
    @staticmethod
    def generate_city_with_state() -> tuple:
        """Generate a random US city with its state."""
        return random.choice(US_CITIES)
    
    @staticmethod
    def generate_state() -> str:
        """Generate a random US state name."""
        return random.choice(list(US_STATES_ABBR.keys()))
    
    @staticmethod
    def generate_state_abbr() -> str:
        """Generate a random US state abbreviation."""
        return random.choice(list(US_STATES_ABBR.values()))
    
    @staticmethod
    def generate_phone_number(region: Optional[str] = None) -> str:
        """
        Generate a random US phone number.
        
        Args:
            region: Optional US region ('Northeast', 'Midwest', 'South', 'West', or None for random)
            
        Returns:
            A formatted US phone number
        """
        if region is None or region not in US_AREA_CODES:
            # Pick a random area code from all regions
            all_area_codes = []
            for codes in US_AREA_CODES.values():
                all_area_codes.extend(codes)
            area_code = random.choice(all_area_codes)
        else:
            # Pick from the specified region
            area_code = random.choice(US_AREA_CODES[region])
        
        # Generate the remaining 7 digits
        prefix = random.randint(200, 999)
        line = random.randint(1000, 9999)
        
        # Format the phone number
        return f"({area_code}) {prefix}-{line}"
    
    @staticmethod
    def generate_street_address() -> str:
        """Generate a random US street address."""
        # Generate house number
        house_number = random.randint(1, 9999)
        
        # Generate street name
        theme = random.choice(US_STREET_THEMES)
        street_name = random.choice(theme)
        
        # Generate street suffix
        street_suffix = random.choice(US_STREET_SUFFIXES)
        
        # Format the street address
        return f"{house_number} {street_name} {street_suffix}"
    
    @staticmethod
    def generate_zip_code() -> str:
        """Generate a random US ZIP code."""
        return f"{random.randint(10000, 99999)}"
    
    @staticmethod
    def generate_zip_code_plus_4() -> str:
        """Generate a random US ZIP+4 code."""
        return f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
    
    @staticmethod
    def generate_address() -> str:
        """Generate a random US address."""
        street = USADataProvider.generate_street_address()
        city, state = random.choice(US_CITIES)
        zip_code = USADataProvider.generate_zip_code()
        
        return f"{street}\n{city}, {US_STATES_ABBR[state]} {zip_code}"
    
    @staticmethod
    def generate_apartment_address() -> str:
        """Generate a random US apartment address."""
        street = USADataProvider.generate_street_address()
        
        # Generate apartment number
        apt_number = random.randint(1, 999)
        apt_prefix = random.choice(["Apt", "Unit", "Suite", "#"])
        
        city, state = random.choice(US_CITIES)
        zip_code = USADataProvider.generate_zip_code()
        
        return f"{street}, {apt_prefix} {apt_number}\n{city}, {US_STATES_ABBR[state]} {zip_code}"
    
    @staticmethod
    def generate_education() -> Dict[str, str]:
        """Generate random US education details."""
        degree = random.choice(US_DEGREES)
        institution = random.choice(US_INSTITUTIONS)
        field = random.choice(US_STUDY_FIELDS)
        
        return {
            'degree': degree,
            'institution': institution,
            'field': field
        }
    
    @staticmethod
    def generate_company() -> str:
        """Generate a random US company name."""
        return random.choice(US_COMPANIES)
    
    @staticmethod
    def generate_job_title() -> str:
        """Generate a random US job title."""
        return random.choice(US_JOB_TITLES)
    
    @staticmethod
    def generate_email(name: Optional[str] = None) -> str:
        """
        Generate a random US email based on name or generate a completely random one.
        
        Args:
            name: Optional name to use as email prefix
            
        Returns:
            A random email address
        """
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
                  "aol.com", "icloud.com", "protonmail.com", "mail.com"]
        
        if name:
            # Clean the name to make it email-friendly
            clean_name = name.lower().replace(" ", ".")
            # Add some randomization if desired
            if random.random() < 0.3:  # 30% chance to add numbers
                clean_name += str(random.randint(1, 9999))
        else:
            # Generate a random name for email
            first = USADataProvider.generate_first_name().lower()
            last = USADataProvider.generate_last_name().lower()
            
            if random.random() < 0.5:  # 50% chance for first.last format
                clean_name = f"{first}.{last}"
            else:  # first+random number
                clean_name = f"{first}{random.randint(1, 9999)}"
        
        domain = random.choice(domains)
        return f"{clean_name}@{domain}"
    
    @staticmethod
    def generate_ssn() -> str:
        """Generate a random US Social Security Number (SSN)."""
        # Note: This generates a random SSN format but doesn't follow
        # actual SSN allocation rules (for privacy/security reasons)
        area = random.randint(100, 899)
        group = random.randint(10, 99)
        serial = random.randint(1000, 9999)
        
        return f"{area}-{group}-{serial}"
    
    @staticmethod
    def generate_credit_card() -> Dict[str, str]:
        """Generate a random US credit card information."""
        # Note: This generates fake credit card formats that won't pass validation
        card_types = ["Visa", "MasterCard", "American Express", "Discover"]
        card_type = random.choice(card_types)
        
        # Generate number pattern based on card type
        if card_type == "American Express":
            # AmEx starts with 34 or 37 and has 15 digits
            prefix = random.choice(["34", "37"])
            digits = ''.join(random.choices('0123456789', k=13))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:10]} {number[10:]}"
        elif card_type == "Visa":
            # Visa starts with 4 and has 16 digits
            prefix = "4"
            digits = ''.join(random.choices('0123456789', k=15))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        elif card_type == "MasterCard":
            # MasterCard starts with 51-55 and has 16 digits
            prefix = f"5{random.randint(1, 5)}"
            digits = ''.join(random.choices('0123456789', k=14))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        else:  # Discover
            # Discover starts with 6011 and has 16 digits
            prefix = "6011"
            digits = ''.join(random.choices('0123456789', k=12))
            number = f"{prefix}{digits}"
            formatted = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        
        # Generate expiration date
        exp_month = random.randint(1, 12)
        exp_year = random.randint(2023, 2030)
        
        # Generate CVV
        cvv = ''.join(random.choices('0123456789', k=3 if card_type != "American Express" else 4))
        
        return {
            'type': card_type,
            'number': number,
            'formatted': formatted,
            'expiration': f"{exp_month:02d}/{exp_year}",
            'cvv': cvv
        }
    
    @staticmethod
    def generate_drivers_license() -> str:
        """Generate a random US driver's license number."""
        # Different states have different formats
        # This is a simplified generic format
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        numbers = ''.join(random.choices('0123456789', k=6))
        
        return f"{letters}-{numbers}"

# ===== Helper Functions =====

def get_generator(data_type: str):
    """
    Get a generator function for specific US data types.
    
    Args:
        data_type: Type of data to generate
        
    Returns:
        A generator function for the specified data type
    """
    # Map data types to provider methods
    provider = USADataProvider()
    
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
        'apartment_address': provider.generate_apartment_address,
        'zip_code': provider.generate_zip_code,
        'zip_plus4': provider.generate_zip_code_plus_4,
        'phone': provider.generate_phone_number,
        'phone_number': provider.generate_phone_number,
        'email': provider.generate_email,
        'ssn': provider.generate_ssn,
        'credit_card': provider.generate_credit_card,
        'drivers_license': provider.generate_drivers_license,
        'company': provider.generate_company,
        'job': provider.generate_job_title,
        'job_title': provider.generate_job_title,
        'education': provider.generate_education
    }
    
    return generators.get(data_type, lambda: f"Unsupported US data type: {data_type}")

def get_supported_data_types() -> List[str]:
    """
    Get a list of all supported US data types.
    
    Returns:
        A list of supported data types
    """
    return list(get_generator('').__globals__['generators'].keys())
