"""
India-specific data generator provider.
This module provides data generators for Indian names, addresses, cities, and more.
"""

import random
from typing import Dict, Any, List, Callable, Optional

# Dictionary of generators for Indian data
generators = {}

# Indian first names (mixed gender)
indian_first_names = [
    # Hindu names
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayaan", "Atharva", 
    "Krishna", "Ishaan", "Shaurya", "Advik", "Rudra", "Pranav", "Advaith", "Aarush",
    "Dhruv", "Kabir", "Ritvik", "Aaryan",
    "Aanya", "Aadhya", "Aadya", "Aaradhya", "Ananya", "Pari", "Anika", "Navya", 
    "Diya", "Avni", "Sara", "Aarohi", "Anvi", "Kiara", "Myra", "Ira", "Disha", 
    "Ishita", "Ahana", "Divya",
    
    # Muslim names
    "Ahmed", "Ali", "Faiz", "Farhan", "Hasan", "Ibrahim", "Kabir", "Omar", "Rehan", "Zain",
    "Aisha", "Fatima", "Iqra", "Noor", "Samira", "Sana", "Yasmin", "Zara", "Aaliyah", "Amara",
    
    # Sikh names
    "Angad", "Arjan", "Daljeet", "Gurpreet", "Harpreet", "Jaspal", "Kirpal", "Manpreet", 
    "Navjot", "Sukhbir",
    "Amrit", "Harleen", "Gurleen", "Jaspreet", "Kiranjot", "Mandeep", "Navpreet", "Parminder", 
    "Simran", "Sukhdeep",
    
    # Christian names
    "Aiden", "Alex", "Chris", "Daniel", "Jacob", "John", "Joshua", "Michael", "Ryan", "Thomas",
    "Anna", "Elizabeth", "Emily", "Mary", "Olivia", "Rebecca", "Sarah", "Sofia", "Susan", "Teresa"
]

# Indian last names
indian_last_names = [
    # Hindu surnames
    "Sharma", "Verma", "Agarwal", "Patil", "Patel", "Singh", "Mishra", "Yadav", "Joshi", 
    "Gupta", "Nair", "Reddy", "Iyer", "Rao", "Chauhan", "Mehta", "Jain", "Shah", "Das", "Pandey",
    
    # Muslim surnames
    "Khan", "Ahmed", "Sheikh", "Pathan", "Shaikh", "Syed", "Qureshi", "Ansari", "Mirza", "Baig",
    
    # Sikh surnames
    "Singh", "Kaur", "Grewal", "Gill", "Dhillon", "Sidhu", "Sandhu", "Brar", "Virk", "Randhawa",
    
    # Christian surnames
    "Thomas", "Joseph", "Fernandes", "D'Souza", "D'Silva", "Pereira", "Rodrigues", "Mathew", 
    "Philip", "George"
]

# Indian cities by state
indian_cities = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur", "Korba", "Durg"],
    "Delhi": ["New Delhi", "Delhi", "Noida", "Ghaziabad", "Faridabad"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala", "Rohtak"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala", "Kullu", "Solan"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Hazaribagh"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubli", "Belgaum"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj"],
    "West Bengal": ["Kolkata", "Asansol", "Siliguri", "Durgapur", "Howrah"]
}

# Flattened list of all cities
all_indian_cities = [city for cities in indian_cities.values() for city in cities]

# Indian states
indian_states = list(indian_cities.keys())

# Indian phone number formats
indian_phone_formats = [
    "+91 9### ######",
    "+91-9###-######",
    "9### ######",
    "9###-######",
    "09### ######",
    "9########",
]

# Indian postal code format (6 digits)
def generate_indian_postal_code():
    return str(random.randint(100000, 999999))

# Street names and types
indian_street_names = [
    "Gandhi", "Nehru", "Patel", "Bose", "Tagore", "Ambedkar", "Shastri", "Azad", "Tilak", 
    "Vivekananda", "Ashoka", "Akbar", "Birla", "Sarabhai", "Tata", "Godrej", "Bajaj", 
    "Jinnah", "Banerjee", "Chatterjee"
]

indian_street_types = [
    "Road", "Street", "Marg", "Nagar", "Colony", "Layout", "Gardens", "Park", "Path", 
    "Lane", "Avenue", "Circle", "Chowk", "Market", "Plaza", "Complex", "Enclave", "Vihar"
]

# ===== GENERATOR FUNCTIONS =====

def generate_indian_name():
    """Generate a full Indian name"""
    first_name = random.choice(indian_first_names)
    last_name = random.choice(indian_last_names)
    return f"{first_name} {last_name}"

def generate_indian_first_name():
    """Generate an Indian first name"""
    return random.choice(indian_first_names)

def generate_indian_last_name():
    """Generate an Indian last name"""
    return random.choice(indian_last_names)

def generate_indian_city():
    """Generate an Indian city name"""
    return random.choice(all_indian_cities)

def generate_indian_state():
    """Generate an Indian state name"""
    return random.choice(indian_states)

def generate_indian_phone_number():
    """Generate an Indian phone number"""
    format_choice = random.choice(indian_phone_formats)
    phone = ""
    for char in format_choice:
        if char == "#":
            phone += str(random.randint(0, 9))
        else:
            phone += char
    return phone

def generate_indian_address():
    """Generate a complete Indian address"""
    building_number = random.randint(1, 999)
    street_name = random.choice(indian_street_names)
    street_type = random.choice(indian_street_types)
    state = random.choice(indian_states)
    city = random.choice(indian_cities[state])
    postal_code = generate_indian_postal_code()
    
    address_formats = [
        f"{building_number}, {street_name} {street_type}, {city}, {state} - {postal_code}",
        f"Flat {random.randint(101, 999)}, {building_number}, {street_name} {street_type}, {city}, {state} - {postal_code}",
        f"House No. {building_number}, {street_name} {street_type}, {city}, {state} - {postal_code}",
        f"{building_number}/{random.randint(1, 99)}, {street_name} {street_type}, {city}, {state} - {postal_code}"
    ]
    
    return random.choice(address_formats)

def generate_indian_company():
    """Generate an Indian company name"""
    company_prefixes = [
        "Bharat", "Indian", "Indo", "Hindustani", "National", "Royal", "Prime", "Star", 
        "Supreme", "Global", "Metro", "City", "Urban", "Coastal", "Heritage", "Modern", 
        "Reliance", "Tata", "Birla", "Bajaj", "Mahindra", "Godrej", "Infosys", "Wipro"
    ]
    
    company_suffixes = [
        "Industries", "Enterprises", "Group", "Corporation", "Limited", "Pvt Ltd", "Technologies", 
        "Solutions", "Systems", "Motors", "Automobiles", "Textiles", "Pharmaceuticals", 
        "Chemicals", "Foods", "Retail", "Consultancy", "Services", "Products", "Traders"
    ]
    
    return f"{random.choice(company_prefixes)} {random.choice(company_suffixes)}"

def generate_indian_job_title():
    """Generate an Indian job title"""
    prefixes = [
        "Senior", "Junior", "Assistant", "Deputy", "Chief", "Principal", "Head", "Lead",
        "Executive", "Associate", "Vice President", "Director", "Manager"
    ]
    
    roles = [
        "Software Engineer", "Developer", "Architect", "Designer", "Analyst", 
        "Consultant", "Accountant", "Auditor", "Teacher", "Professor", 
        "Doctor", "Nurse", "Engineer", "Technician", "Supervisor", "Coordinator",
        "Administrator", "Officer", "Executive", "Specialist"
    ]
    
    # Sometimes include a prefix, sometimes just return the role
    if random.random() < 0.7:
        return f"{random.choice(prefixes)} {random.choice(roles)}"
    else:
        return random.choice(roles)

def generate_indian_email(name=None):
    """Generate an Indian email address"""
    if name is None:
        name = generate_indian_name()
    
    # Clean and format the name for an email
    name_parts = name.lower().split()
    
    email_formats = [
        f"{name_parts[0]}{random.randint(1, 9999)}@gmail.com",
        f"{name_parts[0]}.{name_parts[-1]}@gmail.com",
        f"{name_parts[0]}_{name_parts[-1]}@yahoo.com",
        f"{name_parts[0]}{random.randint(1, 99)}@hotmail.com",
        f"{name_parts[0]}{name_parts[-1][0]}@rediffmail.com",
        f"{name_parts[0]}.{name_parts[-1]}@outlook.com"
    ]
    
    return random.choice(email_formats)

def generate_indian_bank():
    """Generate an Indian bank name"""
    banks = [
        "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank",
        "Punjab National Bank", "Bank of Baroda", "Canara Bank", "Union Bank of India", 
        "Bank of India", "Indian Bank", "Central Bank of India", "IndusInd Bank", "Yes Bank",
        "IDBI Bank", "Federal Bank", "RBL Bank", "South Indian Bank", "Karnataka Bank",
        "Bank of Maharashtra"
    ]
    return random.choice(banks)

def generate_indian_aadhaar():
    """Generate an Indian Aadhaar number (12 digits)"""
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_indian_pan():
    """Generate an Indian PAN (Permanent Account Number)"""
    # Format: AAAPL1234C (5 letters, 4 digits, 1 letter)
    first_five = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
    middle_four = ''.join([str(random.randint(0, 9)) for _ in range(4)])
    last_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    return f"{first_five}{middle_four}{last_letter}"

def generate_indian_college():
    """Generate an Indian college/university name"""
    colleges = [
        "Indian Institute of Technology (IIT), Delhi",
        "Indian Institute of Technology (IIT), Bombay",
        "Indian Institute of Technology (IIT), Madras",
        "Indian Institute of Technology (IIT), Kanpur",
        "Indian Institute of Technology (IIT), Kharagpur",
        "Indian Institute of Science (IISc), Bangalore",
        "Delhi University",
        "Jawaharlal Nehru University (JNU)",
        "Banaras Hindu University (BHU)",
        "Aligarh Muslim University (AMU)",
        "University of Mumbai",
        "University of Calcutta",
        "University of Madras",
        "Anna University",
        "BITS Pilani",
        "National Institute of Technology (NIT), Trichy",
        "National Institute of Technology (NIT), Warangal",
        "National Institute of Technology (NIT), Surathkal",
        "Jadavpur University",
        "Jamia Millia Islamia"
    ]
    return random.choice(colleges)

# Register all generators
generators = {
    'name': generate_indian_name,
    'first_name': generate_indian_first_name,
    'last_name': generate_indian_last_name,
    'city': generate_indian_city,
    'state': generate_indian_state,
    'phone_number': generate_indian_phone_number,
    'address': generate_indian_address,
    'company': generate_indian_company,
    'job': generate_indian_job_title,
    'email': generate_indian_email,
    'bank': generate_indian_bank,
    'aadhaar': generate_indian_aadhaar,
    'pan': generate_indian_pan,
    'college': generate_indian_college,
    'postal_code': generate_indian_postal_code
}

def get_indian_data_generator(data_type: str) -> Optional[Callable]:
    """
    Get a generator function for a specific type of Indian data.
    
    Args:
        data_type: The type of data to generate
        
    Returns:
        A generator function or None if not available
    """
    return generators.get(data_type.lower())

def get_supported_data_types() -> List[str]:
    """
    Get a list of supported data types for Indian data.
    
    Returns:
        A list of supported data type names
    """
    return list(generators.keys())