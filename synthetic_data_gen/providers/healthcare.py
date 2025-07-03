"""
Healthcare data provider for synthetic data generation.
This module contains data and functions to generate realistic healthcare data
such as medical conditions, medications, procedures, and other healthcare-specific information.
"""

import random
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import re

# ===== Medical Conditions and Diagnoses =====

# Common medical conditions and their ICD-10 codes
MEDICAL_CONDITIONS = [
    # Format: (Condition name, ICD-10 code)
    ("Hypertension", "I10"),
    ("Type 2 Diabetes", "E11.9"),
    ("Asthma", "J45.909"),
    ("Chronic Obstructive Pulmonary Disease", "J44.9"),
    ("Congestive Heart Failure", "I50.9"),
    ("Coronary Artery Disease", "I25.10"),
    ("Atrial Fibrillation", "I48.91"),
    ("Osteoarthritis", "M19.90"),
    ("Rheumatoid Arthritis", "M06.9"),
    ("Chronic Kidney Disease", "N18.9"),
    ("Gastroesophageal Reflux Disease", "K21.9"),
    ("Obesity", "E66.9"),
    ("Hyperlipidemia", "E78.5"),
    ("Depression", "F32.9"),
    ("Anxiety Disorder", "F41.9"),
    ("Hypothyroidism", "E03.9"),
    ("Migraine", "G43.909"),
    ("Epilepsy", "G40.909"),
    ("Osteoporosis", "M81.0"),
    ("Anemia", "D64.9"),
    ("Psoriasis", "L40.0"),
    ("Urinary Tract Infection", "N39.0"),
    ("Pneumonia", "J18.9"),
    ("Influenza", "J11.1"),
    ("Acute Bronchitis", "J20.9"),
    ("Acute Sinusitis", "J01.90"),
    ("Allergic Rhinitis", "J30.9"),
    ("Gastroenteritis", "A09"),
    ("Dermatitis", "L30.9"),
    ("Insomnia", "G47.00")
]

# Common chronic conditions
CHRONIC_CONDITIONS = [
    ("Hypertension", "I10"),
    ("Type 2 Diabetes", "E11.9"),
    ("Asthma", "J45.909"),
    ("Chronic Obstructive Pulmonary Disease", "J44.9"),
    ("Congestive Heart Failure", "I50.9"),
    ("Coronary Artery Disease", "I25.10"),
    ("Atrial Fibrillation", "I48.91"),
    ("Osteoarthritis", "M19.90"),
    ("Rheumatoid Arthritis", "M06.9"),
    ("Chronic Kidney Disease", "N18.9"),
    ("Gastroesophageal Reflux Disease", "K21.9"),
    ("Obesity", "E66.9"),
    ("Hyperlipidemia", "E78.5"),
    ("Depression", "F32.9"),
    ("Anxiety Disorder", "F41.9"),
    ("Hypothyroidism", "E03.9"),
    ("Epilepsy", "G40.909"),
    ("Osteoporosis", "M81.0"),
    ("Chronic Anemia", "D64.9"),
    ("Psoriasis", "L40.0")
]

# Common acute conditions
ACUTE_CONDITIONS = [
    ("Urinary Tract Infection", "N39.0"),
    ("Pneumonia", "J18.9"),
    ("Influenza", "J11.1"),
    ("Acute Bronchitis", "J20.9"),
    ("Acute Sinusitis", "J01.90"),
    ("Allergic Rhinitis", "J30.9"),
    ("Gastroenteritis", "A09"),
    ("Dermatitis", "L30.9"),
    ("Insomnia", "G47.00"),
    ("Migraine", "G43.909"),
    ("Acute Upper Respiratory Infection", "J06.9"),
    ("Viral Infection", "B34.9"),
    ("Acute Pharyngitis", "J02.9"),
    ("Acute Tonsillitis", "J03.90"),
    ("Sprain of Ankle", "S93.401"),
    ("Low Back Pain", "M54.5"),
    ("Tension Headache", "G44.209"),
    ("Acute Otitis Media", "H66.90"),
    ("Conjunctivitis", "H10.9"),
    ("Cellulitis", "L03.90")
]

# ===== Medications =====

# Common medications with their categories
MEDICATIONS = [
    # Format: (Medication name, Category, Typical dosage)
    ("Lisinopril", "ACE Inhibitor", "10-40 mg daily"),
    ("Amlodipine", "Calcium Channel Blocker", "5-10 mg daily"),
    ("Metoprolol", "Beta Blocker", "25-100 mg twice daily"),
    ("Atorvastatin", "Statin", "10-80 mg daily"),
    ("Simvastatin", "Statin", "10-40 mg daily"),
    ("Metformin", "Antidiabetic", "500-1000 mg twice daily"),
    ("Levothyroxine", "Thyroid Hormone", "25-200 mcg daily"),
    ("Albuterol", "Bronchodilator", "2 puffs every 4-6 hours as needed"),
    ("Fluticasone", "Corticosteroid", "1-2 sprays in each nostril daily"),
    ("Omeprazole", "Proton Pump Inhibitor", "20-40 mg daily"),
    ("Losartan", "Angiotensin II Receptor Blocker", "25-100 mg daily"),
    ("Hydrochlorothiazide", "Diuretic", "12.5-50 mg daily"),
    ("Furosemide", "Loop Diuretic", "20-80 mg daily"),
    ("Gabapentin", "Anticonvulsant", "300-1200 mg three times daily"),
    ("Sertraline", "SSRI Antidepressant", "50-200 mg daily"),
    ("Escitalopram", "SSRI Antidepressant", "10-20 mg daily"),
    ("Alprazolam", "Benzodiazepine", "0.25-0.5 mg three times daily"),
    ("Ibuprofen", "NSAID", "400-800 mg three times daily"),
    ("Acetaminophen", "Analgesic", "325-650 mg every 4-6 hours"),
    ("Aspirin", "Antiplatelet", "81-325 mg daily"),
    ("Amoxicillin", "Antibiotic", "500 mg three times daily"),
    ("Azithromycin", "Antibiotic", "500 mg on day 1, then 250 mg daily for 4 days"),
    ("Prednisone", "Corticosteroid", "5-60 mg daily"),
    ("Montelukast", "Leukotriene Modifier", "10 mg daily"),
    ("Warfarin", "Anticoagulant", "2-10 mg daily"),
    ("Glipizide", "Antidiabetic", "5-20 mg daily"),
    ("Carvedilol", "Alpha/Beta Blocker", "3.125-25 mg twice daily"),
    ("Pantoprazole", "Proton Pump Inhibitor", "40 mg daily"),
    ("Tramadol", "Opioid Analgesic", "50-100 mg every 4-6 hours"),
    ("Clopidogrel", "Antiplatelet", "75 mg daily")
]

# Map conditions to commonly prescribed medications
CONDITION_MEDICATION_MAP = {
    "Hypertension": ["Lisinopril", "Amlodipine", "Metoprolol", "Losartan", "Hydrochlorothiazide", "Carvedilol"],
    "Type 2 Diabetes": ["Metformin", "Glipizide"],
    "Asthma": ["Albuterol", "Fluticasone", "Montelukast"],
    "Chronic Obstructive Pulmonary Disease": ["Albuterol", "Fluticasone"],
    "Congestive Heart Failure": ["Lisinopril", "Metoprolol", "Furosemide", "Carvedilol"],
    "Coronary Artery Disease": ["Atorvastatin", "Aspirin", "Metoprolol", "Clopidogrel"],
    "Atrial Fibrillation": ["Metoprolol", "Warfarin"],
    "Osteoarthritis": ["Ibuprofen", "Acetaminophen", "Tramadol"],
    "Rheumatoid Arthritis": ["Prednisone", "Ibuprofen"],
    "Chronic Kidney Disease": ["Furosemide", "Losartan"],
    "Gastroesophageal Reflux Disease": ["Omeprazole", "Pantoprazole"],
    "Hyperlipidemia": ["Atorvastatin", "Simvastatin"],
    "Depression": ["Sertraline", "Escitalopram"],
    "Anxiety Disorder": ["Sertraline", "Escitalopram", "Alprazolam"],
    "Hypothyroidism": ["Levothyroxine"],
    "Migraine": ["Ibuprofen", "Acetaminophen"],
    "Epilepsy": ["Gabapentin"],
    "Pneumonia": ["Amoxicillin", "Azithromycin"],
    "Influenza": ["Acetaminophen"],
    "Acute Bronchitis": ["Azithromycin"],
    "Allergic Rhinitis": ["Fluticasone", "Montelukast"]
}

# ===== Procedures =====

# Common medical procedures with CPT codes
PROCEDURES = [
    # Format: (Procedure name, CPT code)
    ("Complete Blood Count", "85025"),
    ("Comprehensive Metabolic Panel", "80053"),
    ("Lipid Panel", "80061"),
    ("Hemoglobin A1c", "83036"),
    ("Thyroid Stimulating Hormone", "84443"),
    ("Electrocardiogram", "93000"),
    ("Chest X-ray", "71046"),
    ("MRI Brain without contrast", "70551"),
    ("CT Scan Abdomen and Pelvis with contrast", "74177"),
    ("Ultrasound Abdomen Complete", "76700"),
    ("Bone Density Study", "77080"),
    ("Colonoscopy", "45378"),
    ("Upper Endoscopy", "43235"),
    ("Pulmonary Function Test", "94060"),
    ("Mammogram", "77067"),
    ("Treadmill Stress Test", "93015"),
    ("Echocardiogram", "93306"),
    ("Lumbar Puncture", "62270"),
    ("Joint Injection", "20610"),
    ("Skin Biopsy", "11100"),
    ("Physical Therapy Evaluation", "97161"),
    ("Psychotherapy 45 minutes", "90834"),
    ("Office Visit, New Patient, Comprehensive", "99204"),
    ("Office Visit, Established Patient, Detailed", "99214"),
    ("Emergency Department Visit, High Complexity", "99285"),
    ("Removal of Impacted Cerumen", "69210"),
    ("Nebulizer Treatment", "94640"),
    ("Influenza Vaccine", "90686"),
    ("Pneumococcal Vaccine", "90732"),
    ("Tetanus-Diphtheria Vaccine", "90714")
]

# ===== Lab Tests =====

# Common lab tests with reference ranges
LAB_TESTS = [
    # Format: (Test name, Unit, Low normal, High normal)
    ("Hemoglobin", "g/dL", 12.0, 16.0),
    ("Hematocrit", "%", 37.0, 47.0),
    ("White Blood Cell Count", "x10^3/uL", 4.5, 11.0),
    ("Platelet Count", "x10^3/uL", 150, 450),
    ("Sodium", "mmol/L", 135, 145),
    ("Potassium", "mmol/L", 3.5, 5.0),
    ("Chloride", "mmol/L", 98, 107),
    ("Carbon Dioxide", "mmol/L", 22, 29),
    ("Blood Urea Nitrogen", "mg/dL", 7, 20),
    ("Creatinine", "mg/dL", 0.6, 1.2),
    ("Glucose", "mg/dL", 70, 99),
    ("Calcium", "mg/dL", 8.5, 10.5),
    ("Total Protein", "g/dL", 6.0, 8.3),
    ("Albumin", "g/dL", 3.5, 5.0),
    ("Total Bilirubin", "mg/dL", 0.1, 1.2),
    ("Alkaline Phosphatase", "U/L", 44, 147),
    ("AST", "U/L", 10, 40),
    ("ALT", "U/L", 7, 56),
    ("Hemoglobin A1c", "%", 4.0, 5.6),
    ("Total Cholesterol", "mg/dL", 125, 200),
    ("HDL Cholesterol", "mg/dL", 40, 60),
    ("LDL Cholesterol", "mg/dL", 0, 100),
    ("Triglycerides", "mg/dL", 0, 150),
    ("TSH", "uIU/mL", 0.4, 4.0),
    ("Free T4", "ng/dL", 0.8, 1.8),
    ("Vitamin D, 25-Hydroxy", "ng/mL", 30, 100),
    ("Vitamin B12", "pg/mL", 200, 900),
    ("Ferritin", "ng/mL", 15, 200),
    ("Iron", "ug/dL", 50, 170),
    ("Prothrombin Time", "seconds", 11.0, 13.5)
]

# ===== Healthcare Providers =====

# Medical specialties
MEDICAL_SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Pediatrics", "Obstetrics/Gynecology",
    "Cardiology", "Dermatology", "Endocrinology", "Gastroenterology", "Hematology",
    "Infectious Disease", "Nephrology", "Neurology", "Oncology", "Ophthalmology",
    "Orthopedics", "Otolaryngology", "Psychiatry", "Pulmonology", "Radiology",
    "Rheumatology", "Urology", "Emergency Medicine", "Anesthesiology", "Physical Medicine"
]

# Healthcare facility types
HEALTHCARE_FACILITIES = [
    "Hospital", "Primary Care Clinic", "Urgent Care", "Specialty Clinic",
    "Ambulatory Surgery Center", "Imaging Center", "Laboratory", "Rehabilitation Center",
    "Long-term Care Facility", "Hospice", "Home Health Agency", "Mental Health Facility",
    "Outpatient Center", "Community Health Center"
]

# ===== Healthcare Class Generator =====

class HealthcareDataProvider:
    """Provider for healthcare-specific data generation."""
    
    @staticmethod
    def generate_diagnosis(chronic: Optional[bool] = None) -> Dict[str, str]:
        """
        Generate a random medical diagnosis.
        
        Args:
            chronic: If True, returns a chronic condition; if False, returns an acute condition;
                   if None, returns any condition
            
        Returns:
            A dictionary with diagnosis name and ICD-10 code
        """
        if chronic is True:
            condition, code = random.choice(CHRONIC_CONDITIONS)
        elif chronic is False:
            condition, code = random.choice(ACUTE_CONDITIONS)
        else:
            condition, code = random.choice(MEDICAL_CONDITIONS)
            
        return {
            'name': condition,
            'code': code
        }
    
    @staticmethod
    def generate_medication(condition: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a random medication, optionally appropriate for a specific condition.
        
        Args:
            condition: Optional medical condition to get appropriate medication for
            
        Returns:
            A dictionary with medication details
        """
        if condition and condition in CONDITION_MEDICATION_MAP:
            # Get a medication appropriate for the condition
            med_options = CONDITION_MEDICATION_MAP[condition]
            med_name = random.choice(med_options)
            
            # Find the full medication details
            for name, category, dosage in MEDICATIONS:
                if name == med_name:
                    return {
                        'name': name,
                        'category': category,
                        'dosage': dosage
                    }
        
        # If no condition or condition not in map, return random medication
        name, category, dosage = random.choice(MEDICATIONS)
        return {
            'name': name,
            'category': category,
            'dosage': dosage
        }
    
    @staticmethod
    def generate_procedure() -> Dict[str, str]:
        """
        Generate a random medical procedure.
        
        Returns:
            A dictionary with procedure name and CPT code
        """
        procedure, code = random.choice(PROCEDURES)
        return {
            'name': procedure,
            'code': code
        }
    
    @staticmethod
    def generate_lab_result(test_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a random lab test result, optionally for a specific test.
        
        Args:
            test_name: Optional specific lab test name
            
        Returns:
            A dictionary with lab test details and result
        """
        if test_name:
            # Find the specified test
            test_info = None
            for name, unit, low, high in LAB_TESTS:
                if name.lower() == test_name.lower():
                    test_info = (name, unit, low, high)
                    break
            
            if not test_info:
                # If test not found, pick a random one
                test_info = random.choice(LAB_TESTS)
        else:
            # Pick a random test
            test_info = random.choice(LAB_TESTS)
        
        name, unit, low, high = test_info
        
        # Generate a result that's within normal range 80% of the time
        if random.random() < 0.8:
            # Within normal range
            result = round(random.uniform(low, high), 1)
            flag = "Normal"
        else:
            # Abnormal result
            if random.random() < 0.5:
                # Below normal
                result = round(random.uniform(low * 0.5, low * 0.9), 1)
                flag = "Low"
            else:
                # Above normal
                result = round(random.uniform(high * 1.1, high * 1.5), 1)
                flag = "High"
        
        return {
            'name': name,
            'result': result,
            'unit': unit,
            'reference_range': f"{low}-{high}",
            'flag': flag
        }
    
    @staticmethod
    def generate_vital_signs() -> Dict[str, Any]:
        """
        Generate a random set of vital signs.
        
        Returns:
            A dictionary with vital sign measurements
        """
        # Generate vital signs with normal ranges most of the time
        temp_f = round(random.uniform(97.0, 99.5), 1)
        heart_rate = random.randint(60, 100)
        resp_rate = random.randint(12, 20)
        
        # Blood pressure components
        systolic = random.randint(110, 140)
        diastolic = random.randint(60, 90)
        
        # Oxygen saturation
        o2_sat = random.randint(95, 100)
        
        # Pain score (0-10)
        pain = random.randint(0, 10)
        
        return {
            'temperature': {
                'value': temp_f,
                'unit': 'F'
            },
            'heart_rate': {
                'value': heart_rate,
                'unit': 'bpm'
            },
            'respiratory_rate': {
                'value': resp_rate,
                'unit': 'breaths/min'
            },
            'blood_pressure': {
                'systolic': systolic,
                'diastolic': diastolic,
                'unit': 'mmHg',
                'formatted': f"{systolic}/{diastolic}"
            },
            'oxygen_saturation': {
                'value': o2_sat,
                'unit': '%'
            },
            'pain': {
                'score': pain,
                'scale': '0-10'
            }
        }
    
    @staticmethod
    def generate_allergy() -> Dict[str, str]:
        """
        Generate a random medical allergy.
        
        Returns:
            A dictionary with allergy details
        """
        # Common allergies
        allergies = [
            # Medication allergies
            ("Penicillin", "Medication", "Rash"),
            ("Sulfa", "Medication", "Hives"),
            ("Aspirin", "Medication", "Respiratory distress"),
            ("Ibuprofen", "Medication", "Swelling"),
            ("Codeine", "Medication", "Nausea"),
            # Food allergies
            ("Peanuts", "Food", "Anaphylaxis"),
            ("Tree nuts", "Food", "Hives"),
            ("Shellfish", "Food", "Swelling"),
            ("Eggs", "Food", "Rash"),
            ("Milk", "Food", "Digestive issues"),
            # Environmental allergies
            ("Pollen", "Environmental", "Sneezing"),
            ("Dust mites", "Environmental", "Congestion"),
            ("Pet dander", "Environmental", "Itchy eyes"),
            ("Mold", "Environmental", "Coughing"),
            ("Latex", "Environmental", "Skin irritation")
        ]
        
        allergen, category, reaction = random.choice(allergies)
        
        # Determine severity
        severity = random.choice(["Mild", "Moderate", "Severe"])
        
        return {
            'allergen': allergen,
            'category': category,
            'reaction': reaction,
            'severity': severity
        }
    
    @staticmethod
    def generate_provider() -> Dict[str, str]:
        """
        Generate a random healthcare provider.
        
        Returns:
            A dictionary with provider details
        """
        # Generate a provider name
        # For simplicity, we'll just concatenate titles and common last names
        titles = ["Dr.", "Dr.", "Dr.", "NP", "PA"]  # Weight towards doctors
        first_names = ["Michael", "Jennifer", "David", "Sarah", "James", "Lisa", "Robert", "Emily", 
                      "John", "Jessica", "William", "Elizabeth", "Richard", "Michelle", "Thomas", 
                      "Amanda", "Daniel", "Rebecca", "Matthew", "Stephanie"]
        last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", 
                     "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
                     "Thompson", "Garcia", "Martinez", "Robinson"]
        
        title = random.choice(titles)
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Generate a specialty
        specialty = random.choice(MEDICAL_SPECIALTIES)
        
        # Generate a facility type
        facility = random.choice(HEALTHCARE_FACILITIES)
        
        return {
            'name': f"{title} {first_name} {last_name}",
            'first_name': first_name,
            'last_name': last_name,
            'title': title,
            'specialty': specialty,
            'facility': facility
        }
    
    @staticmethod
    def generate_encounter() -> Dict[str, Any]:
        """
        Generate a random healthcare encounter.
        
        Returns:
            A dictionary with encounter details
        """
        # Types of encounters
        encounter_types = ["Office Visit", "Emergency Visit", "Hospital Admission", 
                          "Telemedicine", "Outpatient Procedure", "Consultation"]
        
        # Encounter statuses
        statuses = ["Scheduled", "Checked In", "In Progress", "Completed", "Cancelled", "No Show"]
        
        # Generate a date within the past year
        days_ago = random.randint(0, 365)
        encounter_date = datetime.now() - timedelta(days=days_ago)
        
        # Format date and time
        date_str = encounter_date.strftime("%Y-%m-%d")
        time_str = encounter_date.strftime("%H:%M")
        
        # Generate a duration in minutes
        duration = random.choice([15, 30, 45, 60, 90, 120])
        
        # Generate chief complaint
        chief_complaints = [
            "Fever", "Cough", "Shortness of breath", "Chest pain", "Abdominal pain",
            "Headache", "Back pain", "Joint pain", "Dizziness", "Fatigue",
            "Nausea", "Vomiting", "Diarrhea", "Rash", "Sore throat",
            "Ear pain", "Eye pain", "Urinary problems", "Swelling", "Bleeding"
        ]
        
        # Generate a provider and diagnosis
        provider = HealthcareDataProvider.generate_provider()
        diagnosis = HealthcareDataProvider.generate_diagnosis()
        
        return {
            'type': random.choice(encounter_types),
            'date': date_str,
            'time': time_str,
            'duration_minutes': duration,
            'status': random.choice(statuses),
            'chief_complaint': random.choice(chief_complaints),
            'provider': provider,
            'diagnosis': diagnosis
        }
    
    @staticmethod
    def generate_medication_prescription() -> Dict[str, Any]:
        """
        Generate a random medication prescription.
        
        Returns:
            A dictionary with prescription details
        """
        # Generate a medication
        medication = HealthcareDataProvider.generate_medication()
        
        # Generate prescription details
        frequencies = ["Once daily", "Twice daily", "Three times daily", "Four times daily", 
                      "Every 4 hours", "Every 6 hours", "Every 8 hours", "Every 12 hours",
                      "As needed", "Weekly", "Before meals", "At bedtime"]
        
        routes = ["Oral", "Topical", "Inhaled", "Subcutaneous", "Intramuscular", "Intravenous", 
                 "Rectal", "Nasal", "Ophthalmic", "Otic"]
        
        # Extract a basic dose from the medication's typical dosage
        dosage = medication['dosage']
        dose_match = re.search(r'(\d+(?:-\d+)?)\s*(mg|mcg|g|mL)', dosage)
        if dose_match:
            dose = dose_match.group(0)
        else:
            dose = "1 tablet"
        
        # Generate a quantity
        quantity = random.choice([30, 60, 90, 120, 180])
        
        # Generate number of refills
        refills = random.randint(0, 11)
        
        # Generate a date within the past month
        days_ago = random.randint(0, 30)
        rx_date = datetime.now() - timedelta(days=days_ago)
        date_str = rx_date.strftime("%Y-%m-%d")
        
        return {
            'medication': medication['name'],
            'dose': dose,
            'route': random.choice(routes),
            'frequency': random.choice(frequencies),
            'quantity': quantity,
            'refills': refills,
            'date_prescribed': date_str,
            'prescriber': HealthcareDataProvider.generate_provider()['name']
        }
    
    @staticmethod
    def generate_medical_history() -> Dict[str, Any]:
        """
        Generate a random medical history.
        
        Returns:
            A dictionary with medical history details
        """
        # Generate a list of chronic conditions
        num_conditions = random.randint(0, 5)
        conditions = []
        for _ in range(num_conditions):
            conditions.append(HealthcareDataProvider.generate_diagnosis(chronic=True))
        
        # Generate a list of past surgeries
        surgeries = [
            "Appendectomy", "Cholecystectomy", "Hernia repair", "Tonsillectomy", 
            "Knee replacement", "Hip replacement", "Coronary artery bypass",
            "Hysterectomy", "Cesarean section", "Cataract surgery"
        ]
        
        num_surgeries = random.randint(0, 3)
        past_surgeries = []
        for _ in range(num_surgeries):
            # Generate a random date for the surgery
            years_ago = random.randint(1, 20)
            surgery_date = datetime.now() - timedelta(days=365 * years_ago)
            date_str = surgery_date.strftime("%Y-%m-%d")
            
            past_surgeries.append({
                'procedure': random.choice(surgeries),
                'date': date_str
            })
        
        # Generate a list of allergies
        num_allergies = random.randint(0, 3)
        allergies = []
        for _ in range(num_allergies):
            allergies.append(HealthcareDataProvider.generate_allergy())
        
        # Generate family history
        family_conditions = [
            "Diabetes", "Hypertension", "Coronary Artery Disease", "Cancer", 
            "Stroke", "Alzheimer's Disease", "Parkinson's Disease", "Rheumatoid Arthritis",
            "Asthma", "Depression", "Bipolar Disorder", "Schizophrenia",
            "Hemophilia", "Cystic Fibrosis", "Sickle Cell Anemia"
        ]
        
        family_members = ["Mother", "Father", "Sister", "Brother", "Grandmother", "Grandfather"]
        
        num_family_conditions = random.randint(0, 4)
        family_history = []
        for _ in range(num_family_conditions):
            family_history.append({
                'condition': random.choice(family_conditions),
                'relation': random.choice(family_members)
            })
        
        # Generate social history
        smoking_statuses = ["Never smoker", "Former smoker", "Current smoker"]
        alcohol_use = ["None", "Occasional", "Moderate", "Heavy"]
        
        return {
            'chronic_conditions': conditions,
            'past_surgeries': past_surgeries,
            'allergies': allergies,
            'family_history': family_history,
            'social_history': {
                'smoking': random.choice(smoking_statuses),
                'alcohol': random.choice(alcohol_use),
                'occupation': HealthcareDataProvider.generate_occupation()
            }
        }
    
    @staticmethod
    def generate_occupation() -> str:
        """
        Generate a random occupation.
        
        Returns:
            A string with an occupation
        """
        occupations = [
            "Teacher", "Nurse", "Doctor", "Engineer", "Software Developer",
            "Accountant", "Manager", "Sales Representative", "Administrative Assistant",
            "Lawyer", "Chef", "Driver", "Retail Worker", "Construction Worker",
            "Electrician", "Plumber", "Farmer", "Writer", "Artist", "Musician",
            "Pilot", "Flight Attendant", "Firefighter", "Police Officer", "Paramedic",
            "Librarian", "Scientist", "Professor", "Financial Advisor", "Real Estate Agent"
        ]
        
        return random.choice(occupations)

# ===== Helper Functions =====

def get_generator(data_type: str):
    """
    Get a generator function for specific healthcare data types.
    
    Args:
        data_type: Type of data to generate
        
    Returns:
        A generator function for the specified data type
    """
    # Map data types to provider methods
    provider = HealthcareDataProvider()
    
    generators = {
        'diagnosis': provider.generate_diagnosis,
        'medication': provider.generate_medication,
        'procedure': provider.generate_procedure,
        'lab_result': provider.generate_lab_result,
        'vital_signs': provider.generate_vital_signs,
        'allergy': provider.generate_allergy,
        'provider': provider.generate_provider,
        'encounter': provider.generate_encounter,
        'prescription': provider.generate_medication_prescription,
        'medical_history': provider.generate_medical_history,
        'occupation': provider.generate_occupation
    }
    
    return generators.get(data_type, lambda: f"Unsupported healthcare data type: {data_type}")

def get_supported_data_types() -> List[str]:
    """
    Get a list of all supported healthcare data types.
    
    Returns:
        A list of supported data types
    """
    return list(get_generator('').__globals__['generators'].keys())
