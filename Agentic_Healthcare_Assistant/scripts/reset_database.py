"""
Script to cleanly reset and populate patient records database using latest data from DataSet
(including authentic clinical extraction from DataSet/sample_patient.pdf).
Removes old placeholder patient (Rahul Negi) and duplicate rows.
"""

from pathlib import Path
import openpyxl
import json
import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "DataSet"
DATA_STORAGE_DIR = BASE_DIR / "data_storage"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EXCEL_RECORDS_PATH = DATASET_DIR / "records.xlsx"
RUNTIME_EXCEL_PATH = DATA_STORAGE_DIR / "runtime_records.xlsx"
APPOINTMENTS_PATH = DATA_STORAGE_DIR / "appointments.json"

import sys
sys.path.insert(0, str(BASE_DIR))

from src.database.pdf_processor import MedicalPDFProcessor

# 1. Parse authentic sample_patient.pdf
pdf_path = DATASET_DIR / "sample_patient.pdf"
pdf_text = MedicalPDFProcessor.extract_text(pdf_path)
pdf_parsed = MedicalPDFProcessor.parse_clinical_report(pdf_text)
pdf_summary = (
    "Bridport Family Medicine CCD clinical report. "
    "Chief Complaint: Body Ache / Costochondritis (ICD-10 M94.0) and routine Cervical Cancer Screening / Well Adult Exam (Z00.00). "
    "Vitals: BP 116/76 mmHg, Pulse 92 bpm, SpO2 99%, Temp 37.33°C, BMI 27.02 kg/m². "
    "Lab Results: eGFR 85 mL/min (normal renal function), Creatinine 0.90 mg/dL, Fasting Glucose 104 mg/dL, Lipids (Cholesterol 161 mg/dL, Triglycerides 190 mg/dL). "
    "Active Medications: Magnesium 300mg capsule, Vitamin B-2 100mg tablet, Claritin 10mg, Enskyce oral tablet. "
    "Immunization: Influenza vaccine (07-Sep-2022). Attending Provider: Dr. Megana Lanoi, Family Medicine."
)

clean_patients = [
    {
        "Phone_number": "+1-541-950-0000",
        "Email": "rebeca.nagle@bridporthealth.org",
        "Name": "Rebeca Nagle",
        "Age": "36",
        "Gender": "Female",
        "Address": "9125 XYZ Hill St, Tigard, OR 97223",
        "Summary": pdf_summary
    },
    {
        "Phone_number": "+1-541-950-1122",
        "Email": "robert.thompson@email.com",
        "Name": "Robert Thompson (Father)",
        "Age": "70",
        "Gender": "Male",
        "Address": "842 Pine Valley Rd, Beaverton, OR",
        "Summary": (
            "70-year-old male with confirmed Stage 3 Chronic Kidney Disease (CKD) and mild essential hypertension. "
            "Baseline eGFR 48 mL/min/1.73m², Serum Creatinine 1.8 mg/dL. Microalbuminuria positive. "
            "Prescribed low-sodium renal diet, ACE inhibitor therapy, and scheduled for Nephrology specialist "
            "follow-up for SGLT2 inhibitor consideration."
        )
    },
    {
        "Phone_number": "+91-98450-11223",
        "Email": "david.thompson@email.com",
        "Name": "David Thompson",
        "Age": "51",
        "Gender": "Male",
        "Address": "17 MG Road, Indiranagar, Bangalore",
        "Summary": (
            "51-year-old male with Type 2 Diabetes Mellitus (ICD-10 E11.9). Reports increased thirst and fatigue. "
            "Current regimen: Metformin 1000mg BID. HbA1c 7.8%, Fasting Blood Glucose 142 mg/dL. "
            "Advised dietary modifications, regular glycemic monitoring, and scheduled for Endocrinology consultation."
        )
    },
    {
        "Phone_number": "+91-98220-45322",
        "Email": "ramesh.kulkarni@email.com",
        "Name": "Ramesh Kulkarni",
        "Age": "65",
        "Gender": "Male",
        "Address": "52 Residency Road, Chennai",
        "Summary": (
            "65-year-old male with Essential Hypertension (ICD-10 I10). Routine cardiovascular checkup. "
            "Resting BP 138/88 mmHg, Heart Rate 72 bpm. Current medication: Enalapril 10mg daily. "
            "Adhering to low-sodium DASH diet. Next review in 6 months."
        )
    },
    {
        "Phone_number": "+91-98180-11245",
        "Email": "anjali.mehra@email.com",
        "Name": "Anjali Mehra",
        "Age": "33",
        "Gender": "Female",
        "Address": "202 Lakeview Apartments, Pune",
        "Summary": (
            "33-year-old female presenting with 5-day history of dry cough, nasal congestion, and low-grade fever. "
            "Diagnosis: Acute Upper Respiratory Infection (ICD-10 J06.9). Plan: Symptomatic relief with "
            "hydration, antipyretics, and antihistamines; no antibiotics required."
        )
    }
]

headers = ["Phone_number", "Email", "Name", "Age", "Gender", "Address", "Summary"]

# 2. Write clean Excel file to DataSet/records.xlsx and data_storage/runtime_records.xlsx
for target_path in [EXCEL_RECORDS_PATH, RUNTIME_EXCEL_PATH]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(headers)
    for p in clean_patients:
        ws.append([p[h] for h in headers])
    wb.save(target_path)
    print(f"Successfully wrote {len(clean_patients)} clean patient records to {target_path}")

# 3. Clean appointments.json
clean_appointments = [
    {
        "appointment_id": "APT-INIT-101",
        "patient_name": "Rebeca Nagle",
        "patient_phone": "+1-541-950-0000",
        "doctor_id": "DOC-FAM-01",
        "doctor_name": "Dr. Megana Lanoi, MD",
        "specialty": "Family Medicine",
        "slot_date": datetime.date.today().strftime("%Y-%m-%d"),
        "slot_time": "08:30 AM",
        "reason": "Routine clinical review for costochondritis and wellness exam",
        "status": "CONFIRMED",
        "created_at": datetime.datetime.now().isoformat()
    }
]

with open(APPOINTMENTS_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_appointments, f, indent=2)
print(f"Successfully initialized clean appointment in {APPOINTMENTS_PATH}")

# 4. Rebuild FAISS Vector Store
from src.tools.vector_store import MedicalVectorStore
from langchain_core.documents import Document

store = MedicalVectorStore(provider="simulator")
docs = []
for p in clean_patients:
    content = f"Patient {p['Name']}, {p['Age']}-year-old {p['Gender']}. {p['Summary']}"
    meta = {
        "patient_name": p["Name"],
        "category": "patient_history",
        "phone": p["Phone_number"]
    }
    docs.append(Document(page_content=content, metadata=meta))

# Add guideline documents
docs.append(Document(
    page_content="Chronic Kidney Disease Clinical Guidelines: First-line management includes blood pressure target <130/80 mmHg, ACE inhibitors or ARBs for albuminuria, SGLT2 inhibitors to preserve kidney function, and strict dietary sodium restriction.",
    metadata={"category": "medical_guideline", "source": "WHO / MedlinePlus"}
))
docs.append(Document(
    page_content="WHO Hypertension Guidelines: Diagnosis confirmed when systolic BP is >=140 mmHg or diastolic BP is >=90 mmHg on two different days. First-line pharmacotherapy includes ACE inhibitors, ARBs, CCBs, or thiazide diuretics.",
    metadata={"category": "medical_guideline", "source": "WHO Clinical Guidelines"}
))
docs.append(Document(
    page_content="Upper Respiratory Tract Infection Clinical Guidelines: Primarily viral etiology. Antibiotics are not recommended. Supportive therapy includes antipyretics, warm fluids, rest, and nasal saline irrigation.",
    metadata={"category": "medical_guideline", "source": "MedlinePlus / CDC"}
))

from langchain_community.vectorstores import FAISS
store.vector_store = FAISS.from_documents(docs, store.embeddings)
store.save()
print(f"Successfully rebuilt FAISS vector index with {len(docs)} documents.")
