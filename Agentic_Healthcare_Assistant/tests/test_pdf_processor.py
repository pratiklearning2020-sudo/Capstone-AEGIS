"""
Tests for MedicalPDFProcessor (extracting and parsing clinical reports).
"""

from pathlib import Path
from src.config import DATASET_DIR
from src.database.pdf_processor import MedicalPDFProcessor


def test_extract_and_parse_sample_patient_pdf():
    pdf_path = DATASET_DIR / "sample_patient.pdf"
    assert pdf_path.exists(), "DataSet/sample_patient.pdf must exist in the dataset directory"

    text = MedicalPDFProcessor.extract_text(pdf_path)
    assert len(text) > 200
    assert "Rebeca" in text
    assert "Nagle" in text

    parsed = MedicalPDFProcessor.parse_clinical_report(text)
    assert "Rebeca" in parsed["patient_name"]
    assert parsed["gender"].lower() == "female"
    assert "541-950-0000" in parsed["phone"]
    assert len(parsed["vitals"]) > 0

    summary = MedicalPDFProcessor.generate_clinical_summary(parsed)
    assert "Rebeca" in summary
    assert len(summary) > 30


def test_parse_clinical_report_structured_text():
    sample_text = """
    Patient: David Thompson
    DOB: 1973-04-12
    Gender: Male
    Phone: +91-98450-11223
    Address: 17 MG Road, Bangalore

    Subjective Notes:
    Patient reports increased thirst, polyuria, and fatigue over the past 3 weeks.

    Objective Notes:
    Physical exam normal. Vitals: BP 126/82 mmHg, Pulse 74 bpm, BMI 28.4.

    Assessment Notes:
    Type 2 Diabetes Mellitus (ICD-10: E11.9) with suboptimal glycemic control.

    Plan Notes:
    Increase Metformin to 1000mg BID. Order HbA1c and Lipid profile. Dietary counseling.
    """
    parsed = MedicalPDFProcessor.parse_clinical_report(sample_text)
    assert "David" in parsed["patient_name"]
    assert parsed["gender"].lower() == "male"
    assert "Diabetes" in parsed["diagnosis"] or "E11.9" in parsed["diagnosis"]
    assert "Metformin" in parsed["plan"]

    summary = MedicalPDFProcessor.generate_clinical_summary(parsed)
    assert "David" in summary
    assert "Diabetes" in summary
