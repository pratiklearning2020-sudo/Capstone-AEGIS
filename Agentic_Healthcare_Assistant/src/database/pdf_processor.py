"""
PDF Processor for Ingesting Unstructured Medical Records.
Extracts clinical notes from PDFs (e.g. sample_patient.pdf, sample_report_*.pdf)
and parses them into structured clinical sections (Subjective, Objective, Assessment, Plan).
"""

import io
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pypdf


class MedicalPDFProcessor:
    """
    Extracts text and structures clinical sections from patient reports.
    """

    @staticmethod
    def extract_text(pdf_source: Union[str, Path, bytes, io.BytesIO]) -> str:
        """Extracts complete text from a PDF file or byte stream."""
        if isinstance(pdf_source, (str, Path)):
            reader = pypdf.PdfReader(str(pdf_source))
        elif isinstance(pdf_source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(pdf_source))
        else:
            reader = pypdf.PdfReader(pdf_source)

        text_pages = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text_pages.append(content)

        return "\n\n".join(text_pages)

    @staticmethod
    def parse_clinical_report(text: str) -> Dict[str, Any]:
        """
        Parses raw text into standard clinical sections:
        - Patient Info (Name, DOB/Age, Gender, Address, Phone)
        - Reason for Visit / Subjective
        - Objective / Vitals
        - Assessment / Diagnosis
        - Plan / Medications
        """
        report_data = {
            "patient_name": "",
            "dob": "",
            "age": "",
            "gender": "",
            "phone": "",
            "address": "",
            "subjective": "",
            "objective": "",
            "vitals": "",
            "assessment": "",
            "diagnosis": "",
            "plan": "",
            "raw_text": text
        }

        # Extract demographic fields
        name_match = re.search(r"Patient:\s*([^\n\r]+)", text, re.IGNORECASE) or re.search(r"Patient\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
        if name_match:
            report_data["patient_name"] = name_match.group(1).strip()

        dob_match = re.search(r"DOB:\s*([^\n\r]+)", text, re.IGNORECASE) or re.search(r"Date\s+of\s+birth\s*([^\n\r]+)", text, re.IGNORECASE)
        if dob_match:
            report_data["dob"] = dob_match.group(1).strip()

        gender_match = re.search(r"Gender:\s*([^\n\r]+)", text, re.IGNORECASE) or re.search(r"Sex\s+([A-Za-z]+)", text, re.IGNORECASE)
        if gender_match:
            report_data["gender"] = gender_match.group(1).strip()

        phone_match = re.search(r"Phone:\s*([^\n\r]+)", text, re.IGNORECASE) or re.search(r"Tel:\s*([^\n\r]+)", text, re.IGNORECASE)
        if phone_match:
            report_data["phone"] = phone_match.group(1).strip()

        addr_match = re.search(r"Address:\s*([^\n\r]+)", text, re.IGNORECASE)
        if addr_match:
            report_data["address"] = addr_match.group(1).strip()

        # Extract Clinical Sections
        subj_match = re.search(r"Subjective\s+Notes?:?(.*?)(Objective\s+Notes?:|Assessment\s+Notes?:|$)", text, re.DOTALL | re.IGNORECASE)
        if subj_match:
            report_data["subjective"] = subj_match.group(1).strip()

        obj_match = re.search(r"Objective\s+Notes?:?(.*?)(Assessment\s+Notes?:|Plan\s+Notes?:|$)", text, re.DOTALL | re.IGNORECASE)
        if obj_match:
            report_data["objective"] = obj_match.group(1).strip()

        vitals_match = re.search(r"Vitals?:?\s*([^\n\r]+)", text, re.IGNORECASE) or re.search(r"Vital\s+Signs\s+(.*?)(Encounters|$)", text, re.DOTALL | re.IGNORECASE)
        if vitals_match:
            report_data["vitals"] = vitals_match.group(1).strip()

        assess_match = re.search(r"Assessment\s+Notes?:?(.*?)(Plan\s+Notes?:|$)", text, re.DOTALL | re.IGNORECASE) or re.search(r"Assessments\s*:\s*(.*?)(Problems|$)", text, re.DOTALL | re.IGNORECASE)
        if assess_match:
            report_data["assessment"] = assess_match.group(1).strip()

        diag_match = re.search(r"Diagnosis:\s*([^\n\r]+)", text, re.IGNORECASE)
        if diag_match:
            report_data["diagnosis"] = diag_match.group(1).strip()
        elif report_data["assessment"]:
            report_data["diagnosis"] = report_data["assessment"].split("\n")[0].strip()

        plan_match = re.search(r"Plan\s+Notes?:?(.*?)(Follow-up|$)", text, re.DOTALL | re.IGNORECASE) or re.search(r"Plan\s+of\s+Treatment\s*(.*?)(Results|$)", text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            report_data["plan"] = plan_match.group(1).strip()

        return report_data

    @staticmethod
    def generate_clinical_summary(parsed_report: Dict[str, Any]) -> str:
        """Generates a concise clinical summary string suitable for Excel or FAISS vector storage."""
        parts = []
        name = parsed_report.get("patient_name")
        diag = parsed_report.get("diagnosis") or parsed_report.get("assessment")
        subj = parsed_report.get("subjective")
        vitals = parsed_report.get("vitals")
        plan = parsed_report.get("plan")

        if name:
            parts.append(f"Patient {name}")
        if subj:
            parts.append(f"presented with {subj[:140]}...")
        if vitals:
            parts.append(f"Vitals: {vitals[:80]}.")
        if diag:
            parts.append(f"Diagnosis: {diag[:100]}.")
        if plan:
            parts.append(f"Plan: {plan[:120]}.")

        summary = " ".join(parts).strip()
        return summary if summary else parsed_report.get("raw_text", "")[:300]
