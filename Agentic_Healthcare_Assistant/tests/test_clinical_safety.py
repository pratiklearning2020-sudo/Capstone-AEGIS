"""
Unit tests for ClinicalSafetyGuard.
Verifies contraindication detection (e.g. NSAIDs in CKD patients).
"""

from src.tools.clinical_safety import ClinicalSafetyGuard


def test_nsaid_contraindication_in_ckd():
    # Patient with CKD asks about ibuprofen
    alert = ClinicalSafetyGuard.check_safety(
        query="Can I take ibuprofen for my back ache?",
        conditions=["Chronic Kidney Disease (CKD)", "Stage 3 Renal Impairment"]
    )
    assert alert is not None
    assert alert["has_warning"] is True
    assert alert["severity"] == "HIGH"
    assert "ibuprofen" in alert["detected_substance"].lower()
    assert "Acetaminophen" in alert["safe_alternatives"]


def test_safe_medication_no_alert():
    # Safe medication question
    alert = ClinicalSafetyGuard.check_safety(
        query="Can I take acetaminophen for a headache?",
        conditions=["Chronic Kidney Disease (CKD)"]
    )
    assert alert is None


def test_hypertension_decongestant_warning():
    alert = ClinicalSafetyGuard.check_safety(
        query="I have a stuffy nose, can I take pseudoephedrine?",
        conditions=["Essential Hypertension"]
    )
    assert alert is not None
    assert alert["severity"] == "MODERATE"
    assert "decongestants" in alert["warning_title"].lower()


def test_generate_appointment_pass_and_ccd():
    from src.tools.clinical_reports import generate_appointment_pass, generate_patient_ccd_report

    sample_appt = {
        "appointment_id": "APT-NEPH-TEST1",
        "patient_name": "Robert Thompson",
        "doctor_name": "Dr. Aris Thorne, MD",
        "specialty": "Nephrology",
        "clinic": "Renal & Kidney Care Center",
        "slot_date": "2026-09-10",
        "slot_time": "10:30 AM",
        "reason": "CKD Stage 3 clinical follow-up"
    }
    pass_text = generate_appointment_pass(sample_appt)
    assert "APT-NEPH-TEST1" in pass_text
    assert "Dr. Aris Thorne" in pass_text
    assert "Robert Thompson" in pass_text
    assert "PATIENT PREPARATION INSTRUCTIONS" in pass_text

    sample_pt = {
        "Name": "Rebeca Nagle",
        "Age": 36,
        "Gender": "Female",
        "Phone_number": "+1-541-950-0000",
        "Summary": "Active diagnosis of Costochondritis."
    }
    ccd_text = generate_patient_ccd_report(sample_pt)
    assert "CONTINUITY OF CARE DOCUMENT" in ccd_text
    assert "Rebeca Nagle" in ccd_text
    assert "Costochondritis" in ccd_text


def test_emergency_red_flag_detection():
    """Verify acute symptoms trigger emergency clinical alerts."""
    # 1. Cardiac emergency
    cardiac = ClinicalSafetyGuard.check_emergency_red_flags("My dad has crushing chest pain radiating to his left arm")
    assert cardiac is not None
    assert cardiac["is_emergency"] is True
    assert "Cardiac" in cardiac["category"]
    assert "911" in cardiac["action_required"]

    # 2. Stroke emergency (FAST)
    stroke = ClinicalSafetyGuard.check_emergency_red_flags("Patient suddenly has slurred speech and facial droop")
    assert stroke is not None
    assert "Stroke" in stroke["category"]

    # 3. Non-emergency query
    routine = ClinicalSafetyGuard.check_emergency_red_flags("I want to book an appointment for mild skin dryness")
    assert routine is None


def test_drug_drug_interaction_matrix():
    """Verify multi-drug interactions are intercepted."""
    # 1. ACEi + Potassium-sparing diuretic
    ddi = ClinicalSafetyGuard.check_drug_drug_interactions(["Lisinopril 10mg", "Spironolactone 25mg"])
    assert len(ddi) == 1
    assert "Hyperkalemia" in ddi[0]["title"]
    assert ddi[0]["severity"] == "CRITICAL"

    # 2. Anticoagulant + NSAID
    ddi2 = ClinicalSafetyGuard.check_drug_drug_interactions(["Warfarin 5mg", "Ibuprofen 400mg"])
    assert len(ddi2) == 1
    assert "Hemorrhagic" in ddi2[0]["title"]


def test_generate_appointment_ics_calendar():
    """Verify RFC 5545 iCalendar event file generation."""
    from src.tools.clinical_reports import generate_appointment_ics

    sample_appt = {
        "appointment_id": "APT-NEPH-992",
        "patient_name": "Robert Thompson",
        "doctor_name": "Dr. Aris Thorne",
        "specialty": "Nephrology",
        "clinic": "Renal Health Center",
        "slot_date": "2026-09-15",
        "slot_time": "10:00 AM",
        "reason": "CKD Stage 3 Follow-up"
    }
    ics_text = generate_appointment_ics(sample_appt)
    assert "BEGIN:VCALENDAR" in ics_text
    assert "BEGIN:VEVENT" in ics_text
    assert "Dr. Aris Thorne" in ics_text
    assert "END:VCALENDAR" in ics_text


def test_generate_fhir_r4_bundle():
    """Verify HL7 FHIR Release 4 JSON bundle generation."""
    from src.tools.clinical_reports import generate_fhir_r4_bundle

    sample_pt = {
        "Name": "Robert Thompson",
        "Gender": "Male",
        "Phone_number": "+1-541-950-1122",
        "Email": "r.thompson@example.com",
        "Address": "742 Evergreen Terrace",
        "Summary": "Stage 3 Chronic Kidney Disease (CKD)"
    }
    sample_appt = {
        "appointment_id": "APT-78401",
        "doctor_name": "Dr. Aris Thorne",
        "slot_date": "2026-09-15",
        "reason": "Nephrology Consultation"
    }
    bundle = generate_fhir_r4_bundle(sample_pt, sample_appt)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) == 3  # Patient, Condition, Encounter
    resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in resource_types
    assert "Condition" in resource_types
    assert "Encounter" in resource_types

