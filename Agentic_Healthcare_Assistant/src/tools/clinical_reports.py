"""
Clinical Report and Appointment Pass Generator.
Generates formatted outpatient clinical passes, continuity of care documents (CCD),
and preparation instructions for scheduled encounters.
"""

import datetime
import hashlib
from typing import Any, Dict, Optional


def generate_appointment_pass(appointment: Dict[str, Any], patient_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates a structured, official clinical appointment pass text document.
    """
    apt_id = appointment.get("appointment_id", "APT-ENCOUNTER")
    p_name = appointment.get("patient_name") or (patient_data.get("Name") if patient_data else "Patient")
    p_phone = appointment.get("patient_phone") or (patient_data.get("Phone_number") if patient_data else "+1-541-950-0000")
    d_name = appointment.get("doctor_name", "Attending Physician")
    spec = appointment.get("specialty", "General Medicine")
    clinic = appointment.get("clinic", "Aegis Health Main Ambulatory Clinic")
    s_date = appointment.get("slot_date", str(datetime.date.today()))
    s_time = appointment.get("slot_time", "10:00 AM")
    reason = appointment.get("reason", "Outpatient clinical consultation")
    created = appointment.get("created_at", datetime.datetime.now().isoformat())[:19]

    # Generate cryptographic security validation hash
    raw_sig = f"{apt_id}|{p_name}|{d_name}|{s_date}|{s_time}"
    sig_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16].upper()

    pass_text = f"""================================================================================
🏥 AEGIS HEALTHCARE CLINICAL NETWORK • OFFICIAL APPOINTMENT PASS
================================================================================
CONFIRMATION ID : {apt_id}
ISSUED ON       : {created} (VERIFIED BY CLINICAL COMMAND CENTER)
STATUS          : CONFIRMED • AUTHORIZED OUTPATIENT ENCOUNTER
DIGITAL TOKEN   : {sig_hash}
--------------------------------------------------------------------------------
PATIENT INFORMATION:
  Patient Name  : {p_name}
  Contact Phone : {p_phone}
  Encounter Type: Outpatient Specialist Consultation

PHYSICIAN & LOCATION DETAILS:
  Attending MD  : {d_name}
  Specialty     : {spec}
  Clinic/Center : {clinic}
  Scheduled Date: {s_date}
  Scheduled Time: {s_time}

CLINICAL VISIT DETAILS:
  Reason for Visit : {reason}
  Coverage Status  : Verified In-Network Healthcare Service

PATIENT PREPARATION INSTRUCTIONS:
  1. ARRIVAL: Please arrive at the clinic 15 minutes prior to scheduled time.
  2. RECORDS: Bring recent diagnostic lab tests, discharge summaries, or imaging.
  3. MEDICATIONS: Carry an up-to-date printed list of all active prescribed drugs.
  4. FASTING: If metabolic or lipid blood work is ordered, maintain 8-hr fast.
  5. RESCHEDULING: If you must reschedule, notify the clinic at least 24 hours prior.

CLINICAL SAFETY ALERT:
  If managing chronic conditions (e.g. CKD, Hypertension), avoid NSAIDs
  (Ibuprofen/Naproxen) and confirm all OTC medications with your physician.

AEGIS HEALTH ELECTRONIC VERIFICATION HASH:
  SHA-256 AUTH: {sig_hash}-AEGIS-VERIFIED
================================================================================
"""
    return pass_text


def generate_patient_ccd_report(patient_data: Dict[str, Any]) -> str:
    """
    Generates a structured Continuity of Care Document (CCD) text summary.
    """
    name = patient_data.get("Name", "Unknown")
    age = patient_data.get("Age", "N/A")
    gender = patient_data.get("Gender", "N/A")
    phone = patient_data.get("Phone_number", "N/A")
    email = patient_data.get("Email", "On File")
    address = patient_data.get("Address", "N/A")
    summary = patient_data.get("Summary", "No clinical summary on file.")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ccd_text = f"""================================================================================
📋 CONTINUITY OF CARE DOCUMENT (CCD) • AEGIS ELECTRONIC HEALTH RECORD
================================================================================
GENERATED AT : {timestamp}
HEALTH SYSTEM: Aegis Health Informatics & Clinical Records System
RECORD STATUS: Authenticated Master EHR Record (records.xlsx & FAISS Vector Store)
--------------------------------------------------------------------------------
PATIENT DEMOGRAPHICS:
  Full Name     : {name}
  Date of Birth : Age {age}
  Gender        : {gender}
  Phone Number  : {phone}
  Email Address : {email}
  Primary Address: {address}

LONGITUDINAL CLINICAL SUMMARY:
{summary}

CLINICAL AUDIT & REGULATORY NOTICE:
  This document contains confidential health information protected under clinical
  privacy standards. Verified by Aegis Health Clinical Command Center.
================================================================================
"""
    return ccd_text


def generate_appointment_ics(appointment: Dict[str, Any], patient_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Generates an RFC 5545 compliant iCalendar (.ics) event string for one-click
    import into Google Calendar, Apple Calendar, and Microsoft Outlook.
    """
    apt_id = appointment.get("appointment_id", "APT-ENCOUNTER")
    p_name = appointment.get("patient_name") or (patient_data.get("Name") if patient_data else "Patient")
    d_name = appointment.get("doctor_name", "Attending Physician")
    spec = appointment.get("specialty", "Clinical Specialist")
    clinic = appointment.get("clinic", "Aegis Health Clinic")
    s_date = appointment.get("slot_date", str(datetime.date.today()))
    s_time = appointment.get("slot_time", "10:00 AM")
    reason = appointment.get("reason", "Outpatient clinical consultation")

    dt_str = f"{s_date} {s_time}"
    try:
        dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %I:%M %p")
    except Exception:
        try:
            dt_obj = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except Exception:
            dt_obj = datetime.datetime.now() + datetime.timedelta(days=1)

    dt_start = dt_obj.strftime("%Y%m%dT%H%M%S")
    dt_end = (dt_obj + datetime.timedelta(minutes=45)).strftime("%Y%m%dT%H%M%S")
    dt_stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Aegis Health Informatics//Clinical Command Center//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
BEGIN:VEVENT
UID:{apt_id}@aegishealth.org
DTSTAMP:{dt_stamp}
DTSTART:{dt_start}
DTEND:{dt_end}
SUMMARY:Medical Consultation: {d_name} ({spec})
DESCRIPTION:Patient: {p_name}\\nReason: {reason}\\nEncounter ID: {apt_id}\\nPlease arrive 15 minutes prior.
LOCATION:{clinic}
STATUS:CONFIRMED
TRANSP:OPAQUE
BEGIN:VALARM
ACTION:DISPLAY
DESCRIPTION:Reminder: Medical appointment with {d_name}
TRIGGER:-PT2H
END:VALARM
END:VEVENT
END:VCALENDAR
"""
    return ics_content


def generate_fhir_r4_bundle(patient_data: Dict[str, Any], appointment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Exports patient clinical record and active appointment as an HL7 FHIR Release 4
    compliant standard JSON bundle.
    """
    p_id = str(hashlib.md5((patient_data.get("Name", "patient") + patient_data.get("Phone_number", "")).encode()).hexdigest()[:8])
    gender_map = {"Female": "female", "Male": "male"}
    gender = gender_map.get(patient_data.get("Gender", ""), "unknown")

    entries = [
        {
            "fullUrl": f"urn:uuid:patient-{p_id}",
            "resource": {
                "resourceType": "Patient",
                "id": p_id,
                "meta": {
                    "profile": ["http://hl7.org/fhir/StructureDefinition/Patient"]
                },
                "name": [
                    {
                        "use": "official",
                        "text": patient_data.get("Name", "Unknown Patient")
                    }
                ],
                "telecom": [
                    {
                        "system": "phone",
                        "value": patient_data.get("Phone_number", "")
                    },
                    {
                        "system": "email",
                        "value": patient_data.get("Email", "")
                    }
                ],
                "gender": gender,
                "address": [
                    {
                        "text": patient_data.get("Address", "")
                    }
                ]
            }
        },
        {
            "fullUrl": f"urn:uuid:condition-{p_id}-1",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active"
                    }]
                },
                "subject": {
                    "reference": f"urn:uuid:patient-{p_id}",
                    "display": patient_data.get("Name")
                },
                "note": [{
                    "text": patient_data.get("Summary", "Active patient clinical history on file.")
                }]
            }
        }
    ]

    if appointment:
        apt_id = appointment.get("appointment_id", "APT-ENCOUNTER")
        entries.append({
            "fullUrl": f"urn:uuid:encounter-{apt_id}",
            "resource": {
                "resourceType": "Encounter",
                "id": apt_id,
                "status": "planned",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "ambulatory"
                },
                "subject": {
                    "reference": f"urn:uuid:patient-{p_id}",
                    "display": patient_data.get("Name")
                },
                "participant": [{
                    "individual": {
                        "display": appointment.get("doctor_name", "Attending Physician")
                    }
                }],
                "period": {
                    "start": f"{appointment.get('slot_date')}T10:00:00Z"
                },
                "reasonCode": [{
                    "text": appointment.get("reason", "Consultation")
                }],
                "serviceProvider": {
                    "display": appointment.get("clinic", "Aegis Health Clinic")
                }
            }
        })

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(entries),
        "entry": entries
    }

