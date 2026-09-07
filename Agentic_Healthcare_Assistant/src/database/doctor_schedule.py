"""
Doctor Schedule and Appointment Booking API.
Provides doctor directory, slot discovery by specialty, real-time booking,
conflict management, and status tracking.
"""

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import APPOINTMENTS_DB_PATH, DOCTORS_DB_PATH


DEFAULT_DOCTORS = [
    {
        "doctor_id": "DOC-NEPH-01",
        "name": "Dr. Aris Thorne, MD",
        "specialty": "Nephrology",
        "clinic": "Renal & Kidney Care Center, Suite 400",
        "consultation_fee": 150,
        "rating": 4.9,
        "available_slots": ["09:30 AM", "10:30 AM", "02:00 PM", "04:30 PM"]
    },
    {
        "doctor_id": "DOC-NEPH-02",
        "name": "Dr. Priya Sharma, MD",
        "specialty": "Nephrology",
        "clinic": "Metro Nephrology & Dialysis Institute",
        "consultation_fee": 160,
        "rating": 4.8,
        "available_slots": ["11:00 AM", "01:30 PM", "03:30 PM"]
    },
    {
        "doctor_id": "DOC-CARD-01",
        "name": "Dr. Rajesh Patel, MD",
        "specialty": "Cardiology",
        "clinic": "City Heart & Vascular Institute",
        "consultation_fee": 175,
        "rating": 4.9,
        "available_slots": ["09:00 AM", "11:30 AM", "03:00 PM"]
    },
    {
        "doctor_id": "DOC-ENDO-01",
        "name": "Dr. Vikram Sen, MD",
        "specialty": "Endocrinology",
        "clinic": "Advanced Diabetes & Endocrine Center",
        "consultation_fee": 140,
        "rating": 4.7,
        "available_slots": ["10:00 AM", "12:00 PM", "02:30 PM"]
    },
    {
        "doctor_id": "DOC-PULM-01",
        "name": "Dr. Kavita Nair, MD",
        "specialty": "Pulmonology",
        "clinic": "Pulmonary & Respiratory Wellness Clinic",
        "consultation_fee": 145,
        "rating": 4.8,
        "available_slots": ["10:30 AM", "01:00 PM", "04:00 PM"]
    },
    {
        "doctor_id": "DOC-FAM-01",
        "name": "Dr. Megana Lanoi, MD",
        "specialty": "Family Medicine",
        "clinic": "Bridport Family Medicine & Primary Care",
        "consultation_fee": 120,
        "rating": 4.9,
        "available_slots": ["08:30 AM", "10:00 AM", "01:30 PM", "03:00 PM"]
    },
    {
        "doctor_id": "DOC-GER-01",
        "name": "Dr. Harold Vance, MD",
        "specialty": "Geriatrics",
        "clinic": "ElderCare & Comprehensive Geriatric Clinic",
        "consultation_fee": 155,
        "rating": 4.9,
        "available_slots": ["09:30 AM", "11:00 AM", "02:00 PM"]
    }
]


class DoctorScheduleAPI:
    """
    Simulates a hospital doctor scheduling and calendar management system.
    """

    def __init__(
        self,
        doctors_path: Path = DOCTORS_DB_PATH,
        appointments_path: Path = APPOINTMENTS_DB_PATH
    ):
        self.doctors_path = Path(doctors_path)
        self.appointments_path = Path(appointments_path)
        self._initialize_storage()

    def _initialize_storage(self, force_reset: bool = False) -> None:
        self.doctors_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.doctors_path.exists() or force_reset:
            with open(self.doctors_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_DOCTORS, f, indent=2)

        if not self.appointments_path.exists() or force_reset:
            # Seed with one sample initial appointment
            initial_appts = [
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
            with open(self.appointments_path, "w", encoding="utf-8") as f:
                json.dump(initial_appts, f, indent=2)

    def reset_schedule(self) -> None:
        """Resets doctors and appointments to clean baseline state."""
        self._initialize_storage(force_reset=True)

    def list_doctors(self, specialty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns doctors, optionally filtered by specialty (case-insensitive)."""
        with open(self.doctors_path, "r", encoding="utf-8") as f:
            doctors = json.load(f)

        if not specialty:
            return doctors

        spec_clean = specialty.strip().lower()
        return [
            doc for doc in doctors
            if spec_clean in doc.get("specialty", "").lower()
            or doc.get("specialty", "").lower() in spec_clean
        ]

    def get_doctor_by_id(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        doctors = self.list_doctors()
        for d in doctors:
            if d["doctor_id"].upper() == doctor_id.strip().upper():
                return d
        return None

    def list_appointments(
        self,
        status: Optional[str] = None,
        doctor_id: Optional[str] = None,
        patient_phone: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves appointments with deduplication and optional filters."""
        if not self.appointments_path.exists():
            return []

        with open(self.appointments_path, "r", encoding="utf-8") as f:
            try:
                appts = json.load(f)
            except Exception:
                return []

        # Deduplicate identical repeated bookings
        seen_keys = set()
        deduped = []
        for a in reversed(appts):
            k = (
                a.get("patient_name", "").strip().lower(),
                a.get("doctor_id", "").strip(),
                a.get("slot_date", "").strip(),
                a.get("slot_time", "").strip()
            )
            if k not in seen_keys:
                seen_keys.add(k)
                deduped.append(a)
        deduped.reverse()

        filtered = deduped
        if status:
            filtered = [a for a in filtered if a.get("status", "").upper() == status.strip().upper()]
        if doctor_id:
            filtered = [a for a in filtered if a.get("doctor_id", "").upper() == doctor_id.strip().upper()]
        if patient_phone:
            clean_p = patient_phone.replace("-", "").replace(" ", "").replace("+", "")
            filtered = [a for a in filtered if clean_p in a.get("patient_phone", "").replace("-", "").replace(" ", "").replace("+", "")]

        return filtered

    def find_available_slots(
        self,
        specialty: Optional[str] = None,
        date_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds open slots for doctors matching the specialty on the requested date.
        Filters out any slots that are already booked and CONFIRMED.
        """
        matched_doctors = self.list_doctors(specialty=specialty)
        existing_appts = self.list_appointments(status="CONFIRMED")

        booked_keys = {
            f"{a['doctor_id']}_{a['slot_date']}_{a['slot_time']}"
            for a in existing_appts
        }

        if date_str:
            dates_to_check = [date_str]
        else:
            base_d = datetime.date.today() + datetime.timedelta(days=1)
            dates_to_check = [(base_d + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

        for d_str in dates_to_check:
            available = []
            for doc in matched_doctors:
                for slot in doc.get("available_slots", []):
                    key = f"{doc['doctor_id']}_{d_str}_{slot}"
                    if key not in booked_keys:
                        available.append({
                            "doctor_id": doc["doctor_id"],
                            "doctor_name": doc["name"],
                            "specialty": doc["specialty"],
                            "clinic": doc["clinic"],
                            "consultation_fee": doc["consultation_fee"],
                            "date": d_str,
                            "time": slot
                        })
            if available:
                return available

        return []

    def get_relatable_doctors(self, specialty: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recommends close, clinically relatable physicians when a requested specialty
        is not directly available on our hospital roster.
        """
        all_docs = {d["doctor_id"]: dict(d) for d in self.list_doctors()}
        fam_doc = all_docs.get("DOC-FAM-01")   # Dr. Megana Lanoi (Family Medicine)
        ger_doc = all_docs.get("DOC-GER-01")   # Dr. Harold Vance (Geriatrics)
        card_doc = all_docs.get("DOC-CARD-01") # Dr. Rajesh Patel (Cardiology)
        endo_doc = all_docs.get("DOC-ENDO-01") # Dr. Vikram Sen (Endocrinology)
        pulm_doc = all_docs.get("DOC-PULM-01") # Dr. Kavita Nair (Pulmonology)

        spec_lower = (specialty or "").lower().strip()
        relatable = []

        if "neuro" in spec_lower:
            # Neurology
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Initial neurological symptom screening, preliminary cranial nerve & motor reflex exam, "
                    "and expedited formal referral to an affiliated neurology specialist network."
                )
                relatable.append(f_copy)
            if ger_doc:
                g_copy = dict(ger_doc)
                g_copy["relevance_reason"] = (
                    "Specialized evaluation if symptoms involve age-related cognitive changes, memory lapses, "
                    "neuropathy, or balance and gait instability."
                )
                relatable.append(g_copy)

        elif "derm" in spec_lower or "skin" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Primary clinical evaluation of skin lesions, rashes, allergic dermatitis, and outpatient dermatology referral."
                )
                relatable.append(f_copy)

        elif "ortho" in spec_lower or "bone" in spec_lower or "joint" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Initial musculoskeletal assessment, pain management, X-ray ordering, and orthopedic surgery referral."
                )
                relatable.append(f_copy)
            if ger_doc:
                g_copy = dict(ger_doc)
                g_copy["relevance_reason"] = (
                    "Evaluation of osteoarthritis, degenerative joint disease, and fall-risk prevention."
                )
                relatable.append(g_copy)

        elif "psych" in spec_lower or "mental" in spec_lower or "depress" in spec_lower or "anxiet" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Primary mental health assessment, anxiety/depression screening, medical workup, and outpatient psychiatric referral."
                )
                relatable.append(f_copy)

        elif "gastro" in spec_lower or "digest" in spec_lower or "stomach" in spec_lower or "liver" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Primary gastrointestinal assessment, dietary management, and referral to GI endoscopy centers."
                )
                relatable.append(f_copy)
            if endo_doc:
                e_copy = dict(endo_doc)
                e_copy["relevance_reason"] = (
                    "Metabolic and endocrine gastrointestinal disorders (e.g., diabetic gastroparesis, metabolic fatty liver)."
                )
                relatable.append(e_copy)

        elif "ent" in spec_lower or "ear" in spec_lower or "nose" in spec_lower or "throat" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Comprehensive otoscopic and pharyngeal exam, sinusitis care, and ENT specialist referral."
                )
                relatable.append(f_copy)
            if pulm_doc:
                p_copy = dict(pulm_doc)
                p_copy["relevance_reason"] = (
                    "Upper airway, chronic cough, and bronchial allergy evaluation."
                )
                relatable.append(p_copy)

        elif "oncol" in spec_lower or "cancer" in spec_lower or "tumor" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Initial diagnostic workup, baseline blood panels/imaging, and urgent referral to oncology centers."
                )
                relatable.append(f_copy)

        elif "pediatr" in spec_lower or "child" in spec_lower:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Comprehensive primary family care and pediatric wellness examinations."
                )
                relatable.append(f_copy)

        else:
            if fam_doc:
                f_copy = dict(fam_doc)
                f_copy["relevance_reason"] = (
                    "Primary care physician for comprehensive initial health screening, diagnostic workup, and specialist referrals."
                )
                relatable.append(f_copy)
            if ger_doc:
                g_copy = dict(ger_doc)
                g_copy["relevance_reason"] = (
                    "Comprehensive geriatric care and multi-morbidity management for senior patients."
                )
                relatable.append(g_copy)

        return relatable

    def book_appointment(
        self,
        patient_name: str,
        patient_phone: str = "",
        specialty: Optional[str] = None,
        doctor_id: Optional[str] = None,
        slot_date: Optional[str] = None,
        slot_time: Optional[str] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Books an appointment, validating doctor availability and preventing double-booking.
        Strictly enforces specialty matching: if the requested specialist is unavailable,
        DOES NOT book with an unrelated random doctor, but returns relatable alternatives.
        """
        if not slot_date:
            slot_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # 1. Determine target doctor if explicit doctor_id provided
        target_doc = None
        if doctor_id:
            target_doc = self.get_doctor_by_id(doctor_id)
            if not target_doc:
                return {
                    "success": False,
                    "doctor_available": False,
                    "requested_doctor_id": doctor_id,
                    "error": f"Doctor with ID '{doctor_id}' is not found on our active roster.",
                    "available_specialties": sorted(list({d["specialty"] for d in self.list_doctors()})),
                    "relatable_doctors": self.get_relatable_doctors(None),
                    "appointment": None
                }

        # 2. Check if requested specialty exists in our hospital roster
        if not target_doc and specialty:
            matched_specialists = self.list_doctors(specialty=specialty)
            if not matched_specialists:
                # Specialist type is NOT available at this hospital!
                # Do NOT book with anyone else!
                relatable = self.get_relatable_doctors(specialty=specialty)
                all_specialties = sorted(list({d["specialty"] for d in self.list_doctors()}))
                return {
                    "success": False,
                    "doctor_available": False,
                    "requested_specialty": specialty,
                    "error": f"No specialist in '{specialty}' is currently available at our hospital.",
                    "message": (
                        f"We currently do not have an active {specialty} specialist on our immediate staff. "
                        f"To protect patient safety and clinical quality, we do not book with unrelated specialists."
                    ),
                    "available_specialties": all_specialties,
                    "relatable_doctors": relatable,
                    "appointment": None
                }

        chosen_time = slot_time

        if not target_doc:
            # Check open slots for doctors matching the verified specialty
            found_slot = None
            base_date = datetime.datetime.strptime(slot_date, "%Y-%m-%d").date() if slot_date else (datetime.date.today() + datetime.timedelta(days=1))
            for day_offset in range(7):
                curr_date = (base_date + datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
                avail = self.find_available_slots(specialty=specialty, date_str=curr_date)
                if avail:
                    found_slot = avail[0]
                    slot_date = curr_date
                    if not chosen_time:
                        chosen_time = found_slot["time"]
                    target_doc = self.get_doctor_by_id(found_slot["doctor_id"])
                    break

            if not target_doc:
                if specialty:
                    matched = self.list_doctors(specialty=specialty)
                    target_doc = matched[0] if matched else None
                else:
                    # Default to Family Medicine (primary care) when unassigned
                    matched = self.list_doctors(specialty="Family Medicine")
                    target_doc = matched[0] if matched else self.list_doctors()[0]

        if not target_doc:
            return {
                "success": False,
                "doctor_available": False,
                "requested_specialty": specialty,
                "error": f"No doctor could be assigned for specialty '{specialty}'.",
                "relatable_doctors": self.get_relatable_doctors(specialty),
                "appointment": None
            }

        # Determine slot if still not chosen
        if not chosen_time:
            open_slots = self.find_available_slots(specialty=target_doc["specialty"], date_str=slot_date)
            doc_open_slots = [s for s in open_slots if s["doctor_id"] == target_doc["doctor_id"]]
            if doc_open_slots:
                chosen_time = doc_open_slots[0]["time"]
            else:
                chosen_time = target_doc.get("available_slots", ["10:30 AM"])[0]

        # Double-booking check
        existing = self.list_appointments(status="CONFIRMED")
        for a in existing:
            if (
                a.get("doctor_id") == target_doc["doctor_id"]
                and a.get("slot_date") == slot_date
                and a.get("slot_time") == chosen_time
            ):
                return {
                    "success": False,
                    "error": f"Slot {chosen_time} on {slot_date} with {target_doc['name']} is already booked.",
                    "appointment": None
                }

        new_appt_id = f"APT-{target_doc['specialty'][:4].upper()}-{uuid.uuid4().hex[:6].upper()}"
        new_record = {
            "appointment_id": new_appt_id,
            "patient_name": str(patient_name).strip(),
            "patient_phone": str(patient_phone).strip(),
            "doctor_id": target_doc["doctor_id"],
            "doctor_name": target_doc["name"],
            "specialty": target_doc["specialty"],
            "clinic": target_doc["clinic"],
            "slot_date": slot_date,
            "slot_time": chosen_time,
            "reason": str(reason).strip() or f"Consultation for {target_doc['specialty']}",
            "status": "CONFIRMED",
            "created_at": datetime.datetime.now().isoformat()
        }

        # Save to file
        with open(self.appointments_path, "r", encoding="utf-8") as f:
            appts = json.load(f)
        appts.append(new_record)
        with open(self.appointments_path, "w", encoding="utf-8") as f:
            json.dump(appts, f, indent=2)

        return {
            "success": True,
            "appointment_id": new_appt_id,
            "appointment": new_record
        }

    def cancel_appointment(self, appointment_id: str) -> bool:
        """Cancels an existing appointment by ID."""
        with open(self.appointments_path, "r", encoding="utf-8") as f:
            appts = json.load(f)

        found = False
        for a in appts:
            if a.get("appointment_id", "").upper() == appointment_id.strip().upper():
                a["status"] = "CANCELLED"
                a["cancelled_at"] = datetime.datetime.now().isoformat()
                found = True
                break

        if found:
            with open(self.appointments_path, "w", encoding="utf-8") as f:
                json.dump(appts, f, indent=2)
            return True

        return False
