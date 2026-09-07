"""
Tests for DoctorScheduleAPI (slot discovery, booking, conflict detection).
"""

from src.database.doctor_schedule import DoctorScheduleAPI


def test_list_doctors_and_specialties():
    api = DoctorScheduleAPI()
    docs = api.list_doctors()
    assert len(docs) >= 5

    nephs = api.list_doctors("Nephrology")
    assert len(nephs) >= 1
    assert any("Dr. Aris Thorne" in d["name"] for d in nephs)


def test_find_available_slots():
    api = DoctorScheduleAPI()
    slots = api.find_available_slots(specialty="Nephrology")
    assert len(slots) > 0
    assert slots[0]["specialty"] == "Nephrology"


def test_booking_and_double_booking_prevention():
    api = DoctorScheduleAPI()
    # Book slot
    result = api.book_appointment(
        patient_name="John Doe Test",
        specialty="Nephrology",
        slot_date="2026-10-15",
        slot_time="09:30 AM",
        reason="Kidney function test"
    )
    assert result["success"] is True
    appt_id = result["appointment_id"]
    assert appt_id.startswith("APT-")

    # Try booking identical doctor, date, and slot
    doc_id = result["appointment"]["doctor_id"]
    duplicate = api.book_appointment(
        patient_name="Jane Doe Test",
        doctor_id=doc_id,
        slot_date="2026-10-15",
        slot_time="09:30 AM"
    )
    assert duplicate["success"] is False
    assert "already booked" in duplicate["error"].lower()

    # Cancel appointment
    cancelled = api.cancel_appointment(appt_id)
    assert cancelled is True


def test_unavailable_specialty_handling_and_relatable_doctors():
    api = DoctorScheduleAPI()
    initial_appts = len(api.list_appointments())

    # Try booking an unavailable specialist (e.g. Neurologist)
    res = api.book_appointment(
        patient_name="Alex Turner",
        patient_phone="+1-555-0199",
        specialty="Neurology",
        reason="Severe migraine and numbness"
    )

    # Must NOT book with any random doctor!
    assert res["success"] is False
    assert res["doctor_available"] is False
    assert res["appointment"] is None
    assert res["requested_specialty"] == "Neurology"
    assert "no specialist in 'neurology'" in res["error"].lower()

    # Must return relatable doctors (e.g. Family Medicine, Geriatrics)
    relatable = res.get("relatable_doctors", [])
    assert len(relatable) >= 1
    doc_specs = [d["specialty"] for d in relatable]
    assert "Family Medicine" in doc_specs

    # Verify each relatable doctor has a clinical rationale
    for doc in relatable:
        assert "relevance_reason" in doc
        assert len(doc["relevance_reason"]) > 10

    # Ensure no appointment record was persisted
    final_appts = len(api.list_appointments())
    assert final_appts == initial_appts
