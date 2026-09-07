"""
Tests for RecordsManager (Excel EHR database operations).
"""

from src.database.records_manager import RecordsManager


def test_records_initialization_and_list():
    mgr = RecordsManager()
    patients = mgr.get_all_patients()
    assert len(patients) >= 4
    names = [p.get("Name") for p in patients]
    assert "Ramesh Kulkarni" in names or "Rahul Negi" in names or "Anjali Mehra" in names


def test_get_patient_by_name_and_phone():
    mgr = RecordsManager()
    pt = mgr.get_patient("Ramesh Kulkarni")
    assert pt is not None
    assert pt.get("Age") == "65" or "Ramesh" in pt.get("Name", "")

    pt_phone = mgr.get_patient("98220-45322")
    assert pt_phone is not None
    assert "Ramesh" in pt_phone.get("Name", "")


def test_add_and_update_patient():
    mgr = RecordsManager()
    test_name = "Test Evaluation Patient"
    mgr.add_patient(
        name=test_name,
        age=45,
        gender="Male",
        phone="+91-99999-88888",
        summary="Initial test summary"
    )

    found = mgr.get_patient(test_name)
    assert found is not None
    assert found["Age"] == "45"

    updated = mgr.update_patient_summary(test_name, "Updated test summary after visit")
    assert updated is True

    found_updated = mgr.get_patient(test_name)
    assert "Updated test summary" in found_updated.get("Summary", "")
