"""
Doctor Booking Tool.
Exposes appointment scheduling and slot lookup as callable agent tools.
"""

from typing import Any, Dict, List, Optional
from src.database.doctor_schedule import DoctorScheduleAPI


class DoctorBookingTool:
    """Tool enabling the agent to inspect doctor schedules and book appointments."""

    def __init__(self, schedule_api: Optional[DoctorScheduleAPI] = None):
        self.schedule_api = schedule_api or DoctorScheduleAPI()

    def find_slots(self, specialty: str, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Finds open appointment slots for a given specialty."""
        return self.schedule_api.find_available_slots(specialty=specialty, date_str=date)

    def book(
        self,
        patient_name: str,
        patient_phone: str = "",
        specialty: Optional[str] = None,
        doctor_id: Optional[str] = None,
        slot_date: Optional[str] = None,
        slot_time: Optional[str] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """Books an appointment slot with a confirmed appointment ID."""
        return self.schedule_api.book_appointment(
            patient_name=patient_name,
            patient_phone=patient_phone,
            specialty=specialty,
            doctor_id=doctor_id,
            slot_date=slot_date,
            slot_time=slot_time,
            reason=reason
        )
