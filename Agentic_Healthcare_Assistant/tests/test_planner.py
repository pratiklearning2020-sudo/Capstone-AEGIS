"""
Tests for ClinicalPlanner (agent goal decomposition and tool mapping).
"""

from src.agent.planner import ClinicalPlanner
from src.llm_provider import MedicalSimulatorLLM


def test_planner_sub_goals():
    llm = MedicalSimulatorLLM()
    planner = ClinicalPlanner(llm)
    query = "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?"

    sub_goals = planner.plan(query)
    assert len(sub_goals) >= 4

    tools = [sg["tool"] for sg in sub_goals]
    assert "records_manager" in tools or "ehr_database" in tools
    assert "doctor_schedule_api" in tools
    assert "medical_search_rag" in tools


def test_planner_skips_booking_for_informational_query():
    llm = MedicalSimulatorLLM()
    planner = ClinicalPlanner(llm)
    query = "What are the recommended medications and dietary guidelines for chronic kidney disease?"

    sub_goals = planner.plan(query)
    tools = [sg["tool"] for sg in sub_goals]

    # Must NOT contain doctor_schedule_api
    assert "doctor_schedule_api" not in tools
    assert "medical_search_rag" in tools

