"""
End-to-end tests for AgenticHealthcareAssistant.
Tests the exact scenario from the problem statement:
"My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?"
"""

from src.agent.orchestrator import AgenticHealthcareAssistant


def test_benchmark_scenario_workflow():
    assistant = AgenticHealthcareAssistant(provider="simulator")
    query = "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?"

    response_payload = assistant.process_query(query)

    assert "final_response" in response_payload
    final_text = response_payload["final_response"]

    # 1. Verify patient context identification
    assert "father" in final_text.lower() or "70" in final_text.lower() or "kidney" in final_text.lower()

    # 2. Verify appointment booking was triggered
    booking = response_payload["results"]["appointment_booking"]
    assert booking.get("success") is True or "appointment" in final_text.lower()
    assert "nephrolog" in str(booking).lower() or "thorne" in str(booking).lower()

    # 3. Verify RAG search executed
    rag = response_payload["results"]["medical_rag"]
    assert len(rag.get("findings", [])) > 0 or len(rag.get("vector_guidelines", [])) > 0

    # 4. Verify tools were invoked and recorded in audit traces
    tools = response_payload["tools_invoked"]
    assert len(tools) >= 4
    assert any(t["tool"] == "doctor_schedule_api" for t in tools)
    assert any(t["tool"] == "medical_search_rag" for t in tools)

    # 5. Verify memory retained the appointment and patient state
    assert assistant.memory.entity_context["patient_relationship"] == "Father"
    assert assistant.memory.entity_context["patient_age"] == "70"
    assert assistant.memory.entity_context["last_booked_appointment"] is not None


def test_informational_query_does_not_book_doctor():
    """Verify that purely informational or medication queries do NOT trigger unwanted doctor booking."""
    assistant = AgenticHealthcareAssistant(provider="simulator")
    query = "Can I take ibuprofen for headache if I have chronic kidney disease?"

    response_payload = assistant.process_query(query)
    final_text = response_payload["final_response"]
    tools = [t["tool"] for t in response_payload["tools_invoked"]]

    # 1. doctor_schedule_api must NOT have been called
    assert "doctor_schedule_api" not in tools

    # 2. No fake appointment booking card or reservation ID should be in response
    assert "CONFIRMED APPOINTMENT RESERVATION" not in final_text
    assert "APT-" not in final_text

    # 3. Clinical guidance and safety contraindication must be present
    assert "ibuprofen" in final_text.lower()
    assert "acetaminophen" in final_text.lower() or "contraindication" in final_text.lower()

