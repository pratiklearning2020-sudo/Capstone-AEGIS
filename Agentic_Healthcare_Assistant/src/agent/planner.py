"""
Agent Planning and Goal Decomposition Engine.
Interprets multi-step patient queries, extracts clinical intents,
and generates sequential sub-goals mapped to appropriate tools and APIs.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import PLANNER_PROMPT

logger = logging.getLogger(__name__)


class ClinicalPlanner:
    """
    Decomposes multi-intent clinical queries into structured, sequential sub-goals.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def plan(self, user_query: str, patient_context: str = "") -> List[Dict[str, Any]]:
        """
        Generates a sequence of sub-goals to fulfill the user's healthcare request.
        """
        # Formulate prompt
        prompt_text = PLANNER_PROMPT.format(
            user_query=user_query,
            patient_context=patient_context or "No active patient context."
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            content = response.content if hasattr(response, "content") else str(response)
            parsed_plan = self._parse_plan_json(content)
            if parsed_plan:
                if not self.has_booking_intent(user_query):
                    # Strictly enforce: do not book unless user explicitly requested booking
                    parsed_plan = [
                        sg for sg in parsed_plan
                        if sg.get("tool") not in ["doctor_schedule_api", "doctor_booking_tool"]
                        and sg.get("action") not in ["book_appointment", "reserve_slot"]
                    ]
                    for idx, sg in enumerate(parsed_plan):
                        sg["step"] = idx + 1
                return parsed_plan
        except Exception as e:
            logger.warning(f"Error calling LLM for planning ({e}). Using deterministic rule-based planner.")

        return self._generate_rule_based_plan(user_query)

    def _parse_plan_json(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """Extracts and validates JSON sub-goals array from LLM response."""
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                sub_goals = data.get("sub_goals")
                if isinstance(sub_goals, list) and len(sub_goals) > 0:
                    return sub_goals
            except Exception:
                pass
        return None

    @staticmethod
    def has_booking_intent(query: str) -> bool:
        """
        Determines whether the user explicitly requested to book, schedule, or reserve an appointment.
        Prevents booking doctors for purely informational, symptom, or medication queries.
        """
        q_lower = query.lower()

        # Negative check: purely informational questions about doctor/specialty definitions
        if re.search(r"\b(what does a|who is a|explain the role of|difference between)\b", q_lower):
            return False

        # Explicit booking action patterns
        booking_patterns = [
            r"\b(book|booking|reserve)\b",
            r"\bschedule\b.*\b(appointment|consultation|visit|slot|doctor|specialist|checkup)\b",
            r"\b(want|need|like|request)\s+(to\s+)?(book|schedule|make|reserve|get|have)\b",
            r"\b(make|reserve|get|set up)\s+(an?\s+)?(appointment|consultation|slot|visit)\b",
            r"\b(see|consult|visit)\s+(a|an|with|the)?\s*(doctor|physician|specialist|nephrologist|cardiologist|endocrinologist|neurologist|pulmonologist|dermatologist)\b",
            r"\bappointment\s+(for|with)\b",
            r"\bneeds?\s+(an?\s+)?(appointment|checkup|consultation|visit)\b",
            r"\b(check|find|is there)\s+(if\s+)?(a\s+)?doctor\s+is\s+available\b",
        ]

        for pattern in booking_patterns:
            if re.search(pattern, q_lower):
                return True

        return False

    def _generate_rule_based_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Robust rule-based goal decomposition guaranteeing deterministic planning
        for the benchmark scenario and other clinical queries.
        """
        q_lower = query.lower()
        sub_goals = []
        step = 1

        # Step 1: Patient Context Identification
        sub_goals.append({
            "step": step,
            "goal": "Identify patient profile and demographic context",
            "tool": "records_manager",
            "action": "identify_patient_context",
            "description": "Extract patient demographics, conditions, and verify or create EHR profile."
        })
        step += 1

        # Step 2: Medical History Retrieval
        sub_goals.append({
            "step": step,
            "goal": "Retrieve patient medical history and previous records",
            "tool": "ehr_database",
            "action": "retrieve_patient_history",
            "description": "Query EHR database and FAISS vector store for past clinical notes, vitals, and diagnoses."
        })
        step += 1

        # Step 3: Appointment Booking / Doctor Calendar Query (ONLY when explicitly requested)
        if self.has_booking_intent(query):
            specialty = "Family Medicine"
            if "neuro" in q_lower:
                specialty = "Neurology"
            elif "derm" in q_lower or "skin" in q_lower:
                specialty = "Dermatology"
            elif "ortho" in q_lower or "bone" in q_lower or "joint" in q_lower:
                specialty = "Orthopedics"
            elif "psych" in q_lower or "mental" in q_lower:
                specialty = "Psychiatry"
            elif "gastro" in q_lower or "stomach" in q_lower or "digest" in q_lower:
                specialty = "Gastroenterology"
            elif "oncol" in q_lower or "cancer" in q_lower:
                specialty = "Oncology"
            elif "pediatr" in q_lower:
                specialty = "Pediatrics"
            elif re.search(r"\b(ent|ear|nose|throat|otolaryngology)\b", q_lower):
                specialty = "ENT"
            elif "geriatr" in q_lower or "elder" in q_lower:
                specialty = "Geriatrics"
            elif "nephrolog" in q_lower or "kidney" in q_lower or "renal" in q_lower:
                specialty = "Nephrology"
            elif "cardio" in q_lower or "heart" in q_lower:
                specialty = "Cardiology"
            elif "diabet" in q_lower or "endocrin" in q_lower:
                specialty = "Endocrinology"
            elif "pulmon" in q_lower or "cough" in q_lower or "lung" in q_lower:
                specialty = "Pulmonology"
            elif "family" in q_lower or "wellness" in q_lower or "costochondritis" in q_lower:
                specialty = "Family Medicine"

            sub_goals.append({
                "step": step,
                "goal": f"Query doctor calendar and book appointment with {specialty}",
                "tool": "doctor_schedule_api",
                "action": "book_appointment",
                "specialty": specialty,
                "description": f"Discover available slots for {specialty} and confirm reservation."
            })
            step += 1

        # Step 4: Medical Information & RAG Search
        # Trigger RAG for any query that asks for medical info or doesn't have an explicit booking-only intent
        if not self.has_booking_intent(query) or any(w in q_lower for w in ["summarize", "treatment", "methods", "guidelines", "disease", "latest", "what is", "how to treat", "medication", "safe", "take", "ibuprofen", "can i"]):
            disease_term = "chronic kidney disease" if ("kidney" in q_lower or "ckd" in q_lower) else ("diabetes" if "diabet" in q_lower else ("hypertension" if "hypertens" in q_lower or "bp" in q_lower or "blood pressure" in q_lower else "general clinical guidance"))
            sub_goals.append({
                "step": step,
                "goal": f"Search and summarize latest treatment options via RAG pipeline for {disease_term}",
                "tool": "medical_search_rag",
                "action": "search_and_summarize_treatments",
                "condition": disease_term,
                "description": "Fetch up-to-date guidelines from MedlinePlus, PubMed, and WHO via vector RAG."
            })
            step += 1

        # Step 5: Final Response Synthesis
        sub_goals.append({
            "step": step,
            "goal": "Synthesize comprehensive clinical response",
            "tool": "response_synthesizer",
            "action": "generate_final_response",
            "description": "Synthesize patient findings, booking confirmation, and treatment guidelines into a structured report."
        })

        return sub_goals
