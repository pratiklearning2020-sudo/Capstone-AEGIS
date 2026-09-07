"""
Agent Memory and Audit Trace Module.
Maintains:
1. Short-term conversation history.
2. Long-term patient entity context (demographics, diagnoses, appointments).
3. Structured execution traces logging tool usage, sub-goal completion, latency, and status.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AUDIT_LOG_PATH

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Manages short-term conversation context, persistent patient entities,
    and audit traces for observability.
    """

    def __init__(self, audit_log_path: Path = AUDIT_LOG_PATH):
        self.audit_log_path = Path(audit_log_path)
        self.conversation_buffer: List[Dict[str, str]] = []
        self.entity_context: Dict[str, Any] = {
            "current_patient": None,
            "patient_age": None,
            "patient_relationship": None,
            "diagnoses": [],
            "requested_specialty": None,
            "last_booked_appointment": None,
            "active_treatment_notes": None
        }
        self.execution_traces: List[Dict[str, Any]] = []
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the conversational buffer."""
        self.conversation_buffer.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def get_conversation_history(self, max_turns: int = 10) -> str:
        """Formats recent conversational turns into text."""
        recent = self.conversation_buffer[-max_turns:]
        return "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])

    def update_entity(self, key: str, value: Any) -> None:
        """Updates persistent patient entity state."""
        self.entity_context[key] = value

    def get_entity(self, key: str, default: Any = None) -> Any:
        """Retrieves a persistent patient entity value."""
        return self.entity_context.get(key, default)

    def get_patient_context_string(self) -> str:
        """Returns a summarized representation of active patient memory."""
        ctx = self.entity_context
        items = []
        if ctx.get("current_patient"):
            items.append(f"Patient Name: {ctx['current_patient']}")
        if ctx.get("patient_relationship"):
            items.append(f"Relationship: {ctx['patient_relationship']}")
        if ctx.get("patient_age"):
            items.append(f"Age: {ctx['patient_age']}")
        if ctx.get("diagnoses"):
            items.append(f"Known Conditions: {', '.join(ctx['diagnoses'])}")
        if ctx.get("requested_specialty"):
            items.append(f"Specialty Required: {ctx['requested_specialty']}")
        if ctx.get("last_booked_appointment"):
            appt = ctx["last_booked_appointment"]
            items.append(f"Active Appointment: {appt.get('appointment_id')} with {appt.get('doctor_name')} ({appt.get('slot_time')})")

        return "\n".join(items) if items else "No previous patient context recorded in session."

    def log_execution_trace(
        self,
        query: str,
        sub_goals: List[Dict[str, Any]],
        tools_invoked: List[Dict[str, Any]],
        total_latency_ms: float,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Records an agent execution trace for observability, LLMOps monitoring,
        and audit verification.
        """
        trace_record = {
            "trace_id": f"TRC-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "sub_goals": sub_goals,
            "tools_invoked": tools_invoked,
            "latency_ms": round(total_latency_ms, 2),
            "success": success
        }

        self.execution_traces.append(trace_record)

        # Append to JSONL audit file
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")
        except Exception as e:
            logger.warning(f"Failed writing to audit log ({e})")

        return trace_record

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent execution traces."""
        return self.execution_traces[-limit:]

    def clear_session(self) -> None:
        """Clears memory for a new session while preserving file logs."""
        self.conversation_buffer.clear()
        self.entity_context = {
            "current_patient": None,
            "patient_age": None,
            "patient_relationship": None,
            "diagnoses": [],
            "requested_specialty": None,
            "last_booked_appointment": None,
            "active_treatment_notes": None
        }
