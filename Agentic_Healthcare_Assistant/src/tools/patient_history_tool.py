"""
Patient History Tool.
Enables the agent to retrieve and update patient records from both
the structured Excel database and the FAISS semantic vector store.
"""

from typing import Any, Dict, List, Optional
from src.database.records_manager import RecordsManager
from src.tools.vector_store import MedicalVectorStore


class PatientHistoryTool:
    """Tool for retrieving and managing patient EHR and clinical summaries."""

    def __init__(
        self,
        records_mgr: Optional[RecordsManager] = None,
        vector_store: Optional[MedicalVectorStore] = None
    ):
        self.records_mgr = records_mgr or RecordsManager()
        self.vector_store = vector_store or MedicalVectorStore()

    def get_patient_profile(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieves structured patient details from Excel database."""
        return self.records_mgr.get_patient(identifier)

    def search_patient_history(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Searches past diagnoses, notes, and alerts in the FAISS vectorstore."""
        docs = self.vector_store.similarity_search(query, k=k, filter_category="patient_history")
        return [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]

    def update_history_summary(self, identifier: str, new_summary: str) -> bool:
        """Updates patient summary in Excel and adds to FAISS vectorstore."""
        updated = self.records_mgr.update_patient_summary(identifier, new_summary)
        patient = self.records_mgr.get_patient(identifier)
        p_name = patient["Name"] if patient else identifier
        self.vector_store.add_patient_summary(patient_name=p_name, summary=new_summary)
        return updated
