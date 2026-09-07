"""
Vector Store Manager using FAISS.
Indexes and retrieves patient summaries, clinical notes, and medical knowledge.
Provides semantic retrieval for the RAG pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """
    FAISS-based vector store for patient histories and clinical guidelines.
    """

    def __init__(
        self,
        persist_dir: Path = VECTORSTORE_DIR,
        provider: str = "simulator",
        api_key: str = ""
    ):
        self.persist_dir = Path(persist_dir)
        self.embeddings = get_embeddings(provider=provider, api_key=api_key)
        self.vector_store: Optional[FAISS] = None
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Loads saved FAISS index or creates an initial seeded store."""
        index_file = self.persist_dir / "index.faiss"
        if index_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load existing FAISS index ({e}). Creating new index.")

        # Seed initial documents
        initial_docs = [
            Document(
                page_content="Patient Ramesh Kulkarni, 65-year-old male, routine checkup with history of essential hypertension. Vitals stable (BP 130/84). Regular on Telmisartan 40mg OD.",
                metadata={"patient_name": "Ramesh Kulkarni", "category": "patient_history", "phone": "+91-98220-45322"}
            ),
            Document(
                page_content="Patient Anjali Mehra, 33-year-old female, 5-day history of dry cough and mild fever. Diagnosed with Upper Respiratory Infection (J06.9). Symptomatic management with antihistamines and paracetamol.",
                metadata={"patient_name": "Anjali Mehra", "category": "patient_history", "phone": "+91-98180-11245"}
            ),
            Document(
                page_content="Patient David Thompson, 51-year-old male, follow-up for Type 2 Diabetes Mellitus (E11.9). Increased thirst and urination. Plan: Increase metformin dosage to 1000mg BID, HbA1c lab order.",
                metadata={"patient_name": "David Thompson", "category": "patient_history", "phone": "+91-98450-11223"}
            ),
            Document(
                page_content="Patient Rebeca Nagle, 36-year-old female, history of routine wellness and cervical screening. Monitored vitals and lipid profile with Dr. Megana Lanoi.",
                metadata={"patient_name": "Rebeca Nagle", "category": "patient_history", "phone": "+1-541-950-0000"}
            ),
            Document(
                page_content="Senior Patient (Father), 70-year-old male, diagnosed with Chronic Kidney Disease (CKD) Stage 3. Associated hypertension. Requires nephrology consultation, monitoring of eGFR and proteinuria, and renal protective pharmacotherapy (ACEi/ARB, SGLT2i).",
                metadata={"patient_name": "Father (70yo)", "category": "patient_history", "condition": "Chronic Kidney Disease"}
            ),
            Document(
                page_content="Chronic Kidney Disease Clinical Guidelines: First-line management includes blood pressure target <130/80 mmHg, ACE inhibitors or ARBs for albuminuria, SGLT2 inhibitors to preserve kidney function, and strict dietary sodium restriction.",
                metadata={"category": "medical_guideline", "source": "WHO / MedlinePlus"}
            )
        ]

        self.vector_store = FAISS.from_documents(initial_docs, self.embeddings)
        self.save()

    def save(self) -> None:
        """Persists the FAISS index to disk."""
        if self.vector_store:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.persist_dir))

    def add_patient_summary(self, patient_name: str, summary: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Indexes a patient summary or clinical report into the FAISS store."""
        meta = metadata or {}
        meta.update({"patient_name": patient_name, "category": "patient_history"})
        doc = Document(page_content=summary, metadata=meta)

        if self.vector_store:
            self.vector_store.add_documents([doc])
        else:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        self.save()

    def similarity_search(self, query: str, k: int = 3, filter_category: Optional[str] = None) -> List[Document]:
        """Performs semantic similarity search against the vector database."""
        if not self.vector_store:
            return []

        docs = self.vector_store.similarity_search(query, k=k)
        if filter_category:
            docs = [d for d in docs if d.metadata.get("category") == filter_category]
        return docs

    def rebuild_from_records(self, patients: Optional[List[Dict[str, Any]]] = None) -> None:
        """Rebuilds the entire FAISS index using clean patient records and medical guidelines."""
        docs = []
        if patients:
            for p in patients:
                content = f"Patient {p.get('Name')}, {p.get('Age')}-year-old {p.get('Gender')}. {p.get('Summary')}"
                meta = {
                    "patient_name": p.get("Name", ""),
                    "category": "patient_history",
                    "phone": p.get("Phone_number", "")
                }
                docs.append(Document(page_content=content, metadata=meta))

        docs.extend([
            Document(
                page_content="Chronic Kidney Disease Clinical Guidelines: First-line management includes blood pressure target <130/80 mmHg, ACE inhibitors or ARBs for albuminuria, SGLT2 inhibitors to preserve kidney function, and strict dietary sodium restriction.",
                metadata={"category": "medical_guideline", "source": "WHO / MedlinePlus"}
            ),
            Document(
                page_content="WHO Hypertension Guidelines: Diagnosis confirmed when systolic BP is >=140 mmHg or diastolic BP is >=90 mmHg on two different days. First-line pharmacotherapy includes ACE inhibitors, ARBs, CCBs, or thiazide diuretics.",
                metadata={"category": "medical_guideline", "source": "WHO Clinical Guidelines"}
            ),
            Document(
                page_content="Upper Respiratory Tract Infection Clinical Guidelines: Primarily viral etiology. Antibiotics are not recommended. Supportive therapy includes antipyretics, warm fluids, rest, and nasal saline irrigation.",
                metadata={"category": "medical_guideline", "source": "MedlinePlus / CDC"}
            )
        ])

        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.save()

