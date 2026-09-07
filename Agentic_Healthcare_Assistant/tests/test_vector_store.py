"""
Tests for MedicalVectorStore (FAISS indexing and semantic similarity search).
"""

from src.tools.vector_store import MedicalVectorStore


def test_vector_store_initialization_and_search():
    vs = MedicalVectorStore()
    docs = vs.similarity_search("kidney disease nephrology", k=2)
    assert len(docs) > 0
    # One of the documents should mention kidney or nephrology
    matched_text = " ".join([d.page_content.lower() for d in docs])
    assert "kidney" in matched_text or "nephrol" in matched_text


def test_add_and_retrieve_patient_summary():
    vs = MedicalVectorStore()
    vs.add_patient_summary(
        patient_name="Vector Test Patient",
        summary="Patient has acute glomerulonephritis and elevated serum creatinine."
    )
    results = vs.similarity_search("glomerulonephritis creatinine", k=2)
    assert len(results) > 0
    assert any("glomerulonephritis" in r.page_content.lower() for r in results)
