"""
Tests for MedicalSearchTool (MedlinePlus XML, PubMed, and WHO guidelines).
"""

from src.tools.medical_search import MedicalSearchTool


def test_trusted_guidelines_lookup():
    tool = MedicalSearchTool()
    ckd = tool.search_trusted_guidelines("chronic kidney disease")
    assert ckd is not None
    assert "SGLT2" in ckd["summary"] or "Renoprotective" in ckd["summary"]

    diab = tool.search_trusted_guidelines("type 2 diabetes")
    assert diab is not None
    assert "Metformin" in diab["summary"]


def test_consolidated_medical_search():
    tool = MedicalSearchTool()
    res = tool.search("chronic kidney disease latest treatment")
    assert res["results_count"] > 0
    assert len(res["findings"]) > 0
    assert len(res["formatted_summary"]) > 50
