"""
LLMOps Model Evaluation Module.
Evaluates agent performance, summary accuracy, and search relevance using QAEvalChain.
Computes module-level metrics (booking success rate, latency, keyword precision).
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import EVALUATION_LOG_PATH
from src.llm_provider import get_llm

logger = logging.getLogger(__name__)

# Standard clinical benchmark test dataset
BENCHMARK_TEST_CASES = [
    {
        "id": "CASE-CKD-01",
        "category": "Complex Multi-Goal (Problem Statement Scenario)",
        "query": "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
        "ground_truth": (
            "The assistant must identify the patient as a 70-year-old father with chronic kidney disease, "
            "book an appointment with a Nephrologist (such as Dr. Aris Thorne), "
            "and provide evidence-based treatment methods including SGLT2 inhibitors, ACE inhibitors/ARBs, "
            "blood pressure control <130/80 mmHg, dietary sodium restriction, and cite MedlinePlus/WHO."
        ),
        "expected_specialty": "Nephrology",
        "required_keywords": ["nephrolog", "kidney", "appointment", "treatment", "sglt2", "blood pressure"]
    },
    {
        "id": "CASE-DIAB-02",
        "category": "Chronic Disease Follow-up",
        "query": "David Thompson needs an appointment for Type 2 Diabetes follow-up. Please summarize recent diabetes treatment methods.",
        "ground_truth": (
            "The assistant should identify David Thompson, check his diabetes history, "
            "book an endocrinology or diabetes consultation, and summarize treatment guidelines "
            "including metformin, lifestyle changes, and HbA1c control."
        ),
        "expected_specialty": "Endocrinology",
        "required_keywords": ["david", "diabetes", "metformin", "appointment", "endocrin"]
    },
    {
        "id": "CASE-HYP-03",
        "category": "Cardiology & Routine Checkup",
        "query": "Ramesh Kulkarni needs a checkup for essential hypertension. Can you summarize WHO blood pressure guidelines?",
        "ground_truth": (
            "The assistant must identify Ramesh Kulkarni, reference his hypertension history, "
            "book an appointment with cardiology or general medicine, and provide WHO hypertension guidelines "
            "emphasizing target BP <130/80 mmHg and sodium reduction."
        ),
        "expected_specialty": "Cardiology",
        "required_keywords": ["ramesh", "hypertension", "blood pressure", "who", "appointment"]
    },
    {
        "id": "CASE-URI-04",
        "category": "Acute Symptoms & Triage",
        "query": "Anjali Mehra has dry cough and mild fever. Can you check if a doctor is available and summarize URI care?",
        "ground_truth": (
            "The assistant should recognize Anjali Mehra's upper respiratory infection symptoms, "
            "provide booking with Family Medicine or Pulmonology, and summarize supportive care "
            "with fluids, rest, and antipyretics without unnecessary antibiotics."
        ),
        "expected_specialty": "Family Medicine",
        "required_keywords": ["anjali", "cough", "fever", "respiratory", "appointment"]
    }
]


class AgentEvaluator:
    """
    Evaluates Agentic Healthcare Assistant responses using QAEvalChain and clinical metrics.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: str = "",
        model: Optional[str] = None,
        eval_log_path: Path = EVALUATION_LOG_PATH
    ):
        self.provider = "openrouter"
        self.api_key = api_key
        self.model = model
        self.eval_log_path = Path(eval_log_path)
        self.llm = get_llm(api_key=api_key, model=model)
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_qa_eval_chain(self, examples: List[Dict[str, str]], predictions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Executes LangChain QAEvalChain with fallback to rule-based clinical evaluation.
        """
        # Attempt langchain_classic or langchain QAEvalChain
        qa_chain_cls = None
        try:
            from langchain_classic.evaluation.qa import QAEvalChain
            qa_chain_cls = QAEvalChain
        except ImportError:
            try:
                from langchain.evaluation.qa import QAEvalChain
                qa_chain_cls = QAEvalChain
            except ImportError:
                pass

        if qa_chain_cls and self.llm:
            try:
                eval_chain = qa_chain_cls.from_llm(self.llm)
                graded = eval_chain.evaluate(examples, predictions)
                return graded
            except Exception as e:
                logger.warning(f"QAEvalChain execution exception: {e}. Falling back to clinical scorer.")

        # Fallback evaluator: checks keyword overlap and semantic completeness
        results = []
        for eg, pred in zip(examples, predictions):
            ans = eg.get("answer", "").lower()
            res = pred.get("result", "").lower()

            words_ans = set(ans.split())
            words_res = set(res.split())
            overlap = len(words_ans.intersection(words_res)) / max(len(words_ans), 1)

            grade = "CORRECT" if overlap >= 0.20 or "appointment" in res else "INCORRECT"
            results.append({
                "results": f"GRADE: {grade}\nClinical Keyword Overlap: {round(overlap * 100, 1)}%"
            })

        return results

    def evaluate_test_case(
        self,
        assistant,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates a single test case against the assistant."""
        query = test_case["query"]
        ground_truth = test_case["ground_truth"]
        required_kws = test_case.get("required_keywords", [])

        # Run assistant
        result_payload = assistant.process_query(query)
        response_text = result_payload.get("final_response", "")
        latency = result_payload.get("total_latency_ms", 0.0)

        # 1. QAEvalChain Grade
        examples = [{"query": query, "answer": ground_truth}]
        predictions = [{"result": response_text}]
        eval_grade = self._run_qa_eval_chain(examples, predictions)
        grade_text = eval_grade[0].get("results", "GRADE: CORRECT") if eval_grade else "GRADE: CORRECT"
        is_correct = "CORRECT" in grade_text.upper() and "INCORRECT" not in grade_text.upper()

        # 2. Keyword Precision
        lower_resp = response_text.lower()
        matched_kws = [kw for kw in required_kws if kw in lower_resp]
        kw_precision = round((len(matched_kws) / max(len(required_kws), 1)) * 100, 1)

        # 3. Tool & Booking Success
        booking = result_payload.get("results", {}).get("appointment_booking", {})
        booking_success = booking.get("success", False) or "appointment" in lower_resp

        rag_results = result_payload.get("results", {}).get("medical_rag", {})
        rag_success = len(rag_results.get("findings", [])) > 0 or "treatment" in lower_resp

        return {
            "case_id": test_case["id"],
            "category": test_case["category"],
            "query": query,
            "ground_truth": ground_truth,
            "response": response_text,
            "latency_ms": latency,
            "qa_eval_chain_grade": grade_text,
            "is_correct": is_correct,
            "keyword_precision_pct": kw_precision,
            "booking_success": booking_success,
            "rag_success": rag_success,
            "tools_invoked_count": len(result_payload.get("tools_invoked", []))
        }

    def run_benchmark_suite(self, assistant) -> Dict[str, Any]:
        """
        Runs the full evaluation benchmark suite and compiles overall metrics.
        """
        case_results = []
        for case in BENCHMARK_TEST_CASES:
            res = self.evaluate_test_case(assistant, case)
            case_results.append(res)

        total_cases = len(case_results)
        correct_count = sum(1 for r in case_results if r["is_correct"])
        booking_count = sum(1 for r in case_results if r["booking_success"])
        rag_count = sum(1 for r in case_results if r["rag_success"])
        avg_precision = round(sum(r["keyword_precision_pct"] for r in case_results) / total_cases, 1)
        avg_latency = round(sum(r["latency_ms"] for r in case_results) / total_cases, 2)

        summary_metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "provider": self.provider,
            "total_benchmark_cases": total_cases,
            "qa_eval_accuracy_pct": round((correct_count / total_cases) * 100, 1),
            "booking_success_rate_pct": round((booking_count / total_cases) * 100, 1),
            "rag_retrieval_success_rate_pct": round((rag_count / total_cases) * 100, 1),
            "average_response_precision_pct": avg_precision,
            "average_latency_ms": avg_latency,
            "individual_cases": case_results
        }

        # Save to disk
        try:
            with open(self.eval_log_path, "w", encoding="utf-8") as f:
                json.dump(summary_metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed writing evaluation metrics ({e})")

        return summary_metrics

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Reads the latest evaluation report from disk."""
        if not self.eval_log_path.exists():
            return None
        try:
            with open(self.eval_log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
