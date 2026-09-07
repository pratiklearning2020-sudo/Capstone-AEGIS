"""
Tests for AgentEvaluator and LLMOps evaluation pipeline.
"""

from src.agent.orchestrator import AgenticHealthcareAssistant
from src.evaluation.evaluator import AgentEvaluator, BENCHMARK_TEST_CASES


def test_evaluator_benchmark_run():
    assistant = AgenticHealthcareAssistant(provider="simulator")
    evaluator = AgentEvaluator(provider="simulator")

    # Run single case evaluation
    single_res = evaluator.evaluate_test_case(assistant, BENCHMARK_TEST_CASES[0])
    assert single_res["is_correct"] is True
    assert single_res["booking_success"] is True
    assert single_res["keyword_precision_pct"] > 40.0

    # Run complete benchmark suite
    summary = evaluator.run_benchmark_suite(assistant)
    assert summary["total_benchmark_cases"] == len(BENCHMARK_TEST_CASES)
    assert summary["qa_eval_accuracy_pct"] >= 75.0
    assert summary["booking_success_rate_pct"] >= 75.0
    assert summary["average_latency_ms"] > 0
