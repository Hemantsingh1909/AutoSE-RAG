import pytest
from app.eval.metrics import EvaluationMetrics
from app.eval.benchmark_data import BENCHMARK_REQUIREMENTS
from app.eval.experiment_runner import ExperimentRunner


def test_benchmark_dataset_integrity():
    assert len(BENCHMARK_REQUIREMENTS) == 15
    for item in BENCHMARK_REQUIREMENTS:
        assert "id" in item
        assert "requirement" in item
        assert "constraints" in item
        assert "relevant_documents" in item
        assert "hidden_oracle_tests" in item
        assert len(item["constraints"]) >= 3


def test_granular_constraint_coverage():
    code = """
def validate_sensor(val):
    if val < 0.5 or val > 4.5:
        return False, "ELECTRICAL"
    return True, "PLAUSIBLE"
"""
    constraints = [
        {"id": "C1", "description": "Lower boundary 0.5", "check": "0.5"},
        {"id": "C2", "description": "Upper boundary 4.5", "check": "4.5"},
        {"id": "C3", "description": "Detect fault", "check": "ELECTRICAL"},
        {"id": "C4", "description": "Missing check", "check": "UNKNOWN_TAG"},
    ]
    res = EvaluationMetrics.calculate_granular_constraint_coverage(code, constraints)
    assert res["satisfied_count"] == 3
    assert res["coverage_rate"] == 0.75


def test_retrieval_ranking_metrics():
    evidence = [
        {"source": "target.md"},
        {"source": "wrong.md"},
        {"source": "target.md"},
        {"source": "wrong.md"},
    ]
    m = EvaluationMetrics.calculate_retrieval_metrics(evidence, "target.md", top_k=4)
    assert m["precision_at_1"] == 1.0
    assert m["precision_at_4"] == 0.5
    assert m["mrr"] == 1.0
    assert m["ndcg_at_4"] > 0.0


def test_experiment_runner_two_phases():
    runner = ExperimentRunner()
    exp1 = runner.run_retrieval_experiment(max_items=2)
    exp2 = runner.run_generation_experiment(max_items=2)

    assert "summary" in exp1
    assert "summary" in exp2
    assert "tfidf_sparse" in exp1["summary"]
    assert "no_rag" in exp2["summary"]
    assert "agentic_rag" in exp2["summary"]
