import pytest
import json
from pathlib import Path
from app.eval.benchmark_data import BENCHMARK_REQUIREMENTS, BENCHMARK_DIR, REQ_DIR, ORACLE_DIR
from app.eval.metrics import EvaluationMetrics, FailureTaxonomy
from app.engine.agents import CodeGenAgent


def test_layer2_all_requirements_have_valid_schemas():
    assert len(BENCHMARK_REQUIREMENTS) == 15
    for req in BENCHMARK_REQUIREMENTS:
        assert "id" in req
        assert "category" in req
        assert "asil" in req
        assert "requirement" in req
        assert "relevant_documents" in req
        assert "constraints" in req
        assert isinstance(req["constraints"], list)
        assert 3 <= len(req["constraints"]) <= 6, f"Requirement {req['id']} must have between 3 and 6 constraints"
        for c in req["constraints"]:
            assert "id" in c
            assert "description" in c
            assert "check" in c


def test_layer2_all_oracle_files_exist_and_independent():
    for req in BENCHMARK_REQUIREMENTS:
        oracle_path = ORACLE_DIR / f"test_{req['id']}.py"
        assert oracle_path.exists(), f"Oracle test missing for {req['id']}"
        oracle_content = oracle_path.read_text(encoding="utf-8")
        assert "def test_oracle_" in oracle_content, f"Oracle {req['id']} must contain test_oracle_ functions"
        assert "assert" in oracle_content


def test_layer2_all_ground_truth_docs_exist_in_kb():
    kb_dir = Path(__file__).resolve().parent.parent / "knowledge_base"
    existing_docs = {p.name for p in kb_dir.glob("*.md")}

    for req in BENCHMARK_REQUIREMENTS:
        for doc in req["relevant_documents"]:
            assert doc in existing_docs, f"Ground truth doc '{doc}' in {req['id']} does not exist in knowledge_base/"


def test_layer2_anti_leakage_isolation():
    agent = CodeGenAgent()
    req = BENCHMARK_REQUIREMENTS[0]

    # Generator only receives requirement + evidence, NEVER oracle tests or constraint definitions
    gen_out = agent.execute(requirement=req["requirement"], evidence=[])
    code = gen_out["code"]

    # Ensure generated code is standalone and doesn't leak oracle function names
    assert "test_oracle_" not in code
    assert "oracle_file" not in gen_out


def test_layer2_metrics_robustness_on_empty_and_corrupt_data():
    # Empty inputs
    assert EvaluationMetrics.calculate_ast_correctness("") is False
    assert EvaluationMetrics.calculate_hallucination_ugcr("", [], "") == 1.0
    ret_empty = EvaluationMetrics.calculate_retrieval_metrics([], "dummy.md")
    assert ret_empty["mrr"] == 0.0
    assert ret_empty["precision_at_1"] == 0.0

    # Failure taxonomy classification on syntax error
    bad_exec = {"success": False, "stderr": "SyntaxError: invalid syntax", "errors": []}
    f_res = FailureTaxonomy.classify("def broken(", bad_exec, "dummy req")
    assert f_res["type"] == "F1"
