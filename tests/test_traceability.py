import pytest
from app.engine.traceability import TraceabilityEngine


def test_ast_symbol_extraction():
    code = """
class SensorValidator:
    def __init__(self):
        pass

    def check_range(self, val: float) -> bool:
        return val > 0

def standalone_helper():
    return True
"""
    symbols = TraceabilityEngine.extract_ast_symbols(code)
    assert "SensorValidator" in symbols["classes"]
    assert "check_range" in symbols["functions"]
    assert "standalone_helper" in symbols["functions"]


def test_test_case_extraction():
    tests = """
def test_nominal():
    assert True

def test_boundary_upper():
    assert True
"""
    test_cases = TraceabilityEngine.extract_test_cases(tests)
    assert "test_nominal" in test_cases
    assert "test_boundary_upper" in test_cases


def test_traceability_matrix_generation():
    code = "def validate_signal(x): return x > 0"
    tests = "def test_validate_signal(): assert validate_signal(1) is True"
    evidence = [{"id": "doc-1", "source": "safety.md", "text": "Validation rules"}]
    exec_res = {"success": True, "passed_count": 1, "failed_count": 0, "pass_rate": 1.0}

    matrix = TraceabilityEngine.generate_matrix(
        requirement="The system shall validate signal > 0",
        evidence=evidence,
        code=code,
        tests=tests,
        execution_result=exec_res,
        human_approved=True,
    )

    assert matrix["completeness_score"] == 1.0
    assert matrix["verified"] is True
    assert len(matrix["matrix_mappings"]) == 1
