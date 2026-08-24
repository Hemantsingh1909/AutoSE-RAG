import pytest
from app.engine.sandbox import SandboxedExecutor
from app.engine.agents import CodeGenAgent, TestGenAgent, CriticAgent
from app.engine.workflow import AgenticWorkflow
from app.rag.hybrid_retriever import HybridRetriever


def test_sandbox_passing_execution():
    executor = SandboxedExecutor(timeout_seconds=5.0)
    code = """def add(a: int, b: int) -> int:\n    return a + b\n"""
    tests = """def test_add():\n    assert add(2, 3) == 5\n"""
    res = executor.execute_test_suite(code, tests)
    assert res["success"] is True
    assert res["passed_count"] == 1
    assert res["failed_count"] == 0
    assert res["pass_rate"] == 1.0


def test_sandbox_failing_execution():
    executor = SandboxedExecutor(timeout_seconds=5.0)
    code = """def add(a: int, b: int) -> int:\n    return a - b\n"""
    tests = """def test_add():\n    assert add(2, 3) == 5\n"""
    res = executor.execute_test_suite(code, tests)
    assert res["success"] is False
    assert res["failed_count"] == 1
    assert res["pass_rate"] == 0.0


def test_critic_evaluation_pass_and_fail():
    critic = CriticAgent()
    pass_res = {"success": True, "passed_count": 2, "failed_count": 0, "pass_rate": 1.0}
    verdict_pass = critic.evaluate("dummy req", "dummy code", "dummy tests", pass_res)
    assert verdict_pass["verdict"] == "ACCEPT"
    assert verdict_pass["repair_needed"] is False

    fail_res = {"success": False, "passed_count": 0, "failed_count": 1, "pass_rate": 0.0, "errors": ["AssertionError: 2 != 3"]}
    verdict_fail = critic.evaluate("dummy req", "dummy code", "dummy tests", fail_res)
    assert verdict_fail["verdict"] == "REJECT"
    assert verdict_fail["repair_needed"] is True


def test_agentic_workflow_end_to_end():
    retriever = HybridRetriever()
    workflow = AgenticWorkflow(retriever)
    req = "The system shall validate dual redundant accelerator pedal position sensors between 0.5V and 4.5V."
    result = workflow.run_pipeline(req, top_k=3, require_human_approval=False)

    assert result["status"] == "COMPLETED"
    assert result["execution_result"]["success"] is True
    assert result["traceability"]["completeness_score"] >= 0.7
    assert len(result["pipeline_trace"]) >= 4
