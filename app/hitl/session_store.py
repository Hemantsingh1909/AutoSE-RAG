import uuid
import time
from typing import Dict, Any, Optional, List
from app.engine.sandbox import SandboxedExecutor
from app.engine.traceability import TraceabilityEngine
from app.engine.agents import RevisionAgent, TestGenAgent, CriticAgent, CodeGenAgent


class HITLSessionStore:
    """
    In-memory session manager for Human-in-the-Loop approval workflows.
    """
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.sandbox = SandboxedExecutor()
        self.critic = CriticAgent()
        self.code_gen = CodeGenAgent()
        self.test_gen = TestGenAgent()
        self.revision_agent = RevisionAgent(self.code_gen)

    def create_session(self, workflow_output: Dict[str, Any]) -> str:
        session_id = f"hitl-{uuid.uuid4().hex[:8]}"
        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "status": workflow_output.get("status", "PENDING_APPROVAL"),
            "data": workflow_output,
            "human_feedback": None,
            "decision": None,
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return list(self.sessions.values())

    def approve_session(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        data = session["data"]
        code = data["code"]
        tests = data["tests"]

        # Run sandboxed execution
        exec_res = self.sandbox.execute_test_suite(code, tests)
        critic_rev = self.critic.evaluate(data["requirement"], code, tests, exec_res)
        traceability = TraceabilityEngine.generate_matrix(
            requirement=data["requirement"],
            evidence=data["evidence"],
            code=code,
            tests=tests,
            execution_result=exec_res,
            human_approved=True,
        )

        data["pipeline_trace"].append({
            "stage": "hitl_approval",
            "status": "approved",
            "message": "Human reviewer approved artifacts for automated verification.",
        })
        data["pipeline_trace"].append({
            "stage": "sandbox_execution",
            "status": "passed" if exec_res["success"] else "failed",
            "pass_rate": exec_res["pass_rate"],
        })

        session["status"] = "APPROVED_AND_CERTIFIED" if exec_res["success"] else "APPROVED_WITH_TEST_FAILURES"
        session["decision"] = "APPROVED"
        data["status"] = session["status"]
        data["execution_result"] = exec_res
        data["critic_review"] = critic_rev
        data["traceability"] = traceability

        return session

    def reject_and_refine(self, session_id: str, feedback: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        data = session["data"]
        session["human_feedback"] = feedback
        session["decision"] = "REJECTED_WITH_FEEDBACK"

        # Apply human revision
        revised = self.revision_agent.revise(
            data["requirement"], data["evidence"], data["code"], critique=feedback
        )
        new_code = revised["code"]
        new_tests = self.test_gen.execute(data["requirement"], new_code)

        data["code"] = new_code
        data["tests"] = new_tests
        data["plan"] = revised["plan"]

        data["pipeline_trace"].append({
            "stage": "hitl_rejection_and_refinement",
            "status": "revised",
            "human_feedback": feedback,
            "message": "AI revised code and tests based on human reviewer feedback.",
        })

        session["status"] = "PENDING_APPROVAL"
        return session
