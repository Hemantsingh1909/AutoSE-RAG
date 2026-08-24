from typing import Dict, Any, List, Optional
import time
from .agents import RetrieverAgent, CodeGenAgent, TestGenAgent, CriticAgent, RevisionAgent
from .sandbox import SandboxedExecutor
from .traceability import TraceabilityEngine


class AgenticWorkflow:
    """
    End-to-end self-refining agentic workflow for AI software engineering.
    """
    def __init__(self, retriever):
        self.retriever_agent = RetrieverAgent(retriever)
        self.code_gen_agent = CodeGenAgent()
        self.test_gen_agent = TestGenAgent()
        self.sandbox_executor = SandboxedExecutor()
        self.critic_agent = CriticAgent()
        self.revision_agent = RevisionAgent(self.code_gen_agent)

    def run_pipeline(
        self,
        requirement: str,
        top_k: int = 4,
        max_revisions: int = 2,
        require_human_approval: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the autonomous agent pipeline.
        If require_human_approval is True, stops at PENDING_APPROVAL state before executing tests.
        """
        start_time = time.time()
        pipeline_trace = []

        # 1. Retrieval
        evidence = self.retriever_agent.execute(requirement, top_k=top_k)
        pipeline_trace.append({
            "stage": "retrieval",
            "status": "completed",
            "message": f"Retrieved {len(evidence)} evidence chunks via Hybrid RRF.",
            "evidence_count": len(evidence),
        })

        # 2. Code Generation
        gen_output = self.code_gen_agent.execute(requirement, evidence)
        code = gen_output["code"]
        plan = gen_output["plan"]
        safety_notes = gen_output["safety_notes"]
        pipeline_trace.append({
            "stage": "code_generation",
            "status": "completed",
            "message": "Synthesized safety-compliant implementation.",
            "plan": plan,
        })

        # 3. Test Generation
        tests = self.test_gen_agent.execute(requirement, code)
        pipeline_trace.append({
            "stage": "test_generation",
            "status": "completed",
            "message": "Generated standalone pytest suite covering boundary and failure modes.",
        })

        # If HITL gate is enabled, return intermediate state for review
        if require_human_approval:
            return {
                "status": "PENDING_APPROVAL",
                "requirement": requirement,
                "evidence": evidence,
                "plan": plan,
                "code": code,
                "tests": tests,
                "safety_notes": safety_notes,
                "pipeline_trace": pipeline_trace,
                "execution_result": None,
                "critic_review": None,
                "traceability": None,
            }

        # 4. Sandboxed Execution & Critic Self-Refining Loop
        revision_count = 0
        current_code = code
        current_tests = tests
        exec_result = self.sandbox_executor.execute_test_suite(current_code, current_tests)
        critic_review = self.critic_agent.evaluate(requirement, current_code, current_tests, exec_result)

        pipeline_trace.append({
            "stage": "sandbox_execution",
            "status": "passed" if exec_result["success"] else "failed",
            "pass_rate": exec_result["pass_rate"],
            "passed_count": exec_result["passed_count"],
            "failed_count": exec_result["failed_count"],
        })
        pipeline_trace.append({
            "stage": "critic_review",
            "verdict": critic_review["verdict"],
            "critique": critic_review["critique"],
        })

        # Iterative revision loop if needed
        while critic_review["repair_needed"] and revision_count < max_revisions:
            revision_count += 1
            pipeline_trace.append({
                "stage": f"revision_iteration_{revision_count}",
                "status": "refining",
                "message": f"Auto-repairing implementation based on critic trace (Attempt {revision_count}/{max_revisions}).",
            })

            revised_output = self.revision_agent.revise(
                requirement, evidence, current_code, critic_review["critique"]
            )
            current_code = revised_output["code"]
            # Re-generate tests if code structure changed
            current_tests = self.test_gen_agent.execute(requirement, current_code)

            exec_result = self.sandbox_executor.execute_test_suite(current_code, current_tests)
            critic_review = self.critic_agent.evaluate(requirement, current_code, current_tests, exec_result)

            pipeline_trace.append({
                "stage": f"sandbox_execution_rev_{revision_count}",
                "status": "passed" if exec_result["success"] else "failed",
                "pass_rate": exec_result["pass_rate"],
            })

        # 5. Build Traceability Matrix
        traceability = TraceabilityEngine.generate_matrix(
            requirement=requirement,
            evidence=evidence,
            code=current_code,
            tests=current_tests,
            execution_result=exec_result,
            human_approved=False,
        )

        pipeline_trace.append({
            "stage": "traceability_matrix",
            "status": "certified" if exec_result["success"] else "verification_needed",
            "completeness_score": traceability["completeness_score"],
        })

        duration = round(time.time() - start_time, 3)

        return {
            "status": "COMPLETED",
            "requirement": requirement,
            "evidence": evidence,
            "plan": plan,
            "code": current_code,
            "tests": current_tests,
            "safety_notes": safety_notes,
            "revisions_count": revision_count,
            "execution_result": exec_result,
            "critic_review": critic_review,
            "traceability": traceability,
            "pipeline_trace": pipeline_trace,
            "duration_seconds": duration,
        }
