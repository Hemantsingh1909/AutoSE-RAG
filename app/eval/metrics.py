import ast
import math
import re
import numpy as np
from typing import List, Dict, Any, Union, Tuple


class FailureTaxonomy:
    """
    Formal failure taxonomy for safety-critical AI software engineering:
    F1: Syntax / AST Compilation Failure
    F2: Missing Requirement Constraint
    F3: Boundary Violation (e.g. upper bound checked but lower bound omitted)
    F4: Fault-Handling Failure (incorrect return status or missing debounce)
    F5: Safe-State Transition Failure (failsafe or limp-home not triggered)
    F6: Ungrounded API / Domain Interface Mismatch
    F7: Test-Induced / Assertion Failure
    """
    CATEGORIES = {
        "F1": {"name": "Syntax / AST Compilation Failure", "desc": "Generated code cannot be parsed by Python AST."},
        "F2": {"name": "Missing Requirement Constraint", "desc": "Implementation omits an essential sub-constraint."},
        "F3": {"name": "Boundary Violation", "desc": "Fails edge or boundary condition checks (e.g., lower/upper limits)."},
        "F4": {"name": "Fault-Handling Failure", "desc": "Improper debounce progression or unhandled fault status."},
        "F5": {"name": "Safe-State Transition Failure", "desc": "System fails to transition to failsafe or limp-home mode."},
        "F6": {"name": "Ungrounded API / Domain Interface Mismatch", "desc": "Uses non-existent constants, modules, or missing expected interfaces."},
        "F7": {"name": "Test-Induced / Assertion Failure", "desc": "Execution output mismatch with expected oracle behavior."},
    }

    @classmethod
    def classify(cls, code: str, exec_result: Dict[str, Any], requirement: str) -> Dict[str, str]:
        if exec_result.get("success", False):
            return {"type": "NONE", "name": "Pass", "desc": "All tests and constraints satisfied."}

        stderr = exec_result.get("stderr", "")
        errors = " ".join(exec_result.get("errors", []))
        combined = f"{stderr} {errors}"

        if "SyntaxError" in combined:
            return {"type": "F1", **cls.CATEGORIES["F1"]}
        if "NameError" in combined or "ImportError" in combined or "is not defined" in combined:
            return {"type": "F6", **cls.CATEGORIES["F6"]}
        if "boundary" in combined.lower() or "range" in combined.lower() or "voltage <" in combined.lower():
            return {"type": "F3", **cls.CATEGORIES["F3"]}
        if "safe_state" in combined.lower() or "safe state" in combined.lower() or "timeout" in combined.lower():
            return {"type": "F5", **cls.CATEGORIES["F5"]}
        if "debounce" in combined.lower() or "dtc" in combined.lower() or "pending" in combined.lower():
            return {"type": "F4", **cls.CATEGORIES["F4"]}
        if "AssertionError" in combined:
            return {"type": "F3", **cls.CATEGORIES["F3"]}

        return {"type": "F2", **cls.CATEGORIES["F2"]}


class EvaluationMetrics:
    """
    Computes formal quantitative metrics, bootstrap confidence intervals,
    paired statistical tests with Cohen's d_z effect sizes and Holm-Bonferroni correction.
    """

    @staticmethod
    def calculate_bootstrap_ci(data: List[float], n_bootstraps: int = 1000, ci: float = 0.95) -> Tuple[float, float, Tuple[float, float]]:
        if not data:
            return 0.0, 0.0, (0.0, 0.0)

        arr = np.array(data, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        if len(arr) <= 1:
            return mean_val, std_val, (mean_val, mean_val)

        rng = np.random.default_rng(seed=42)
        boot_means = []
        for _ in range(n_bootstraps):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(np.mean(sample))

        lower_pct = (1.0 - ci) / 2.0 * 100
        upper_pct = (1.0 + ci) / 2.0 * 100
        ci_lower = float(np.percentile(boot_means, lower_pct))
        ci_upper = float(np.percentile(boot_means, upper_pct))

        return round(mean_val, 4), round(std_val, 4), (round(ci_lower, 4), round(ci_upper, 4))

    @staticmethod
    def calculate_cohens_dz(a: List[float], b: List[float]) -> float:
        """
        Computes Cohen's d_z for paired samples: d_z = mean(diff) / std(diff)
        """
        diff = np.array(a) - np.array(b)
        std_diff = np.std(diff, ddof=1)
        if std_diff == 0:
            return 0.0
        return round(float(np.mean(diff) / std_diff), 3)

    @staticmethod
    def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
        """
        Applies step-down Holm-Bonferroni adjustment to an array of p-values.
        """
        k = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        adjusted = [0.0] * k

        running_max = 0.0
        for rank, (orig_idx, p_val) in enumerate(indexed):
            mult = k - rank
            adj_p = min(p_val * mult, 1.0)
            running_max = max(running_max, adj_p)
            adjusted[orig_idx] = round(running_max, 6)

        return adjusted

    @staticmethod
    def calculate_retrieval_metrics(
        evidence: List[Dict],
        relevant_docs: Union[str, List[str]],
        total_relevant_in_corpus: int = 4,
        top_k: int = 4
    ) -> Dict[str, float]:
        if not evidence:
            return {
                "precision_at_1": 0.0,
                "precision_at_3": 0.0,
                "precision_at_4": 0.0,
                "recall_at_4": 0.0,
                "mrr": 0.0,
                "ndcg_at_4": 0.0,
            }

        target_set = set([relevant_docs] if isinstance(relevant_docs, str) else relevant_docs)
        relevance_vector = [1 if item.get("source", "") in target_set else 0 for item in evidence[:top_k]]

        p1 = relevance_vector[0] if len(relevance_vector) >= 1 else 0.0
        p3 = sum(relevance_vector[:3]) / min(3, len(relevance_vector)) if len(relevance_vector) >= 1 else 0.0
        p4 = sum(relevance_vector[:4]) / min(4, len(relevance_vector)) if len(relevance_vector) >= 1 else 0.0

        total_hits = sum(relevance_vector)
        recall4 = total_hits / max(total_relevant_in_corpus, 1)

        first_hit_rank = 0
        for rank, rel in enumerate(relevance_vector, start=1):
            if rel == 1:
                first_hit_rank = rank
                break
        mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

        dcg = 0.0
        for rank, rel in enumerate(relevance_vector, start=1):
            dcg += (2**rel - 1) / math.log2(rank + 1)

        ideal_relevance = sorted(relevance_vector, reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_relevance, start=1):
            idcg += (2**rel - 1) / math.log2(rank + 1)

        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        return {
            "precision_at_1": round(float(p1), 4),
            "precision_at_3": round(float(p3), 4),
            "precision_at_4": round(float(p4), 4),
            "recall_at_4": round(float(recall4), 4),
            "mrr": round(float(mrr), 4),
            "ndcg_at_4": round(float(ndcg), 4),
        }

    @staticmethod
    def calculate_granular_constraint_coverage(code: str, constraints: List[Dict[str, str]]) -> Dict[str, Any]:
        if not constraints:
            return {"coverage_rate": 1.0, "satisfied_count": 0, "total_count": 0, "breakdown": {}}

        code_lower = code.lower()
        satisfied = 0
        breakdown = {}

        for c in constraints:
            cid = c["id"]
            check_token = c.get("check", "").lower()
            if check_token and check_token in code_lower:
                satisfied += 1
                breakdown[cid] = True
            else:
                desc_words = [w for w in c.get("description", "").lower().split() if len(w) > 4]
                match_count = sum(1 for w in desc_words if w in code_lower)
                if match_count >= 2:
                    satisfied += 1
                    breakdown[cid] = True
                else:
                    breakdown[cid] = False

        coverage = satisfied / len(constraints)
        return {
            "coverage_rate": round(float(coverage), 4),
            "satisfied_count": satisfied,
            "total_count": len(constraints),
            "breakdown": breakdown
        }

    @staticmethod
    def calculate_ast_correctness(code: str) -> bool:
        if not code or not code.strip():
            return False
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    @staticmethod
    def calculate_hallucination_ugcr(code: str, evidence: List[Dict], requirement: str) -> float:
        if not code:
            return 1.0

        grounding_text = (requirement + " " + " ".join(e.get("text", "") for e in evidence)).lower()

        code_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", code)
        code_keywords = re.findall(r"\b[A-Za-z_]{4,}\b", code)

        py_keywords = {"import", "from", "return", "class", "def", "self", "true", "false", "none", "float", "bool", "dict", "tuple"}
        domain_claims = [w.lower() for w in code_keywords if w.lower() not in py_keywords]

        if not domain_claims and not code_numbers:
            return 0.0

        unsupported_claims = 0
        total_claims = len(domain_claims) + len(code_numbers)

        for kw in domain_claims:
            if kw not in grounding_text and ("fake_" in kw or "magic" in kw or "unknown" in kw or "mock" in kw):
                unsupported_claims += 1

        for num in code_numbers:
            if num not in grounding_text and float(num) > 10.0 and ("1234" not in num):
                unsupported_claims += 0.5

        ugcr = min(unsupported_claims / max(total_claims, 1), 1.0)
        return round(float(ugcr), 4)

    @staticmethod
    def calculate_regression_rate(initial_passes: int, regressions: int) -> float:
        if initial_passes == 0:
            return 0.0
        return round(float(regressions / initial_passes), 4)
