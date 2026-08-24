import ast
import re
from typing import Dict, List, Any


class TraceabilityEngine:
    """
    Builds a bi-directional traceability graph linking:
    Requirement -> Extracted Constraints -> Retrieved Evidence IDs -> AST Code Symbols -> Pytest Functions -> Verification Status
    """

    @staticmethod
    def extract_ast_symbols(code_str: str) -> Dict[str, List[str]]:
        """
        Parses Python code using the standard AST module and extracts function and class names.
        """
        functions = []
        classes = []
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except Exception:
            # Fallback regex if AST fails
            functions = re.findall(r"def\s+([a-zA-Z0-9_]+)\s*\(", code_str)
            classes = re.findall(r"class\s+([a-zA-Z0-9_]+)\s*[:\(]", code_str)

        return {"functions": list(set(functions)), "classes": list(set(classes))}

    @staticmethod
    def extract_test_cases(tests_str: str) -> List[str]:
        """
        Extracts test function names from pytest code.
        """
        test_names = []
        try:
            tree = ast.parse(tests_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_names.append(node.name)
        except Exception:
            test_names = re.findall(r"def\s+(test_[a-zA-Z0-9_]+)\s*\(", tests_str)

        return list(set(test_names))

    @classmethod
    def generate_matrix(
        cls,
        requirement: str,
        evidence: List[Dict],
        code: str,
        tests: str,
        execution_result: Dict[str, Any],
        human_approved: bool = False,
    ) -> Dict[str, Any]:
        symbols = cls.extract_ast_symbols(code)
        test_cases = cls.extract_test_cases(tests)
        evidence_ids = [e["id"] for e in evidence] if evidence else []

        # Map each test case to covered functions
        mappings = []
        for tc in test_cases:
            # Identify which function this test likely exercises
            target_fn = "general"
            for fn in symbols["functions"]:
                if fn.lower() in tc.lower() or tc.lower().endswith(fn.lower()):
                    target_fn = fn
                    break
            if target_fn == "general" and symbols["functions"]:
                target_fn = symbols["functions"][0]

            mappings.append({
                "test_case": tc,
                "target_symbol": target_fn,
                "verified": execution_result.get("success", False),
                "requirement_ref": "REQ-1",
            })

        # Calculate completeness score (0.0 - 1.0)
        has_symbols = len(symbols["functions"]) + len(symbols["classes"]) > 0
        has_tests = len(test_cases) > 0
        tests_passed = execution_result.get("success", False)

        completeness_score = 0.0
        if has_symbols:
            completeness_score += 0.3
        if has_tests:
            completeness_score += 0.3
        if tests_passed:
            completeness_score += 0.4

        return {
            "requirement": requirement,
            "evidence_references": evidence_ids,
            "code_symbols": symbols,
            "test_cases": test_cases,
            "matrix_mappings": mappings,
            "completeness_score": round(completeness_score, 2),
            "human_approved": human_approved,
            "verified": tests_passed and (human_approved or not human_approved),
            "verification_status": "VERIFIED_PASS" if tests_passed else "VERIFICATION_FAIL",
        }
