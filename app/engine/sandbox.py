import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


class SandboxedExecutor:
    """
    Executes generated Python code and pytest suites in a sandboxed, isolated subprocess
    with execution timeouts and output capturing.
    """
    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds

    def execute_test_suite(self, code: str, tests: str) -> Dict[str, Any]:
        """
        Writes code and tests into a temporary test directory, executes pytest,
        and parses results.
        """
        with tempfile.TemporaryDirectory(prefix="agent_sandbox_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            module_file = tmp_path / "solution.py"
            test_file = tmp_path / "test_solution.py"

            # Write implementation
            module_file.write_text(code, encoding="utf-8")

            # Prepare test file: import solution symbols
            test_header = "from solution import *\nimport pytest\n\n"
            test_file.write_text(test_header + tests, encoding="utf-8")

            python_exe = sys.executable

            cmd = [
                python_exe,
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "-o",
                "pythonpath=.",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode

                # Parse pytest outcome
                passed_count = 0
                failed_count = 0
                error_lines = []

                for line in stdout.splitlines():
                    if " PASSED" in line:
                        passed_count += 1
                    elif " FAILED" in line:
                        failed_count += 1
                    elif " ERROR" in line:
                        failed_count += 1
                    if "FAILED " in line or "AssertionError:" in line or "SyntaxError:" in line or "NameError:" in line:
                        error_lines.append(line.strip())

                all_passed = (returncode == 0) and (failed_count == 0) and (passed_count > 0)

                return {
                    "success": all_passed,
                    "returncode": returncode,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "total_tests": passed_count + failed_count,
                    "pass_rate": round(passed_count / max(passed_count + failed_count, 1), 4),
                    "stdout": stdout,
                    "stderr": stderr,
                    "errors": error_lines[:5],
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "returncode": -1,
                    "passed_count": 0,
                    "failed_count": 1,
                    "total_tests": 1,
                    "pass_rate": 0.0,
                    "stdout": "",
                    "stderr": f"Execution timed out after {self.timeout_seconds}s (infinite loop or hang guard triggered)",
                    "errors": ["TimeoutExpired: Execution exceeded safety time limit."],
                }
            except Exception as ex:
                return {
                    "success": False,
                    "returncode": -2,
                    "passed_count": 0,
                    "failed_count": 1,
                    "total_tests": 1,
                    "pass_rate": 0.0,
                    "stdout": "",
                    "stderr": str(ex),
                    "errors": [f"Execution exception: {str(ex)}"],
                }
