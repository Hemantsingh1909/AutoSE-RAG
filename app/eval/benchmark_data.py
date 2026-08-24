import json
from pathlib import Path
from typing import List, Dict, Any

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark"
REQ_DIR = BENCHMARK_DIR / "requirements"
ORACLE_DIR = BENCHMARK_DIR / "oracle"


def load_benchmark_dataset() -> List[Dict[str, Any]]:
    """
    Loads all benchmark requirements and pairs them with their independent oracle test suites.
    """
    dataset = []
    for req_file in sorted(REQ_DIR.glob("*.json")):
        req_data = json.loads(req_file.read_text(encoding="utf-8"))
        req_id = req_data["id"]

        # Load paired independent test oracle
        oracle_file = ORACLE_DIR / f"test_{req_id}.py"
        oracle_code = oracle_file.read_text(encoding="utf-8") if oracle_file.exists() else ""

        req_data["oracle_file"] = str(oracle_file)
        req_data["hidden_oracle_tests"] = oracle_code
        dataset.append(req_data)

    return dataset


BENCHMARK_REQUIREMENTS = load_benchmark_dataset()
