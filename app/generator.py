import os
from typing import List, Dict
from openai import OpenAI

SYSTEM = """You are an AI software engineering research assistant.
Use ONLY the supplied retrieved evidence for domain-specific claims.
Do not invent standards, requirements or APIs.
Return JSON with keys: implementation_plan, code, tests, risks, traceability_notes.
The code should be concise and executable Python where possible.
Tests must include normal, boundary and invalid-input cases when applicable.
For safety/security-sensitive systems, explicitly recommend human review.
"""


def generate(requirement: str, evidence: List[Dict[str, str]]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "implementation_plan": [
                "Validate the input before downstream processing.",
                "Reject invalid values and create a diagnostic event.",
                "Preserve traceability from the requirement to verification tests.",
            ],
            "code": "def validate_sensor(value, minimum, maximum):\n    if value < minimum or value > maximum:\n        return False, 'diagnostic: invalid sensor value'\n    return True, 'ok'\n",
            "tests": [
                "Valid value inside range is accepted.",
                "Minimum boundary is accepted.",
                "Maximum boundary is accepted.",
                "Value below minimum is rejected and produces a diagnostic event.",
                "Value above maximum is rejected and produces a diagnostic event.",
            ],
            "risks": ["Prototype output must be reviewed by a qualified engineer before integration."],
            "traceability_notes": "Baseline deterministic fallback; no LLM call was made.",
        }

    client = OpenAI(api_key=api_key)
    context = "\n\n".join(
        f"[{e['id']}] source={e['source']} score={e['score']}\n{e['text']}" for e in evidence
    )
    prompt = f"Requirement:\n{requirement}\n\nRetrieved evidence:\n{context}"
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    )
    import json
    return json.loads(response.choices[0].message.content)
