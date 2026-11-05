
import json

ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

REQUIRED_KEYS = {
    "incident_id",
    "severity",
    "symptoms",
    "evidence",
    "possible_causes",
    "confirmed_root_cause",
    "recommended_actions",
    "requires_human_review"
}


def parse_response(response):
    """Convert JSON string into a Python dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {error}") from error


def validate_schema(data):
    """Validate required fields and their data types."""

    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")

    missing_keys = REQUIRED_KEYS - data.keys()

    if missing_keys:
        raise ValueError(f"Missing keys: {missing_keys}")

    if not isinstance(data["incident_id"], str):
        raise ValueError("incident_id must be a string")

    if data["severity"] not in ALLOWED_SEVERITIES:
        raise ValueError("Invalid severity")

    list_fields = [
        "symptoms",
        "evidence",
        "possible_causes",
        "recommended_actions"
    ]

    for field in list_fields:
        if not isinstance(data[field], list):
            raise ValueError(f"{field} must be a list")

        if not all(isinstance(item, str) for item in data[field]):
            raise ValueError(f"{field} must contain only strings")

    if (
        data["confirmed_root_cause"] is not None
        and not isinstance(data["confirmed_root_cause"], str)
    ):
        raise ValueError("confirmed_root_cause must be string or null")

    if not isinstance(data["requires_human_review"], bool):
        raise ValueError("requires_human_review must be boolean")

    return True


def validate_business_rules(data):
    """Validate application-specific rules."""

    if (
        data["severity"] == "CRITICAL"
        and not data["requires_human_review"]
    ):
        raise ValueError("Critical incidents require human review")

    if (
        data["confirmed_root_cause"] is None
        and not data["requires_human_review"]
    ):
        raise ValueError("Unknown root cause requires human review")

    return True


def process_llm_response(response):
    """Complete validation pipeline."""

    data = parse_response(response)
    validate_schema(data)
    validate_business_rules(data)

    return data


# Simulated LLM response
llm_response = """
{
    "incident_id": "INC-2041",
    "severity": "HIGH",
    "symptoms": ["HTTP 500 errors increased to 38%"],
    "evidence": ["Database connection timeouts"],
    "possible_causes": ["Database connectivity issue"],
    "confirmed_root_cause": null,
    "recommended_actions": ["Investigate database connections"],
    "requires_human_review": true
}
"""

try:
    result = process_llm_response(llm_response)
    print("Response accepted:")
    print(json.dumps(result, indent=4))

except ValueError as error:
    print("Response rejected:", error)
