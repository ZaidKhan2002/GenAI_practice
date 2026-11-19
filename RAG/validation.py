import re

from decimal import Decimal, InvalidOperation
from generation import create_rag_prompt, generate_answer

# STEP 1: CHECK REQUESTED YEAR

def has_year_evidence(query, retrieved_chunks):

    match = re.search(r"\b(?:19|20)\d{2}\b", query)

    # No year requested: let the pipeline continue
    if not match:
        return True

    requested_year = match.group()

    for chunk in retrieved_chunks:
        text = chunk.get("chunk_text", "")
        if re.search(rf"\b{re.escape(requested_year)}\b", text):
            return True

    return False


# STEP 2: NORMALIZE NUMERIC VALUES

def normalize_numeric_value(value):

    if value is None:
        return None

    normalized = str(value).replace("$", "").replace(",", "").strip()

    try:
        return Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None


def extract_numeric_values(text):

    numbers = re.findall(
        r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])",
        text
    )

    extracted_values = []

    for number in numbers:
        try:
            extracted_values.append(Decimal(number.replace(",", "")))
        except InvalidOperation:
            continue

    return extracted_values


# STEP 3: VALIDATE GENERATED FINANCIAL VALUE

def validate_generated_value(response, retrieved_chunks):

    metric = response.get("metric")
    value = response.get("value")
    year = response.get("year")

    if not metric or value is None or year is None:
        return False

    expected_number = normalize_numeric_value(value)

    if expected_number is None:
        return False

    for chunk in retrieved_chunks:

        text = chunk.get("chunk_text", "")

        if metric.lower() not in text.lower():
            continue

        if not re.search(rf"\b{re.escape(str(year))}\b", text):
            continue

        if expected_number in extract_numeric_values(text):
            return True

    return False


# STEP 4: VALIDATE SOURCE CITATIONS

def validate_sources(response, retrieved_chunks):

    metric = response.get("metric")
    value = response.get("value")
    year = response.get("year")
    sources = response.get("sources", [])

    if not metric or value is None or year is None or not sources:
        return False

    normalized_value = normalize_numeric_value(value)

    if normalized_value is None:
        return False

    supporting_chunks = []

    for chunk in retrieved_chunks:

        text = chunk.get("chunk_text", "")

        if metric.lower() not in text.lower():
            continue

        if not re.search(rf"\b{re.escape(str(year))}\b", text):
            continue

        if normalized_value in extract_numeric_values(text):
            supporting_chunks.append(chunk)

    if not supporting_chunks:
        return False

    for source in sources:

        if not isinstance(source, dict):
            return False

        source_document = source.get("document_id")
        source_page = source.get("page")

        if source_document is None or source_page is None:
            return False

        source_exists = any(
            source_document == chunk.get("document_id")
            and source_page == chunk.get("page")
            for chunk in supporting_chunks
        )

        if not source_exists:
            return False

    return True


# STEP 5: VALIDATE METRIC, YEAR AND UNIT

def validate_metric_year_unit(response, retrieved_chunks):

    if not validate_generated_value(response, retrieved_chunks):
        return False

    metric = response.get("metric")
    year = response.get("year")
    unit = response.get("unit")

    if not metric or year is None or not unit:
        return False

    unit_aliases = {
        "millions": ["in millions", "million"],
        "billions": ["in billions", "billion"],
        "thousands": ["in thousands", "thousand"],
        "units": ["units", "unit"],
        "percentage": ["%", "percent", "percentage"],
    }

    expected_units = unit_aliases.get(unit.lower(), [unit.lower()])

    for chunk in retrieved_chunks:

        text = chunk.get("chunk_text", "").lower()

        metric_found = metric.lower() in text

        year_found = bool(re.search(rf"\b{re.escape(str(year))}\b", text))

        unit_found = any(alias in text for alias in expected_units)

        if metric_found and year_found and unit_found:
            return True

    return False


# STEP 6: VALIDATE MULTIPLE SUPPORTING EVIDENCE

def validate_multiple_evidence(expected_evidence, retrieved_chunks):

    if not expected_evidence:
        return False

    for evidence in expected_evidence:

        if not isinstance(evidence, dict):
            return False

        metric = evidence.get("metric")
        value = evidence.get("value")
        year = evidence.get("year")

        if not metric or value is None:
            return False

        expected_number = normalize_numeric_value(value)

        if expected_number is None:
            return False

        evidence_found = False

        for chunk in retrieved_chunks:

            text = chunk.get("chunk_text", "")

            if metric.lower() not in text.lower():
                continue

            if year is not None:
                if not re.search(rf"\b{re.escape(str(year))}\b", text):
                    continue

            if expected_number in extract_numeric_values(text):
                evidence_found = True
                break

        if not evidence_found:
            return False

    return True


# STEP 7: VALIDATE SOURCES FOR CALCULATIONS

def validate_calculation_sources(sources, supporting_evidence, retrieved_chunks):

    if not sources or not supporting_evidence:
        return False

    evidence_chunks = []

    for evidence in supporting_evidence:

        metric = evidence.get("metric")
        value = evidence.get("value")
        year = evidence.get("year")

        if not metric or value is None:
            return False

        expected_number = normalize_numeric_value(value)

        if expected_number is None:
            return False

        matching_chunks = []

        for chunk in retrieved_chunks:

            text = chunk.get("chunk_text", "")

            if metric.lower() not in text.lower():
                continue

            if year is not None:
                if not re.search(rf"\b{re.escape(str(year))}\b", text):
                    continue

            if expected_number in extract_numeric_values(text):
                matching_chunks.append(chunk)

        if not matching_chunks:
            return False

        evidence_chunks.extend(matching_chunks)

    for source in sources:

        if not isinstance(source, dict):
            return False

        source_document = source.get("document_id")
        source_page = source.get("page")

        if source_document is None or source_page is None:
            return False

        source_exists = any(
            source_document == chunk.get("document_id")
            and source_page == chunk.get("page")
            for chunk in evidence_chunks
        )

        if not source_exists:
            return False

    return True


# STEP 8: CREATE ABSTENTION RESPONSE

def create_abstention_response(reason):

    return {
        "answer": reason,
        "metric": None,
        "value": None,
        "year": None,
        "unit": None,
        "sources": [],
        "supporting_evidence": [],
        "calculation": {"operation": None, "result": None},
        "insufficient_evidence": True
    }


# STEP 9: VALIDATE CALCULATIONS
# (defined before answer_rag_query for readability)

def validate_calculation(value, evidence_values, operation="subtract"):

    if len(evidence_values) != 2:
        return False

    first, second = evidence_values

    # Accept both the prompt's names ("subtract", "add") and the old ones.
    if operation in ("subtract", "subtraction"):
        # Compare magnitudes so the order the LLM lists the figures in
        # does not matter (increases/differences are reported as positive).
        return abs(value) == abs(first - second)

    if operation in ("add", "addition"):
        return value == first + second

    return False


# STEP 10: MAIN RAG ANSWERING PIPELINE

def answer_rag_query(query, retrieved_chunks):

    if not retrieved_chunks:
        return create_abstention_response("No relevant information was retrieved.")

    if not has_year_evidence(query, retrieved_chunks):
        return create_abstention_response(
            "The retrieved document context does not contain "
            "financial information for the requested year."
        )

    prompt = create_rag_prompt(query, retrieved_chunks)
    response = generate_answer(prompt)

    if response.get("error") is not None:
        return response

    if response.get("insufficient_evidence"):
        return response

    supporting_evidence = response.get("supporting_evidence", [])
    sources = response.get("sources", [])
    calculation = response.get("calculation", {})

    if not isinstance(calculation, dict):
        return create_abstention_response(
            "The generated calculation format is invalid."
        )

    if not validate_multiple_evidence(supporting_evidence, retrieved_chunks):
        return create_abstention_response(
            "One or more supporting financial figures could not be verified."
        )

    # ---------------- CALCULATION VALIDATION ----------------

    operation = calculation.get("operation")

    if operation is not None:

        if not validate_calculation_sources(
            sources, supporting_evidence, retrieved_chunks
        ):
            return create_abstention_response(
                "The calculation source citations could not be verified."
            )

        evidence_values = []

        for item in supporting_evidence:

            numeric_value = normalize_numeric_value(item.get("value"))

            if numeric_value is None:
                return create_abstention_response(
                    "The supporting calculation values are invalid."
                )

            # Only integer financial calculations are supported
            if numeric_value != numeric_value.to_integral_value():
                return create_abstention_response(
                    "The calculation contains unsupported decimal values."
                )

            evidence_values.append(int(numeric_value))

        calculated_result = normalize_numeric_value(calculation.get("result"))

        if calculated_result is None:
            return create_abstention_response("The calculated result is invalid.")

        if calculated_result != calculated_result.to_integral_value():
            return create_abstention_response(
                "The calculated result is unsupported."
            )

        if not validate_calculation(
            int(calculated_result), evidence_values, operation
        ):
            return create_abstention_response(
                "The calculated result could not be verified."
            )

        return response

    # ---------------- STANDARD VALUE VALIDATION ----------------

    if not validate_generated_value(response, retrieved_chunks):
        return create_abstention_response(
            "The generated financial value could not be verified "
            "against the retrieved document evidence."
        )

    if not validate_sources(response, retrieved_chunks):
        return create_abstention_response(
            "The generated source citations could not be verified "
            "against the retrieved document evidence."
        )

    if not validate_metric_year_unit(response, retrieved_chunks):
        return create_abstention_response(
            "The generated metric, year or unit could not be verified."
        )

    return response