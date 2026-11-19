import json
import re
import ollama

MODEL_NAME = "qwen3.5:4b"


# STEP 1: CREATE RAG PROMPT

def create_rag_prompt(query, retrieved_chunks):

    context = ""

    for rank, result in enumerate(retrieved_chunks, start=1):
        context += f"""
Source {rank}:
Document: {result['document_id']}
Page: {result['page']}
Chunk type: {result['chunk_type']}

Content:
{result['chunk_text']}
"""

    prompt = f"""
You are a financial document analysis assistant.

Answer the question using ONLY the provided context.

Rules:
1. Never invent financial figures or reporting periods.
2. Identify the exact financial metric requested.
3. Prefer the primary consolidated financial statement.
4. Do not mix figures from different statements.
5. Preserve the original financial value and unit.
6. Include only sources that support your answer.
7. If evidence is insufficient, do not guess.
8. Return valid JSON only. Do not use Markdown code fences.
9. Cite the minimum number of sources needed to support the answer.
10. For questions requiring multiple financial metrics, include every required metric in supporting_evidence.
11. Each supporting metric must have its own source document and page number.
12. For calculation questions, calculate only when all required financial values are available.
13. Only perform a calculation when requested by the question.
14. Preserve the reporting unit during calculations.
15. The top-level metric and value should contain the primary metric and final result when a calculation is requested.
16. For single-metric questions, supporting_evidence must still contain the supporting financial figure.
17. If evidence is insufficient, return an empty supporting_evidence array and null calculation fields.
18. For questions without calculations, calculation MUST remain an object. Set calculation.operation and calculation.result to null. Never set calculation itself to null.
19. For calculation questions, operation must be "subtract" (differences or increases) or "add" (totals or combined amounts).
20. For calculations, supporting_evidence must contain the two figures used. For "subtract", list the larger or later figure first.
21. Every cited page must exist in the provided context.
22. Every supporting_evidence item must be traceable to its document and page in the context.
23. Do not use a financial value merely because it appears in the context. Ensure it belongs to the requested metric and reporting year.
24. Successful answers must set error to null.
25. The calculation field MUST ALWAYS be a JSON object containing exactly operation and result. Never return calculation as null.

Required JSON structure:

{{
    "answer": "string",
    "metric": "string or null",
    "value": "string or null",
    "year": "integer or null",
    "unit": "string or null",
    "sources": [
        {{
            "document_id": "string",
            "page": "integer"
        }}
    ],
    "supporting_evidence": [
        {{
            "metric": "string",
            "value": "string",
            "year": "integer",
            "document_id": "string",
            "page": "integer"
        }}
    ],
    "calculation": {{
        "operation": "subtract, add or null",
        "result": "string or null"
    }},
    "insufficient_evidence": "boolean",
    "error": null
}}

If evidence is insufficient:
- Set insufficient_evidence to true.
- Set answer to a brief explanation.
- Set metric, value, year and unit to null.
- Set sources to an empty array.
- Set supporting_evidence to an empty array.
- Set calculation.operation and calculation.result to null.
- Set error to null.

Context:
{context}

Question:
{query}
"""

    return prompt


# STEP 2: PARSE AND VALIDATE MODEL RESPONSE

def parse_model_response(raw_response):

    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"LLM returned invalid JSON: {error}")

    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")

    # Small models sometimes drop this field; it is always null on success
    data.setdefault("error", None)

    required_fields = {
        "answer", "metric", "value", "year", "unit", "sources",
        "supporting_evidence", "calculation", "insufficient_evidence", "error"
    }

    if set(data.keys()) != required_fields:
        raise ValueError(
            f"Missing or unexpected fields: {set(data.keys()) ^ required_fields}"
        )

    if not isinstance(data["answer"], str):
        raise ValueError("answer must be a string")

    for field in ["metric", "value", "unit"]:
        if data[field] is not None and not isinstance(data[field], str):
            raise ValueError(f"{field} must be a string or null")

    if data["year"] is not None and type(data["year"]) is not int:
        raise ValueError("year must be an integer or null")

    if not isinstance(data["sources"], list):
        raise ValueError("sources must be a list")

    if not isinstance(data["supporting_evidence"], list):
        raise ValueError("supporting_evidence must be a list")

    if type(data["insufficient_evidence"]) is not bool:
        raise ValueError("insufficient_evidence must be a boolean")

    if data["error"] is not None and not isinstance(data["error"], str):
        raise ValueError("error must be a string or null")

    for source in data["sources"]:

        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")

        if set(source.keys()) != {"document_id", "page"}:
            raise ValueError("Invalid source fields")

        if (
            not isinstance(source["document_id"], str)
            or not source["document_id"].strip()
            or type(source["page"]) is not int
        ):
            raise ValueError("Invalid source information")

    for evidence in data["supporting_evidence"]:

        if not isinstance(evidence, dict):
            raise ValueError("Each evidence item must be an object")

        if set(evidence.keys()) != {
            "metric", "value", "year", "document_id", "page"
        }:
            raise ValueError("Missing or unexpected evidence fields")

        if (
            not isinstance(evidence["metric"], str)
            or not evidence["metric"].strip()
            or not isinstance(evidence["value"], str)
            or not evidence["value"].strip()
            or type(evidence["year"]) is not int
            or not isinstance(evidence["document_id"], str)
            or not evidence["document_id"].strip()
            or type(evidence["page"]) is not int
        ):
            raise ValueError("Invalid supporting evidence")

    calculation = data["calculation"]

    if not isinstance(calculation, dict):
        raise ValueError("calculation must be an object")

    if set(calculation.keys()) != {"operation", "result"}:
        raise ValueError("Invalid calculation fields")

    if calculation["operation"] not in ["subtract", "add", None]:
        raise ValueError("Unsupported calculation operation")

    if calculation["result"] is not None and not isinstance(calculation["result"], str):
        raise ValueError("Calculation result must be a string or null")

    if calculation["operation"] is None and calculation["result"] is not None:
        raise ValueError("Result must be null when operation is null")

    if calculation["operation"] is not None and calculation["result"] is None:
        raise ValueError("Result is required when an operation is specified")

    if data["insufficient_evidence"]:

        if (
            data["metric"] is not None
            or data["value"] is not None
            or data["year"] is not None
            or data["unit"] is not None
            or data["sources"]
            or data["supporting_evidence"]
            or calculation["operation"] is not None
            or calculation["result"] is not None
        ):
            raise ValueError("Invalid insufficient-evidence response")

    if not data["insufficient_evidence"]:

        if data["error"] is not None:
            raise ValueError("Successful response must have error set to null")

        if (
            not data["metric"]
            or data["value"] is None
            or data["year"] is None
            or not data["unit"]
            or not data["sources"]
            or not data["supporting_evidence"]
        ):
            raise ValueError(
                "Successful response is missing required financial evidence"
            )

    return data


# STEP 3: CREATE GENERATION ERROR RESPONSE

def create_generation_error_response(error):

    return {
        "answer": "Unable to generate an answer due to a processing error.",
        "metric": None,
        "value": None,
        "year": None,
        "unit": None,
        "sources": [],
        "insufficient_evidence": False,
        "supporting_evidence": [],
        "calculation": {"operation": None, "result": None},
        "error": str(error)
    }


# STEP 4: GENERATE ANSWER USING OLLAMA

def _chat(prompt):
    """Call Ollama; try think=False first, fall back if unsupported."""

    kwargs = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "options": {"temperature": 0},
    }

    try:
        return ollama.chat(think=False, **kwargs)
    except Exception:
        return ollama.chat(**kwargs)


def _clean_llm_output(raw):
    """Remove <think> blocks and Markdown code fences."""

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# def generate_answer(prompt):

#     try:
#         response = _chat(prompt)
#         raw_response = response["message"]["content"]
#         return parse_model_response(_clean_llm_output(raw_response))

#     except Exception as error:
#         return create_generation_error_response(error)

def generate_answer(prompt):

    try:
        response = _chat(prompt)
        raw_response = response["message"]["content"]

        print("\nRAW LLM RESPONSE:\n")
        print(raw_response)

        cleaned_response = _clean_llm_output(raw_response)

        print("\nCLEANED LLM RESPONSE:\n")
        print(cleaned_response)

        return parse_model_response(cleaned_response)

    except Exception as error:
        return create_generation_error_response(error)