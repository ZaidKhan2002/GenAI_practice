# Day 13 — Prompt Engineering Interview Questions

## Fundamentals

### 1. What is prompt engineering?
Designing instructions and context to guide an LLM towards desired outputs without modifying its model weights.

### 2. What are the components of a good prompt?
Role, task, context, constraints, examples and output format.

### 3. What is zero-shot prompting?
Performing a task without supplying examples.

### 4. What is few-shot prompting?
Providing multiple examples to demonstrate the desired input-output pattern.

### 5. Does few-shot prompting fine-tune an LLM?
No. It provides examples in the current context. Model weights remain unchanged.

## Prompt Design

### 6. Why should prompts be specific?
Specific prompts reduce ambiguity and make the expected behaviour clearer.

### 7. What are delimiters?
Markers used to separate instructions from data, such as XML-like tags, triple quotes and code fences.

### 8. Do delimiters prevent prompt injection?
No. They improve input separation but do not guarantee security.

### 9. What is Chain-of-Thought prompting?
A prompting approach that encourages structured intermediate problem-solving. A concise rationale or structured analysis is often more suitable for application outputs.

## Structured Outputs

### 10. What is the difference between JSON mode and structured output?
JSON mode generally ensures valid JSON syntax. Structured output additionally constrains the response to a defined schema.

### 11. Does valid JSON guarantee correct information?
No. JSON validity only establishes syntactic correctness.

### 12. Why is schema validation required?
To ensure required fields, expected types, allowed values and other structural constraints are satisfied.

### 13. Why is business validation separate from schema validation?
Schema validation checks structure. Business validation checks whether the response follows application-specific rules.

### 14. Why use null for an unknown root cause?
It explicitly represents missing or unconfirmed information and discourages the model from inventing a conclusion.

## Security and Reliability

### 15. What is prompt injection?
An attack where malicious content attempts to manipulate an LLM's instructions or behaviour.

### 16. Why is backend security necessary?
Prompts cannot enforce authentication, authorisation or reliable access control. These controls must be implemented in the application.

### 17. What is least-privilege tool access?
Giving an LLM or agent only the minimum tool permissions required for its task.

### 18. Why do prompts require regression testing?
Small wording changes can alter model behaviour. Regression tests detect unexpected changes in output quality, formatting and security.

### 19. How should LLM output be handled in a production backend?
Parse it, validate its schema, apply business rules, enforce permissions and require human approval where appropriate.

### 20. Can structured output eliminate hallucinations?
No. It controls response structure, not factual correctness.