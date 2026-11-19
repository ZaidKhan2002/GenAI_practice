from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500
    )

class QuestionResponse(BaseModel):
    question: str
    message: str

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    return {
        "question": request.question,
        "message": "Question received successfully"
    }