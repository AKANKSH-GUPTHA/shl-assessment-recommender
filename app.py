from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from services.recommender import recommend_assessments

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.post("/chat")
def chat(request: ChatRequest):

    conversation_text = " ".join(
        [m.content for m in request.messages if m.role == "user"]
    )

    if len(conversation_text.split()) < 5:
        return {
            "reply": "Please provide more details. What is the role, experience level, and key skills or technologies required?",
            "recommendations": [],
            "end_of_conversation": False
        }

    if "compare" in conversation_text.lower():
        return {
            "reply": "Please specify the two SHL assessments you would like compared.",
            "recommendations": [],
            "end_of_conversation": False
        }

    recommendations = recommend_assessments(conversation_text)

    return {
        "reply": "Here are relevant SHL assessments.",
        "recommendations": recommendations,
        "end_of_conversation": True
    }

@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommender API is running"
    }