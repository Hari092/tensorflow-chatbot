
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot import chatbot_response

app = FastAPI()

income_sources = {}
deductions = {}
age = None

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    global income_sources, deductions, age
    response = chatbot_response(message.message, income_sources, deductions, age)
    return {"response": response}
