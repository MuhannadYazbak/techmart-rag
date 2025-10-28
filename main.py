from fastapi import FastAPI, Request
from app.dynamic_agent import build_agent
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

load_dotenv()

from pydantic import BaseModel

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for React, etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    user_id: int
    
@app.post("/ask")
async def ask(data: AskRequest):
    agent = build_agent(data.user_id)
    response = agent.invoke(data.query)
    return {"response": response.content}


# @app.post("/ask")
# async def ask(request: Request):
#     data = await request.json()
#     query = data["query"]
#     user_id = data.get("user_id", 2)
#     agent = build_agent(user_id)
#     response = agent.invoke(query)
#     return {"response": response.content}