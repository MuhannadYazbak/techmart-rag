from fastapi import FastAPI, Request
from app.dynamic_agent import build_agent
from dotenv import load_dotenv
app = FastAPI()

load_dotenv()

from pydantic import BaseModel

class AskRequest(BaseModel):
    query: str
    user_id: int = 2
    
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