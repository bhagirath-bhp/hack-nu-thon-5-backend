from fastapi import FastAPI , Header
from typing import Annotated, List, Union

from fastapi.middleware.cors import CORSMiddleware
import os 
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Customized Chatbot", description="Customized Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.post("/ask" ,  summary="pass value using header")
async def read_items(user_query:str , company_name:str):
    return {"response": "ask" }


@app.get("/")
async def index():
    return {"message": "Hello World use //extract for api"}


# if __name__ == "__main__":
#     print(ask_ai("what is my name ?"))
