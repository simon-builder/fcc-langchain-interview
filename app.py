from fastapi import FastAPI

from src.helper import llm_pipeline

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

