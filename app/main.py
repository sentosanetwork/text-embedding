from fastapi import FastAPI
from .models import TextInput
from .services import get_score_embeddings, get_text_embeddings

app = FastAPI()

@app.post("/score-embedding")
async def get_score_embeddings_api(input: TextInput):
    scores = get_score_embeddings(input.texts)
    return {"data": scores}

@app.post("/text-embedding")
async def get_text_embeddings_api(input: TextInput):
    embeddings = get_text_embeddings(input.texts)
    return {"data": embeddings}

@app.get("/")
def read_root():
    return {"message": "Hugging Face Multilingual Embeddings API"}
