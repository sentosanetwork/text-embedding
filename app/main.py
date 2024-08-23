from fastapi import FastAPI
from app.models import TextsInput, ScoresOutput, EmbeddingsOutput
from app.services import get_score_embeddings, get_text_embeddings

app = FastAPI()

@app.post("/get-score-embeddings", response_model=ScoresOutput)
async def compute_similarity_scores(input_data: TextsInput):
    scores = get_score_embeddings(input_data.texts)
    return {"scores": scores}

@app.post("/get-text-embeddings", response_model=EmbeddingsOutput)
async def compute_text_embeddings(input_data: TextsInput):
    embeddings = get_text_embeddings(input_data.texts)
    return {"embeddings": embeddings}
