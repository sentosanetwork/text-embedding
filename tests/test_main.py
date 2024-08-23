from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_score_embeddings():
    response = client.post(
        "/get-score-embeddings",
        json={
            "texts": [
                "query: how much protein should a female eat",
                "query: Công thức nấu ăn bí ngô tự làm",
                "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
                "passage: 1. Bí ngô xào sợi Nguyên liệu: nửa quả bí ngô mềm Gia vị: hành, muối, đường, cốt gà"
            ]
        }
    )
    assert response.status_code == 200
    assert "data" in response.json()

def test_get_text_embeddings():
    response = client.post(
        "/get-text-embeddings",
        json={
            "texts": [
                "query: how much protein should a female eat",
                "query: Công thức nấu ăn bí ngô tự làm"
            ]
        }
    )
    assert response.status_code == 200
    assert "data" in response.json()
