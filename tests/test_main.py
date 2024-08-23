import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_get_score_embeddings():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/score-embedding",
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

@pytest.mark.asyncio
async def test_get_text_embeddings():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/text-embedding",
            json={
                "texts": [
                    "query: how much protein should a female eat",
                    "query: Công thức nấu ăn bí ngô tự làm"
                ]
            }
        )
    assert response.status_code == 200
    assert "data" in response.json()
