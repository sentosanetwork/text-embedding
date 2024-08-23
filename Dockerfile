FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app/ /app/app

EXPOSE 8700

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8700"]
