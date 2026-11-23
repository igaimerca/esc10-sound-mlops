FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
COPY app ./app
COPY models ./models
COPY data ./data
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
