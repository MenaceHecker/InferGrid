# Build from the repository root:
# docker build -f async-queue/consumer/Dockerfile -t inference-worker .
FROM python:3.11-slim

WORKDIR /app

COPY async-queue/consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker and the model loaders it depends on
COPY async-queue/consumer/worker.py .
COPY inference-api/app/__init__.py ./app/__init__.py
COPY inference-api/app/models/ ./app/models/
COPY inference-api/models/ ./models/

CMD ["python", "worker.py"]
