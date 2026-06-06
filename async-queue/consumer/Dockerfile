FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker and the model loaders it depends on
COPY worker.py .
COPY ../inference-api/app/models/ ./app/models/
COPY ../inference-api/app/schemas.py ./app/schemas.py

CMD ["python", "worker.py"]