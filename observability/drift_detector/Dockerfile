FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detector.py .
COPY register_baseline.py .

# baseline.npy is mounted via a Kubernetes ConfigMap or volume at runtime
# Default path matches BASELINE_PATH env var default in detector.py

CMD ["python", "detector.py"]