"""
inference-api/app/auth.py

API key authentication for all inference API endpoints.
Keys are stored in a Kubernetes Secret and injected via the API_KEY env var.

Usage:
  Add `api_key: str = Depends(verify_api_key)` to any route that requires auth.
  The client passes the key in the X-API-Key header.

Key rotation:
  1. Generate a new key: openssl rand -hex 32
  2. Update the Kubernetes Secret:
     kubectl create secret generic inference-api-secret \
       --namespace infergrid \
       --from-literal=API_KEY=<new_key> \
       --dry-run=client -o yaml | kubectl apply -f -
  3. Restart the deployment to pick up the new value:
     kubectl rollout restart deployment/inference-api -n infergrid

Health and ready endpoints are exempt from auth so Kubernetes probes
can reach them without credentials.
"""

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Endpoints that bypass auth -- Kubernetes liveness/readiness probes
# and Prometheus scraping hit these without credentials
EXEMPT_PATHS = {"/health", "/ready", "/metrics"}


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    Raises 401 if the header is missing.
    Raises 403 if the key is present but incorrect.
    Returns the key on success.

    If API_KEY env var is not set (local dev without a secret), auth is
    skipped and a warning is logged. This prevents the API from being
    completely locked out in environments where the secret is not configured.
    """
    if not API_KEY:
        import logging

        logging.getLogger(__name__).warning(
            "API_KEY env var not set so authentication is disabled. "
            "Set API_KEY in your Kubernetes Secret before exposing this service."
        )
        return ""

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key
