"""
Webhook dispatcher — best-effort POST, never raises.
"""

import requests


def fire_webhook(url: str, payload: dict, timeout: int = 5) -> None:
    """POST payload to url. Failure is silent — webhook must never crash the pipeline."""
    try:
        requests.post(url, json=payload, timeout=timeout)
    except Exception:
        pass
