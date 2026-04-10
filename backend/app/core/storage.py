"""
S3-compatible object storage client (Neva Objects).
"""

import boto3
from botocore.client import Config

from app.core.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
        )
    return _client


def upload_file(local_path: str, key: str) -> str:
    """Upload a local file to S3 and return its public URL."""
    get_client().upload_file(
        local_path,
        settings.s3_bucket,
        key,
        ExtraArgs={"ACL": "public-read"},
    )
    return f"{settings.s3_endpoint_url}/{settings.s3_bucket}/{key}"
