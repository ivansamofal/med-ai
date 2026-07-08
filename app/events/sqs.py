import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.config import settings

_client = None
_queue_url: str | None = None


def get_sqs_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "sqs",
            region_name=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    return _client


def _resolve_queue_url() -> str:
    """Look up the queue URL, creating the queue if we're pointed at LocalStack.

    Real AWS environments are expected to provision the queue out-of-band (e.g.
    Terraform); we only auto-create when a custom (LocalStack) endpoint is set,
    to keep `docker-compose up && make test` a one-command local workflow.
    """
    global _queue_url
    if _queue_url is not None:
        return _queue_url

    client = get_sqs_client()
    try:
        _queue_url = client.get_queue_url(QueueName=settings.sqs_lab_result_queue_name)["QueueUrl"]
    except ClientError:
        if settings.aws_endpoint_url is None:
            raise
        _queue_url = client.create_queue(QueueName=settings.sqs_lab_result_queue_name)["QueueUrl"]
    return _queue_url


def _publish_sync(event: dict[str, Any]) -> str:
    client = get_sqs_client()
    queue_url = _resolve_queue_url()
    response = client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
    return response["MessageId"]


async def publish_lab_result_created(lab_result_id: str, patient_id: str, is_abnormal: bool) -> str:
    """Publish the `lab_result_created` event that downstream services (the
    Phase 3 recommendation worker) consume. boto3 is sync, so we push the call
    to a worker thread to keep the FastAPI event loop unblocked.
    """
    event = {
        "event_type": "lab_result_created",
        "lab_result_id": lab_result_id,
        "patient_id": patient_id,
        "is_abnormal": is_abnormal,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    return await asyncio.to_thread(_publish_sync, event)


def reset_sqs_state() -> None:
    """Test-only: drop cached client/queue URL so fixtures get a clean slate."""
    global _client, _queue_url
    _client = None
    _queue_url = None
