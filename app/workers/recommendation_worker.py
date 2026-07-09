"""SQS consumer: picks up `lab_result_created` events and runs the
recommendation chain for each one. Long-polls forever by default; retry/
backoff on failure is Phase 7 — for now a failed message is simply left
un-deleted so SQS's visibility timeout redelivers it.
"""

import asyncio
import json

from app.events.sqs import get_sqs_client, resolve_queue_url
from app.logging import get_logger
from app.recommendations.chain import generate_recommendation

logger = get_logger(__name__)


async def _handle_message(message: dict) -> None:
    event = json.loads(message["Body"])
    lab_result_id = event["lab_result_id"]
    recommendation_id = await generate_recommendation(lab_result_id)
    logger.info(
        "recommendation_generated",
        lab_result_id=lab_result_id,
        recommendation_id=recommendation_id,
    )


async def run_worker(max_messages: int | None = None) -> None:
    """Long-poll the queue. Pass `max_messages` to process a bounded number
    and return (used by tests/CLI); omit it to run forever.
    """
    client = get_sqs_client()
    queue_url = resolve_queue_url()
    processed = 0

    while max_messages is None or processed < max_messages:
        response = await asyncio.to_thread(
            client.receive_message,
            QueueUrl=queue_url,
            WaitTimeSeconds=10,
            MaxNumberOfMessages=1,
        )
        messages = response.get("Messages", [])
        if not messages:
            continue

        for message in messages:
            try:
                await _handle_message(message)
            except Exception:
                logger.exception("recommendation_generation_failed", message_id=message["MessageId"])
                continue

            await asyncio.to_thread(
                client.delete_message,
                QueueUrl=queue_url,
                ReceiptHandle=message["ReceiptHandle"],
            )
            processed += 1


if __name__ == "__main__":
    asyncio.run(run_worker())
