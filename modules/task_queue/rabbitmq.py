"""RabbitMQ publisher and consumer helpers for match jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from modules.task_queue.config import TaskQueueSettings, load_task_queue_settings

JobMessageHandler = Callable[[str], Awaitable[None]]


async def publish_match_job(
    job_id: str,
    *,
    settings: TaskQueueSettings | None = None,
) -> None:
    """Publish one durable match-job message."""
    await publish_match_jobs([job_id], settings=settings)


async def publish_match_jobs(
    job_ids: list[str],
    *,
    settings: TaskQueueSettings | None = None,
) -> None:
    """Publish durable match-job messages using one queue connection."""
    normalized_job_ids = [job_id.strip() for job_id in job_ids if job_id.strip()]
    if not normalized_job_ids:
        return

    resolved_settings = settings or load_task_queue_settings()
    connection = await aio_pika.connect_robust(resolved_settings.amqp_url)
    async with connection:
        channel = await connection.channel()
        await channel.declare_queue(resolved_settings.queue_name, durable=True)
        for job_id in normalized_job_ids:
            message = aio_pika.Message(
                body=job_id.encode("utf-8"),
                content_type="text/plain",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await channel.default_exchange.publish(
                message,
                routing_key=resolved_settings.queue_name,
            )


async def consume_match_jobs(
    handler: JobMessageHandler,
    *,
    settings: TaskQueueSettings | None = None,
    prefetch_count: int = 1,
) -> None:
    """Consume durable match-job messages forever."""
    resolved_settings = settings or load_task_queue_settings()
    connection = await aio_pika.connect_robust(resolved_settings.amqp_url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=prefetch_count)
        queue = await channel.declare_queue(
            resolved_settings.queue_name,
            durable=True,
        )
        async with queue.iterator() as queue_iterator:
            async for message in queue_iterator:
                await _handle_message(message, handler)


async def _handle_message(
    message: AbstractIncomingMessage,
    handler: JobMessageHandler,
) -> None:
    async with message.process(requeue=False):
        job_id = message.body.decode("utf-8").strip()
        if job_id:
            await handler(job_id)
