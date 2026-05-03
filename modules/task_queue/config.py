"""RabbitMQ task queue configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os

from modules.error.common import ConfigurationError


@dataclass(frozen=True, slots=True)
class TaskQueueSettings:
    """Validated durable task queue settings."""

    amqp_url: str
    queue_name: str


def load_task_queue_settings() -> TaskQueueSettings:
    """Load RabbitMQ settings from environment."""
    amqp_url = os.getenv("AMQP_URL", "").strip()
    if not amqp_url:
        raise ConfigurationError("AMQP_URL must not be empty.")
    queue_name = os.getenv("MATCH_QUEUE_NAME", "career.match_jobs").strip()
    if not queue_name:
        raise ConfigurationError("MATCH_QUEUE_NAME must not be empty.")
    return TaskQueueSettings(amqp_url=amqp_url, queue_name=queue_name)
