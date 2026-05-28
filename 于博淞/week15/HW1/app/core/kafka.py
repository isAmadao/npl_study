"""Module-level Kafka producer singleton — created once, reused across requests."""
import json

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from app.core.config import settings

_producer: KafkaProducer | None = None


def get_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode(),
                acks="all",           # wait for leader + replicas
                retries=3,
                request_timeout_ms=10_000,
            )
        except NoBrokersAvailable as e:
            raise RuntimeError(
                f"Kafka not reachable at {settings.kafka_bootstrap_servers}. "
                "Start Kafka first (docker compose up -d kafka)."
            ) from e
    return _producer


def close_producer() -> None:
    global _producer
    if _producer is not None:
        _producer.close()
        _producer = None
