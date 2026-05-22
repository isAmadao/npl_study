from __future__ import annotations

import json
import time
from pathlib import Path

from rag_mvp.config import settings
from rag_mvp.types import DispatchResult, FileTask


class TaskQueue:
    def __init__(self) -> None:
        settings.ensure_directories()

    def publish(self, task: FileTask) -> DispatchResult:
        try:
            self._publish_kafka(task)
            return DispatchResult(backend="kafka", message="任务已投递到 Kafka。")
        except Exception as exc:
            self._publish_local(task)
            return DispatchResult(
                backend="local",
                message=f"Kafka 不可用，已自动写入本地队列。原因：{exc}",
            )

    def get_next_task(self) -> FileTask | None:
        kafka_task = self._poll_kafka()
        if kafka_task is not None:
            return kafka_task
        return self._poll_local()

    def _publish_kafka(self, task: FileTask) -> None:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            request_timeout_ms=2000,
        )
        future = producer.send(settings.kafka_topic, task.to_dict())
        future.get(timeout=3)
        producer.flush()
        producer.close()

    def _publish_local(self, task: FileTask) -> None:
        filename = f"{int(time.time() * 1000)}_{task.file_id}.json"
        file_path = settings.local_queue_dir / filename
        file_path.write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _poll_kafka(self) -> FileTask | None:
        from kafka import KafkaConsumer

        try:
            consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id=settings.kafka_group_id,
                consumer_timeout_ms=1500,
                request_timeout_ms=2000,
            )
        except Exception:
            return None

        try:
            for message in consumer:
                return FileTask.from_dict(message.value)
        finally:
            consumer.close()
        return None

    def _poll_local(self) -> FileTask | None:
        queue_files = sorted(Path(settings.local_queue_dir).glob("*.json"))
        if not queue_files:
            return None
        queue_file = queue_files[0]
        payload = json.loads(queue_file.read_text(encoding="utf-8"))
        queue_file.unlink(missing_ok=True)
        return FileTask.from_dict(payload)

