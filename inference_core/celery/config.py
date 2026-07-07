"""
Celery Configuration
"""

import os

from kombu import Exchange, Queue


class CeleryConfig:
    """Celery configuration class"""

    # Broker settings
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # Connection pool settings
    broker_pool_limit = 20
    broker_connection_retry_on_startup = True
    broker_connection_retry = True
    broker_connection_max_retries = 10

    # Task settings
    task_serializer = "json"
    result_serializer = "json"
    accept_content = ["json"]
    timezone = "UTC"
    enable_utc = True

    # Performance optimizations
    task_acks_late = True
    worker_prefetch_multiplier = 1
    task_reject_on_worker_lost = True

    # Retry configuration
    task_default_retry_delay = 60
    task_max_retries = 3

    # Result backend settings
    result_expires = 3600  # 1 hour
    result_persistent = True

    # Queue configuration with priority
    task_routes = {
        "inference_core.celery.tasks.batch_tasks.*": {"queue": "batch_tasks"},
        "inference_core.celery.tasks.email_tasks.*": {"queue": "mail"},
        "inference_core.celery.tasks.embedding_tasks.*": {"queue": "embeddings"},
        # PII detection runs on its own opt-in prefork worker (--profile
        # local-pii); by-name route so it is not swept into 'embeddings'.
        "pii.detect": {"queue": "pii"},
    }

    task_default_queue = "default"
    task_queues = [
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("batch_tasks", Exchange("batch_tasks"), routing_key="batch_tasks"),
        Queue("mail", Exchange("mail"), routing_key="mail"),
        Queue("embeddings", Exchange("embeddings"), routing_key="embeddings"),
        Queue("pii", Exchange("pii"), routing_key="pii"),
    ]

    # Monitoring
    worker_send_task_events = True
    task_send_sent_event = True

    # Redis-specific optimizations
    broker_transport_options = {
        "visibility_timeout": 3600,  # 1 hour
        "fanout_prefix": True,
        "fanout_patterns": True,
    }

    result_backend_transport_options = {"retry_policy": {"timeout": 5.0}}


# Beat schedule configuration
def get_beat_schedule():
    """Get the beat schedule configuration with dynamic polling interval"""
    try:
        from inference_core.llm.config import get_llm_config

        llm_config = get_llm_config()

        poll_interval = llm_config.batch_config.default_poll_interval_seconds
    except Exception:
        # Fallback to default if config is not available
        poll_interval = 30

    return {
        "batch-poll": {
            "task": "batch.poll",
            "schedule": float(poll_interval),
            "options": {"queue": "batch_tasks"},
        },
        "batch-dispatch": {
            "task": "batch.dispatch",
            # Prefer faster dispatch than poll (submit quickly, then poll at its own cadence)
            "schedule": float(
                getattr(
                    llm_config.batch_config, "default_dispatch_interval_seconds", 15
                )
                if "llm_config" in globals()
                else 15
            ),
            "options": {"queue": "batch_tasks"},
        },
    }


# Set beat schedule on the class
CeleryConfig.beat_schedule = get_beat_schedule()
