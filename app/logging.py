import logging

import structlog

# Field names that must never reach a log line unredacted. Extend as new
# PHI-bearing fields are introduced (e.g. once patient names/DOB show up).
_PHI_FIELDS = {"patient_id", "value", "raw_payload", "notes"}


def _redact_phi(_logger: object, _method_name: str, event_dict: dict) -> dict:
    for field in _PHI_FIELDS:
        if field in event_dict:
            event_dict[field] = "[REDACTED]"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_phi,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(*args, **kwargs) -> structlog.BoundLogger:
    return structlog.get_logger(*args, **kwargs)
