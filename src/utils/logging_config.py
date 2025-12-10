"""
Structured logging configuration for stock agent system.

Provides consistent, structured logging across all components with:
- JSON formatting for easy parsing
- Context-rich log entries
- Performance tracking
- Error categorization
"""

import structlog
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output logs in JSON format for production
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class AgentLogger:
    """Logger with specialized methods for agent operations."""

    def __init__(self, agent_name: str):
        self.logger = get_logger(agent_name)
        self.agent_name = agent_name

    def log_tool_call(
        self,
        tool_name: str,
        ticker: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log a tool invocation."""
        self.logger.info(
            "tool_call",
            agent=self.agent_name,
            tool=tool_name,
            ticker=ticker,
            success=success,
            latency_ms=latency_ms,
            error=error,
            **kwargs
        )

    def log_agent_decision(
        self,
        ticker: str,
        decision: str,
        confidence: float,
        reasoning: str,
        **kwargs
    ):
        """Log an agent's decision."""
        self.logger.info(
            "agent_decision",
            agent=self.agent_name,
            ticker=ticker,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            **kwargs
        )

    def log_error(
        self,
        error_type: str,
        error_msg: str,
        ticker: Optional[str] = None,
        **kwargs
    ):
        """Log an error with context."""
        self.logger.error(
            "agent_error",
            agent=self.agent_name,
            error_type=error_type,
            error_msg=error_msg,
            ticker=ticker,
            **kwargs
        )

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        **kwargs
    ):
        """Log performance metrics."""
        self.logger.info(
            "performance",
            agent=self.agent_name,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )


class ToolLogger:
    """Logger with specialized methods for tool operations."""

    def __init__(self, tool_name: str):
        self.logger = get_logger(tool_name)
        self.tool_name = tool_name

    def log_fetch(
        self,
        ticker: str,
        data_type: str,
        success: bool,
        latency_ms: float,
        records_fetched: Optional[int] = None,
        error: Optional[str] = None,
        data_age_seconds: Optional[float] = None,
        source: Optional[str] = None,
        attempt: Optional[int] = None,
    ):
        """Log a data fetch operation."""
        self.logger.info(
            "data_fetch",
            tool=self.tool_name,
            ticker=ticker,
            data_type=data_type,
            success=success,
            latency_ms=latency_ms,
            records_fetched=records_fetched,
            error=error,
            data_age_seconds=data_age_seconds,
            source=source,
            attempt=attempt,
        )

    def log_validation(
        self,
        ticker: str,
        validation_type: str,
        passed: bool,
        reason: Optional[str] = None,
        **kwargs
    ):
        """Log a validation check."""
        self.logger.info(
            "data_validation",
            tool=self.tool_name,
            ticker=ticker,
            validation_type=validation_type,
            passed=passed,
            reason=reason,
            **kwargs
        )

    def log_retry(
        self,
        ticker: str,
        attempt: int,
        max_attempts: int,
        error: str,
        retryable: bool = True,
    ):
        """Log a retry attempt."""
        self.logger.warning(
            "retry_attempt",
            tool=self.tool_name,
            ticker=ticker,
            attempt=attempt,
            max_attempts=max_attempts,
            error=error,
            retryable=retryable,
        )


# Initialize logging on module import with default settings
configure_logging()
