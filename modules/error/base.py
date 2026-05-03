"""Base wrapped error type."""


class WrappedError(Exception):
    """Base exception that can wrap another error."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        """Initialize wrapped error with message and optional cause."""
        super().__init__(message)
        self.cause = cause
