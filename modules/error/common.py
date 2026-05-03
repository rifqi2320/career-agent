from modules.error.base import BaseError


class IncorrectCombinationError(BaseError):
    """Raised when an incorrect combination of options is provided."""


class UnknownOptionsError(BaseError):
    """Raised when an unrecognized option is provided."""


class ValidationError(BaseError, ValueError):
    """Raised when validation of input data fails."""


class ConfigurationError(BaseError):
    """Raised when runtime configuration is missing or invalid."""


class DependencyError(BaseError):
    """Raised when an internal or external dependency cannot be used."""


class ToolError(BaseError):
    """Base error for tool-facing failures."""

    retriable: bool = False

    def __init__(
        self,
        message: str,
        *,
        original_error: BaseException | None = None,
        retriable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        if retriable is not None:
            self.retriable = retriable

    @property
    def original_error_type(self) -> str | None:
        """Return the original exception type name when this wraps another error."""
        if self.original_error is None:
            return None
        return type(self.original_error).__name__


class RetryableToolError(ToolError):
    """Raised when a tool failure may succeed on retry."""

    retriable = True


class NonRetryableToolError(ToolError):
    """Raised when retrying the same tool call is not expected to help."""

    retriable = False


class ToolInputError(NonRetryableToolError):
    """Raised when tool input or state is invalid."""


class ToolExecutionError(NonRetryableToolError):
    """Raised when a tool dependency fails after valid input was accepted."""


class ToolTimeoutError(RetryableToolError):
    """Raised when a tool call times out."""


class UnexpectedToolError(NonRetryableToolError):
    """Raised when an unexpected exception crosses the tool boundary."""


class RetryableModelOutputError(RetryableToolError):
    """Raised when model output is malformed and a retry may succeed."""
