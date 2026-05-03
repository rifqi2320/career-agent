from modules.error.base import BaseError


class IncorrectCombinationError(BaseError):
    """Raised when an incorrect combination of options is provided."""

class UnknownOptionsError(BaseError):
    """Raised when an unrecognized option is provided."""

class ValidationError(BaseError):
    """Raised when validation of input data fails."""


class RetryableModelOutputError(BaseError):
    """Raised when model output is malformed and a retry may succeed."""
