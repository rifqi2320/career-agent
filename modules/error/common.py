"""Common domain errors."""

from modules.error.base import WrappedError


class UnknownOptionsError(WrappedError):
    """Raised when unknown options are provided."""


class IncorrectCombinationError(WrappedError):
    """Raised when incompatible options are provided together."""