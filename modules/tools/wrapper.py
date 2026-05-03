"""Helpers to unwrap `safe_result` results for tool-facing responses."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

ToolWrappedOutput = T | dict[str, str]


def _unwrap_safe_result(result: Any) -> Any:
    """Unwrap `safe_result` values, mapping failures to `{\"error\": ...}`."""
    is_err_method = getattr(result, "is_err", None)
    if callable(is_err_method):
        if result.is_err():
            return {"error": str(result.error)}
        return result.value
    return result


def wrap_safe_tool(
    func: Any,
) -> Any:
    """Wrap a `@safe` or `@safe_async` function for tool-friendly output."""
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolWrappedOutput[Any]:
            result = await func(*args, **kwargs)
            return _unwrap_safe_result(result)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolWrappedOutput[Any]:
        result = func(*args, **kwargs)
        return _unwrap_safe_result(result)

    return sync_wrapper
