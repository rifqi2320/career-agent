from safe_result import safe
from urllib.parse import urlparse, ParseResult

MAXIMUM_URL_LENGTH = 4096

@safe
def validate_url(url: str) -> ParseResult:
    if len(url) > MAXIMUM_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length of {MAXIMUM_URL_LENGTH} characters.")

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        msg = "Invalid URL. Provide an absolute URL such as https://example.com"
        raise ValueError(msg)
    if parsed.scheme not in {"http", "https"}:
        msg = "Unsupported URL scheme. Only http and https are supported"
        raise ValueError(msg)
    return parsed