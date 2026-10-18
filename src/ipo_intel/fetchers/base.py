from __future__ import annotations


class FetchError(Exception):
    """Raised when a source can't be read. The pipeline records it as a warning and carries on."""


class Fetcher:
    name = "base"

    def fetch(self, ipo_name: str) -> dict:
        """Return a dict of standard field names (see schema.py) found for this IPO."""
        raise NotImplementedError
