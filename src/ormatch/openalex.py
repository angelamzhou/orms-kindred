"""Minimal OpenAlex client: retries/backoff, cursor paging, field projection.

The API key is read from OPENALEX_API_KEY (via .env or the environment) and is
only ever sent as a query parameter; it is never logged.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Iterator

import requests

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
CA_BUNDLE = "/root/.ccr/ca-bundle.crt"

WORK_FIELDS = [
    "id",
    "doi",
    "title",
    "publication_year",
    "primary_location",
    "authorships",
    "abstract_inverted_index",
    "open_access",
    "best_oa_location",
    "referenced_works_count",
    "cited_by_count",
]

RETRY_STATUSES = {429, 500, 502, 503, 504}


class OpenAlexError(RuntimeError):
    pass


def reconstruct_abstract(inv: dict[str, list[int]] | None) -> str | None:
    """Turn an abstract_inverted_index into plain text."""
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort()
    return " ".join(w for _, w in positions)


class OpenAlexClient:
    def __init__(
        self,
        api_key: str | None = None,
        mailto: str | None = None,
        max_tries: int = 5,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY")
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO")
        self.max_tries = max_tries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "ormatch/0.1 (python-requests)"
        self._verify: bool | str = True
        self.calls = 0

    # -- low level -----------------------------------------------------
    def _params(self, params: dict[str, Any]) -> dict[str, Any]:
        p = {k: v for k, v in params.items() if v is not None}
        if self.api_key:
            p["api_key"] = self.api_key
        if self.mailto:
            p["mailto"] = self.mailto
        return p

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(1, self.max_tries + 1):
            try:
                self.calls += 1
                r = self.session.get(url, params=self._params(params), timeout=self.timeout, verify=self._verify)
            except requests.exceptions.SSLError as e:
                if self._verify is True and os.path.exists(CA_BUNDLE):
                    log.warning("TLS verification failed; retrying with CA bundle %s", CA_BUNDLE)
                    self._verify = CA_BUNDLE
                    continue
                last_err = e
            except requests.RequestException as e:
                last_err = e
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code in RETRY_STATUSES:
                    retry_after = r.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    log.warning("HTTP %s on %s (attempt %d/%d); sleeping %.1fs", r.status_code, path, attempt, self.max_tries, wait)
                    time.sleep(wait + random.uniform(0, 0.5))
                    delay = min(delay * 2, 60)
                    continue
                # non-retryable
                raise OpenAlexError(f"HTTP {r.status_code} for {path}: {r.text[:300]}")
            log.warning("request error on %s (attempt %d/%d): %s", path, attempt, self.max_tries, type(last_err).__name__)
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 60)
        raise OpenAlexError(f"gave up after {self.max_tries} tries on {path}: {last_err}")

    # -- paging ----------------------------------------------------------
    def paginate(
        self,
        endpoint: str,
        filter: str | None = None,
        select: list[str] | None = None,
        per_page: int = 100,
        cursor: str = "*",
        sort: str | None = None,
        **extra: Any,
    ) -> Iterator[tuple[list[dict[str, Any]], str | None]]:
        """Yield (page_results, next_cursor) tuples until exhausted.

        Yielding the cursor alongside each page lets callers persist it after
        writing the page, so a rerun can resume exactly where it stopped.
        """
        while cursor:
            data = self.get(
                endpoint,
                filter=filter,
                select=",".join(select) if select else None,
                per_page=min(per_page, 100),
                cursor=cursor,
                sort=sort,
                **extra,
            )
            results = data.get("results", [])
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not results:
                yield results, None
                return
            yield results, next_cursor
            cursor = next_cursor

    def works(self, filter: str, select: list[str] | None = None, cursor: str = "*", **kw: Any):
        return self.paginate("works", filter=filter, select=select or WORK_FIELDS, cursor=cursor, **kw)

    def count_works(self, filter: str) -> int:
        data = self.get("works", filter=filter, per_page=1, select="id")
        return int(data["meta"]["count"])
