"""A polite HTTP client: honours robots.txt, rate-limits, retries, and keeps raw responses for debugging."""
from __future__ import annotations

import hashlib
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .. import config
from .base import FetchError


class PoliteSession:
    def __init__(self, min_delay: float | None = None, respect_robots: bool = True, keep_raw: bool = True):
        self.min_delay = config.MIN_REQUEST_DELAY_S if min_delay is None else min_delay
        self.respect_robots = respect_robots
        self.keep_raw = keep_raw
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"})
        retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self.s.mount("http://", HTTPAdapter(max_retries=retry))
        self._last = 0.0
        self._robots: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        host = urlparse(url)
        base = f"{host.scheme}://{host.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            try:
                r = self.s.get(f"{base}/robots.txt", timeout=15)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
                self._robots[base] = rp
            except requests.RequestException:
                self._robots[base] = None  # can't read robots.txt; the real request will fail loudly if the site is down
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(config.USER_AGENT, url)

    def get(self, url: str, as_json: bool = False):
        if not self.allowed(url):
            raise FetchError(f"robots.txt disallows fetching {url}")
        wait = self.min_delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.get(url, timeout=25)
        except requests.RequestException as e:
            raise FetchError(f"{url}: {e}") from e
        finally:
            self._last = time.time()
        if r.status_code >= 400:
            raise FetchError(f"{url}: HTTP {r.status_code}")
        if self.keep_raw:
            config.ensure_dirs()
            path = config.RAW_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:12]}.txt"
            path.write_text(f"{url}\n\n{r.text}", encoding="utf-8")
        try:
            return r.json() if as_json else r.text
        except ValueError as e:
            raise FetchError(f"{url}: response is not JSON") from e
