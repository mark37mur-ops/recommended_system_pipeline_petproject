"""Synthetic visitors for the monitoring e2e (stdlib only): traffic for every dashboard panel.

Per visitor (own cookie jar):
- every 4th visitor is a cold-start one: rates 2 onboarding movies, gets popular-by-genre
  recommendations (path=cold_start);
- the others rate 5-8 onboarding movies (path=full from the model), then rate 2-3 of the movies
  they were recommended (the online metric: "rated within an hour of being shown"), and ask
  again (cache miss after the new ratings, then a cache hit).
Plus anonymous traffic: recommendations without a cookie, searches and a few unknown URLs
(404, endpoint label "unmatched").

Ratings (RATE_LIMIT_PER_MINUTE, default 30) and recommendations
(RECOMMENDATIONS_RATE_LIMIT_PER_MINUTE, default 120) are rate-limited per IP: on 429 the
script waits for Retry-After. Raise the limits for a quick run, e.g. RATE_LIMIT_PER_MINUTE=600
in the environment of `docker compose up`.

    python infra/scripts/traffic.py --base-url http://localhost:8000 --visitors 8
"""

import argparse
import collections
import http.cookiejar
import json
import random
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, stats: collections.Counter, cookies: bool = True) -> None:
        handlers = (
            [urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())] if cookies else []
        )
        self.opener = urllib.request.build_opener(*handlers)
        self.base = base
        self.stats = stats

    def call(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if body is not None else {}
        for _ in range(5):
            request = urllib.request.Request(self.base + path, data=data, headers=headers)
            try:
                with self.opener.open(request, timeout=30) as response:
                    status, raw = response.status, response.read()
            except urllib.error.HTTPError as exc:
                status, raw = exc.code, exc.read()
                if status == 429:
                    wait = int(exc.headers.get("Retry-After", "60"))
                    print(f"  429 on {path}, waiting {wait}s (rate limit)")
                    self.stats["429"] += 1
                    time.sleep(wait)
                    continue
            self.stats[str(status)] += 1
            try:
                return status, json.loads(raw) if raw else {}
            except ValueError:
                return status, {}
        raise RuntimeError(f"{path}: still rate-limited")


def visitor(base, rng, stats, paths, cold: bool) -> None:
    client = Client(base, stats)
    _, onboarding = client.call("/movies/onboarding")
    pool = [m["movie_id"] for m in onboarding["items"]]
    for movie_id in rng.sample(pool, 2 if cold else rng.randint(5, 8)):
        client.call("/ratings", {"movie_id": movie_id, "rating": rng.randint(3, 5)})
    _, recs = client.call("/recommendations?k=10")
    paths[recs["path"]] += 1
    if cold:
        return
    shown = [m["movie_id"] for m in recs["items"]]
    for movie_id in rng.sample(shown, rng.randint(2, 3)):  # engagement with what was shown
        client.call("/ratings", {"movie_id": movie_id, "rating": rng.randint(3, 5)})
    for _ in range(2):  # miss (ratings invalidated the cache), then hit
        _, recs = client.call("/recommendations?k=10")
        paths[recs["path"] + (" (cached)" if recs["cached"] else "")] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--visitors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    base = args.base_url.rstrip("/")
    stats: collections.Counter = collections.Counter()
    paths: collections.Counter = collections.Counter()

    anonymous = Client(base, stats, cookies=False)
    for i in range(args.visitors):
        visitor(base, rng, stats, paths, cold=i % 4 == 0)
        _, recs = anonymous.call("/recommendations?k=10")
        paths["anonymous " + recs["path"]] += 1
        anonymous.call("/movies/search?q=" + rng.choice(["star", "love", "matrix", "toy"]))
        anonymous.call(f"/no-such-page/{i}")
    print("HTTP status counts:", dict(sorted(stats.items())))
    print("recommendation paths:", dict(sorted(paths.items())))


if __name__ == "__main__":
    main()
