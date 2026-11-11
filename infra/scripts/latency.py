"""Rough latency probe for a running API (stdlib only).

Creates `--users` fresh visitors; each rates 5 onboarding movies (POST /ratings), then asks
for recommendations twice: the first GET is a cache miss (fold-in ALS + ranker), the second a
cache hit. Prints p50 / p95 / max per request type. Visitors share one IP, so run the API with
RATE_LIMIT_PER_MINUTE raised for the probe (default limit is 30 ratings/min per IP).

    python infra/scripts/latency.py --base-url http://localhost:8000 --users 100
"""

import argparse
import http.cookiejar
import json
import random
import statistics
import time
import urllib.request


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def timed(opener, request) -> tuple[float, dict]:
    start = time.perf_counter()
    with opener.open(request, timeout=30) as response:
        body = json.loads(response.read())
    return (time.perf_counter() - start) * 1000, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    base = args.base_url.rstrip("/")

    anonymous = urllib.request.build_opener()
    pool = [m["movie_id"] for m in timed(anonymous, f"{base}/movies/onboarding")[1]["items"]]
    timings: dict[str, list[float]] = {"POST /ratings": [], "GET /recommendations (miss)": []}
    timings["GET /recommendations (hit)"] = []
    paths: dict[str, int] = {}
    for _ in range(args.users):
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        for movie_id in rng.sample(pool, 5):
            body = json.dumps({"movie_id": movie_id, "rating": rng.randint(3, 5)}).encode()
            request = urllib.request.Request(
                f"{base}/ratings", data=body, headers={"Content-Type": "application/json"}
            )
            timings["POST /ratings"].append(timed(opener, request)[0])
        for kind in ("miss", "hit"):
            ms, recs = timed(opener, f"{base}/recommendations?k=10")
            expected_cached = kind == "hit"
            if recs["cached"] != expected_cached:
                raise SystemExit(f"expected cached={expected_cached}, got {recs}")
            timings[f"GET /recommendations ({kind})"].append(ms)
            paths[recs["path"]] = paths.get(recs["path"], 0) + 1

    print(f"{args.users} visitors, paths served: {paths}")
    print(f"{'request':32} {'n':>5} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
    for name, values in timings.items():
        print(
            f"{name:32} {len(values):5d} {statistics.median(values):8.1f} "
            f"{percentile(values, 0.95):8.1f} {max(values):8.1f}"
        )


if __name__ == "__main__":
    main()
