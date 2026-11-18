"""Bounded offline catalog enrichment; credentials are read only by the CLI."""

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from recsys.db.schema import make_engine
from recsys.movie_metadata import (
    CatalogEntry,
    MovieMetadata,
    load_enrichment_catalog,
    load_metadata,
    match_movie,
    normalize_title,
    poster_url,
    upsert_metadata,
)


class EnrichmentError(Exception):
    """A credential-free operational error."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a Bearer header to a redirect target.


@dataclass
class _RequestBudget:
    remaining: int = 3


class TmdbClient:
    def __init__(self, token: str, *, transport=None, sleep=time.sleep, clock=time.monotonic):
        self._token = token
        self._transport = transport or build_opener(_NoRedirect()).open
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def search(self, title: str, year: int) -> list[dict]:
        return self._search(title, year, _RequestBudget())

    def _search(self, title: str, year: int, budget: _RequestBudget) -> list[dict]:
        query = urlencode(
            {"query": title, "year": year, "language": "ru-RU", "include_adult": "false"}
        )
        return self._get(f"search/movie?{query}", "results", budget)["results"]

    def resolve(self, title: str, year: int) -> dict | None:
        """Resolve exact titles/verified aliases within one aggregate HTTP budget."""
        budget = _RequestBudget()
        query = normalize_title(title.split("(", 1)[0])
        candidates = self._search(query, year, budget)
        match = match_movie(title, year, candidates)
        if match is not None:
            return match
        if any(match_movie(title, year, [item]) is not None for item in candidates):
            return None  # Already ambiguous exact aliases must never be overwritten.

        eligible = {}
        for item in candidates:
            if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] <= 0:
                continue
            date = item.get("release_date")
            if (
                isinstance(date, str)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
                and int(date[:4]) == year
            ):
                eligible[item["id"]] = item

        # Never accept the first alias while leaving a competitor unchecked.
        if len(eligible) > budget.remaining:
            return None
        verified = []
        for movie_id, item in eligible.items():
            if not budget.remaining:
                return None
            payload = self._get(f"movie/{movie_id}/alternative_titles", "titles", budget)
            if type(payload.get("id")) is not int or payload["id"] != movie_id:
                raise EnrichmentError("Invalid TMDB alternative titles response")
            verified.append({**item, "alternative_titles": payload})
        return match_movie(title, year, verified)

    def _get(self, path: str, field: str, budget: _RequestBudget) -> dict:
        request = Request(
            f"https://api.themoviedb.org/3/{path}",
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
        )
        for attempt in range(budget.remaining):
            now = self._clock()
            if self._last_request is not None:
                wait = 0.5 - (now - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            delay = float(2**attempt)
            budget.remaining -= 1
            try:
                with self._transport(request, timeout=10) as response:
                    data = response.read(2_000_001)
                if len(data) > 2_000_000:
                    raise EnrichmentError("TMDB response too large")
                payload = json.loads(data)
                if not isinstance(payload, dict) or not isinstance(payload.get(field), list):
                    raise EnrichmentError("Invalid TMDB response")
                return payload
            except HTTPError as error:
                code = error.code
                header = (error.headers or {}).get("Retry-After", "0")
                error.close()
                if code not in (429,) and not 500 <= code <= 599:
                    raise EnrichmentError(f"TMDB HTTP {code}") from None
                try:
                    retry_after = float(header)
                    if math.isfinite(retry_after):
                        delay = max(delay, min(30, max(0, retry_after)))
                except (TypeError, ValueError):
                    pass
            except (URLError, TimeoutError, OSError):
                pass
            except (ValueError, TypeError):
                raise EnrichmentError("Invalid TMDB response") from None
            if not budget.remaining:
                raise EnrichmentError("TMDB request failed after 3 attempts") from None
            self._sleep(delay)
        raise EnrichmentError("TMDB request failed")  # pragma: no cover


def build_queue(
    catalog: list[CatalogEntry],
    existing: dict[int, MovieMetadata],
    priority_ids: list[int],
    limit: int,
    *,
    refresh: bool = False,
    after_movie_id: int | None = None,
) -> list[CatalogEntry]:
    if not 1 <= limit <= 4000:
        raise ValueError("limit must be between 1 and 4000")
    by_id = {entry.movie_id: entry for entry in catalog}
    ordered = [by_id[i] for i in dict.fromkeys(priority_ids) if i in by_id]
    ordered.extend(sorted(catalog, key=lambda entry: (-entry.n_ratings, entry.movie_id)))
    ordered = list({entry.movie_id: entry for entry in ordered}.values())
    if after_movie_id is not None:
        if type(after_movie_id) is not int or after_movie_id not in by_id:
            raise ValueError("cursor movie id is absent from catalog")
        cursor = next(i for i, entry in enumerate(ordered) if entry.movie_id == after_movie_id)
        ordered = ordered[cursor + 1 :]
    seen = set() if refresh else set(existing)
    queue = []
    for entry in ordered:
        if entry.movie_id not in seen:
            seen.add(entry.movie_id)
            queue.append(entry)
            if len(queue) == limit:
                break
    return queue


def enrich(
    engine,
    client: TmdbClient,
    *,
    limit=100,
    priority_ids=(),
    refresh=False,
    dry_run=False,
    after_movie_id: int | None = None,
):
    queue = build_queue(
        load_enrichment_catalog(engine),
        load_metadata(engine),
        list(priority_ids),
        limit,
        refresh=refresh,
        after_movie_id=after_movie_id,
    )
    summary = {"processed": 0, "matched": 0, "unmatched": 0, "errors": 0}
    for entry in queue:
        summary["processed"] += 1
        if entry.year is None:
            summary["unmatched"] += 1
            continue
        try:
            item = client.resolve(entry.title, entry.year)
            if item is None:
                summary["unmatched"] += 1
                continue
            localized = item.get("title")
            title_ru = (
                localized.strip()
                if isinstance(localized, str) and any("\u0400" <= c <= "\u04ff" for c in localized)
                else None
            )
            path = item.get("poster_path")
            metadata = MovieMetadata(
                entry.movie_id, item["id"], path if poster_url(path) else None, title_ru
            )
            if not dry_run:
                upsert_metadata(engine, metadata)
            summary["matched"] += 1
        except Exception:
            # Transport/SQL exceptions may contain credentials: never log their text or repr.
            summary["errors"] += 1
    return summary


def _limit(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 4000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 4000")
    return number


def _priority_ids(value: str) -> list[int]:
    try:
        ids = [int(part) for part in value.split(",")]
        if not ids or any(i <= 0 for i in ids):
            raise ValueError
        return list(dict.fromkeys(ids))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "priority ids must be positive comma-separated integers"
        ) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Enrich existing movies with TMDB metadata")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--limit", type=_limit, default=100)
    parser.add_argument("--priority-ids", type=_priority_ids, default=[])
    parser.add_argument(
        "--after-movie-id",
        type=int,
        help="Continue after this catalog movie ID in the same priority/popularity order",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    engine = None
    try:
        token = args.token_file.read_text(encoding="utf-8-sig").strip()
        if not token or any(c.isspace() for c in token):
            raise EnrichmentError("Invalid token file")
        engine = make_engine(os.environ["DATABASE_URL"])
        summary = enrich(
            engine,
            TmdbClient(token),
            limit=args.limit,
            priority_ids=args.priority_ids,
            refresh=args.refresh,
            dry_run=args.dry_run,
            after_movie_id=args.after_movie_id,
        )
    except Exception:
        summary = {"errors": 1, "status": "enrichment failed"}
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                summary = {"errors": 1, "status": "enrichment failed"}
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
