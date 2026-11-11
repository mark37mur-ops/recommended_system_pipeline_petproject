"""Movie catalog held in memory: search, onboarding and the cold-start rules (spec 2.4).

~3.9k movies with their rating counts fit in a few MB, so search, popularity and cold start
need no database round trip and keep working while Postgres is down. The catalog is an
immutable snapshot; `CatalogHolder.refresh` swaps in a new one atomically.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

HIGH_RATING = 4  # cold start: genres of movies rated at least this count as liked
POPULAR_POOL = 500  # onboarding and cold start only pick from the most rated movies

_ARTICLE = re.compile(r"^(?P<name>.+), (?P<article>The|A|An)(?P<rest> \(.+\))?$")


def display_title(raw: str) -> str:
    """MovieLens stores "Matrix, The"; show "The Matrix" (alternative titles kept)."""
    match = _ARTICLE.match(raw)
    if match is None:
        return raw
    return f"{match['article']} {match['name']}{match['rest'] or ''}"


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str  # as stored (MovieLens form)
    year: int | None
    genres: tuple[str, ...]
    n_ratings: int = 0

    @property
    def display_title(self) -> str:
        return display_title(self.title)


class Catalog:
    def __init__(self, movies: Iterable[Movie]) -> None:
        self.by_popularity: list[Movie] = sorted(movies, key=lambda m: (-m.n_ratings, m.movie_id))
        self._by_id = {m.movie_id: m for m in self.by_popularity}
        self._search_keys = [
            (m, _normalize(m.display_title), _normalize(m.title)) for m in self.by_popularity
        ]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, movie_id: object) -> bool:
        return movie_id in self._by_id

    def get(self, movie_id: int) -> Movie | None:
        return self._by_id.get(movie_id)

    def popular(self, k: int, exclude: Iterable[int] = ()) -> list[int]:
        excluded = set(exclude)
        out: list[int] = []
        for movie in self.by_popularity:
            if len(out) >= k:
                break
            if movie.movie_id not in excluded:
                out.append(movie.movie_id)
        return out

    def search(self, query: str, limit: int = 20) -> list[Movie]:
        """Case-insensitive substring match on the shown and the stored title; titles that start
        with the query come first, each group in popularity order."""
        q = _normalize(query)
        if not q:
            return []
        prefix, inner = [], []
        for movie, shown, stored in self._search_keys:
            if shown.startswith(q) or stored.startswith(q):
                prefix.append(movie)
            elif q in shown or q in stored:
                inner.append(movie)
            if len(prefix) >= limit:
                break
        return (prefix + inner)[:limit]

    def onboarding(self, n: int = 20, pool: int = POPULAR_POOL) -> list[Movie]:
        """~n well-known movies of different genres for a visitor's first ratings.

        Round-robin over genres (ordered by how many ratings their top-`pool` movies have),
        taking the most rated not yet picked movie of each genre. ML-1M lists a movie's genres
        alphabetically, so "first genre" would not give variety; any genre of a movie counts."""
        top = self.by_popularity[:pool]
        weight: Counter[str] = Counter()
        for movie in top:
            for genre in movie.genres:
                weight[genre] += movie.n_ratings
        genres = sorted(weight, key=lambda g: (-weight[g], g))
        picks: list[Movie] = []
        picked: set[int] = set()
        while len(picks) < n:
            progressed = False
            for genre in genres:
                if len(picks) >= n:
                    break
                movie = next(
                    (m for m in top if genre in m.genres and m.movie_id not in picked), None
                )
                if movie is not None:
                    picks.append(movie)
                    picked.add(movie.movie_id)
                    progressed = True
            if not progressed:
                break
        return picks

    def genre_popular(
        self, ratings: Sequence[tuple[int, int]], k: int, pool: int = POPULAR_POOL
    ) -> list[int]:
        """Cold start for 1-4 ratings: popular movies in the genres the visitor rated high.

        Genre weight = number of the visitor's ratings >= HIGH_RATING on movies of that genre;
        a candidate (top-`pool` popular, not rated) scores the sum of weights of its genres,
        ties by popularity. No high rating -> plain popularity. Short lists are padded with
        popular movies."""
        rated = {movie_id for movie_id, _ in ratings}
        weights: Counter[str] = Counter()
        for movie_id, rating in ratings:
            movie = self._by_id.get(movie_id)
            if movie is not None and rating >= HIGH_RATING:
                weights.update(movie.genres)
        if not weights:
            return self.popular(k, exclude=rated)
        scored = []
        for movie in self.by_popularity[:pool]:
            if movie.movie_id in rated:
                continue
            score = sum(weights[g] for g in movie.genres)
            if score > 0:
                scored.append((score, movie.movie_id))
        scored.sort(key=lambda item: -item[0])  # stable: popularity order within a score
        out = [movie_id for _, movie_id in scored[:k]]
        if len(out) < k:
            out += self.popular(k - len(out), exclude=rated | set(out))
        return out


class MovieSource(Protocol):
    def load_movies(self) -> list[Movie]: ...


class CatalogHolder:
    """Current catalog snapshot; `refresh` builds a new one and swaps the reference."""

    def __init__(self) -> None:
        self.current = Catalog([])

    def refresh(self, source: MovieSource) -> int:
        catalog = Catalog(source.load_movies())
        self.current = catalog
        return len(catalog)
