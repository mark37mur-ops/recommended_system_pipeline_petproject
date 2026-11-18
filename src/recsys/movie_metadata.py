"""Strict catalog matching and isolated, additive TMDB metadata storage."""

import re
import unicodedata
from dataclasses import dataclass

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert

from recsys.db.schema import movie_metadata, movies, ratings


@dataclass(frozen=True)
class MovieMetadata:
    movie_id: int
    tmdb_id: int
    poster_path: str | None
    title_ru: str | None
    source: str = "tmdb"


@dataclass(frozen=True)
class CatalogEntry:
    movie_id: int
    title: str
    year: int | None
    n_ratings: int


def normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title).casefold().strip()
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    title = title.rstrip(".!? ")
    article = re.fullmatch(
        r"(.+),\s*(the|an|a|le|la|les|el|los|las|il|lo|gli|i|der|die|das)", title
    )
    if article:
        title = f"{article[2]} {article[1]}"
    return " ".join("".join(c if c.isalnum() else " " for c in title).split())


def _titles(title: str) -> set[str]:
    parts = [re.sub(r"\([^)]*\)", "", title)]
    for part in re.findall(r"\(([^)]*)\)", title):
        if re.fullmatch(r"\s*\d{4}\s*", part):
            continue
        if re.match(r"^a\.k\.a\.\s*", part, flags=re.IGNORECASE):
            # Commas separate aliases, except a trailing MovieLens article.
            parts.extend(re.split(r",(?!\s*(?:the|a|an)\s*(?:,|$))\s*", part, flags=re.IGNORECASE))
        else:
            parts.append(part)
    return {
        key
        for part in parts
        if (key := normalize_title(re.sub(r"^a\.k\.a\.\s*", "", part, flags=re.IGNORECASE)))
    }


def match_movie(title: str, year: int | None, candidates: list[dict]) -> dict | None:
    """Accept one exact title/alias AND release year; ambiguity never picks a poster."""
    if year is None:
        return None
    wanted = _titles(title)
    matches = {}
    for item in candidates:
        if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] <= 0:
            continue
        date = item.get("release_date")
        if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            continue
        if int(date[:4]) != year:
            continue
        names = [item.get("title"), item.get("original_title")]
        alternatives = item.get("alternative_titles", [])
        if isinstance(alternatives, dict):
            alternatives = alternatives.get("titles", [])
        if isinstance(alternatives, list):
            names.extend(a.get("title") if isinstance(a, dict) else a for a in alternatives)
        keys = set().union(*(_titles(n) for n in names if isinstance(n, str)))
        if wanted & keys:
            matches[item["id"]] = item
    return next(iter(matches.values())) if len(matches) == 1 else None


def poster_url(poster_path: str | None) -> str | None:
    if not isinstance(poster_path, str) or not re.fullmatch(
        r"/[A-Za-z0-9_-]+\.(jpg|jpeg|png|webp)", poster_path
    ):
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def load_metadata(engine: Engine) -> dict[int, MovieMetadata]:
    with engine.connect() as conn:
        rows = conn.execute(select(movie_metadata)).mappings()
        return {
            row["movie_id"]: MovieMetadata(
                **{k: row[k] for k in ("movie_id", "tmdb_id", "poster_path", "title_ru", "source")}
            )
            for row in rows
        }


def upsert_metadata(engine: Engine, item: MovieMetadata) -> None:
    stmt = insert(movie_metadata).values(
        movie_id=item.movie_id,
        tmdb_id=item.tmdb_id,
        poster_path=item.poster_path,
        title_ru=item.title_ru,
        source=item.source,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[movie_metadata.c.movie_id],
        set_={
            **{
                k: getattr(stmt.excluded, k)
                for k in ("tmdb_id", "poster_path", "title_ru", "source")
            },
            "updated_at": func.now(),
        },
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def load_enrichment_catalog(engine: Engine) -> list[CatalogEntry]:
    count = func.count(ratings.c.movie_id).label("n_ratings")
    stmt = (
        select(movies.c.movie_id, movies.c.title, movies.c.year, count)
        .select_from(movies.outerjoin(ratings, movies.c.movie_id == ratings.c.movie_id))
        .group_by(movies.c.movie_id, movies.c.title, movies.c.year)
        .order_by(count.desc(), movies.c.movie_id)
    )
    with engine.connect() as conn:
        return [CatalogEntry(**row) for row in conn.execute(stmt).mappings()]
