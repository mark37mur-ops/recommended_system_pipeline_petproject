"""Local metadata snapshot: no TMDB requests on user request paths."""

from sqlalchemy import Engine

from recsys.movie_metadata import MovieMetadata, load_metadata, normalize_title


class MetadataHolder:
    def __init__(self) -> None:
        self.current: dict[int, MovieMetadata] = {}

    def refresh(self, engine: Engine) -> int:
        snapshot = load_metadata(engine)
        self.current = snapshot
        return len(snapshot)

    def get(self, movie_id: int) -> MovieMetadata | None:
        return self.current.get(movie_id)

    def search(self, query: str, limit: int = 20) -> list[int]:
        query = normalize_title(query)
        if not query or limit <= 0:
            return []
        snapshot = self.current
        return [
            movie_id
            for movie_id, item in snapshot.items()
            if item.title_ru and query in normalize_title(item.title_ru)
        ][:limit]
