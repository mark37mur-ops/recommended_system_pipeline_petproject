"""Request and response models (spec 3.4 point 1: Pydantic validation)."""

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from recsys_api.catalog import Movie


class RatingIn(BaseModel):
    """Strict integers: 4.0, "4" or true are rejected (the model takes integer ratings)."""

    model_config = ConfigDict(extra="forbid")

    movie_id: StrictInt = Field(gt=0)
    rating: StrictInt = Field(ge=1, le=5)


class RatingOut(BaseModel):
    movie_id: int
    rating: int
    n_ratings: int


class MovieOut(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    user_rating: int | None = None

    @classmethod
    def of(cls, movie: Movie, user_rating: int | None = None) -> "MovieOut":
        return cls(
            movie_id=movie.movie_id,
            title=movie.display_title,
            year=movie.year,
            genres=list(movie.genres),
            user_rating=user_rating,
        )


class MovieList(BaseModel):
    items: list[MovieOut]


class RecommendationsOut(BaseModel):
    items: list[MovieOut]
    path: str  # full | cold_start | fallback
    model_version: str | None
    n_ratings: int | None
    min_ratings: int
    cached: bool


class ReloadOut(BaseModel):
    status: str  # loaded | unchanged | failed | busy
    version: str | None
    previous_version: str | None
    error: str | None
    catalog_size: int | None
