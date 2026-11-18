import json
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from recsys import enrich_movies as cli
from recsys.movie_metadata import CatalogEntry, MovieMetadata


class Response:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return self.data[:size]


class Transport:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return Response(result)


def result(title="The Thing"):
    return {
        "results": [
            {
                "id": 1091,
                "title": "Нечто",
                "original_title": title,
                "release_date": "1982-06-25",
                "poster_path": "/thing.jpg",
            }
        ]
    }


def test_transport_contract_and_retry_bounds():
    transport = Transport(
        [
            HTTPError("url", 429, "secret", {"Retry-After": "1000"}, None),
            HTTPError("url", 503, "secret", {}, None),
            result(),
        ]
    )
    sleeps = []
    client = cli.TmdbClient("private-token", transport=transport, sleep=sleeps.append)
    assert client.search("The Thing", 1982)[0]["id"] == 1091
    assert len(transport.calls) == 3
    assert 30 in sleeps and 2 in sleeps
    request, timeout = transport.calls[0]
    assert timeout == 10
    assert request.get_header("Authorization") == "Bearer private-token"
    assert "private-token" not in request.full_url
    assert parse_qs(urlparse(request.full_url).query) == {
        "query": ["The Thing"],
        "year": ["1982"],
        "language": ["ru-RU"],
        "include_adult": ["false"],
    }


@pytest.mark.parametrize("code,attempts", [(401, 1), (403, 1), (429, 3), (500, 3)])
def test_failures_safe_and_bounded(code, attempts):
    transport = Transport([HTTPError("secret-url", code, "private-token", {}, None)] * 3)
    client = cli.TmdbClient("private-token", transport=transport, sleep=lambda _: None)
    with pytest.raises(cli.EnrichmentError) as error:
        client.search("The Thing", 1982)
    assert "private-token" not in str(error.value)
    assert "secret-url" not in str(error.value)
    assert len(transport.calls) == attempts


def test_timeout_and_network_retries_and_rate_limit():
    transport = Transport(
        [TimeoutError("private-token"), URLError("private-token"), result(), result()]
    )
    sleeps = []
    client = cli.TmdbClient(
        "private-token", transport=transport, sleep=sleeps.append, clock=lambda: 0
    )
    client.search("Thing", 1982)
    client.search("Thing", 1982)
    assert 1 in sleeps and 2 in sleeps and 0.5 in sleeps


def test_queue_priority_popularity_and_existing():
    entries = [
        CatalogEntry(1, "One", 1982, 1),
        CatalogEntry(2, "Two", 1982, 20),
        CatalogEntry(3, "Three", 1982, 10),
    ]
    existing = {2: MovieMetadata(2, 2, None, None)}
    assert [e.movie_id for e in cli.build_queue(entries, existing, [3, 3, 999], 3)] == [3, 1]
    assert [e.movie_id for e in cli.build_queue(entries, existing, [], 3, refresh=True)] == [
        2,
        3,
        1,
    ]


def test_cursor_uses_full_priority_order_before_skipping_existing():
    entries = [CatalogEntry(i, str(i), 1982, 10) for i in (1, 2, 3, 4)]
    existing = {3: MovieMetadata(3, 3, None, None)}
    assert [
        e.movie_id for e in cli.build_queue(entries, existing, [4, 3, 4], 4, after_movie_id=3)
    ] == [1, 2]
    assert [
        e.movie_id
        for e in cli.build_queue(entries, existing, [4, 3, 4], 4, after_movie_id=4, refresh=True)
    ] == [3, 1, 2]
    assert cli.build_queue(entries, existing, [4, 3], 4, after_movie_id=2) == []


def test_two_batches_advance_past_unmatched_and_no_cursor_retries(monkeypatch):
    entries = [
        CatalogEntry(1, "The Thing", 1982, 100),
        CatalogEntry(2, "No Match", 1982, 1),
        CatalogEntry(3, "Already Saved", 1982, 200),
    ]
    saved = {3: MovieMetadata(3, 99, None, None)}
    monkeypatch.setattr(cli, "load_enrichment_catalog", lambda _: entries)
    monkeypatch.setattr(cli, "load_metadata", lambda _: saved.copy())
    monkeypatch.setattr(cli, "upsert_metadata", lambda _, item: saved.update({item.movie_id: item}))
    transport = Transport([{"results": []}, result(), {"results": []}])
    client = cli.TmdbClient("fake-token", transport=transport, sleep=lambda _: None)
    first = cli.enrich(None, client, limit=1, priority_ids=[2, 3, 2])
    assert first == {"processed": 1, "matched": 0, "unmatched": 1, "errors": 0}
    second = cli.enrich(None, client, limit=1, priority_ids=[2, 3, 2], after_movie_id=2)
    assert second == {"processed": 1, "matched": 1, "unmatched": 0, "errors": 0}
    assert saved[1] == MovieMetadata(1, 1091, "/thing.jpg", "Нечто")
    assert set(saved) == {1, 3}
    assert cli.enrich(None, client, limit=1, priority_ids=[2, 3, 2]) == first
    assert [parse_qs(urlparse(req.full_url).query)["query"] for req, _ in transport.calls] == [
        ["no match"],
        ["the thing"],
        ["no match"],
    ]


def test_absent_cursor_rejected_before_requests_or_writes(monkeypatch):
    monkeypatch.setattr(cli, "load_enrichment_catalog", lambda _: [CatalogEntry(1, "One", 1982, 1)])
    monkeypatch.setattr(cli, "load_metadata", lambda _: {})
    writes = []
    monkeypatch.setattr(cli, "upsert_metadata", lambda *args: writes.append(args))
    transport = Transport([])
    with pytest.raises(ValueError, match="cursor"):
        cli.enrich(None, cli.TmdbClient("fake-token", transport=transport), after_movie_id=999)
    assert transport.calls == [] and writes == []


def test_cli_forwards_cursor(tmp_path, monkeypatch, capsys):
    token = tmp_path / "fake-token.txt"
    token.write_text("fake-token")
    monkeypatch.setenv("DATABASE_URL", "fake-dsn")

    class Engine:
        def dispose(self):
            pass

    monkeypatch.setattr(cli, "make_engine", lambda _: Engine())
    received = []

    def enrich(*args, **kwargs):
        received.append(kwargs["after_movie_id"])
        return {"errors": 0}

    monkeypatch.setattr(cli, "enrich", enrich)
    assert cli.main(["--token-file", str(token), "--after-movie-id", "42"]) == 0
    assert received == [42]
    assert json.loads(capsys.readouterr().out) == {"errors": 0}


def test_dry_run_and_repeat_no_writes(monkeypatch):
    writes = []
    monkeypatch.setattr(
        cli, "load_enrichment_catalog", lambda _: [CatalogEntry(1, "Thing, The", 1982, 9)]
    )
    monkeypatch.setattr(cli, "load_metadata", lambda _: {})
    monkeypatch.setattr(cli, "upsert_metadata", lambda *args: writes.append(args))
    transport = Transport([result()])
    summary = cli.enrich(None, cli.TmdbClient("token", transport=transport), limit=1, dry_run=True)
    assert summary["matched"] == 1 and not writes
    transport = Transport([result()])
    cli.enrich(None, cli.TmdbClient("token", transport=transport), limit=1)
    assert writes[0][1] == MovieMetadata(1, 1091, "/thing.jpg", "Нечто")
    monkeypatch.setattr(cli, "load_metadata", lambda _: {1: writes[0][1]})
    transport = Transport([])
    assert cli.enrich(None, cli.TmdbClient("token", transport=transport), limit=1)["processed"] == 0
    assert not transport.calls


def test_main_safe_errors_and_argument_validation(tmp_path, monkeypatch, capsys):
    token = tmp_path / "token.txt"
    token.write_text("private-token")
    monkeypatch.setenv("DATABASE_URL", "private-dsn")
    monkeypatch.setattr(
        cli, "make_engine", lambda _: (_ for _ in ()).throw(RuntimeError("private-dsn"))
    )
    assert cli.main(["--token-file", str(token)]) == 1
    output = capsys.readouterr()
    assert "private-token" not in output.out + output.err
    assert "private-dsn" not in output.out + output.err
    for limit in ("0", "4001"):
        with pytest.raises(SystemExit):
            cli.main(["--token-file", str(token), "--limit", limit])


def test_main_disposal_failure_is_safe(tmp_path, monkeypatch, capsys):
    token = tmp_path / "fake-token.txt"
    token.write_text("fake-private-token")
    monkeypatch.setenv("DATABASE_URL", "fake-private-dsn")

    class Engine:
        def dispose(self):
            raise RuntimeError("fake-private-dsn")

    monkeypatch.setattr(cli, "make_engine", lambda _: Engine())
    monkeypatch.setattr(cli, "enrich", lambda *a, **kw: {"errors": 0})
    assert cli.main(["--token-file", str(token)]) == 1
    output = capsys.readouterr()
    assert "fake-private" not in output.out + output.err


def test_total_attempts_limit_no_pagination_and_unmatched_preserved(monkeypatch):
    entries = [CatalogEntry(i, "The Thing", 1982, 10) for i in range(1, 5)]
    saved = MovieMetadata(1, 1091, "/old.jpg", "Нечто")
    monkeypatch.setattr(cli, "load_enrichment_catalog", lambda _: entries)
    monkeypatch.setattr(cli, "load_metadata", lambda _: {1: saved})
    writes = []
    monkeypatch.setattr(cli, "upsert_metadata", lambda *args: writes.append(args))
    transport = Transport([TimeoutError("fake-private-token")] * 6)
    summary = cli.enrich(
        None, cli.TmdbClient("fake-token", transport=transport, sleep=lambda _: None), limit=2
    )
    assert summary == {"processed": 2, "matched": 0, "unmatched": 0, "errors": 2}
    assert len(transport.calls) == 6 and writes == []
    transport = Transport([{"results": [], "total_pages": 20}])
    summary = cli.enrich(
        None, cli.TmdbClient("fake-token", transport=transport), limit=1, refresh=True
    )
    assert summary["unmatched"] == 1 and len(transport.calls) == 1 and writes == []


@pytest.mark.parametrize("payload", [[], {}, {"results": "invalid"}])
def test_malformed_payload_is_safe(payload):
    with pytest.raises(cli.EnrichmentError):
        cli.TmdbClient("fake-token", transport=Transport([payload])).search("Thing", 1982)


def test_missing_year_makes_no_request(monkeypatch):
    monkeypatch.setattr(
        cli, "load_enrichment_catalog", lambda _: [CatalogEntry(1, "Thing", None, 1)]
    )
    monkeypatch.setattr(cli, "load_metadata", lambda _: {})
    transport = Transport([])
    assert cli.enrich(None, cli.TmdbClient("fake-token", transport=transport))["unmatched"] == 1
    assert transport.calls == []


ONBOARDING_TITLES = [
    ("Star Wars: Episode IV - A New Hope", "Star Wars", 1977),
    ("Star Wars: Episode V - The Empire Strikes Back", "The Empire Strikes Back", 1980),
    ("Star Wars: Episode VI - Return of the Jedi", "Return of the Jedi", 1983),
]


def alias_candidate(original_title="Star Wars", year=1977, tmdb_id=700):
    return {
        "id": tmdb_id,
        "title": "Локальное название",
        "original_title": original_title,
        "release_date": f"{year}-05-25",
        "poster_path": "/verified.jpg",
    }


def alias_detail(title, tmdb_id=700):
    return {"id": tmdb_id, "titles": [{"iso_3166_1": "US", "title": title, "type": ""}]}


def run_alias_enrichment(monkeypatch, title, year, outcomes, *, count=1):
    entries = [CatalogEntry(i, title, year, 10) for i in range(1, count + 1)]
    monkeypatch.setattr(cli, "load_enrichment_catalog", lambda _: entries)
    monkeypatch.setattr(cli, "load_metadata", lambda _: {})
    writes = []
    monkeypatch.setattr(cli, "upsert_metadata", lambda *args: writes.append(args[1]))
    transport = Transport(outcomes)
    sleeps = []
    client = cli.TmdbClient(
        "fake-private-token", transport=transport, sleep=sleeps.append, clock=lambda: 0
    )
    return cli.enrich(None, client, limit=count), writes, transport, sleeps


@pytest.mark.parametrize("canonical,original,year", ONBOARDING_TITLES)
def test_onboarding_title_requires_verified_exact_alias(monkeypatch, canonical, original, year):
    summary, writes, transport, sleeps = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [{"results": [alias_candidate(original, year)]}, alias_detail(canonical)],
    )
    assert summary == {"processed": 1, "matched": 1, "unmatched": 0, "errors": 0}
    assert writes == [MovieMetadata(1, 700, "/verified.jpg", "Локальное название")]
    assert len(transport.calls) == 2 and sleeps == [0.5]
    request, timeout = transport.calls[1]
    assert request.full_url == "https://api.themoviedb.org/3/movie/700/alternative_titles"
    assert request.get_header("Authorization") == "Bearer fake-private-token" and timeout == 10


@pytest.mark.parametrize("canonical,original,year", ONBOARDING_TITLES)
@pytest.mark.parametrize("verified_alias", [None, "A New Hope", "Unrelated Film"])
def test_onboarding_never_infers_unverified_alias(
    monkeypatch, canonical, original, year, verified_alias
):
    detail = alias_detail(verified_alias) if verified_alias else {"id": 700, "titles": []}
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, [{"results": [alias_candidate(original, year)]}, detail]
    )
    assert summary["matched"] == 0 and summary["unmatched"] == 1 and writes == []
    assert len(transport.calls) == 2


@pytest.mark.parametrize("canonical,original,year", ONBOARDING_TITLES)
def test_verified_alias_cannot_match_wrong_year_remake(monkeypatch, canonical, original, year):
    remake = alias_candidate(original, year + 20)
    remake["alternative_titles"] = alias_detail(canonical)
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, [{"results": [remake]}]
    )
    assert summary["unmatched"] == 1 and writes == [] and len(transport.calls) == 1


def test_explicit_exact_alias_reused_without_detail(monkeypatch):
    canonical, original, year = ONBOARDING_TITLES[0]
    item = alias_candidate(original, year)
    item["alternative_titles"] = alias_detail(canonical)
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, [{"results": [item]}]
    )
    assert summary["matched"] == 1 and len(writes) == 1 and len(transport.calls) == 1


def test_multiple_verified_alias_candidates_rejected(monkeypatch):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [
            {"results": [alias_candidate(original, year), alias_candidate("Other", year, 701)]},
            alias_detail(canonical),
            alias_detail(canonical, 701),
        ],
    )
    assert summary["matched"] == 0 and summary["unmatched"] == 1 and writes == []
    assert len(transport.calls) == 3


@pytest.mark.parametrize(
    "retry_stage", ["search", "detail", "search_exhausted", "detail_exhausted"]
)
def test_search_and_detail_share_three_attempt_budget_per_movie(monkeypatch, retry_stage):
    canonical, original, year = ONBOARDING_TITLES[0]
    search = {"results": [alias_candidate(original, year)]}
    failure = TimeoutError("fake-private-token")
    outcomes = {
        "search": [failure, search, alias_detail(canonical)],
        "detail": [search, failure, alias_detail(canonical)],
        "search_exhausted": [failure, failure, search],
        "detail_exhausted": [search, failure, failure],
    }[retry_stage]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, outcomes * 4, count=4
    )
    assert len(transport.calls) == 3 * summary["processed"] == 12
    if retry_stage in ("search", "detail"):
        assert summary["matched"] == 4 and len(writes) == 4
        assert all(item.tmdb_id == 700 for item in writes)
    else:
        assert summary["matched"] == 0 and writes == []
        assert summary["unmatched"] + summary["errors"] == 4
    for offset in range(0, 12, 3):
        paths = [urlparse(req.full_url).path for req, _ in transport.calls[offset : offset + 3]]
        assert paths[0] == "/3/search/movie"
        assert (
            paths.count("/3/movie/700/alternative_titles")
            == {"search": 1, "detail": 2, "search_exhausted": 0, "detail_exhausted": 2}[retry_stage]
        )


def test_budget_never_accepts_first_alias_with_unverified_competitor(monkeypatch):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [
            TimeoutError("fake-private-token"),
            {"results": [alias_candidate(original, year), alias_candidate("Other", year, 701)]},
            alias_detail(canonical),
        ],
    )
    assert summary["matched"] == 0 and writes == [] and len(transport.calls) <= 3


@pytest.mark.parametrize(
    "detail",
    [
        {"titles": alias_detail(ONBOARDING_TITLES[0][0])["titles"]},
        alias_detail(ONBOARDING_TITLES[0][0], 701),
        {"id": 700, "titles": "bad"},
    ],
)
def test_alias_response_must_belong_to_requested_movie(monkeypatch, detail):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, [{"results": [alias_candidate(original, year)]}, detail]
    )
    assert summary["matched"] == 0 and writes == [] and len(transport.calls) == 2
    assert summary["errors"] == 1


def test_existing_alias_ambiguity_cannot_be_erased_by_detail(monkeypatch):
    canonical, original, year = ONBOARDING_TITLES[0]
    first = alias_candidate(original, year)
    second = alias_candidate("Other", year, 701)
    first["alternative_titles"] = alias_detail(canonical)
    second["alternative_titles"] = alias_detail(canonical, 701)
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [{"results": [first, second]}, alias_detail(canonical), {"id": 701, "titles": []}],
    )
    assert summary["unmatched"] == 1 and writes == [] and len(transport.calls) == 1


def test_detail_retry_cannot_leave_competitor_unverified(monkeypatch):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [
            {"results": [alias_candidate(original, year), alias_candidate("Other", year, 701)]},
            TimeoutError("fake-private-token"),
            alias_detail(canonical),
        ],
    )
    assert summary["matched"] == 0 and summary["unmatched"] == 1 and writes == []
    assert len(transport.calls) == 3


@pytest.mark.parametrize("movie_id", [True, -1, 0, "700", "../secret", None])
def test_invalid_candidate_id_never_reaches_detail_url(monkeypatch, movie_id):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, _ = run_alias_enrichment(
        monkeypatch, canonical, year, [{"results": [alias_candidate(original, year, movie_id)]}]
    )
    assert summary["unmatched"] == 1 and writes == [] and len(transport.calls) == 1


@pytest.mark.parametrize("code,attempts", [(401, 2), (403, 2), (302, 2), (429, 3), (503, 3)])
def test_detail_errors_preserve_safe_retry_and_total_budget(monkeypatch, code, attempts):
    canonical, original, year = ONBOARDING_TITLES[0]
    summary, writes, transport, sleeps = run_alias_enrichment(
        monkeypatch,
        canonical,
        year,
        [
            {"results": [alias_candidate(original, year)]},
            HTTPError("fake-private-url", code, "fake-private-token", {"Retry-After": "99"}, None),
            HTTPError("fake-private-url", code, "fake-private-token", {}, None),
        ],
    )
    assert summary == {"processed": 1, "matched": 0, "unmatched": 0, "errors": 1}
    assert writes == [] and len(transport.calls) == attempts
    assert "fake-private" not in json.dumps(summary)
    if code == 429:
        assert 30 in sleeps
