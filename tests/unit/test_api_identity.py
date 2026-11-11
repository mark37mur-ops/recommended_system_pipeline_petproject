import pytest

from recsys_api.identity import CookieSigner


def test_round_trip():
    signer = CookieSigner("secret")
    value = signer.sign(1_000_042)
    assert value.startswith("1000042.") and signer.verify(value) == 1_000_042


@pytest.mark.parametrize(
    "tamper",
    [
        lambda v: "1000043." + v.split(".")[1],  # someone else's id with my signature
        lambda v: v[:-1] + ("A" if v[-1] != "A" else "B"),
        lambda v: v.split(".")[0],
        lambda v: "",
        lambda v: "abc.def",
        lambda v: "-5." + v.split(".")[1],
        lambda v: "9" * 40 + "." + v.split(".")[1],
    ],
)
def test_tampered_values_are_rejected(tamper):
    signer = CookieSigner("secret")
    assert signer.verify(tamper(signer.sign(1_000_042))) is None


def test_other_secret_rejects_and_none_is_none():
    assert CookieSigner("other").verify(CookieSigner("secret").sign(7)) is None
    assert CookieSigner("secret").verify(None) is None


def test_empty_secret_is_refused():
    with pytest.raises(ValueError):
        CookieSigner("")
