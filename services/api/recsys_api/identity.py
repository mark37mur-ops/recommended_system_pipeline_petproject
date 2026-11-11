"""Anonymous visitors: the user id lives in a signed cookie (spec 1: no registration).

Value format `{user_id}.{signature}`, signature = first 16 bytes of HMAC-SHA256 over the id,
urlsafe base64. A visitor cannot switch to somebody else's id, and the id is known without a
database lookup (recommendations can be served from Redis while Postgres is down).
Only ids of the web range are accepted: a cookie for a MovieLens user (id below
WEB_USER_ID_START) is treated like no cookie at all, so it can never touch training data.
"""

import base64
import hashlib
import hmac

from recsys.db.schema import WEB_USER_ID_START

COOKIE_NAME = "uid"
_MAX_ID_DIGITS = 12


class CookieSigner:
    def __init__(self, secret: str, min_user_id: int = WEB_USER_ID_START) -> None:
        if not secret:
            raise ValueError("cookie secret must not be empty")
        self._key = secret.encode()
        self._min_user_id = min_user_id

    def _signature(self, user_id: int) -> str:
        digest = hmac.new(self._key, f"uid:{user_id}".encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest[:16]).rstrip(b"=").decode()

    def sign(self, user_id: int) -> str:
        return f"{user_id}.{self._signature(user_id)}"

    def verify(self, value: str | None) -> int | None:
        """Web user id if `value` was produced by `sign` with this secret, else None."""
        if not value:
            return None
        raw_id, _, signature = value.partition(".")
        if not raw_id.isdigit() or len(raw_id) > _MAX_ID_DIGITS or not signature:
            return None
        user_id = int(raw_id)
        if user_id < self._min_user_id:
            return None
        if not hmac.compare_digest(signature, self._signature(user_id)):
            return None
        return user_id
