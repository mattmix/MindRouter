############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/db.py: Direct MariaDB access for the DLP
# harness — app_config snapshot/apply/restore and the alert/
# request queries no HTTP surface exposes.
#
# DLP alerts have no JSON API (dashboard session routes
# only), and DLP config is hot-reloaded from app_config on
# every scanned item, so direct DB reads/writes are both
# necessary and immediately effective. Guarded local-only
# by default: this must never point at prod by accident.
#
############################################################

"""MariaDB access layer for the DLP evaluation harness."""

import json
import os
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

import pymysql
import pymysql.cursors

from dlp_harness.constants import DLP_CONFIG_KEYS, SCANNER_ERROR_CATEGORY

# Sentinel for "key did not exist at snapshot time" — restore deletes it.
MISSING = object()

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

_IN_CHUNK = 1000


class HarnessDB:
    """One pymysql connection (autocommit) with harness-domain helpers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "mindrouter",
        password: Optional[str] = None,
        database: str = "mindrouter",
        allow_remote: bool = False,
    ):
        if host not in _LOCAL_HOSTS and not allow_remote:
            raise RuntimeError(
                f"HarnessDB refuses non-local host {host!r} without allow_remote=True "
                "(this layer writes app_config and deletes alert rows)"
            )
        self._params = dict(
            host=host,
            port=port,
            user=user,
            password=password if password is not None
                     else os.environ.get("MYSQL_PASSWORD", "mindrouter_password"),
            database=database,
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
        )
        self._conn = pymysql.connect(**self._params)

    @classmethod
    def from_database_url(cls, url: Optional[str] = None,
                          allow_remote: bool = False) -> "HarnessDB":
        """Build from a SQLAlchemy-style mysql URL (default: $DATABASE_URL).

        On the prod host the app's own /opt/mindrouter/.env DATABASE_URL (or
        the container's env) is the source of truth for credentials — the
        harness should never carry its own copy of prod secrets.
        """
        url = url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("no database URL given and DATABASE_URL is unset")
        parsed = urlparse(url)
        if not parsed.scheme.startswith("mysql"):
            raise RuntimeError(f"unsupported database URL scheme {parsed.scheme!r}")
        return cls(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=unquote(parsed.username or "mindrouter"),
            password=unquote(parsed.password) if parsed.password else None,
            database=(parsed.path or "/mindrouter").lstrip("/"),
            allow_remote=allow_remote,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- low level ----------------------------------------------------------

    def _cursor(self):
        self._conn.ping(reconnect=True)
        return self._conn.cursor()

    def query(self, sql: str, params: Sequence = ()) -> List[dict]:
        with self._cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def execute(self, sql: str, params: Sequence = ()) -> int:
        with self._cursor() as cur:
            return cur.execute(sql, params)

    def db_now(self):
        """DB-server clock; all lag math uses DB timestamps exclusively."""
        return self.query("SELECT NOW(6) AS now")[0]["now"]

    # -- app_config (mirrors crud.set_config: value column is json.dumps) ---

    def get_config_raw(self, key: str) -> Optional[str]:
        rows = self.query("SELECT value FROM app_config WHERE `key`=%s", (key,))
        return rows[0]["value"] if rows else None

    def get_config(self, key: str, default: Any = None) -> Any:
        raw = self.get_config_raw(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return default

    def set_config(self, key: str, value: Any) -> None:
        raw = json.dumps(value)
        self.execute(
            "INSERT INTO app_config (`key`, value, description) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE value=VALUES(value)",
            (key, raw, "dlp_harness"),
        )

    def delete_config(self, key: str) -> None:
        self.execute("DELETE FROM app_config WHERE `key`=%s", (key,))

    def snapshot_dlp_config(self) -> Dict[str, Any]:
        """Raw values (or MISSING) for every DLP key, for exact restore."""
        return {k: (self.get_config_raw(k) if self.get_config_raw(k) is not None else MISSING)
                for k in DLP_CONFIG_KEYS}

    def snapshot_to_json(self, snap: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """JSON-safe form of a snapshot (MISSING -> None marker dict)."""
        return {k: (None if v is MISSING else v) for k, v in snap.items()}

    def snapshot_from_json(self, data: Dict[str, Optional[str]]) -> Dict[str, Any]:
        return {k: (MISSING if v is None else v) for k, v in data.items()}

    def restore_dlp_config(self, snap: Dict[str, Any]) -> None:
        for key, raw in snap.items():
            if raw is MISSING:
                self.delete_config(key)
            else:
                self.execute(
                    "INSERT INTO app_config (`key`, value) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE value=VALUES(value)",
                    (key, raw),
                )

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        for k, v in overrides.items():
            self.set_config(k, v)

    # -- requests / responses / alerts --------------------------------------

    @staticmethod
    def _chunks(seq):
        seq = list(seq)
        for i in range(0, len(seq), _IN_CHUNK):
            yield seq[i:i + _IN_CHUNK]

    def fetch_requests_by_uuids(self, uuids: Sequence[str]) -> List[dict]:
        out: List[dict] = []
        for chunk in self._chunks(uuids):
            ph = ",".join(["%s"] * len(chunk))
            out.extend(self.query(
                "SELECT id, request_uuid, user_id, status, model, is_streaming, "
                "       created_at, queued_at, started_at, completed_at, "
                "       queue_delay_ms, processing_time_ms, total_time_ms, "
                "       prompt_tokens, completion_tokens, error_code "
                f"FROM requests WHERE request_uuid IN ({ph})", chunk))
        return out

    def fetch_alerts_by_request_ids(self, request_ids: Sequence[int]) -> List[dict]:
        out: List[dict] = []
        for chunk in self._chunks(request_ids):
            ph = ",".join(["%s"] * len(chunk))
            out.extend(self.query(
                "SELECT id, request_id, user_id, severity, scanner, categories, "
                "       entities, confidence, scan_latency_ms, scanned_at, detail "
                f"FROM dlp_alerts WHERE request_id IN ({ph})", chunk))
        for row in out:
            for col in ("categories", "entities"):
                if isinstance(row.get(col), str):
                    try:
                        row[col] = json.loads(row[col])
                    except (ValueError, TypeError):
                        pass
        return out

    def fetch_scan_lags_ms(self, request_ids: Sequence[int]) -> List[dict]:
        """Per-alert lag: scanned_at - requests.completed_at, DB-side math."""
        out: List[dict] = []
        for chunk in self._chunks(request_ids):
            ph = ",".join(["%s"] * len(chunk))
            out.extend(self.query(
                "SELECT a.request_id, a.id AS alert_id, a.scan_latency_ms, "
                "       CAST(TIMESTAMPDIFF(MICROSECOND, r.completed_at, a.scanned_at)/1000.0 AS DOUBLE) AS lag_ms "
                "FROM dlp_alerts a JOIN requests r ON r.id = a.request_id "
                f"WHERE a.request_id IN ({ph}) AND r.completed_at IS NOT NULL", chunk))
        return out

    def count_alerts_for_request_ids(self, request_ids: Sequence[int],
                                     exclude_scanner_errors: bool = True) -> int:
        total = 0
        for chunk in self._chunks(request_ids):
            ph = ",".join(["%s"] * len(chunk))
            sql = f"SELECT COUNT(*) AS n FROM dlp_alerts WHERE request_id IN ({ph})"
            if exclude_scanner_errors:
                sql += " AND (categories IS NULL OR categories NOT LIKE %s)"
                total += self.query(sql, list(chunk) + [f'%{SCANNER_ERROR_CATEGORY}%'])[0]["n"]
            else:
                total += self.query(sql, chunk)[0]["n"]
        return total

    def purge_alerts_for_request_ids(self, request_ids: Sequence[int]) -> int:
        n = 0
        for chunk in self._chunks(request_ids):
            ph = ",".join(["%s"] * len(chunk))
            n += self.execute(f"DELETE FROM dlp_alerts WHERE request_id IN ({ph})", chunk)
        return n

    def scanner_error_alerts_since(self, since) -> List[dict]:
        return self.query(
            "SELECT id, scanner, severity, detail, scanned_at FROM dlp_alerts "
            "WHERE scanned_at >= %s AND categories LIKE %s",
            (since, f'%{SCANNER_ERROR_CATEGORY}%'))
