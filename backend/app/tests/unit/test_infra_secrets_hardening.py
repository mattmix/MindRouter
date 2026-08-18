"""Structural guards for ws11-infra-secrets hardening.

These are pure text/structure assertions over infra config files (shell,
compose, Dockerfiles, env examples). They intentionally do NOT import the app,
so there is no sys.modules pollution risk. Repo root is derived from this
file's location: .../backend/app/tests/unit/<this file>.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(rel: str) -> str:
    p = REPO_ROOT / rel
    assert p.is_file(), f"expected file missing: {rel}"
    return p.read_text()


# ---------------------------------------------------------------------------
# F06 — sidecar keys no longer committed
# ---------------------------------------------------------------------------

# The twelve keys that used to live in an in-repo `case` table. They are
# compromised and must never reappear in the working tree.
_LEAKED_SIDECAR_KEYS = [
    "b0447d8d7efefac27a761783447e37dce366395b802f69c5e578014966507953",
    "dd4bb8de54cb3ee1f732b1d21ac428180b0a76c6e5b740dcde5daf8e33d5fcbc",
    "udiydhdy7d7d7hdjhxhxhxhxh67sdgo",
    "ce6187b6c36098a4a23f34c62b5112d2304000b4a9dd616fefec502e3a588428",
    "a2663b63a2b036a88f9bb56a332dfd019f34c6398b9825ea0ec5aa940adf4830",
    "67be9f8d222c05656af048d6dd81368237890ce43aab039a66736bd9429ca4b6",
    "Y8wzaoMh1aqYfOrSqJtTpyzvCauu9gyEEFDuoMh_tcc",
    "Y8wza1oMh1aqYfOrSqJtTpyzvCauu9gyEEFDuoMh_xxxt",
    "491023f96203f67cf3d86bf81aacb98604b657f78b53b955407d35f51f3006ef",
    "71f3997b67b423a52e243b081413f64591f1bf64272bf0e58f4cd03bc7506ee3",
    "f0426bf2f23aa5d6810ebc233b4944bee462a83bd4eb7ebfffe6581cacbbd431",
    "5871a710564c92d079a42ecdbbb3b0183f5a7accbf2ffc368f90421bf082896d",
]


@pytest.mark.parametrize("leaked", _LEAKED_SIDECAR_KEYS)
def test_no_hardcoded_sidecar_key_in_deploy_script(leaked):
    script = _read("sidecar/deploy_sidecars.sh")
    assert leaked not in script, "a compromised sidecar key is still committed"


def test_deploy_script_keeps_env_and_keyfile_resolution():
    script = _read("sidecar/deploy_sidecars.sh")
    # Env-var and keyfile resolution paths must remain.
    assert "SIDECAR_KEY_$(printf" in script
    assert "SIDECAR_KEY_FILE" in script


def test_deploy_script_hard_errors_without_key():
    script = _read("sidecar/deploy_sidecars.sh")
    # Hard error replacing the old case table.
    assert "no key for" in script
    # No node_key fallback echoing a bare secret table remains.
    assert "UNKNOWN_NODE" not in script


def test_deploy_script_documents_rotation():
    script = _read("sidecar/deploy_sidecars.sh")
    assert "COMPROMISED" in script.upper()
    assert "ROTATE" in script.upper()


# ---------------------------------------------------------------------------
# F22/L1 — Redis requires a password
# ---------------------------------------------------------------------------


def test_dev_compose_redis_requires_password():
    compose = _read("docker-compose.yml")
    assert "--requirepass" in compose
    # The app must connect with credentials (redis://:<pass>@...), not the old
    # passwordless URL.
    assert "redis://127.0.0.1:6379/0" not in compose
    assert "redis://:${REDIS_PASSWORD" in compose


def test_prod_compose_redis_fails_without_password():
    compose = _read("docker-compose.prod.yml")
    # Fail-closed interpolation form, and no well-known default.
    assert "REDIS_PASSWORD:?" in compose
    assert "changeme" not in compose


# ---------------------------------------------------------------------------
# L6 — non-root aux images
# ---------------------------------------------------------------------------


def test_sidecar_image_runs_non_root():
    df = _read("sidecar/Dockerfile.sidecar")
    assert "useradd" in df
    assert "USER appuser" in df


# ---------------------------------------------------------------------------
# Env examples document the new settings
# ---------------------------------------------------------------------------

_NEW_SETTING_KEYS = [
    "ENABLE_API_DOCS",
    "CSRF_TRUSTED_ORIGINS",
    "IMAGE_URL_BLOCK_PRIVATE",
    "OCR_MAX_FRAMES",
    "CHAT_UPLOAD_MAX_UNCOMPRESSED_MB",
    "MAX_COMPLETIONS_N",
    "LOGIN_MAX_ATTEMPTS_PER_WINDOW",
    "LOGIN_ATTEMPT_WINDOW_SECONDS",
    "RATE_LIMIT_LOCAL_FALLBACK",
    "TOKENIZE_MAX_INPUT_CHARS",
]


@pytest.mark.parametrize("key", _NEW_SETTING_KEYS)
def test_dev_env_example_documents_new_settings(key):
    assert key in _read(".env.example")


@pytest.mark.parametrize("key", _NEW_SETTING_KEYS)
def test_prod_env_example_documents_new_settings(key):
    assert key in _read(".env.prod.example")


def test_prod_env_example_hardening_values():
    prod = _read(".env.prod.example")
    assert "ENABLE_API_DOCS=false" in prod
    assert "SESSION_COOKIE_SECURE=true" in prod
    assert "REDIS_PASSWORD=" in prod


def test_dev_env_example_documents_redis_password():
    dev = _read(".env.example")
    assert "REDIS_PASSWORD=" in dev
