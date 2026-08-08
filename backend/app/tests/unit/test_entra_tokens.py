############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_entra_tokens.py: Entra id_token verification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Verification of Entra id_tokens forwarded by a registered app.

These tests sign REAL tokens with a throwaway RSA key and serve a REAL JWKS,
rather than mocking the decode. The property under test is forgery
resistance, and a mocked verifier cannot demonstrate it — every negative case
here would pass trivially against a stub.

The rejections that matter most:
  * wrong audience — otherwise ANY app in the tenant could provision accounts
  * wrong issuer/tenant — otherwise any Entra tenant could
  * bad signature, wrong key, `alg: none` — the classic JWT forgeries
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt
from jose.utils import base64url_encode

_APP = Path(__file__).resolve().parents[2]

TENANT = "11111111-2222-3333-4444-555555555555"
APP_CLIENT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_CLIENT_ID = "99999999-8888-7777-6666-555555555555"


def _load_module():
    saved = {k: sys.modules.get(k) for k in
             ("backend", "backend.app", "backend.app.logging_config")}
    sys.modules.setdefault("backend", MagicMock())
    sys.modules.setdefault("backend.app", MagicMock())
    sys.modules["backend.app.logging_config"] = MagicMock(
        get_logger=MagicMock(return_value=MagicMock())
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "entra_tokens_under_test", _APP / "security" / "entra_tokens.py",
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


et = _load_module()


def _make_rsa(kid):
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key().public_numbers()

    def _b64(n):
        raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
        return base64url_encode(raw).decode()

    jwk = {"kty": "RSA", "kid": kid, "use": "sig", "alg": "RS256",
           "n": _b64(pub.n), "e": _b64(pub.e)}
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return pem, jwk


_PEM, _JWK = _make_rsa("test-key-1")
_OTHER_PEM, _OTHER_JWK = _make_rsa("other-key")


def _claims(**over):
    now = int(time.time())
    c = {
        "iss": f"https://login.microsoftonline.com/{TENANT}/v2.0",
        "aud": APP_CLIENT_ID,
        "tid": TENANT,
        "oid": "abc-oid-123",
        "sub": "pairwise-sub",
        "email": "Alice@uidaho.edu",
        "name": "Alice Example",
        "iat": now - 10,
        "nbf": now - 10,
        "exp": now + 3600,
        "nonce": "the-apps-own-nonce",
    }
    c.update(over)
    return c


def _token(pem=_PEM, kid="test-key-1", **over):
    return jwt.encode(_claims(**over), pem, algorithm="RS256",
                      headers={"kid": kid})


@pytest.fixture(autouse=True)
def _serve_jwks(monkeypatch):
    """Serve a real JWKS containing only the good key."""
    et._reset_jwks_cache_for_tests()

    async def _fake_fetch(tenant_id):
        assert tenant_id == TENANT
        return {_JWK["kid"]: _JWK}

    monkeypatch.setattr(et, "_fetch_jwks", _fake_fetch)
    yield
    et._reset_jwks_cache_for_tests()


class TestAcceptsAGenuineToken:
    @pytest.mark.asyncio
    async def test_valid_token_yields_identity(self):
        ident = await et.verify_entra_id_token(_token(), APP_CLIENT_ID, TENANT)
        assert ident.oid == "abc-oid-123"
        assert ident.tid == TENANT
        assert ident.display_name == "Alice Example"

    @pytest.mark.asyncio
    async def test_email_is_lowercased(self):
        """users.email is the linking key and is compared lowercased."""
        ident = await et.verify_entra_id_token(_token(), APP_CLIENT_ID, TENANT)
        assert ident.email == "alice@uidaho.edu"

    @pytest.mark.asyncio
    async def test_v1_issuer_form_accepted(self):
        tok = _token(iss=f"https://sts.windows.net/{TENANT}/")
        ident = await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)
        assert ident.oid == "abc-oid-123"

    @pytest.mark.asyncio
    async def test_nonce_is_not_required(self):
        """MindRouter did not issue the app's sign-in request, so it cannot
        validate the nonce; audience+issuer+signature do the binding."""
        c = _claims()
        c.pop("nonce")
        tok = jwt.encode(c, _PEM, algorithm="RS256", headers={"kid": "test-key-1"})
        assert await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)


class TestRejectsForgeries:
    @pytest.mark.asyncio
    async def test_wrong_audience_rejected(self):
        """THE critical check: a token minted for a different app in the same
        tenant must not work, or every app in the tenant can provision."""
        tok = _token(aud=OTHER_CLIENT_ID)
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_token_for_this_app_rejected_when_checking_another(self):
        """Symmetric case: registering app B must not accept app A's tokens."""
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(_token(), OTHER_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_foreign_issuer_rejected(self):
        tok = _token(iss="https://login.microsoftonline.com/evil-tenant/v2.0")
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_signature_by_unknown_key_rejected(self):
        """Signed with a well-formed RSA key that is not in the tenant JWKS."""
        tok = _token(pem=_OTHER_PEM, kid="other-key")
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_key_substitution_rejected(self):
        """Claims a known kid but is signed with a different private key."""
        tok = _token(pem=_OTHER_PEM, kid="test-key-1")
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_alg_none_rejected(self):
        """The classic JWT forgery."""
        import base64

        def _seg(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        tok = f"{_seg({'alg': 'none', 'kid': 'test-key-1'})}.{_seg(_claims())}."
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_expired_rejected(self):
        """Expired beyond the clock-skew allowance."""
        now = int(time.time())
        stale = et.LEEWAY_SECONDS + 60
        tok = _token(exp=now - stale, nbf=now - stale - 100, iat=now - stale - 100)
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_clock_skew_leeway_is_deliberate(self):
        """A token just past expiry IS accepted, within LEEWAY_SECONDS. This
        is intentional — Entra and this host can disagree slightly — so pin it
        rather than let it be an accident of the library's defaults."""
        now = int(time.time())
        tok = _token(exp=now - 5, nbf=now - 100, iat=now - 100)
        assert await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)
        assert et.LEEWAY_SECONDS <= 300, "clock-skew allowance should stay tight"

    @pytest.mark.asyncio
    async def test_not_yet_valid_rejected(self):
        now = int(time.time())
        ahead = et.LEEWAY_SECONDS + 60
        tok = _token(nbf=now + ahead, iat=now + ahead, exp=now + ahead + 3600)
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_tampered_payload_rejected(self):
        tok = _token()
        head, payload, sig = tok.split(".")
        import base64
        bad = _claims(oid="somebody-else")
        payload2 = base64.urlsafe_b64encode(
            json.dumps(bad).encode()).rstrip(b"=").decode()
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(f"{head}.{payload2}.{sig}",
                                           APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_mismatched_tid_rejected(self):
        tok = _token(tid="a-different-tenant")
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_missing_oid_rejected(self):
        """`sub` is pairwise per application and useless to a second relying
        party, so without oid there is nothing to key an account on."""
        c = _claims()
        c.pop("oid")
        tok = jwt.encode(c, _PEM, algorithm="RS256", headers={"kid": "test-key-1"})
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(tok, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_garbage_rejected(self):
        for junk in ("", "not-a-token", "a.b.c"):
            with pytest.raises(et.EntraTokenError):
                await et.verify_entra_id_token(junk, APP_CLIENT_ID, TENANT)

    @pytest.mark.asyncio
    async def test_missing_registration_details_rejected(self):
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(_token(), "", TENANT)
        with pytest.raises(et.EntraTokenError):
            await et.verify_entra_id_token(_token(), APP_CLIENT_ID, "")


class TestJwksHandling:
    @pytest.mark.asyncio
    async def test_keys_are_cached_not_refetched_per_request(self, monkeypatch):
        calls = {"n": 0}

        async def _counting(tenant_id):
            calls["n"] += 1
            return {_JWK["kid"]: _JWK}

        et._reset_jwks_cache_for_tests()
        monkeypatch.setattr(et, "_fetch_jwks", _counting)
        for _ in range(5):
            await et.verify_entra_id_token(_token(), APP_CLIENT_ID, TENANT)
        assert calls["n"] == 1, "JWKS should be fetched once, not per request"

    @pytest.mark.asyncio
    async def test_unknown_kid_does_not_amplify_requests(self, monkeypatch):
        """A stream of tokens bearing bogus kids must not turn into unbounded
        refetches against Entra."""
        calls = {"n": 0}

        async def _counting(tenant_id):
            calls["n"] += 1
            return {_JWK["kid"]: _JWK}

        et._reset_jwks_cache_for_tests()
        monkeypatch.setattr(et, "_fetch_jwks", _counting)
        await et.verify_entra_id_token(_token(), APP_CLIENT_ID, TENANT)  # warms cache

        for _ in range(10):
            with pytest.raises(et.EntraTokenError):
                await et.verify_entra_id_token(
                    _token(pem=_OTHER_PEM, kid="rotated-or-bogus"),
                    APP_CLIENT_ID, TENANT,
                )
        assert calls["n"] <= 2, f"unknown kid caused {calls['n']} JWKS fetches"


class TestDoesNotReuseTheUnverifiedDecoder:
    def test_module_does_not_call_the_sso_helper(self):
        src = (_APP / "security" / "entra_tokens.py").read_text()
        assert "_decode_id_token_claims" not in src.split('"""', 2)[2], (
            "the unverified SSO claim decoder must never be used on a token "
            "that arrives from a third party"
        )

    def test_signature_verification_is_on(self):
        src = (_APP / "security" / "entra_tokens.py").read_text()
        assert '"verify_signature": True' in src
        assert '"verify_aud": True' in src
        assert '"verify_iss": True' in src
