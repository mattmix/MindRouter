"""Tests for the universal SSO provider framework (Google / generic OIDC / SAML).

The sso modules import backend.app.db/settings/logging at module level, so
they are spec-loaded under their real dotted names with those dependencies
stubbed (save/restore hygiene) — see MEMORY.md "Import Chain Gotcha".
Azure AD regression coverage lives at the bottom: its driver and routes must
remain untouched by the framework.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SSO_DIR = _APP_DIR / "dashboard" / "sso"


class FakeSettings:
    """Configurable stand-in for backend.app.settings.Settings."""

    def __init__(self, **kw):
        self.secret_key = "test-secret-key"
        self.azure_ad_client_id = None
        self.azure_ad_tenant_id = None
        self.google_sso_client_id = None
        self.google_sso_client_secret = None
        self.google_sso_redirect_uri = None
        self.google_sso_hosted_domain = None
        self.google_sso_default_group = "other"
        self.oidc_sso_issuer = None
        self.oidc_sso_client_id = None
        self.oidc_sso_client_secret = None
        self.oidc_sso_redirect_uri = None
        self.oidc_sso_display_name = "SSO"
        self.oidc_sso_scopes = "openid profile email"
        self.oidc_sso_default_group = "other"
        self.saml_sp_entity_id = None
        self.saml_sp_acs_url = None
        self.saml_idp_metadata_url = None
        self.saml_idp_entity_id = None
        self.saml_idp_sso_url = None
        self.saml_idp_x509_cert = None
        self.saml_display_name = "SSO"
        self.saml_default_group = "other"
        self.saml_sp_x509_cert = None
        self.saml_sp_private_key = None
        self.saml_authn_requests_signed = False
        self.saml_want_assertions_encrypted = False
        self.saml_attr_email = "mail"
        self.saml_attr_name = "displayName"
        self.saml_attr_username = "eduPersonPrincipalName"
        self.app_base_url = "https://mr.example.edu"
        for k, v in kw.items():
            setattr(self, k, v)

    # Mirror the real Settings properties.
    @property
    def azure_ad_enabled(self):
        return bool(self.azure_ad_client_id and self.azure_ad_tenant_id)

    @property
    def google_sso_enabled(self):
        return bool(self.google_sso_client_id and self.google_sso_client_secret)

    @property
    def oidc_sso_enabled(self):
        return bool(self.oidc_sso_issuer and self.oidc_sso_client_id and self.oidc_sso_client_secret)

    @property
    def saml_sso_enabled(self):
        return bool(
            self.saml_sp_entity_id
            and (self.saml_idp_metadata_url
                 or (self.saml_idp_entity_id and self.saml_idp_sso_url and self.saml_idp_x509_cert))
        )


_CURRENT = {"settings": FakeSettings()}


def _use(settings: FakeSettings):
    _CURRENT["settings"] = settings


def _load_sso_modules():
    """Stub deps, spec-load base/oidc/saml/registry under real dotted names."""
    saved = {}

    def stub(name, mod):
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = mod

    settings_mod = types.ModuleType("backend.app.settings")
    settings_mod.get_settings = lambda: _CURRENT["settings"]
    logging_mod = types.ModuleType("backend.app.logging_config")
    logging_mod.get_logger = lambda *_: MagicMock()
    crud_mod = types.ModuleType("backend.app.db.crud")
    session_mod = types.ModuleType("backend.app.db.session")
    session_mod.get_async_db = lambda: None
    db_pkg = types.ModuleType("backend.app.db")
    db_pkg.crud = crud_mod

    for name, mod in [
        ("backend", types.ModuleType("backend")),
        ("backend.app", types.ModuleType("backend.app")),
        ("backend.app.db", db_pkg),
        ("backend.app.db.crud", crud_mod),
        ("backend.app.db.session", session_mod),
        ("backend.app.logging_config", logging_mod),
        ("backend.app.settings", settings_mod),
        ("backend.app.dashboard", types.ModuleType("backend.app.dashboard")),
        ("backend.app.dashboard.sso", types.ModuleType("backend.app.dashboard.sso")),
    ]:
        stub(name, mod)

    mods = {}
    for modname, filename in [
        ("backend.app.dashboard.sso.base", "base.py"),
        ("backend.app.dashboard.sso.oidc", "oidc.py"),
        ("backend.app.dashboard.sso.saml", "saml.py"),
        ("backend.app.dashboard.sso.registry", "registry.py"),
    ]:
        spec = importlib.util.spec_from_file_location(modname, _SSO_DIR / filename)
        mod = importlib.util.module_from_spec(spec)
        stub(modname, mod)
        spec.loader.exec_module(mod)
        mods[modname.rsplit(".", 1)[1]] = mod

    return mods, saved, crud_mod


@pytest.fixture(scope="module")
def sso():
    mods, saved, crud_mod = _load_sso_modules()
    mods["_crud"] = crud_mod
    yield mods
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ── Settings gating (real Settings class) ────────────────────────

def _real_settings(**kw):
    spec = importlib.util.spec_from_file_location("mr2_settings_under_test", _APP_DIR / "settings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Settings(secret_key="x", database_url="mysql+pymysql://u:p@h/db", **kw)


def test_settings_gating_google_oidc_saml():
    s = _real_settings()
    assert not s.google_sso_enabled and not s.oidc_sso_enabled and not s.saml_sso_enabled

    assert _real_settings(google_sso_client_id="a", google_sso_client_secret="b").google_sso_enabled
    assert not _real_settings(google_sso_client_id="a").google_sso_enabled

    assert _real_settings(
        oidc_sso_issuer="https://idp", oidc_sso_client_id="a", oidc_sso_client_secret="b"
    ).oidc_sso_enabled

    assert _real_settings(
        saml_sp_entity_id="https://mr/sp", saml_idp_metadata_url="https://idp/metadata"
    ).saml_sso_enabled
    assert _real_settings(
        saml_sp_entity_id="https://mr/sp", saml_idp_entity_id="e",
        saml_idp_sso_url="https://idp/sso", saml_idp_x509_cert="CERT",
    ).saml_sso_enabled
    assert not _real_settings(saml_sp_entity_id="https://mr/sp").saml_sso_enabled


def test_settings_sso_enabled_aggregate():
    assert not _real_settings().sso_enabled
    assert _real_settings(google_sso_client_id="a", google_sso_client_secret="b").sso_enabled
    assert _real_settings(azure_ad_client_id="a", azure_ad_tenant_id="t").sso_enabled


# ── Provider registry ─────────────────────────────────────────────

def test_registry_azure_label_uses_org_name(sso):
    _use(FakeSettings(azure_ad_client_id="a", azure_ad_tenant_id="t"))
    provs = sso["registry"].enabled_providers(org_name="University of Idaho")
    assert [p.id for p in provs] == ["azure"]
    assert provs[0].label == "University of Idaho"
    assert provs[0].login_url == "/login/azure"

    provs = sso["registry"].enabled_providers(org_name=None)
    assert provs[0].label == "SSO"


def test_registry_all_providers_order_and_labels(sso):
    _use(FakeSettings(
        azure_ad_client_id="a", azure_ad_tenant_id="t",
        google_sso_client_id="g", google_sso_client_secret="s",
        oidc_sso_issuer="https://idp", oidc_sso_client_id="c", oidc_sso_client_secret="s",
        oidc_sso_display_name="Okta",
        saml_sp_entity_id="https://mr/sp", saml_idp_metadata_url="https://idp/md",
        saml_display_name="InCommon",
    ))
    provs = sso["registry"].enabled_providers(org_name="U of I")
    assert [p.id for p in provs] == ["azure", "saml", "oidc", "google"]
    labels = {p.id: p.label for p in provs}
    assert labels == {"azure": "U of I", "saml": "InCommon", "oidc": "Okta", "google": "Google"}


def test_registry_saml_only_inherits_org_name(sso):
    _use(FakeSettings(saml_sp_entity_id="https://mr/sp", saml_idp_metadata_url="https://idp/md"))
    provs = sso["registry"].enabled_providers(org_name="Harvard")
    assert [p.id for p in provs] == ["saml"]
    assert provs[0].label == "Harvard"


def test_registry_routes_exist(sso):
    paths = {r.path for r in sso["registry"].sso_router.routes}
    for expected in ("/login/google", "/login/google/authorized", "/login/oidc",
                     "/login/oidc/authorized", "/login/saml", "/login/saml/acs", "/saml/metadata"):
        assert expected in paths


# ── OIDC driver ───────────────────────────────────────────────────

def _google_cfg(sso, **kw):
    _use(FakeSettings(google_sso_client_id="g", google_sso_client_secret="s", **kw))
    return sso["oidc"].google_config()


def test_oidc_profile_from_claims_happy(sso):
    cfg = _google_cfg(sso)
    p = sso["oidc"].profile_from_claims(cfg, {
        "sub": "115", "email": "Goober@Example.COM", "email_verified": True,
        "name": "Goober McGee", "preferred_username": "goober",
    })
    assert p.provider == "google" and p.subject == "115"
    assert p.email == "Goober@Example.COM"  # lowercased later, in provisioning
    assert p.display_name == "Goober McGee" and p.username_hint == "goober"


def test_oidc_profile_rejects_missing_or_unverified(sso):
    cfg = _google_cfg(sso)
    f = sso["oidc"].profile_from_claims
    assert f(cfg, {"email": "a@b.c"}) is None                       # no sub
    assert f(cfg, {"sub": "1"}) is None                             # no email
    assert f(cfg, {"sub": "1", "email": "a@b.c", "email_verified": False}) is None
    # IdPs that omit email_verified are trusted
    assert f(cfg, {"sub": "1", "email": "a@b.c"}) is not None


def test_oidc_email_verified_string_forms_do_not_fail_open(sso):
    """Regression: some IdPs send email_verified as a string; a bare
    `is False` check would treat "false" as verified."""
    cfg = _google_cfg(sso)
    f = sso["oidc"].profile_from_claims
    for falsey in ("false", "False", "0", 0, "no", ""):
        assert f(cfg, {"sub": "1", "email": "a@b.c", "email_verified": falsey}) is None, falsey
    for truthy in (True, "true", "True", "1", 1):
        assert f(cfg, {"sub": "1", "email": "a@b.c", "email_verified": truthy}) is not None, truthy


def test_oidc_hosted_domain_enforced(sso):
    cfg = _google_cfg(sso, google_sso_hosted_domain="uidaho.edu")
    f = sso["oidc"].profile_from_claims
    assert f(cfg, {"sub": "1", "email": "x@uidaho.edu", "hd": "uidaho.edu"}) is not None
    assert f(cfg, {"sub": "1", "email": "x@gmail.com"}) is None
    assert f(cfg, {"sub": "1", "email": "x@evil.com", "hd": "evil.com"}) is None


def test_oidc_absolute_redirect_uri_uses_public_base_url(sso):
    """Regression: request.base_url reports http:// behind an untrusted proxy,
    and IdPs match the redirect URI exactly -> use the configured public URL."""
    _use(FakeSettings(app_base_url="https://mr.example.edu"))
    req = MagicMock()
    req.base_url = "http://localhost:8000/"      # what an untrusted proxy yields
    f = sso["oidc"]._absolute_redirect_uri
    assert f(req, "/login/google/authorized") == "https://mr.example.edu/login/google/authorized"
    assert f(req, "https://other.example/cb") == "https://other.example/cb"


def test_public_base_url_fails_closed_and_ignores_host(sso):
    """public_base_url returns the configured origin and NEVER trusts request
    headers; when APP_BASE_URL is empty it raises SSOConfigError. (2.9.24)"""
    base = sso["base"]
    req = MagicMock()
    req.headers = {"host": "attacker.example", "x-forwarded-host": "attacker.example"}
    req.url.netloc = "attacker.example"
    # Configured -> returns it, trailing slash stripped, headers ignored.
    _use(FakeSettings(app_base_url="https://mr.example.edu/"))
    assert base.public_base_url(req) == "https://mr.example.edu"
    # Unset -> fail closed (no Host fallback).
    _use(FakeSettings(app_base_url=""))
    with pytest.raises(base.SSOConfigError):
        base.public_base_url(req)


def test_oidc_redirect_uri_fails_closed_without_app_base_url(sso):
    # Empty app_base_url must NOT fall back to the attacker-controllable Host /
    # X-Forwarded-* headers — it fails closed (SSOConfigError), which begin_login
    # turns into an error redirect. (2.9.24 SSO redirect hardening.)
    _use(FakeSettings(app_base_url=""))
    req = MagicMock()
    req.url.scheme = "http"
    req.url.netloc = "internal:8000"
    req.headers = {"x-forwarded-proto": "https", "host": "mr.example.edu"}
    with pytest.raises(sso["base"].SSOConfigError):
        sso["oidc"]._absolute_redirect_uri(req, "/login/oidc/authorized")
    # With app_base_url set it resolves against the configured origin, ignoring
    # the request headers entirely.
    _use(FakeSettings(app_base_url="https://mr.example.edu"))
    assert sso["oidc"]._absolute_redirect_uri(req, "/login/oidc/authorized") == \
        "https://mr.example.edu/login/oidc/authorized"
    # An already-absolute redirect_uri is returned unchanged.
    assert sso["oidc"]._absolute_redirect_uri(req, "https://x/y") == "https://x/y"


def test_generic_config_strips_trailing_slash(sso):
    _use(FakeSettings(oidc_sso_issuer="https://cilogon.org/", oidc_sso_client_id="c",
                      oidc_sso_client_secret="s"))
    cfg = sso["oidc"].generic_config()
    assert cfg.issuer == "https://cilogon.org"
    assert cfg.provider_id == "oidc"


# ── CSRF state ────────────────────────────────────────────────────

def test_state_round_trip_and_tamper(sso):
    _use(FakeSettings())
    base = sso["base"]
    state = base.new_signed_state()
    assert base.validate_state(state, state)
    assert not base.validate_state(state, None)
    assert not base.validate_state(None, state)
    assert not base.validate_state(state + "x", state + "x")  # bad signature
    assert not base.validate_state(state, "different")


# ── JIT provisioning ─────────────────────────────────────────────

def _profile(sso, **kw):
    d = dict(provider="google", subject="sub-1", email="New@Example.EDU",
             display_name="New User", username_hint="newuser")
    d.update(kw)
    return sso["base"].SSOProfile(**d)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_jit_existing_by_subject_updates(sso):
    crud = sso["_crud"]
    existing = MagicMock(sso_provider="google", sso_subject="sub-1")
    crud.get_user_by_sso_subject = AsyncMock(return_value=existing)
    crud.get_user_by_email = AsyncMock()
    db = MagicMock(flush=AsyncMock())

    user = _run(sso["base"].find_or_create_sso_user(db, _profile(sso), "other"))
    assert user is existing
    assert user.full_name == "New User"
    crud.get_user_by_email.assert_not_called()


def test_jit_links_existing_local_account_by_email(sso):
    crud = sso["_crud"]
    local = MagicMock(sso_provider=None, sso_subject=None, azure_oid=None,
                      password_hash="argon2hash")
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=local)
    db = MagicMock(flush=AsyncMock())

    user = _run(sso["base"].find_or_create_sso_user(db, _profile(sso), "other"))
    assert user is local
    assert user.sso_provider == "google" and user.sso_subject == "sub-1"
    assert user.password_hash == "argon2hash"  # local login keeps working
    crud.get_user_by_email.assert_awaited_once_with(db, "new@example.edu")


def test_jit_refuses_to_hijack_azure_account_by_email(sso):
    """CRITICAL regression: a second IdP asserting an Azure user's email must
    NOT inherit that account (admin takeover with no password)."""
    crud = sso["_crud"]
    azure_admin = MagicMock(sso_provider=None, sso_subject=None,
                            azure_oid="00000000-1111-2222-3333-444444444444")
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=azure_admin)
    db = MagicMock(flush=AsyncMock())

    user = _run(sso["base"].find_or_create_sso_user(db, _profile(sso), "other"))
    assert user is None
    assert azure_admin.sso_provider is None      # identity not stamped
    assert azure_admin.sso_subject is None


def test_jit_refuses_cross_provider_email_takeover(sso):
    """An account already claimed by provider A is not adoptable by provider B."""
    crud = sso["_crud"]
    other_idp_user = MagicMock(sso_provider="saml", sso_subject="saml-subject", azure_oid=None)
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=other_idp_user)
    db = MagicMock(flush=AsyncMock())

    user = _run(sso["base"].find_or_create_sso_user(db, _profile(sso, provider="oidc"), "other"))
    assert user is None
    assert other_idp_user.sso_provider == "saml"       # untouched
    assert other_idp_user.sso_subject == "saml-subject"


def test_jit_creates_new_user_with_group_quota(sso):
    crud = sso["_crud"]
    group = MagicMock(id=7, rpm_limit=30)
    group.name = "other"
    created = MagicMock()
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=None)
    crud.get_group_by_name = AsyncMock(return_value=group)
    crud.get_user_by_username = AsyncMock(return_value=None)
    crud.create_user = AsyncMock(return_value=created)
    crud.create_quota = AsyncMock()
    db = MagicMock(flush=AsyncMock())

    user = _run(sso["base"].find_or_create_sso_user(db, _profile(sso), "other"))
    assert user is created
    kwargs = crud.create_user.await_args.kwargs
    assert kwargs["username"] == "newuser"          # username_hint prefix wins
    assert kwargs["email"] == "new@example.edu"
    assert kwargs["password_hash"] is None
    assert kwargs["group_id"] == 7
    assert created.sso_provider == "google" and created.sso_subject == "sub-1"
    crud.create_quota.assert_awaited_once()


def test_jit_username_collision_gets_suffix(sso):
    crud = sso["_crud"]
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=None)
    crud.get_group_by_name = AsyncMock(return_value=MagicMock(id=7, rpm_limit=30, name="other"))
    crud.get_user_by_username = AsyncMock(return_value=MagicMock())  # taken
    crud.create_user = AsyncMock(return_value=MagicMock())
    crud.create_quota = AsyncMock()
    db = MagicMock(flush=AsyncMock())

    _run(sso["base"].find_or_create_sso_user(db, _profile(sso, subject="subject-42"), "other"))
    # Suffix = first 8 chars of the subject, mirroring the Azure driver.
    assert crud.create_user.await_args.kwargs["username"] == "newuser_subject-"


def test_jit_refuses_when_default_group_missing(sso):
    """A *_DEFAULT_GROUP naming a group that does not exist must refuse
    cleanly, not pass group_id=None into a NOT NULL column (which
    surfaced as a bare HTTP 500 at the IdP callback)."""
    crud = sso["_crud"]
    crud.get_user_by_sso_subject = AsyncMock(return_value=None)
    crud.get_user_by_email = AsyncMock(return_value=None)
    crud.get_group_by_name = AsyncMock(return_value=None)  # group absent
    crud.get_user_by_username = AsyncMock(return_value=None)
    crud.create_user = AsyncMock(return_value=MagicMock())
    crud.create_quota = AsyncMock()
    db = MagicMock(flush=AsyncMock())

    result = _run(
        sso["base"].find_or_create_sso_user(db, _profile(sso), "no-such-group")
    )
    assert result is None, "must refuse rather than provision without a group"
    crud.create_user.assert_not_awaited()
    crud.create_quota.assert_not_awaited()


def test_jit_rejects_missing_email_or_subject(sso):
    db = MagicMock()
    assert _run(sso["base"].find_or_create_sso_user(db, _profile(sso, email=""), "other")) is None
    assert _run(sso["base"].find_or_create_sso_user(db, _profile(sso, subject=""), "other")) is None


# ── SAML driver ───────────────────────────────────────────────────

def test_saml_profile_eduperson_mapping(sso):
    _use(FakeSettings())
    p = sso["saml"].profile_from_assertion(
        {"mail": ["vandal@uidaho.edu"], "displayName": ["Joe Vandal"],
         "eduPersonPrincipalName": ["vandal@uidaho.edu"]},
        nameid="AAdzZWNyZXRJRA==", nameid_format="urn:...:persistent",
    )
    assert p.provider == "saml"
    assert p.subject == "AAdzZWNyZXRJRA=="       # persistent NameID preferred
    assert p.email == "vandal@uidaho.edu"
    assert p.display_name == "Joe Vandal"
    assert p.username_hint == "vandal@uidaho.edu"


def test_saml_profile_nameid_email_fallback(sso):
    _use(FakeSettings())
    p = sso["saml"].profile_from_assertion({}, nameid="user@adfs.example.com", nameid_format=None)
    assert p.email == "user@adfs.example.com"
    assert p.subject == "user@adfs.example.com"


def test_saml_profile_requires_email(sso):
    _use(FakeSettings())
    assert sso["saml"].profile_from_assertion({}, nameid="opaque-id", nameid_format=None) is None


def test_saml_settings_explicit_idp(sso):
    _use(FakeSettings(
        saml_sp_entity_id="https://mr.example.edu/sp",
        saml_idp_entity_id="https://idp.example.edu/idp",
        saml_idp_sso_url="https://idp.example.edu/sso",
        saml_idp_x509_cert="MIICERT",
    ))
    cfg = sso["saml"].build_saml_settings("https://mr.example.edu")
    assert cfg["strict"] is True
    assert cfg["sp"]["entityId"] == "https://mr.example.edu/sp"
    assert cfg["sp"]["assertionConsumerService"]["url"] == "https://mr.example.edu/login/saml/acs"
    assert cfg["idp"]["entityId"] == "https://idp.example.edu/idp"
    assert cfg["security"]["wantAssertionsSigned"] is True


def test_saml_settings_none_when_unconfigured(sso):
    _use(FakeSettings())
    assert sso["saml"].build_saml_settings("https://mr.example.edu") is None


def _saml_configured():
    return FakeSettings(
        saml_sp_entity_id="https://mr.example.edu/sp",
        saml_idp_entity_id="https://idp/idp", saml_idp_sso_url="https://idp/sso",
        saml_idp_x509_cert="CERT",
    )


def test_saml_security_rejects_deprecated_algorithms(sso):
    """SHA-1 signatures must not be accepted. Note python3-saml has no
    unsolicited-response setting (that php-saml key is inert) — SP-initiated
    enforcement is asserted by the handle_acs tests below."""
    _use(_saml_configured())
    sec = sso["saml"].build_saml_settings("https://mr.example.edu")["security"]
    assert sec["rejectDeprecatedAlgorithm"] is True
    assert "rejectUnsolicitedResponsesWithInResponseTo" not in sec


def _acs_request(sso, cookie_value=None):
    request = MagicMock()
    request.method = "POST"
    request.url.scheme = "https"
    request.url.netloc = "mr.example.edu"
    request.url.path = "/login/saml/acs"
    request.query_params = {}
    request.headers = {"host": "mr.example.edu"}
    request.cookies = {sso["saml"].REQUEST_ID_COOKIE: cookie_value} if cookie_value else {}
    request.form = AsyncMock(return_value={"SAMLResponse": "base64blob"})
    return request


def _install_fake_onelogin(sso, monkey, in_response_to, authenticated=True):
    """Patch _import_onelogin with a stand-in that mimics python3-saml's
    conditional InResponseTo behavior."""
    auth = MagicMock()
    auth.process_response = MagicMock()
    auth.get_errors.return_value = []
    auth.is_authenticated.return_value = authenticated
    auth.get_last_response_in_response_to.return_value = in_response_to
    auth.get_attributes.return_value = {"mail": ["victim@example.edu"]}
    auth.get_nameid.return_value = "nameid-1"
    auth.get_nameid_format.return_value = "persistent"
    monkey.append((sso["saml"], "_import_onelogin", sso["saml"]._import_onelogin))
    sso["saml"]._import_onelogin = lambda: (lambda req, settings: auth, MagicMock())
    return auth


def test_saml_acs_rejects_unsolicited_without_cookie(sso):
    """CRITICAL regression: an IdP-initiated POST to the ACS (no AuthnRequest
    cookie) is a login-CSRF vector and must never reach provisioning."""
    _use(_saml_configured())
    monkey = []
    provisioned = []
    _install_fake_onelogin(sso, monkey, in_response_to="whatever")
    orig_find = sso["saml"].find_or_create_sso_user
    sso["saml"].find_or_create_sso_user = AsyncMock(side_effect=lambda *a, **k: provisioned.append(a))
    try:
        resp = _run(sso["saml"].handle_acs(_acs_request(sso), MagicMock()))
        assert resp.status_code == 302
        assert "must+start+from+this+site" in resp.headers["location"]
        assert provisioned == []          # never provisioned/logged in
    finally:
        sso["saml"].find_or_create_sso_user = orig_find
        for mod, name, val in monkey:
            setattr(mod, name, val)


def test_saml_acs_rejects_response_without_matching_in_response_to(sso):
    """A valid cookie plus a response that omits/echoes a different
    InResponseTo must be refused — python3-saml skips that comparison itself."""
    _use(_saml_configured())
    monkey = []
    provisioned = []
    _install_fake_onelogin(sso, monkey, in_response_to=None)  # unsolicited-style
    cookie = sso["base"].state_serializer().dumps("our-request-id")
    orig_find = sso["saml"].find_or_create_sso_user
    sso["saml"].find_or_create_sso_user = AsyncMock(side_effect=lambda *a, **k: provisioned.append(a))
    try:
        resp = _run(sso["saml"].handle_acs(_acs_request(sso, cookie), MagicMock()))
        assert resp.status_code == 302
        assert "did+not+match" in resp.headers["location"]
        assert provisioned == []
    finally:
        sso["saml"].find_or_create_sso_user = orig_find
        for mod, name, val in monkey:
            setattr(mod, name, val)


def test_saml_acs_accepts_matching_sp_initiated_response(sso):
    """CONTROL: a genuine SP-initiated round trip still logs in."""
    _use(_saml_configured())
    monkey = []
    _install_fake_onelogin(sso, monkey, in_response_to="our-request-id")
    cookie = sso["base"].state_serializer().dumps("our-request-id")
    orig_find, orig_finish = sso["saml"].find_or_create_sso_user, sso["saml"].finish_login
    sso["saml"].find_or_create_sso_user = AsyncMock(return_value=MagicMock(is_active=True))
    sso["saml"].finish_login = AsyncMock(return_value="LOGGED_IN")
    try:
        resp = _run(sso["saml"].handle_acs(_acs_request(sso, cookie), MagicMock()))
        assert resp == "LOGGED_IN"
        sso["saml"].find_or_create_sso_user.assert_awaited_once()
    finally:
        sso["saml"].find_or_create_sso_user, sso["saml"].finish_login = orig_find, orig_finish
        for mod, name, val in monkey:
            setattr(mod, name, val)


def test_saml_metadata_url_must_be_https(sso):
    """Regression: the metadata document carries the IdP signing cert (the only
    trust anchor), so plain http must be refused."""
    _use(FakeSettings(saml_sp_entity_id="https://mr.example.edu/sp",
                      saml_idp_metadata_url="http://idp.example.edu/metadata"))
    assert sso["saml"].build_saml_settings("https://mr.example.edu") is None


def test_saml_request_dict_ignores_forwarded_host(sso):
    """Regression: python3-saml validates Destination/Recipient against this
    host, so a client-supplied X-Forwarded-Host must not influence it."""
    _use(FakeSettings(app_base_url="https://mr.example.edu"))
    request = MagicMock()
    request.method = "GET"
    request.url.scheme = "http"
    request.url.netloc = "internal:8000"
    request.url.path = "/login/saml/acs"
    request.query_params = {}
    request.headers = {"x-forwarded-host": "evil.example.com", "host": "evil.example.com"}

    req = _run(sso["saml"]._prepare_fastapi_request(request))
    assert req["http_host"] == "mr.example.edu"
    assert req["https"] == "on"


def test_saml_request_id_cookie_defined(sso):
    assert sso["saml"].REQUEST_ID_COOKIE == "saml_request_id"
    src = (_SSO_DIR / "saml.py").read_text()
    # begin_login stores the signed AuthnRequest id; ACS feeds it to process_response
    assert "auth.get_last_request_id()" in src
    assert "auth.process_response(request_id=request_id)" in src


def test_saml_request_id_cookie_survives_cross_site_post(sso):
    """CRITICAL regression: the IdP returns the assertion via a CROSS-SITE POST
    to our ACS. Browsers withhold SameSite=Lax cookies on cross-site POSTs, so
    a "lax" cookie here would make handle_acs reject every login from an IdP on
    a different registrable domain. Must be SameSite=None + Secure."""
    src = (_SSO_DIR / "saml.py").read_text()
    setcookie = src.split("key=REQUEST_ID_COOKIE")[1].split("max_age=")[0]
    assert 'samesite="none"' in setcookie
    assert "secure=True" in setcookie
    assert 'samesite="lax"' not in setcookie


# ── Migration + model + deploy-file contracts ─────────────────────

_REPO = Path(__file__).resolve().parents[4]


def test_migration_068_contract():
    src = (_REPO / "backend/app/db/migrations/versions/20260730_000000_068_add_generic_sso_identity.py").read_text()
    assert 'revision = "068"' in src and 'down_revision = "067"' in src
    assert '"sso_provider"' in src and '"sso_subject"' in src
    assert "uq_users_sso_identity" in src and "unique=True" in src


def test_models_account_type_covers_generic_sso():
    src = (_REPO / "backend/app/db/models.py").read_text()
    assert "self.azure_oid or self.sso_subject" in src
    assert "sso_provider" in src and "sso_subject" in src


def test_azure_driver_routes_and_oid_semantics():
    """The Azure AD flow keeps its own routes and azure_oid identity column."""
    src = (_REPO / "backend/app/dashboard/azure_auth.py").read_text()
    assert '@azure_router.get("/login/azure")' in src
    assert '@azure_router.get("/login/azure/authorized")' in src
    assert "find_or_create_azure_user" in src
    assert "user.azure_oid = azure_oid" in src
    assert "user.sso_subject" not in src  # Azure never writes the shared column


def test_azure_email_link_guard_matches_shared_driver():
    """CRITICAL: the Azure driver must refuse to adopt an account already
    claimed by another provider, exactly as sso/base.py does. Without this,
    enabling Azure alongside SAML/OIDC/Google lets an Azure-asserted email
    inherit an account provisioned by that other provider."""
    src = (_REPO / "backend/app/dashboard/azure_auth.py").read_text()
    body = src.split("async def find_or_create_azure_user")[1]
    # The email fallback must be guarded, not a bare assignment.
    assert "candidate = await crud.get_user_by_email" in body
    assert "candidate.azure_oid or candidate.sso_provider" in body
    assert "sso_email_link_refused" in body
    # and the guard must return before any identity is stamped
    guard = body.split("candidate.azure_oid or candidate.sso_provider")[1]
    assert "return None" in guard.split("user = candidate")[0]


def test_both_drivers_agree_on_link_policy():
    """Consistency: both drivers gate email-linking on the same two fields, so
    the documented rule is true of every provider."""
    azure = (_REPO / "backend/app/dashboard/azure_auth.py").read_text()
    shared = (_SSO_DIR / "base.py").read_text()
    for src in (azure, shared):
        assert "candidate.azure_oid or candidate.sso_provider" in src
        assert "sso_email_link_refused" in src


def test_docker_compose_passes_all_sso_env_vars():
    compose = (_REPO / "docker-compose.yml").read_text()
    for var in [
        "GOOGLE_SSO_CLIENT_ID", "GOOGLE_SSO_CLIENT_SECRET", "GOOGLE_SSO_REDIRECT_URI",
        "GOOGLE_SSO_HOSTED_DOMAIN", "GOOGLE_SSO_DEFAULT_GROUP",
        "OIDC_SSO_ISSUER", "OIDC_SSO_CLIENT_ID", "OIDC_SSO_CLIENT_SECRET",
        "OIDC_SSO_REDIRECT_URI", "OIDC_SSO_DISPLAY_NAME", "OIDC_SSO_SCOPES", "OIDC_SSO_DEFAULT_GROUP",
        "SAML_SP_ENTITY_ID", "SAML_SP_ACS_URL", "SAML_IDP_METADATA_URL", "SAML_IDP_ENTITY_ID",
        "SAML_IDP_SSO_URL", "SAML_IDP_X509_CERT", "SAML_DISPLAY_NAME", "SAML_DEFAULT_GROUP",
        "SAML_ATTR_EMAIL", "SAML_ATTR_NAME", "SAML_ATTR_USERNAME",
    ]:
        assert f"- {var}=${{{var}:-" in compose, f"{var} missing from docker-compose environment"


def test_dockerfile_installs_saml_deps():
    dockerfile = (_REPO / "Dockerfile").read_text()
    assert "libxmlsec1-dev" in dockerfile
    assert ".[saml]" in dockerfile
    pyproject = (_REPO / "pyproject.toml").read_text()
    assert "python3-saml" in pyproject


# ── 2.9.6: OIDC callback hardening for external IdPs ─────────────


def test_id_token_claims_decoded_without_verification(sso):
    """ADFS-style IdPs return a userinfo document containing only `sub`;
    email/name live in the id_token. We merge them so provisioning works."""
    import base64
    import json as _json

    oidc = sso["oidc"]
    payload = {"sub": "abc", "email": "person@utah.edu", "name": "A Person"}
    raw = base64.urlsafe_b64encode(_json.dumps(payload).encode()).rstrip(b"=").decode()
    token = f"header.{raw}.signature"

    assert oidc._decode_id_token_claims(token) == payload


def test_id_token_decode_is_total(sso):
    """Malformed tokens must degrade to {}, never raise into the callback."""
    oidc = sso["oidc"]
    for bad in ("", "notatoken", "a.b", "a.!!!!.c", "a." + "x" * 7 + ".c"):
        assert oidc._decode_id_token_claims(bad) == {}


def test_email_verified_helper_matches_profile_rule(sso):
    """The helper that picks the error message must agree with the rule
    that actually rejects the profile — otherwise we'd report the wrong
    cause. Absent claim is trusted; string forms must not fail open."""
    oidc = sso["oidc"]
    assert oidc._email_is_verified({}) is True
    assert oidc._email_is_verified({"email_verified": True}) is True
    assert oidc._email_is_verified({"email_verified": "true"}) is True
    assert oidc._email_is_verified({"email_verified": "1"}) is True
    assert oidc._email_is_verified({"email_verified": False}) is False
    assert oidc._email_is_verified({"email_verified": "false"}) is False
    assert oidc._email_is_verified({"email_verified": "0"}) is False


def test_token_exchange_falls_back_to_basic_auth(sso):
    """Okta and friends register clients as client_secret_basic by
    default and reject credentials in the POST body."""
    src = (_APP_DIR / "dashboard" / "sso" / "oidc.py").read_text()
    assert "auth=(cfg.client_id, cfg.client_secret)" in src
    # The failure log must carry the response body: OIDC error payloads
    # hold the actionable detail a bare status code hides.
    assert "body=token_response.text[:500]" in src
    assert "basic_body=basic_response.text[:500]" in src


def test_profile_failure_messages_are_distinct(sso):
    """One generic 'missing a verified email' sent operators down the
    wrong path; the three causes must be reported separately."""
    src = (_APP_DIR / "dashboard" / "sso" / "oidc.py").read_text()
    assert "missing+required+claims" in src
    assert "not+verified+at+the+identity+provider" in src
    assert "not+in+an+allowed+domain" in src


def test_stdlib_logger_modules_use_no_structlog_kwargs():
    """Modules bound to a stdlib logger must not use structlog-style
    keyword fields: `logging.Logger.error(msg, key=...)` raises TypeError
    at call time, which is worst inside an error-handling guard (2.9.6
    regression — the Azure missing-group guard crashed instead of
    returning, producing the very 500 it was written to prevent)."""
    import ast

    _STDLIB_OK = {"exc_info", "extra", "stack_info", "stacklevel"}
    _LEVELS = {"debug", "info", "warning", "warn", "error", "critical", "exception"}

    offenders = []
    # Whole app, not just dashboard/: services/email_service.py also binds a
    # stdlib logger, and 2.9.9 added a DLP alert-notification call into it.
    for path in sorted(_APP_DIR.rglob("*.py")):
        if "tests" in path.parts:
            continue
        src = path.read_text()
        # Only modules that bind the STDLIB logger are at risk.
        if "logging.getLogger(" not in src:
            continue
        if "from backend.app.logging_config import get_logger" in src:
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in _LEVELS):
                continue
            if not (isinstance(fn.value, ast.Name) and fn.value.id == "logger"):
                continue
            bad = [k.arg for k in node.keywords if k.arg not in _STDLIB_OK]
            if bad:
                offenders.append(
                    f"{path.relative_to(_APP_DIR)}:{node.lineno} logger.{fn.attr}(..., {', '.join(bad)}=)"
                )

    assert not offenders, (
        "structlog-style kwargs on a stdlib logger will raise TypeError when called: "
        + "; ".join(offenders)
    )


def test_azure_missing_group_guard_logs_safely():
    """The Azure guard's log line must survive a real stdlib logger."""
    import logging

    src = (_REPO / "backend/app/dashboard/azure_auth.py").read_text()
    assert "sso_default_group_missing provider=azure" in src, "guard log line missing"
    # Execute the exact %-format the guard uses.
    logging.getLogger("test_azure_guard").error(
        "sso_default_group_missing provider=azure group=%s"
        " (create the group or fix AZURE_AD_DEFAULT_GROUP)",
        "other",
    )


# ── 2.9.7: SAML SP key pair ──────────────────────────────────────

_TEST_CERT = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIIBfakeCERTIFICATEbodyForUnitTestsOnly==\n"
    "-----END CERTIFICATE-----"
)
_TEST_KEY = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIBfakePRIVATEKEYbodyForUnitTestsOnly==\n"
    "-----END PRIVATE KEY-----"
)


def _saml_settings(sso, **kw):
    _use(FakeSettings(
        saml_sp_entity_id="https://mr.example.edu/sp",
        saml_idp_entity_id="https://idp.example.edu/idp",
        saml_idp_sso_url="https://idp.example.edu/sso",
        saml_idp_x509_cert="IDPCERT",
        **kw,
    ))
    return sso["saml"].build_saml_settings("https://mr.example.edu")


def test_saml_sp_has_no_key_material_by_default(sso):
    """Unchanged default: no SP key pair, and the two dependent security
    options stay absent so python3-saml cannot refuse the config."""
    cfg = _saml_settings(sso)
    assert "x509cert" not in cfg["sp"]
    assert "privateKey" not in cfg["sp"]
    assert "authnRequestsSigned" not in cfg["security"]
    assert "wantAssertionsEncrypted" not in cfg["security"]


def test_saml_sp_keypair_is_wired_through(sso):
    cfg = _saml_settings(
        sso, saml_sp_x509_cert=_TEST_CERT, saml_sp_private_key=_TEST_KEY
    )
    assert cfg["sp"]["x509cert"] == _TEST_CERT
    assert cfg["sp"]["privateKey"] == _TEST_KEY
    # Present but off unless explicitly enabled.
    assert cfg["security"]["authnRequestsSigned"] is False
    assert cfg["security"]["wantAssertionsEncrypted"] is False


def test_saml_security_options_enable_with_keypair(sso):
    cfg = _saml_settings(
        sso,
        saml_sp_x509_cert=_TEST_CERT,
        saml_sp_private_key=_TEST_KEY,
        saml_authn_requests_signed=True,
        saml_want_assertions_encrypted=True,
    )
    assert cfg["security"]["authnRequestsSigned"] is True
    assert cfg["security"]["wantAssertionsEncrypted"] is True


def test_saml_security_options_ignored_without_keypair(sso):
    """python3-saml raises sp_cert_not_found_and_required if either flag
    is on without a key pair, which would take SAML down entirely — so
    they must be dropped, not passed through."""
    cfg = _saml_settings(
        sso, saml_authn_requests_signed=True, saml_want_assertions_encrypted=True
    )
    assert "authnRequestsSigned" not in cfg["security"]
    assert "wantAssertionsEncrypted" not in cfg["security"]
    # A cert with no key is equally unusable for signing/decryption.
    cfg = _saml_settings(
        sso, saml_sp_x509_cert=_TEST_CERT, saml_authn_requests_signed=True
    )
    assert "authnRequestsSigned" not in cfg["security"]
    assert cfg["sp"]["x509cert"] == _TEST_CERT  # still published in metadata


def test_saml_pem_accepts_escaped_newlines(sso):
    """A multi-line PEM survives a single-line .env entry as \\n escapes."""
    inline = _TEST_CERT.replace("\n", "\\n")
    cfg = _saml_settings(sso, saml_sp_x509_cert=inline, saml_sp_private_key=_TEST_KEY)
    assert cfg["sp"]["x509cert"] == _TEST_CERT
    assert "\\n" not in cfg["sp"]["x509cert"]


def test_saml_pem_accepts_file_path(sso, tmp_path):
    cert_file = tmp_path / "sp.crt"
    cert_file.write_text(_TEST_CERT + "\n")
    key_file = tmp_path / "sp.key"
    key_file.write_text(_TEST_KEY + "\n")
    cfg = _saml_settings(
        sso, saml_sp_x509_cert=str(cert_file), saml_sp_private_key=str(key_file)
    )
    assert cfg["sp"]["x509cert"] == _TEST_CERT
    assert cfg["sp"]["privateKey"] == _TEST_KEY


def test_saml_pem_unreadable_path_degrades_without_raising(sso):
    """A bad path must not take the provider down — metadata simply has
    no KeyDescriptor, and the dependent options stay off."""
    cfg = _saml_settings(
        sso,
        saml_sp_x509_cert="/nonexistent/sp.crt",
        saml_sp_private_key="/nonexistent/sp.key",
        saml_want_assertions_encrypted=True,
    )
    assert "x509cert" not in cfg["sp"]
    assert "wantAssertionsEncrypted" not in cfg["security"]


def test_saml_pem_never_logged(sso):
    """BEHAVIORAL: no shape of bad input may reach the log.

    An earlier grep-based version of this test enumerated kwarg names and
    missed a real leak (`path=raw` echoed a base64-packed private key), so
    this asserts on what is actually emitted instead.
    """
    saml = sso["saml"]
    calls = []

    class _Recorder:
        def __getattr__(self, _level):
            def _log(event, **fields):
                calls.append((event, fields))
            return _log

    original_logger = saml.logger
    saml.logger = _Recorder()
    try:
        sentinel = "SUPERSECRETKEYMATERIAL"
        # Every non-PEM shape an operator might supply: base64-packed PEM,
        # a bare key body, a plain wrong path, and a path-shaped value.
        for bad in (
            sentinel,
            sentinel * 200,                      # too long to be a path (ENAMETOOLONG)
            f"MIIB{sentinel}==",                 # headerless base64 body
            f"/nonexistent/{sentinel}.key",      # path-shaped, missing
        ):
            assert saml._load_pem(bad, "SAML_SP_PRIVATE_KEY") is None

        blob = repr(calls)
        assert sentinel not in blob, f"PEM material leaked into logs: {blob[:300]}"
        assert calls, "failures must still be logged (diagnosability)"
        # The record must stay useful without the value.
        assert any("value_sha256_8" in f for _, f in calls)
    finally:
        saml.logger = original_logger


def test_saml_pem_non_utf8_file_degrades(sso, tmp_path):
    """A DER/PKCS#12 file (or any non-UTF-8 blob) must not 500 every SAML
    endpoint — UnicodeDecodeError has to be caught alongside OSError."""
    der = tmp_path / "sp.der"
    der.write_bytes(b"\x30\x82\x01\x0a\xff\xfe\x00binary-not-utf8")
    cfg = _saml_settings(sso, saml_sp_x509_cert=str(der), saml_sp_private_key=str(der))
    assert "x509cert" not in cfg["sp"]
    assert "privateKey" not in cfg["sp"]


def test_saml_sp_keypair_env_passthrough():
    """Settings the app reads must be in docker-compose (MEMORY.md gotcha)."""
    compose = (_REPO / "docker-compose.yml").read_text()
    for var in (
        "SAML_SP_X509_CERT",
        "SAML_SP_PRIVATE_KEY",
        "SAML_AUTHN_REQUESTS_SIGNED",
        "SAML_WANT_ASSERTIONS_ENCRYPTED",
    ):
        assert f"- {var}=${{{var}:-" in compose, f"{var} not passed through"
