############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_apps_provisioning.py: Registered-app session endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The registered-app provisioning endpoint.

Two credentials are required and neither is sufficient alone: the app's own
`apps:provision` key proves the caller is that app's server, and the user's
Entra id_token proves that person just authenticated. Requiring both means a
compromised app can only reach users actually signing in to it.

The properties defended here are the ones whose absence would be a security
hole rather than a bug:

  * identity comes only from the verified token, never the request body
  * a credential cannot act for an app it does not belong to
  * the key issued is inference-only even when its owner is an administrator
  * the cross-provider account-takeover guard still applies
  * app-provisioned users are marked for later group classification

Source-contract style is used where behaviour would require standing up the
DB/telemetry import chain; the pure logic is executed.
"""

import re
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[2]
_SRC = (_APP / "api" / "apps_api.py").read_text()


class TestIdentityComesOnlyFromTheToken:
    def test_request_model_accepts_nothing_but_the_token(self):
        """An app asserting who its users are is the attack this design
        exists to prevent — so the body must carry no identity at all."""
        i = _SRC.index("class AppSessionRequest")
        block = _SRC[i:_SRC.index("class AppSessionResponse")]
        for forbidden in ("email", "username", "oid", "user_id", "upn", "subject"):
            assert f"{forbidden}:" not in block, (
                f"AppSessionRequest must not accept {forbidden} from the caller"
            )
        assert "id_token: str" in block

    def test_user_lookup_keys_on_verified_oid(self):
        assert "identity.oid" in _SRC
        i = _SRC.index("def _provision_user")
        block = _SRC[i:_SRC.index("async def _issue_or_reuse_key")]
        assert '"id": identity.oid' in block

    def test_token_is_verified_not_decoded(self):
        assert "verify_entra_id_token" in _SRC
        assert "_decode_id_token_claims" not in _SRC


class TestNamespaceEnforcement:
    def test_credential_must_belong_to_the_app(self):
        """Without this, any registered app's credential could provision for
        every other app — the scope would be global, not bounded."""
        assert "caller_key.app_id != app.id" in _SRC
        i = _SRC.index("caller_key.app_id != app.id")
        assert "403" in _SRC[i:i + 500] or "HTTP_403_FORBIDDEN" in _SRC[i:i + 500]

    def test_endpoint_requires_the_provision_scope(self):
        assert "require_scope(SCOPE_APP_PROVISION)" in _SRC

    def test_unknown_and_disabled_apps_are_indistinguishable(self):
        """A caller holding another app's credential should not be able to
        enumerate which apps exist here."""
        i = _SRC.index("app is None or app.status")
        block = _SRC[i:i + 400]
        assert "404" in block
        assert "Unknown application" in block


class TestIssuedKeyIsBounded:
    def test_key_is_inference_only(self):
        """An administrator using a first-party app must not hand that app an
        admin credential — the DLP-key failure mode."""
        assert "format_scopes(APP_USER_KEY_SCOPES)" in _SRC

    def test_key_is_hidden_from_the_owner(self):
        assert "key_row.hidden = True" in _SRC

    def test_key_expires(self):
        """A credential that is both invisible to its owner and immortal is
        the worst case for revocation."""
        assert "expires_at=datetime.now(timezone.utc) + timedelta(days=ttl_days)" in _SRC
        assert "app.key_ttl_days" in _SRC

    def test_key_is_namespaced_to_the_app(self):
        assert "key_row.app_id = app.id" in _SRC

    def test_live_keys_per_user_are_capped(self):
        assert "MAX_LIVE_KEYS_PER_APP_USER" in _SRC
        i = _SRC.index("live >= MAX_LIVE_KEYS_PER_APP_USER")
        assert "revoke_app_keys" in _SRC[i:i + 400]


class TestRotationSemantics:
    def test_reuses_a_key_with_enough_life_left(self):
        """Rotating on every call would invalidate a user's other concurrent
        sessions; never rotating would leave a leaked key useful for its full
        TTL. Reuse while it outlasts the session about to start."""
        assert "REUSE_IF_REMAINING_FRACTION" in _SRC
        i = _SRC.index("remaining >")
        assert "return None, existing, False" in _SRC[i:i + 300]

    def test_plaintext_is_only_returned_when_freshly_minted(self):
        """MindRouter stores hashes only, so a reused key cannot be re-shown —
        the caller keeps what it already holds."""
        i = _SRC.index("def _issue_or_reuse_key")
        block = _SRC[i:]
        assert "return None, existing, False" in block
        assert "return full_key, key_row, True" in block

    def test_naive_expiry_from_mariadb_is_handled(self):
        """MariaDB returns naive datetimes; comparing one to an aware now()
        raises TypeError."""
        assert "def _as_utc(" in _SRC
        assert "value.tzinfo is None" in _SRC
        assert "_as_utc(existing.expires_at)" in _SRC

    def test_the_response_never_mixes_naive_and_aware_timestamps(self):
        """A minted key carries the aware datetime just constructed; a reused
        one carries what MariaDB returned. Serialized raw, the same field would
        sometimes have a UTC offset and sometimes not, and a client parsing the
        offset-less form as local time believes the key outlives its expiry."""
        assert "expires_at=_as_utc(key_row.expires_at)" in _SRC
        assert "expires_at=key_row.expires_at," not in _SRC

    def test_an_app_that_lost_its_key_cache_can_recover(self):
        """Reuse hands back a key whose plaintext can never be shown again. An
        app that restarted without persistent storage would otherwise have no
        way to obtain a usable key until the old one aged past half its TTL —
        two weeks, at the default."""
        assert "force_rotate: bool = False" in _SRC
        i = _SRC.index("def _issue_or_reuse_key")
        block = _SRC[i:]
        assert "if force_rotate and existing is not None:" in block
        assert "existing = None" in block

    def test_forcing_rotation_defaults_off_and_claims_no_identity(self):
        """It only asks for a key the caller is already entitled to, so it does
        not weaken the rule that identity comes solely from the token. Default
        off so an app that never sets it keeps the cheap reuse path."""
        i = _SRC.index("class AppSessionRequest")
        block = _SRC[i:_SRC.index("class AppSessionResponse")]
        assert "force_rotate: bool = False" in block
        fields = re.findall(r"^\s{4}(\w+):", block, re.M)
        assert set(fields) == {"id_token", "force_rotate"}, fields

    def test_reuse_returns_null_not_an_empty_string(self):
        """An app assigning the field straight into its key store should get an
        obviously-absent value, not a plausible-looking empty one."""
        i = _SRC.index("class AppSessionResponse")
        block = _SRC[i:_SRC.index("@router.post")]
        assert "api_key: Optional[str] = None" in block
        assert 'api_key=api_key_str or ""' not in _SRC

    def test_forced_rotation_is_throttled(self):
        """Minting runs Argon2 by design. Without a ceiling, an app that set
        force_rotate on every call would turn each sign-in into deliberate CPU
        work — and the recovery case it exists for happens once."""
        assert "MIN_FORCED_ROTATE_INTERVAL_SECONDS" in _SRC
        i = _SRC.index("if force_rotate and existing is not None:")
        block = _SRC[i:i + 1600]
        assert "429" in block or "HTTP_429_TOO_MANY_REQUESTS" in block
        assert "Retry-After" in block, "a throttle should say when to come back"

    def test_throttle_handles_naive_timestamps(self):
        """created_at comes back naive from MariaDB; subtracting it from an
        aware now() raises TypeError inside the guard."""
        i = _SRC.index("if force_rotate and existing is not None:")
        block = _SRC[i:i + 1600]
        assert "_as_utc(existing.created_at)" in block

    def test_response_identifies_which_key_is_live(self):
        """Lets a caller check the key it holds against the one MindRouter
        considers current, and force a rotation when they disagree."""
        assert "key_prefix=key_row.key_prefix" in _SRC


class TestReusesTheSsoSecurityRules:
    def test_provisioning_delegates_to_the_azure_driver(self):
        """The cross-provider takeover guard lives in that predicate. A second
        provisioning door that reimplemented it would be the bypass."""
        assert "find_or_create_azure_user" in _SRC

    def test_refusal_is_surfaced_not_swallowed(self):
        i = _SRC.index("user is None")
        block = _SRC[i:i + 700]
        assert "409" in block or "HTTP_409_CONFLICT" in block

    def test_inactive_users_are_refused(self):
        assert "not user.is_active" in _SRC

    def test_job_title_is_absent_and_that_is_deliberate(self):
        i = _SRC.index("def _provision_user")
        block = _SRC[i:i + 2200]
        assert "jobTitle" in block, "the omission should be documented in place"
        assert '"jobTitle":' not in block, "an id_token does not carry jobTitle"

    def test_new_users_are_marked_unclassified(self):
        assert "user.group_classified = False" in _SRC

    def test_creation_is_decided_on_both_keys_the_driver_matches_on(self):
        """The driver has TWO non-creating outcomes: the oid, and — for an
        account no provider has claimed — the email. Deciding on the oid alone
        classifies an ADOPTED local account as new and marks it for
        re-grouping, which is exactly what migration 074 promises never happens
        to an existing user. The account that best fits that description is the
        local bootstrap admin, whose group is the deployment's way back in."""
        i = _SRC.index("def _provision_user")
        block = _SRC[i:_SRC.index("async def _issue_or_reuse_key")]
        assert "get_user_by_azure_oid" in block
        assert "get_user_by_email" in block
        oid_at = block.index("get_user_by_azure_oid")
        email_at = block.index("get_user_by_email")
        created_at = block.index("created = existing is None")
        assert oid_at < created_at and email_at < created_at, (
            "both lookups must resolve before created-ness is decided"
        )

    def test_a_token_with_no_email_is_its_own_error(self):
        """The `email` claim is optional in Entra and off by default. Folding
        that into the takeover 409 sends an integrator hunting a duplicate
        account that does not exist."""
        i = _SRC.index("def _provision_user")
        block = _SRC[i:_SRC.index("async def _issue_or_reuse_key")]
        assert "if not identity.email:" in block
        assert "HTTP_400_BAD_REQUEST" in block

    def test_a_lost_provisioning_race_is_retried(self):
        """users.username and users.email are UNIQUE and provisioning is a
        read-then-insert with no lock. Prod runs two uvicorn workers, so one
        user opening the app in two tabs can have both decide the account does
        not exist."""
        assert "except IntegrityError" in _SRC
        i = _SRC.index("except IntegrityError")
        block = _SRC[i:i + 700]
        assert "db.rollback()" in block
        assert "_provision_user" in block

    def test_audit_records_the_forwarded_address(self):
        """Behind nginx, request.client.host is the proxy — identical for every
        caller. The whole point of this row is which app server called."""
        assert "ip_address=get_client_ip(request)" in _SRC
        code = [ln for ln in _SRC.splitlines() if not ln.lstrip().startswith("#")]
        assert not [ln for ln in code if "request.client.host" in ln], (
            "the raw peer address must not be recorded"
        )


class TestGroupClassificationFix:
    """The trap: group is chosen once at provisioning and never revisited, and
    an id_token carries no jobTitle — so an app-provisioned faculty member
    would sit in the default group forever."""

    def _azure_src(self):
        return (_APP / "dashboard" / "azure_auth.py").read_text()

    def test_reclassifies_only_unclassified_users(self):
        src = self._azure_src()
        assert "not getattr(user, \"group_classified\", True)" in src
        assert "user.group_classified = True" in src

    def test_requires_job_title_to_reclassify(self):
        src = self._azure_src()
        i = src.index('group_classified", True)')
        assert "and job_title" in src[i:i + 200]

    def test_uses_stdlib_log_formatting(self):
        """azure_auth binds a STDLIB logger; structlog kwargs raise TypeError
        at call time (the 2.9.6 regression)."""
        src = self._azure_src()
        i = src.index("user_group_classified")
        call = src[i - 40:i + 400]
        assert "%s" in call
        assert "user_id=user.id" not in call

    def test_a_privileged_group_is_never_overwritten(self):
        """jobTitle maps to exactly four group names and `admin` is not among
        them, so an administrator who reached MindRouter through an app first
        would be demoted on their first direct sign-in — silently, because
        losing admin produces no error, only 403s later."""
        src = self._azure_src()
        assert "current_group.is_admin or current_group.has_admin_read" in src
        i = src.index('group_classified", True)')
        assert "not privileged" in src[i:i + 200]

    def test_an_admin_choosing_a_group_settles_the_classification(self):
        """Otherwise an account an administrator deliberately placed keeps the
        flag and is re-grouped from jobTitle on the next sign-in."""
        crud_src = (_APP / "db" / "crud.py").read_text()
        i = crud_src.index("async def update_user")
        block = crud_src[i:i + 1400]
        assert 'kwargs.get("group_id")' in block
        assert "user.group_classified = True" in block

    def test_reclassification_resyncs_the_rate_limit(self):
        """Token budget and scheduler weight are read from the group at request
        time, but RPM is copied into the user's quota row at creation. Without
        this the fix is two-thirds applied and the rest is invisible."""
        src = self._azure_src()
        i = src.index("user.group_classified = True")
        block = src[i - 400:i + 600]
        assert "get_user_quota" in block
        assert "rpm_limit" in block

    def test_default_is_classified_so_nobody_is_reassigned_on_upgrade(self):
        mig = (_APP / "db" / "migrations" / "versions"
               / "20260809_000000_074_add_group_classified_flag.py").read_text()
        assert re.search(r'^revision = "074"', mig, re.M)
        assert re.search(r'^down_revision = "073"', mig, re.M)
        assert 'server_default=sa.text("1")' in mig


class TestAuditTrail:
    def test_every_session_is_logged(self):
        assert "apps.session" in _SRC
        assert "log_admin_action" in _SRC

    def test_actor_is_the_app_not_a_person(self):
        i = _SRC.index("log_admin_action")
        assert "user_id=None" in _SRC[i:i + 400]

    def test_token_rejection_reason_is_logged_but_not_returned(self):
        """The reason describes why verification failed and would help tune a
        forgery, so it is logged and a generic message is returned."""
        i = _SRC.index("EntraTokenError as e")
        block = _SRC[i:i + 700]
        assert "reason=str(e)" in block
        assert "was not accepted" in block
        assert "detail=str(e)" not in block
