############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_apps_admin_panel.py: Admin -> Applications panel
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The admin surface for registered applications.

An application's credential can create MindRouter accounts and mint keys for
them, so the operator surface has to make that privilege visible and — the part
that is easy to get wrong — actually revocable.

The properties defended here are the ones whose absence would be a security
hole rather than a cosmetic gap:

  * disabling an app revokes its keys, rather than only refusing new sessions
  * rotating a credential revokes the previous one BEFORE minting the next
  * the credential carries the provisioning scope and nothing else
  * read-only auditors cannot reach any mutating route
  * the Entra ids that steer JWKS retrieval are validated, not trusted
  * app-minted keys stay out of the owner's own key list

Source-contract and AST style is used throughout: importing the module pulls
the dashboard routes package and with it the whole db/telemetry chain, which
the project forbids at test-module level. The validation regexes are lifted out
of the source and executed for real.
"""

import ast
import re
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parents[2]
_ROUTES = _APP / "dashboard" / "apps_routes.py"
_SRC = _ROUTES.read_text()
_TREE = ast.parse(_SRC)
_TEMPLATES = _APP / "dashboard" / "templates"


def _compiled_pattern(name: str):
    """Compile a module-level `NAME = re.compile("...")` without importing.

    Adjacent string literals are merged by the parser, so a pattern split
    across source lines still arrives here as one constant.
    """
    for node in _TREE.body:
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == name for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            return re.compile(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"{name} not found in apps_routes.py")


def _function(name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in apps_routes.py")


def _calls(func) -> list:
    """Names of every function called inside `func`, in source order."""
    names = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.append((node.lineno, fn.id))
            elif isinstance(fn, ast.Attribute):
                names.append((node.lineno, fn.attr))
    return [n for _, n in sorted(names)]


def _routes():
    """(path, handler_name) for every route in the module."""
    found = []
    for node in _TREE.body:
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr in ("get", "post") and dec.args:
                    found.append((dec.func.attr, ast.literal_eval(dec.args[0]), node))
    return found


class TestDisablingRevokes:
    """`apps.status` alone only refuses NEW sessions. Every key the app already
    minted keeps working until it lapses — a month, at the default TTL — after
    the operator believes they cut it off."""

    def test_disable_revokes_every_key_the_app_holds(self):
        fn = _function("admin_set_app_status")
        assert "revoke_app_keys" in _calls(fn)

    def test_revoke_is_conditional_on_disabling(self):
        """Enabling must not revoke: an operator re-enabling an app should not
        silently destroy credentials in the same click."""
        fn = _function("admin_set_app_status")
        guarded = [
            n for n in ast.walk(fn)
            if isinstance(n, ast.If) and "revoke_app_keys" in _calls(n)
        ]
        assert guarded, "revoke_app_keys must sit inside a status test"

    def test_only_known_statuses_are_accepted(self):
        assert '("active", "disabled")' in _SRC


class TestCredentialRotation:
    def test_previous_credential_is_revoked_before_the_new_one_is_minted(self):
        """Revoke-then-issue, in one transaction. Two live provisioning
        credentials means a leaked one still works after the operator believes
        they replaced it."""
        fn = _function("admin_issue_app_credential")
        calls = _calls(fn)
        assert "revoke_app_provision_keys" in calls
        assert "generate_api_key" in calls
        assert calls.index("revoke_app_provision_keys") < calls.index("generate_api_key")

    def test_credential_carries_only_the_provisioning_scope(self):
        assert "format_scopes(APP_CREDENTIAL_SCOPES)" in _SRC
        assert "SCOPE_ADMIN" not in _SRC
        assert "SCOPE_INFERENCE" not in _SRC

    def test_credential_is_hidden_and_expiring(self):
        fn = _function("admin_issue_app_credential")
        block = ast.get_source_segment(_SRC, fn)
        assert "key_row.hidden = True" in block
        assert "expires_at=expires_at" in block

    def test_credential_is_namespaced_to_its_app(self):
        block = ast.get_source_segment(_SRC, _function("admin_issue_app_credential"))
        assert "key_row.app_id = app.id" in block

    def test_unconfigured_apps_cannot_be_given_a_credential(self):
        """A credential for an app with no Entra registration could never
        verify a token, so issuing one only creates a live secret nothing
        checks."""
        block = ast.get_source_segment(_SRC, _function("admin_issue_app_credential"))
        assert "app.entra_client_id" in block and "app.entra_tenant_id" in block

    def test_plaintext_is_rendered_never_redirected(self):
        """A redirect would put the credential in the URL, and from there into
        browser history and the access log."""
        block = ast.get_source_segment(_SRC, _function("admin_issue_app_credential"))
        assert "TemplateResponse" in block
        assert "api_key=" not in block or "RedirectResponse(f\"/admin/apps?success" not in block


class TestDeregistration:
    def test_requires_the_slug_typed_exactly(self):
        block = ast.get_source_segment(_SRC, _function("admin_delete_app"))
        assert "confirm_slug" in block
        assert "!= app.slug" in block

    def test_keys_are_revoked_and_detached_before_the_row_is_removed(self):
        """delete_app would otherwise trip the api_keys.app_id foreign key, and
        deleting the keys instead would orphan the request history that
        explains what the app did."""
        calls = _calls(_function("admin_delete_app"))
        assert calls.index("revoke_app_keys") < calls.index("delete_app")
        assert calls.index("detach_app_keys") < calls.index("delete_app")


class TestEntraIdsAreValidated:
    """Both ids are interpolated into the JWKS URL and into the pinned issuer
    string, so a value that is not a GUID either is a typo or is steering key
    retrieval somewhere else."""

    def test_guid_pattern_accepts_a_real_guid(self):
        guid = _compiled_pattern("_GUID_RE")
        assert guid.match("6c9c9c3f-1e3f-4a3a-9a1b-0d2f5a7b8c9d")
        assert guid.match("6C9C9C3F-1E3F-4A3A-9A1B-0D2F5A7B8C9D")

    @pytest.mark.parametrize(
        "bad",
        [
            "../../evil",
            "common",
            "contoso.onmicrosoft.com",
            "6c9c9c3f-1e3f-4a3a-9a1b-0d2f5a7b8c9d/../x",
            "6c9c9c3f-1e3f-4a3a-9a1b-0d2f5a7b8c9d\n",
            "",
            "6c9c9c3f1e3f4a3a9a1b0d2f5a7b8c9d",
        ],
    )
    def test_guid_pattern_rejects_everything_else(self, bad):
        guid = _compiled_pattern("_GUID_RE")
        assert not guid.match(bad), f"{bad!r} must not be accepted as an Entra id"

    def test_both_create_and_update_validate_both_ids(self):
        for name in ("admin_create_app", "admin_update_app"):
            block = ast.get_source_segment(_SRC, _function(name))
            assert "_GUID_RE.match(entra_client_id)" in block, name
            assert "_GUID_RE.match(entra_tenant_id)" in block, name


class TestSlugValidation:
    def test_slug_pattern_accepts_ordinary_slugs(self):
        slug = _compiled_pattern("_SLUG_RE")
        for good in ("vandalchat", "vandal-chat", "app2", "a1"):
            assert slug.match(good), good

    @pytest.mark.parametrize(
        "bad",
        ["../etc", "Vandal", "-lead", "trail-", "has space", "a", "", "sl/ash", "dot.dot"],
    )
    def test_slug_pattern_rejects_anything_awkward_in_a_url(self, bad):
        slug = _compiled_pattern("_SLUG_RE")
        assert not slug.match(bad), f"{bad!r} must not be accepted as a slug"


class TestAuthorization:
    def test_every_mutating_route_requires_full_admin(self):
        """Auditors have read-only admin. A route that checked _require_admin_read
        would let them mint and revoke application credentials."""
        offenders = []
        for method, path, fn in _routes():
            if method != "post":
                continue
            calls = _calls(fn)
            if "_require_admin" not in calls:
                offenders.append(f"{path} -> {fn.name} does not call _require_admin")
            if "_require_admin_read" in calls:
                offenders.append(f"{path} -> {fn.name} settles for read-only admin")
        assert not offenders, "\n".join(offenders)

    def test_the_listing_route_allows_auditors(self):
        gets = [fn for method, _, fn in _routes() if method == "get"]
        assert gets, "expected a listing route"
        for fn in gets:
            assert "_require_admin_read" in _calls(fn)

    def test_admin_helpers_require_an_active_account(self):
        """A deactivated admin's surviving session cookie must not still
        administer applications."""
        for name in ("_require_admin", "_require_admin_read"):
            block = ast.get_source_segment(_SRC, _function(name))
            assert "user.is_active" in block, name


class TestErrorsAreStatic:
    def test_error_redirects_are_url_encoded(self):
        block = ast.get_source_segment(_SRC, _function("_err"))
        assert "quote_plus(message)" in block

    def test_no_exception_text_reaches_the_redirect(self):
        """A SQLAlchemy error stringifies its statement and bound parameters,
        which is how secrets reach browser history and access logs."""
        assert "_err(str(" not in _SRC
        assert "_err(f\"{e" not in _SRC
        for node in ast.walk(_TREE):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_err":
                for arg in node.args:
                    assert isinstance(arg, (ast.Constant, ast.JoinedStr)), (
                        "_err takes a literal message"
                    )

    def test_audit_log_records_every_mutation(self):
        for method, path, fn in _routes():
            if method != "post":
                continue
            assert "log_admin_action" in _calls(fn), f"{path} is unaudited"


class TestHiddenKeysStayOutOfTheOwnersList:
    """The dashboard chat, image, and video pages take the FIRST key
    get_user_api_keys returns and make requests with it. An app-minted key in
    that list would attribute a user's web-UI traffic to the app."""

    def _crud(self):
        return (_APP / "db" / "crud.py").read_text()

    def test_listing_excludes_hidden_keys_by_default(self):
        src = self._crud()
        i = src.index("async def get_user_api_keys")
        block = src[i:i + 1200]
        assert "include_hidden: bool = False" in block
        assert "ApiKey.hidden.is_(False)" in block

    def test_the_key_allowance_ignores_app_minted_keys(self):
        src = self._crud()
        i = src.index("async def count_user_active_api_keys")
        block = src[i:i + 700]
        assert "ApiKey.hidden.is_(False)" in block


class TestPanelIsReachable:
    def test_sidebar_links_to_the_panel(self):
        sidebar = (_TEMPLATES / "admin" / "_sidebar.html").read_text()
        assert 'href="/admin/apps"' in sidebar
        assert "active == 'apps'" in sidebar

    def test_page_marks_itself_active(self):
        page = (_TEMPLATES / "admin" / "apps.html").read_text()
        assert 'set active = "apps"' in page

    def test_router_is_registered(self):
        main = (_APP / "main.py").read_text()
        assert "from backend.app.dashboard.apps_routes import apps_admin_router" in main
        assert "app.include_router(apps_admin_router)" in main

    def test_mutating_controls_are_hidden_from_auditors(self):
        """Read-only admins reach this page; showing them buttons that will
        bounce is how an auditor concludes the panel is broken."""
        page = (_TEMPLATES / "admin" / "apps.html").read_text()
        assert "is_read_only" in page

    def test_credential_page_does_not_leak_into_a_url(self):
        page = (_TEMPLATES / "admin" / "app_credential.html").read_text()
        assert "{{ api_key }}" in page
        assert "?key=" not in page and "&key=" not in page
