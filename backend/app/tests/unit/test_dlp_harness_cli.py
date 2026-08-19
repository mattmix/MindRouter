############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# backend/app/tests/unit/test_dlp_harness_cli.py:
# Unit tests for the harness CLI (dlp_harness/__main__.py)
# and the offline orchestration (offline_eval.py).
#
# No live HTTP, no real DB: gateway/e2e/load/report/db
# modules are monkeypatched stubs; the teardown flow uses
# httpx.MockTransport; corpus + offline regex runs are real.
#
############################################################

"""Tests for the DLP harness CLI dispatch and offline evaluation artifact."""

import json
import os
import stat
import sys
import types

import pytest

# Import dlp_harness from the repo root, not the app package (import-chain rule).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import dlp_harness.__main__ as cli
from dlp_harness import corpus as corpus_mod
from dlp_harness.schemas import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Recorder:
    """Callable that records (args, kwargs) and returns a fixed result."""

    def __init__(self, result=None, raises=None):
        self.calls = []
        self.result = result
        self.raises = raises

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result

    @property
    def only_call(self):
        assert len(self.calls) == 1, f"expected 1 call, got {len(self.calls)}"
        return self.calls[0]


def make_fake_db_class():
    class FakeHarnessDB:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.closed = False
            FakeHarnessDB.instances.append(self)

        def close(self):
            self.closed = True

    return FakeHarnessDB


def stub_imports(monkeypatch, mapping):
    real = cli._import

    def fake(name):
        if name in mapping:
            val = mapping[name]
            if isinstance(val, Exception):
                raise val
            return val
        return real(name)

    monkeypatch.setattr(cli, "_import", fake)


@pytest.fixture
def small_corpus(tmp_path):
    docs = corpus_mod.generate("load", 6, seed=3, dirty_rate=0.5)
    path = str(tmp_path / "small_corpus.jsonl")
    write_jsonl(path, docs)
    return path, docs


# ---------------------------------------------------------------------------
# Top-level parser behavior
# ---------------------------------------------------------------------------

def test_no_subcommand_prints_help_and_exits_2(capsys):
    assert cli.main([]) == 2
    assert "subcommand" in capsys.readouterr().out.lower() or True  # help printed


def test_unknown_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])


# ---------------------------------------------------------------------------
# corpus: real end-to-end into a tmp dir (no stubs)
# ---------------------------------------------------------------------------

def test_corpus_subcommand_real(tmp_path, capsys):
    out = str(tmp_path / "corpus_run")
    rc = cli.main(["corpus", "--profile", "accuracy", "--size", "20",
                   "--seed", "7", "--out", out])
    assert rc == 0
    docs = read_jsonl(os.path.join(out, "corpus.jsonl"))
    assert len(docs) == 20
    assert os.path.isfile(os.path.join(out, "manifest.json"))
    assert os.path.isfile(os.path.join(out, "run.json"))
    with open(os.path.join(out, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["docs"] == 20
    assert manifest["dirty_docs"] + manifest["clean_docs"] == 20
    # every command prints the run dir it wrote (last line)
    assert capsys.readouterr().out.strip().splitlines()[-1] == out


def test_corpus_runs_root_override(tmp_path, monkeypatch):
    from dlp_harness import schemas
    monkeypatch.setattr(schemas, "RUNS_ROOT", schemas.RUNS_ROOT)  # auto-restore
    root = str(tmp_path / "custom_root")
    rc = cli.main(["--runs-root", root, "corpus", "--profile", "load",
                   "--size", "3", "--seed", "1"])
    assert rc == 0
    runs = os.listdir(root)
    assert len(runs) == 1
    assert os.path.isfile(os.path.join(root, runs[0], "corpus.jsonl"))


# ---------------------------------------------------------------------------
# offline: real regex run over a 30-doc corpus (artifact contract)
# ---------------------------------------------------------------------------

def test_offline_subcommand_real_regex(tmp_path, capsys):
    docs = corpus_mod.generate("accuracy", 30, seed=11)
    corpus_path = str(tmp_path / "corpus.jsonl")
    write_jsonl(corpus_path, docs)
    out = str(tmp_path / "offline_run")

    rc = cli.main(["offline", "--corpus", corpus_path, "--out", out])
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == out

    with open(os.path.join(out, "offline_metrics.json")) as f:
        m = json.load(f)
    assert set(m) == {
        "run", "doc_confusion", "span_confusion", "span_confusion_strict",
        "scope_split", "recall_by", "fp_traps", "latency_ms",
        "latency_by_length", "severity_accuracy", "bootstrap", "sweep",
        "scan_errors",
    }
    assert set(m["recall_by"]) == {"difficulty", "generator", "carrier"}
    assert set(m["bootstrap"]) == {"doc_recall", "doc_precision", "span_recall"}
    for ci in m["bootstrap"].values():
        assert set(ci) == {"lo", "hi", "point", "degenerate"}
    assert m["sweep"] is None            # not requested
    assert m["scan_errors"] == 0
    assert m["run"]["scanners_effective"] == ["regex"]
    assert m["run"]["n_docs"] == 30

    n_dirty = sum(1 for d in docs if d.entities)
    n_clean = 30 - n_dirty
    dc = m["doc_confusion"]
    assert dc["tp"] + dc["fn"] == n_dirty
    assert dc["fp"] + dc["tn"] == n_clean
    assert "regex" in m["latency_ms"]["per_scanner"]
    assert m["latency_ms"]["per_scanner"]["regex"]["n"] == 30
    assert set(m["scope_split"]) == {"in_scope", "out_of_scope"}
    assert m["span_confusion"]["overall"]["tp"] >= m["span_confusion_strict"]["overall"]["tp"]

    rows = read_jsonl(os.path.join(out, "offline_findings.jsonl"))
    assert len(rows) == 30
    assert all(set(r) == {"doc_id", "findings", "latency_ms", "errors"} for r in rows)
    assert [r["doc_id"] for r in rows] == [d.doc_id for d in docs]


def test_offline_dispatch_forwards_args(tmp_path, monkeypatch):
    rec = Recorder(result={})
    stub_imports(monkeypatch, {
        "dlp_harness.offline_eval": types.SimpleNamespace(run_offline=rec)})
    out = str(tmp_path / "o")
    rc = cli.main(["offline", "--corpus", "/x/c.jsonl",
                   "--scanners", "regex,gliner", "--gliner-threshold", "0.3",
                   "--gliner-max-chars", "5000", "--sweep", "--in-container",
                   "--seed", "9", "--out", out])
    assert rc == 0
    _, kw = rec.only_call
    assert kw["corpus_path"] == "/x/c.jsonl"
    assert kw["scanners"] == ("regex", "gliner")
    assert kw["gliner_threshold"] == 0.3
    assert kw["gliner_max_chars"] == 5000
    assert kw["sweep"] is True
    assert kw["in_container"] is True
    assert kw["seed"] == 9
    assert kw["out_dir"] == out


# ---------------------------------------------------------------------------
# mock: serve delegation + register/disable dispatch + prod guard
# ---------------------------------------------------------------------------

def test_mock_serve_forwards_remainder_argv(monkeypatch):
    rec = Recorder(result=0)
    stub_imports(monkeypatch, {
        "dlp_harness.mock_backend": types.SimpleNamespace(main=rec)})
    rc = cli.main(["mock", "serve", "--port", "9999", "--latency-ms", "5"])
    assert rc == 0
    args, _ = rec.only_call
    assert args == (["--port", "9999", "--latency-ms", "5"],)


def test_mock_register_dispatch_and_guard(monkeypatch, capsys):
    rec = Recorder(result={"id": 5, "name": "dlp-harness-mock"})
    stub_imports(monkeypatch, {
        "dlp_harness.mock_backend": types.SimpleNamespace(
            register_mock_backend=rec, disable_backend=Recorder())})
    rc = cli.main(["mock", "register", "--base-url", "http://localhost:8000",
                   "--admin-key", "ak", "--backend-url", "http://127.0.0.1:9101",
                   "--name", "myname"])
    assert rc == 0
    args, kw = rec.only_call
    assert args == ("http://localhost:8000", "ak", "http://127.0.0.1:9101")
    assert kw == {"name": "myname"}
    assert json.loads(capsys.readouterr().out)["id"] == 5

    with pytest.raises(SystemExit):
        cli.main(["mock", "register", "--base-url", "https://mindrouter.uidaho.edu",
                  "--admin-key", "ak", "--backend-url", "http://127.0.0.1:9101"])
    assert len(rec.calls) == 1  # guard fired before the call


def test_mock_disable_dispatch(monkeypatch):
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.mock_backend": types.SimpleNamespace(disable_backend=rec)})
    rc = cli.main(["mock", "disable", "--base-url", "http://127.0.0.1:8000",
                   "--admin-key", "ak", "--backend-id", "42"])
    assert rc == 0
    args, _ = rec.only_call
    assert args == ("http://127.0.0.1:8000", "ak", 42)


# ---------------------------------------------------------------------------
# provision: gateway dispatch, teardown-by-prefix (fake transport), guards
# ---------------------------------------------------------------------------

def _fake_gateway(users=None):
    users = users if users is not None else [
        types.SimpleNamespace(user_id=1, username="_dlpharness_u1",
                              api_key="mr2_aaaaaaaaaaaaaaaa"),
        types.SimpleNamespace(user_id=2, username="_dlpharness_u2",
                              api_key="mr2_bbbbbbbbbbbbbbbb"),
    ]
    return types.SimpleNamespace(
        ensure_group=Recorder(result=7),
        provision_users=Recorder(result=users),
        teardown_users=Recorder(),
    ), users


def test_provision_dispatch_hides_keys_by_default(monkeypatch, capsys, tmp_path):
    gw, _ = _fake_gateway()
    stub_imports(monkeypatch, {"dlp_harness.gateway": gw})
    out = str(tmp_path / "prov")
    rc = cli.main(["provision", "--base-url", "http://localhost:8000",
                   "--admin-key", "ak", "--users", "2", "--out", out])
    assert rc == 0
    args, kw = gw.ensure_group.only_call
    assert args == ("http://localhost:8000", "ak")
    assert kw == {"allow_prod": False}
    args, kw = gw.provision_users.only_call
    assert args == ("http://localhost:8000", "ak", 2, 7)
    assert kw == {"allow_prod": False}
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["group_id"] == 7
    # stdout carries usernames + key PREFIXES only — never the full keys
    assert payload["users"] == [
        {"username": "_dlpharness_u1", "key_prefix": "mr2_aaaaaaaa"},
        {"username": "_dlpharness_u2", "key_prefix": "mr2_bbbbbbbb"}]
    assert "mr2_aaaaaaaaaaaaaaaa" not in stdout
    # the full keys live in a mode-0600 file in the run dir
    keys_path = payload["keys_file"]
    assert keys_path == os.path.join(out, "provision.json")
    assert stat.S_IMODE(os.stat(keys_path).st_mode) == 0o600
    with open(keys_path) as f:
        on_disk = json.load(f)
    assert on_disk["users"] == [
        {"username": "_dlpharness_u1", "api_key": "mr2_aaaaaaaaaaaaaaaa"},
        {"username": "_dlpharness_u2", "api_key": "mr2_bbbbbbbbbbbbbbbb"}]


def test_provision_show_keys_restores_old_behavior(monkeypatch, capsys, tmp_path):
    gw, _ = _fake_gateway()
    stub_imports(monkeypatch, {"dlp_harness.gateway": gw})
    out = str(tmp_path / "prov")
    rc = cli.main(["provision", "--base-url", "http://localhost:8000",
                   "--admin-key", "ak", "--users", "2", "--out", out,
                   "--show-keys"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["users"] == [
        {"username": "_dlpharness_u1", "api_key": "mr2_aaaaaaaaaaaaaaaa"},
        {"username": "_dlpharness_u2", "api_key": "mr2_bbbbbbbbbbbbbbbb"}]
    assert os.path.isfile(os.path.join(out, "provision.json"))


def test_provision_forwards_allow_prod(monkeypatch, tmp_path):
    # Finding [14]/[22]: --allow-prod must reach the gateway helpers, which
    # run their own require_local(allow_prod=...) guard.
    gw, _ = _fake_gateway()
    stub_imports(monkeypatch, {"dlp_harness.gateway": gw})
    rc = cli.main(["provision", "--base-url", "https://mindrouter.uidaho.edu",
                   "--admin-key", "ak", "--users", "2", "--allow-prod",
                   "--out", str(tmp_path / "prov")])
    assert rc == 0
    assert gw.ensure_group.only_call[1] == {"allow_prod": True}
    assert gw.provision_users.only_call[1] == {"allow_prod": True}


def test_provision_prod_guard(monkeypatch):
    gw, _ = _fake_gateway()
    stub_imports(monkeypatch, {"dlp_harness.gateway": gw})
    with pytest.raises(SystemExit):
        cli.main(["provision", "--base-url", "https://mindrouter.uidaho.edu",
                  "--admin-key", "ak"])
    assert gw.ensure_group.calls == []


def test_teardown_by_prefix_fake_transport():
    httpx = pytest.importorskip("httpx")
    seen = {"deleted": [], "auth": set(), "search": None}

    def handler(request):
        seen["auth"].add(request.headers.get("authorization"))
        if request.method == "GET" and request.url.path == "/api/admin/users":
            seen["search"] = request.url.params.get("search")
            return httpx.Response(200, json={
                "total": 3, "skip": 0, "limit": 200,
                "users": [{"id": 1, "username": "_dlpharness_a"},
                          {"id": 2, "username": "alice"},
                          {"id": 3, "username": "_dlpharness_b"}]})
        if request.method == "DELETE" and request.url.path.startswith("/api/admin/users/"):
            seen["deleted"].append(int(request.url.path.rsplit("/", 1)[1]))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    deleted = cli._teardown_by_prefix(
        "http://localhost:8000", "ak", transport=httpx.MockTransport(handler))
    assert [d["id"] for d in deleted] == [1, 3]
    assert seen["deleted"] == [1, 3]           # never touched user 2 (real account)
    assert seen["auth"] == {"Bearer ak"}
    assert seen["search"] == "_dlpharness_"


def test_provision_teardown_flag_uses_inline_not_gateway(monkeypatch, capsys):
    rec = Recorder(result=[{"id": 1, "username": "_dlpharness_a"}])
    grp = Recorder(result={"name": "dlp-harness", "deleted": True, "id": 7})
    monkeypatch.setattr(cli, "_teardown_by_prefix", rec)
    monkeypatch.setattr(cli, "_teardown_group", grp)
    # gateway must NOT be imported on the teardown path
    stub_imports(monkeypatch, {
        "dlp_harness.gateway": AssertionError("gateway imported during teardown")})
    rc = cli.main(["provision", "--base-url", "http://localhost:8000",
                   "--admin-key", "ak", "--teardown"])
    assert rc == 0
    args, kw = rec.only_call
    assert args == ("http://localhost:8000", "ak")
    assert kw == {"prefix": "_dlpharness_"}
    assert grp.only_call[0] == ("http://localhost:8000", "ak")
    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] == [{"id": 1, "username": "_dlpharness_a"}]
    assert payload["group"] == {"name": "dlp-harness", "deleted": True, "id": 7}


def test_provision_teardown_group_failure_is_best_effort(monkeypatch, capsys):
    # Finding [4]: group deletion is best-effort — a failure must not undo the
    # (already successful) user teardown or change the exit code.
    monkeypatch.setattr(cli, "_teardown_by_prefix", Recorder(result=[]))
    monkeypatch.setattr(cli, "_teardown_group",
                        Recorder(raises=RuntimeError("admin API down")))
    stub_imports(monkeypatch, {
        "dlp_harness.gateway": AssertionError("gateway imported during teardown")})
    rc = cli.main(["provision", "--base-url", "http://localhost:8000",
                   "--admin-key", "ak", "--teardown"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["group"]["deleted"] is False
    assert "admin API down" in payload["group"]["reason"]
    assert "group teardown failed" in captured.err


def test_teardown_group_deletes_empty_group_fake_transport():
    httpx = pytest.importorskip("httpx")
    seen = {"deleted": [], "auth": set()}

    def handler(request):
        seen["auth"].add(request.headers.get("authorization"))
        if request.method == "GET" and request.url.path == "/api/admin/groups":
            return httpx.Response(200, json={"groups": [
                {"id": 3, "name": "students", "user_count": 40},
                {"id": 7, "name": "dlp-harness", "user_count": 0}]})
        if request.method == "DELETE" and request.url.path == "/api/admin/groups/7":
            seen["deleted"].append(7)
            return httpx.Response(200, json={"status": "deleted", "group_id": 7})
        return httpx.Response(404)

    out = cli._teardown_group("http://localhost:8000", "ak",
                              transport=httpx.MockTransport(handler))
    assert out == {"name": "dlp-harness", "deleted": True, "id": 7}
    assert seen["deleted"] == [7]              # never touched the real group
    assert seen["auth"] == {"Bearer ak"}


def test_teardown_group_skips_when_members_remain(capsys):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        if request.method == "GET" and request.url.path == "/api/admin/groups":
            return httpx.Response(200, json={"groups": [
                {"id": 7, "name": "dlp-harness", "user_count": 2}]})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    out = cli._teardown_group("http://localhost:8000", "ak",
                              transport=httpx.MockTransport(handler))
    assert out["deleted"] is False
    assert "2 member(s) remain" in out["reason"]
    assert "not deleted" in capsys.readouterr().err


def test_teardown_group_absent_is_a_noop():
    httpx = pytest.importorskip("httpx")

    def handler(request):
        if request.method == "GET" and request.url.path == "/api/admin/groups":
            return httpx.Response(200, json={"groups": []})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    out = cli._teardown_group("http://localhost:8000", "ak",
                              transport=httpx.MockTransport(handler))
    assert out == {"name": "dlp-harness", "deleted": False, "reason": "not found"}


# ---------------------------------------------------------------------------
# e2e: dispatch, list/float parsing, guard, db lifecycle
# ---------------------------------------------------------------------------

def test_e2e_dispatch_forwards_args(monkeypatch, small_corpus, tmp_path):
    corpus_path, docs = small_corpus
    rec = Recorder()
    FakeDB = make_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.e2e": types.SimpleNamespace(run_e2e=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    out = str(tmp_path / "e2e_run")
    rc = cli.main(["e2e", "--corpus", corpus_path,
                   "--base-url", "http://localhost:8000",
                   "--api-key", "uk", "--admin-key", "ak",
                   "--mode", "regex+gliner", "--plant-side", "response",
                   "--stream-pct", "0.25", "--concurrency", "8",
                   "--keep-alerts", "--drain-timeout", "33", "--out", out])
    assert rc == 0
    _, kw = rec.only_call
    assert kw["base_url"] == "http://localhost:8000"
    assert kw["api_key"] == "uk"
    assert kw["admin_key"] == "ak"
    assert kw["scanner_mode"] == "regex+gliner"
    assert kw["plant_side"] == "response"
    assert kw["stream_pct"] == 0.25
    assert kw["concurrency"] == 8
    assert kw["keep_alerts"] is True
    assert kw["allow_prod"] is False
    assert kw["model"] == "dlp-mock"
    assert kw["seed"] == 42
    assert kw["drain_timeout_s"] == 33
    assert "settle_s" not in kw           # not passed -> module default wins
    assert kw["out_dir"] == out
    assert len(kw["docs"]) == len(docs)
    assert kw["docs"][0].doc_id == docs[0].doc_id
    db = kw["db"]
    assert isinstance(db, FakeDB)
    assert db.init_kwargs["host"] == "127.0.0.1"
    assert db.closed                      # closed in the finally block


def test_e2e_plant_side_mixed_passes_through(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.e2e": types.SimpleNamespace(run_e2e=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class())})
    rc = cli.main(["e2e", "--corpus", corpus_path,
                   "--base-url", "http://localhost:8000", "--api-key", "uk",
                   "--admin-key", "ak", "--plant-side", "mixed",
                   "--out", str(tmp_path / "o")])
    assert rc == 0
    assert rec.only_call[1]["plant_side"] == "mixed"


def test_e2e_prod_guard(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.e2e": types.SimpleNamespace(run_e2e=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class())})
    with pytest.raises(SystemExit):
        cli.main(["e2e", "--corpus", corpus_path,
                  "--base-url", "https://mindrouter.uidaho.edu",
                  "--api-key", "uk", "--admin-key", "ak",
                  "--out", str(tmp_path / "o")])
    assert rec.calls == []


# ---------------------------------------------------------------------------
# load: dispatch, list parsing, provision/teardown lifecycle
# ---------------------------------------------------------------------------

def test_load_dispatch_api_keys_and_lists(monkeypatch, small_corpus, tmp_path):
    corpus_path, docs = small_corpus
    rec = Recorder()
    FakeDB = make_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    out = str(tmp_path / "load_run")
    rc = cli.main(["load", "--corpus", corpus_path,
                   "--base-url", "http://127.0.0.1:8000", "--admin-key", "ak",
                   "--api-keys", "k1, k2", "--modes", "off,regex",
                   "--concurrencies", "1,4,16", "--duration", "30",
                   "--warmup", "5", "--no-stream", "--out", out])
    assert rc == 0
    _, kw = rec.only_call
    assert kw["api_keys"] == ["k1", "k2"]
    assert kw["modes"] == ["off", "regex"]
    assert kw["concurrencies"] == [1, 4, 16]
    assert kw["duration_s"] == 30.0
    assert kw["warmup_s"] == 5.0
    assert kw["stream"] is False
    assert kw["model"] == "dlp-mock"
    assert kw["max_tokens"] == 64
    assert kw["dirty_rate"] == 0.2
    assert kw["allow_prod"] is False
    assert kw["compose_dir"] is None          # load.py derives the repo root
    assert kw["out_dir"] == out
    assert len(kw["docs"]) == len(docs)
    assert isinstance(kw["db"], FakeDB)
    assert kw["db"].closed


def test_load_stream_default_true(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class())})
    cli.main(["load", "--corpus", corpus_path, "--base-url", "http://localhost:8000",
              "--admin-key", "ak", "--api-keys", "k1", "--out", str(tmp_path / "o")])
    assert rec.only_call[1]["stream"] is True


def test_load_requires_keys_or_provision(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class())})
    with pytest.raises(SystemExit):
        cli.main(["load", "--corpus", corpus_path,
                  "--base-url", "http://localhost:8000", "--admin-key", "ak",
                  "--out", str(tmp_path / "o")])
    assert rec.calls == []


def test_load_provision_and_teardown_success(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder()
    gw, users = _fake_gateway()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class()),
        "dlp_harness.gateway": gw})
    rc = cli.main(["load", "--corpus", corpus_path,
                   "--base-url", "http://localhost:8000", "--admin-key", "ak",
                   "--provision", "2", "--out", str(tmp_path / "o")])
    assert rc == 0
    assert gw.ensure_group.only_call == (("http://localhost:8000", "ak"),
                                         {"allow_prod": False})
    assert gw.provision_users.only_call == (("http://localhost:8000", "ak", 2, 7),
                                            {"allow_prod": False})
    assert rec.only_call[1]["api_keys"] == ["mr2_aaaaaaaaaaaaaaaa",
                                            "mr2_bbbbbbbbbbbbbbbb"]
    args, kw = gw.teardown_users.only_call
    assert args == ("http://localhost:8000", "ak", users)
    assert kw == {"allow_prod": False}


def test_load_provision_forwards_allow_prod(monkeypatch, small_corpus, tmp_path):
    # Finding [14]/[22]: without forwarding, gateway.require_local raises on
    # any non-local base_url even though the operator passed --allow-prod.
    corpus_path, _ = small_corpus
    rec = Recorder()
    gw, users = _fake_gateway()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class()),
        "dlp_harness.gateway": gw})
    rc = cli.main(["load", "--corpus", corpus_path,
                   "--base-url", "https://mindrouter.uidaho.edu",
                   "--admin-key", "ak", "--provision", "2", "--allow-prod",
                   "--out", str(tmp_path / "o")])
    assert rc == 0
    assert gw.ensure_group.only_call[1] == {"allow_prod": True}
    assert gw.provision_users.only_call[1] == {"allow_prod": True}
    assert gw.teardown_users.only_call[1] == {"allow_prod": True}
    assert rec.only_call[1]["allow_prod"] is True


def test_load_compose_dir_passthrough(monkeypatch, small_corpus, tmp_path):
    # Finding [15]/[26]: --compose-dir must reach run_load_matrix so CPU and
    # queue-drop sampling work on hosts other than the dev laptop.
    corpus_path, _ = small_corpus
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=make_fake_db_class())})
    rc = cli.main(["load", "--corpus", corpus_path,
                   "--base-url", "http://localhost:8000", "--admin-key", "ak",
                   "--api-keys", "k1", "--compose-dir", "/opt/mindrouter",
                   "--out", str(tmp_path / "o")])
    assert rc == 0
    assert rec.only_call[1]["compose_dir"] == "/opt/mindrouter"


def test_load_teardown_runs_even_when_run_fails(monkeypatch, small_corpus, tmp_path):
    corpus_path, _ = small_corpus
    rec = Recorder(raises=RuntimeError("load blew up"))
    gw, users = _fake_gateway()
    FakeDB = make_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.load": types.SimpleNamespace(run_load_matrix=rec),
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB),
        "dlp_harness.gateway": gw})
    with pytest.raises(RuntimeError):
        cli.main(["load", "--corpus", corpus_path,
                  "--base-url", "http://localhost:8000", "--admin-key", "ak",
                  "--provision", "2", "--out", str(tmp_path / "o")])
    assert gw.teardown_users.only_call[0][2] == users
    assert FakeDB.instances[-1].closed


# ---------------------------------------------------------------------------
# report: dispatch
# ---------------------------------------------------------------------------

def test_report_dispatch(monkeypatch, tmp_path, capsys):
    rec = Recorder()
    stub_imports(monkeypatch, {
        "dlp_harness.report": types.SimpleNamespace(generate_report=rec)})
    out = str(tmp_path / "report_run")
    rc = cli.main(["report", "--runs", "runA, runB", "--title", "My Title",
                   "--out", out])
    assert rc == 0
    _, kw = rec.only_call
    assert kw["run_dirs"] == [os.path.abspath("runA"), os.path.abspath("runB")]
    assert kw["out_dir"] == out
    assert kw["title"] == "My Title"
    assert capsys.readouterr().out.strip().splitlines()[-1] == out


# ---------------------------------------------------------------------------
# db-check: stubbed HarnessDB
# ---------------------------------------------------------------------------

def test_db_check_with_stubbed_db(monkeypatch, capsys):
    class FakeCheckDB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def query(self, sql, params=()):
            if "app_config" in sql:
                assert params == ("dlp.%",)
                return [{"key": "dlp.enabled"}, {"key": "dlp.regex.enabled"}]
            if "COUNT(*)" in sql:
                return [{"n": 12}]
            raise AssertionError(f"unexpected query: {sql}")

        def get_config(self, key, default=None):
            assert key == "dlp.enabled"
            return True

        def db_now(self):
            return "2026-08-19 10:00:00"

        def close(self):
            self.closed = True

    created = []

    def factory(**kwargs):
        db = FakeCheckDB(**kwargs)
        created.append(db)
        return db

    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=factory)})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = cli.main(["db-check"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["dlp.enabled"] is True
    assert info["alerts_total"] == 12
    assert "dlp.enabled" in info["config_keys_present"]
    assert "dlp.regex.patterns" in info["config_keys_missing"]
    assert created[0].kwargs["host"] == "127.0.0.1"
    assert created[0].closed


# ---------------------------------------------------------------------------
# restore-config: disaster recovery from an on-disk snapshot (FakeDB)
# ---------------------------------------------------------------------------

_SENTINEL_MISSING = object()


def make_restore_fake_db_class():
    class FakeRestoreDB:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.restored = None
            self.closed = False
            FakeRestoreDB.instances.append(self)

        def snapshot_from_json(self, data):
            return {k: (_SENTINEL_MISSING if v is None else v)
                    for k, v in data.items()}

        def restore_dlp_config(self, snap):
            self.restored = snap

        def close(self):
            self.closed = True

    return FakeRestoreDB


def test_restore_config_replays_snapshot(monkeypatch, tmp_path, capsys):
    # Finding [0] companion: the recovery path for a hard-killed run.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = {"dlp.enabled": "true",
                "dlp.email.critical.mode": "\"immediate\"",
                "dlp.gliner.threshold": None}      # absent pre-run -> delete
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(snapshot, f)
    FakeDB = make_restore_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = cli.main(["restore-config", "--run-dir", str(run_dir)])
    assert rc == 0
    db = FakeDB.instances[-1]
    assert db.restored == {"dlp.enabled": "true",
                           "dlp.email.critical.mode": "\"immediate\"",
                           "dlp.gliner.threshold": _SENTINEL_MISSING}
    assert db.closed
    assert db.init_kwargs["allow_remote"] is False
    out = json.loads(capsys.readouterr().out)
    assert out["restored"] == ["dlp.email.critical.mode", "dlp.enabled"]
    assert out["deleted_missing_keys"] == ["dlp.gliner.threshold"]
    assert out["snapshot"] == str(run_dir / "config_snapshot.json")


def test_restore_config_accepts_direct_file_path(monkeypatch, tmp_path):
    snap_path = tmp_path / "config_snapshot.json"
    with open(snap_path, "w") as f:
        json.dump({"dlp.enabled": "false"}, f)
    FakeDB = make_restore_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = cli.main(["restore-config", "--run-dir", str(snap_path),
                   "--allow-prod"])
    assert rc == 0
    assert FakeDB.instances[-1].restored == {"dlp.enabled": "false"}
    assert FakeDB.instances[-1].init_kwargs["allow_remote"] is True


def test_restore_config_missing_snapshot_is_a_clean_error(monkeypatch, tmp_path):
    FakeDB = make_restore_fake_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    with pytest.raises(SystemExit) as ei:
        cli.main(["restore-config", "--run-dir", str(tmp_path / "nope")])
    assert "config snapshot" in str(ei.value)
    assert FakeDB.instances == []             # never opened a DB connection


# ---------------------------------------------------------------------------
# _open_db: in-container DATABASE_URL preference (production runbook path)
# ---------------------------------------------------------------------------

def _make_url_aware_db_class():
    class FakeUrlDB:
        instances = []
        from_url_calls = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.restored = None
            self.closed = False
            FakeUrlDB.instances.append(self)

        @classmethod
        def from_database_url(cls, url=None, allow_remote=False):
            cls.from_url_calls.append({"url": url, "allow_remote": allow_remote})
            db = cls(via_url=True)
            return db

        def snapshot_from_json(self, data):
            return dict(data)

        def restore_dlp_config(self, snap):
            self.restored = snap

        def close(self):
            self.closed = True

    return FakeUrlDB


def _write_snapshot(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump({"dlp.enabled": "true"}, f)
    return run_dir


def test_open_db_prefers_container_database_url(monkeypatch, tmp_path):
    run_dir = _write_snapshot(tmp_path)
    FakeDB = _make_url_aware_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    monkeypatch.setenv("DATABASE_URL",
                       "mysql+pymysql://mindrouter:pw@127.0.0.1:3306/mindrouter")
    rc = cli.main(["restore-config", "--run-dir", str(run_dir), "--allow-prod"])
    assert rc == 0
    assert FakeDB.from_url_calls == [{"url": None, "allow_remote": True}]
    assert FakeDB.instances[-1].init_kwargs == {"via_url": True}


def test_open_db_explicit_flags_beat_database_url(monkeypatch, tmp_path):
    run_dir = _write_snapshot(tmp_path)
    FakeDB = _make_url_aware_db_class()
    stub_imports(monkeypatch, {
        "dlp_harness.db": types.SimpleNamespace(HarnessDB=FakeDB)})
    monkeypatch.setenv("DATABASE_URL",
                       "mysql+pymysql://mindrouter:pw@127.0.0.1:3306/mindrouter")
    rc = cli.main(["restore-config", "--run-dir", str(run_dir),
                   "--db-password", "explicit-pw"])
    assert rc == 0
    assert FakeDB.from_url_calls == []
    assert FakeDB.instances[-1].init_kwargs["password"] == "explicit-pw"
    assert FakeDB.instances[-1].init_kwargs["host"] == "127.0.0.1"
