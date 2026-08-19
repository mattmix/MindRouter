############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# tests/unit/test_dlp_harness_e2e.py: Unit tests for the
# DLP harness end-to-end modules (dlp_harness/gateway.py +
# dlp_harness/e2e.py). No live HTTP, no real DB: the gateway
# is a fake ASGI app behind httpx.ASGITransport and the DB
# is a dict-backed fake with the HarnessDB method surface.
#
############################################################

import asyncio
import base64
import json
import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import httpx  # noqa: E402

from dlp_harness import e2e as e2e_mod  # noqa: E402
from dlp_harness import gateway as gw  # noqa: E402
from dlp_harness.constants import (  # noqa: E402
    GLINER_DEFAULT_SCOPE,
    REGEX_BUILTIN_SCOPE,
    SAFE_RUN_OVERRIDES,
    SCANNER_ERROR_CATEGORY,
    SCANNER_MODES,
    SEVERITY_RULES_OVERRIDE,
)
from dlp_harness.mock_backend import REPLY_B64_MARKER  # noqa: E402
from dlp_harness.schemas import GroundTruthEntity, LabeledDocument  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731


def _doc(doc_id, text, kinds=()):
    """kinds: iterable of (category, generator) ground-truth pairs."""
    entities = [GroundTruthEntity(category=c, text="x", start=0, end=1, generator=g)
                for c, g in kinds]
    return LabeledDocument(doc_id=doc_id, text=text, entities=entities)


# ---------------------------------------------------------------------------
# Fake gateway environment (shared by the ASGI app and the FakeDB)
# ---------------------------------------------------------------------------

class FakeEnv:
    """State shared by the fake gateway app and the FakeDB.

    The fake 'scanner' runs synchronously at request time, so alerts already
    exist by the time the harness drains — the drain loop settles on the
    first stable poll.
    """

    def __init__(self):
        self.next_rid = 0
        self.next_alert_id = 100
        self.rid_by_uuid = {}
        self.messages = {}      # rid -> stored prompt text (audit capture)
        self.responses = {}     # rid -> stored response text
        self.alerts = []
        self.last_content = None
        self.audit_prompts_on = True
        self.audit_responses_on = True
        self.stream_own_ids = False      # real-backend behavior: chatcmpl-* chunk ids
        self.stream_error_event = False  # emit an SSE error event mid-stream

    def register(self, prompt_text, response_text, scanned_text):
        self.next_rid += 1
        rid = self.next_rid
        uuid = f"uuid-{rid:04d}"
        self.rid_by_uuid[uuid] = rid
        if self.audit_prompts_on:
            self.messages[rid] = prompt_text
        if self.audit_responses_on:
            self.responses[rid] = response_text
        self._scan(rid, scanned_text)
        return rid, uuid

    def _alert(self, rid, scanner, cats, severity, confidence):
        self.next_alert_id += 1
        self.alerts.append({
            "id": self.next_alert_id, "request_id": rid, "user_id": 1,
            "severity": severity, "scanner": scanner, "categories": list(cats),
            "entities": [{"masked": "***"}], "confidence": confidence,
            "scan_latency_ms": 5.0, "scanned_at": None, "detail": None,
        })

    def _scan(self, rid, text):
        if re.search(r"\d{3}-\d{2}-\d{4}", text):
            self._alert(rid, "regex", ["Social Security Number"], "major", 0.99)
        if "FPTRIG" in text:
            self._alert(rid, "regex", ["Email"], "minor", 0.8)
        if "MULTITRIG" in text:
            self._alert(rid, "gliner", ["person"], "moderate", 0.7)
        if "SCANERRTRIG" in text:
            self._alert(rid, "gliner", [SCANNER_ERROR_CATEGORY], "minor", 0.0)


def make_gateway_app(env: FakeEnv):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/admin/backends")
    async def backends():
        return {"backends": []}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        content = body["messages"][-1]["content"]
        env.last_content = content
        if REPLY_B64_MARKER in content:
            payload = content.split(REPLY_B64_MARKER, 1)[1].strip()
            reply = base64.b64decode(payload).decode("utf-8")
            scanned = content + "\n" + reply
        else:
            reply = "A perfectly neutral reply."
            scanned = content
        rid, uuid = env.register(content, reply, scanned)

        if body.get("stream"):
            # Mirrors the real gateway: request_uuid stamped into every chunk
            # — unless stream_own_ids simulates a REAL backend that stamps
            # its own chatcmpl-* ids (which resolve to no requests row).
            chunk_id = f"chatcmpl-{rid}" if env.stream_own_ids else uuid
            base = {"id": chunk_id, "object": "chat.completion.chunk",
                    "model": "dlp-mock"}

            def sse(obj):
                return f"data: {json.dumps(obj)}\n\n"

            async def gen():
                yield sse(dict(base, choices=[{"index": 0,
                                               "delta": {"role": "assistant"},
                                               "finish_reason": None}]))
                if env.stream_error_event:
                    yield sse({"error": {"message": "backend exploded",
                                         "type": "server_error"}})
                    yield "data: [DONE]\n\n"
                    return
                for part in (reply[: len(reply) // 2], reply[len(reply) // 2:]):
                    yield sse(dict(base, choices=[{"index": 0,
                                                   "delta": {"content": part},
                                                   "finish_reason": None}]))
                yield sse(dict(base, choices=[{"index": 0, "delta": {},
                                               "finish_reason": "stop"}]))
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return {"id": uuid, "object": "chat.completion", "model": "dlp-mock",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    return app


class FakeDB:
    """Dict-backed stand-in implementing the HarnessDB surface e2e.py uses."""

    def __init__(self, env: FakeEnv):
        self.env = env
        self.snapshots_taken = 0
        self.applied = []
        self.config_sets = []
        self.restore_calls = []
        self.purge_calls = []
        self.fail_fetch_alerts = False
        self.ops = []           # coarse op sequence for cleanup-ordering asserts

    def snapshot_dlp_config(self):
        self.snapshots_taken += 1
        return {"dlp.enabled": "true", "dlp.regex.enabled": "false"}

    def snapshot_to_json(self, snap):
        return dict(snap)

    def apply_overrides(self, overrides):
        self.applied.append(dict(overrides))

    def set_config(self, key, value):
        self.config_sets.append((key, value))

    def restore_dlp_config(self, snap):
        self.ops.append("restore")
        self.restore_calls.append(dict(snap))

    def fetch_requests_by_uuids(self, uuids):
        return [{"id": self.env.rid_by_uuid[u], "request_uuid": u, "status": "completed"}
                for u in uuids if u in self.env.rid_by_uuid]

    def query(self, sql, params=()):
        if "FROM requests" in sql and "messages" in sql:
            return [{"messages": self.env.messages.get(params[0])}]
        if "FROM responses" in sql:
            return [{"content": self.env.responses.get(params[0])}]
        raise AssertionError(f"unexpected SQL in FakeDB.query: {sql}")

    def fetch_alerts_by_request_ids(self, request_ids):
        if self.fail_fetch_alerts:
            self.ops.append("fetch_fail")
            raise RuntimeError("boom: fetch_alerts failed")
        ids = set(request_ids)
        return [dict(a) for a in self.env.alerts if a["request_id"] in ids]

    def fetch_scan_lags_ms(self, request_ids):
        ids = set(request_ids)
        return [{"request_id": a["request_id"], "alert_id": a["id"],
                 "scan_latency_ms": a["scan_latency_ms"], "lag_ms": 123.0}
                for a in self.env.alerts if a["request_id"] in ids]

    def count_alerts_for_request_ids(self, request_ids, exclude_scanner_errors=True):
        self.ops.append("count")
        ids = set(request_ids)
        return sum(1 for a in self.env.alerts
                   if a["request_id"] in ids
                   and not (exclude_scanner_errors
                            and SCANNER_ERROR_CATEGORY in a["categories"]))

    def purge_alerts_for_request_ids(self, request_ids):
        self.ops.append("purge")
        ids = set(request_ids)
        self.purge_calls.append(sorted(ids))
        before = len(self.env.alerts)
        self.env.alerts = [a for a in self.env.alerts if a["request_id"] not in ids]
        return before - len(self.env.alerts)


# ---------------------------------------------------------------------------
# send_chat / plant_side
# ---------------------------------------------------------------------------

def _client_for(env):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=make_gateway_app(env)))


def test_send_chat_nonstream_extracts_request_uuid():
    env = FakeEnv()

    async def go():
        async with _client_for(env) as client:
            return await gw.send_chat(client, "http://localhost", "k", "dlp-mock",
                                      "hello there", stream=False, plant_side="prompt")
    res = asyncio.run(go())
    assert res.ok and res.status_code == 200 and res.error is None
    assert res.request_uuid == "uuid-0001"
    assert res.ttft_ms is None and res.e2e_ms > 0


def test_send_chat_stream_extracts_uuid_from_first_chunk():
    env = FakeEnv()

    async def go():
        async with _client_for(env) as client:
            return await gw.send_chat(client, "http://localhost", "k", "dlp-mock",
                                      "hello stream", stream=True, plant_side="prompt")
    res = asyncio.run(go())
    assert res.ok and res.status_code == 200
    assert res.request_uuid == "uuid-0001"
    assert res.ttfb_ms is not None and res.ttft_ms is not None
    assert res.e2e_ms >= res.ttft_ms >= res.ttfb_ms


def test_send_chat_stream_error_event_marks_failure():
    # The gateway reports mid-stream backend failure as an SSE error event on
    # an HTTP 200 stream; such requests are FAILED and never DLP-scanned, so
    # they must come back ok=False or they would score as fake coverage misses.
    env = FakeEnv()
    env.stream_error_event = True

    async def go():
        async with _client_for(env) as client:
            return await gw.send_chat(client, "http://localhost", "k", "dlp-mock",
                                      "hello", stream=True, plant_side="prompt")
    res = asyncio.run(go())
    assert res.ok is False
    assert res.status_code == 200
    assert "backend exploded" in res.error
    assert res.request_uuid == "uuid-0001"   # id from earlier chunks still captured


def test_send_chat_never_raises_on_transport_error():
    async def go():
        async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")))) as client:
            return await gw.send_chat(client, "http://localhost", "k", "m",
                                      "x", stream=False, plant_side="prompt")
    res = asyncio.run(go())
    assert not res.ok and res.status_code is None and "ConnectError" in res.error


def test_plant_side_response_hides_raw_text_behind_base64():
    secret = "My SSN is 123-45-6789 and my email is bob@example.com"
    msgs = gw.build_messages(secret, "response")
    content = msgs[0]["content"]
    assert REPLY_B64_MARKER in content
    assert "123-45-6789" not in content and "bob@example.com" not in content
    payload = content.split(REPLY_B64_MARKER, 1)[1].strip()
    assert base64.b64decode(payload).decode("utf-8") == secret

    # Through the wire: the fake backend sees only the encoded payload...
    env = FakeEnv()

    async def go():
        async with _client_for(env) as client:
            return await gw.send_chat(client, "http://localhost", "k", "dlp-mock",
                                      secret, stream=False, plant_side="response")
    res = asyncio.run(go())
    assert res.ok
    assert "123-45-6789" not in env.last_content
    # ...and the decoded plaintext exists only on the response side.
    assert env.responses[1] == secret

    with pytest.raises(ValueError):
        gw.build_messages("x", "sideways")


def test_plant_side_echo_plants_text_verbatim_in_prompt():
    secret = "My SSN is 123-45-6789"
    msgs = gw.build_messages(secret, "echo")
    assert msgs[0]["content"] == (
        "Repeat the following text exactly, with no commentary: " + secret)


# ---------------------------------------------------------------------------
# Scope map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [
    ("off", set()),
    ("regex", set(REGEX_BUILTIN_SCOPE)),
    ("gliner", set(GLINER_DEFAULT_SCOPE)),
    ("regex+gliner", set(REGEX_BUILTIN_SCOPE) | set(GLINER_DEFAULT_SCOPE)),
])
def test_active_scope(mode, expected):
    assert e2e_mod.active_scope(mode) == expected


def test_active_scope_rejects_unknown_mode():
    with pytest.raises(ValueError):
        e2e_mod.active_scope("llm-only")


@pytest.mark.parametrize("cat,gen,mode,expected", [
    ("ssn", "ssn.dashed", "regex", True),
    ("ssn", "ssn.spaced", "regex", False),      # regex-invisible variant
    ("ssn", "ssn.bare9", "regex", False),
    ("ssn", "", "regex", False),                # unknown variant: not visible
    ("ssn", "ssn.spaced", "gliner", True),      # gliner scope is category-level
    ("ssn", "ssn.spaced", "regex+gliner", True),
    ("date_of_birth", "date_of_birth.bare", "regex", False),
    ("date_of_birth", "date_of_birth.prefixed", "regex", True),
    ("email", "email.plain", "regex", True),    # None in the map = all variants
    ("person", "person.name", "regex", False),
    # person dropped from GLiNER defaults 2026-08-19 (precision 0.34)
    ("person", "person.name", "gliner", False),
    ("keyword", "keyword.word", "regex", False),
    ("ssn", "ssn.dashed", "off", False),
])
def test_entity_in_scope_variant_matrix(cat, gen, mode, expected):
    assert e2e_mod.entity_in_scope(cat, gen, mode) is expected


# ---------------------------------------------------------------------------
# Full run: scoring pipeline
# ---------------------------------------------------------------------------

def _corpus():
    return [
        _doc("d1-ssn-hit", "Employee record: SSN 444-55-6666 on file.",
             [("ssn", "ssn.dashed")]),
        _doc("d2-ssn-miss", "Employee record: SSN 444 55 6666 on file.",
             [("ssn", "ssn.spaced")]),
        _doc("d3-person-oos", "Please talk to Jordan about the rollout.",
             [("person", "person.name")]),
        _doc("d4-clean-fp", "Totally clean text but FPTRIG lurks here."),
        _doc("d5-clean", "Nothing sensitive in this sentence at all."),
        _doc("d6-clean-scanerr", "Clean text that hits SCANERRTRIG only."),
        _doc("d7-ssn-multi", "SSN 777-88-9999 plus MULTITRIG marker.",
             [("ssn", "ssn.dashed")]),
    ]


def _run(env, db, tmp_path, **kw):
    kwargs = dict(
        docs=_corpus(), base_url="http://localhost", api_key="uk", admin_key="ak",
        db=db, out_dir=str(tmp_path / "run"), scanner_mode="regex",
        plant_side="prompt", stream_pct=0.0, concurrency=4,
        drain_timeout_s=5.0, settle_s=0.05, seed=7, progress=QUIET,
        transport=httpx.ASGITransport(app=make_gateway_app(env)),
        poll_interval_s=0.01,
    )
    kwargs.update(kw)
    return e2e_mod.run_e2e(**kwargs)


def test_run_e2e_scoring_end_to_end(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    metrics = _run(env, db, tmp_path)

    # send
    assert metrics["send"] == {**metrics["send"], "n_sent": 7, "n_ok": 7, "n_failed": 0}
    assert metrics["send"]["client_latency_ms"]["n"] == 7

    # coverage: dirty = d1,d2,d3,d7; alerted = d1,d7
    cov = metrics["coverage"]
    assert cov["dirty_sent"] == 4 and cov["dirty_alerted"] == 2
    assert cov["rate"] == pytest.approx(0.5)
    # in-scope (regex) is VARIANT-aware: d3 (person) and d2 (spaced SSN —
    # regex-invisible) both drop out; the scanner catches everything it
    # could possibly see.
    assert cov["in_scope_dirty_sent"] == 2 and cov["in_scope_dirty_alerted"] == 2
    assert cov["in_scope_rate"] == pytest.approx(1.0)

    # d6's scanner-error alert degrades the doc OUT of the clean denominator
    fp = metrics["clean_fp"]
    assert fp["clean_sent"] == 2 and fp["clean_alerted"] == 1
    assert fp["rate"] == pytest.approx(1 / 2)
    assert cov["degraded_docs"] == 1
    assert metrics["scanner_error_alerts"] == 1
    assert "lower bound" in metrics["notes"]["scanner_error_note"]

    # per-category: ssn expected d1,d2,d7 detected d1,d7; person 0/1
    pcd = metrics["per_category_detection"]
    assert pcd["ssn"] == {"expected": 3, "detected": 2, "recall": pytest.approx(2 / 3)}
    assert pcd["person"] == {"expected": 1, "detected": 0, "recall": 0.0}

    # severity: both alerted dirty docs are SSN -> expected major, predicted major
    sev = metrics["severity"]
    assert sev["matrix"]["major"]["major"] == 2
    assert sev["exact_match_rate"] == pytest.approx(1.0)

    # multi-alert on d7: regex alert kept (lower id), gliner extra counted
    assert metrics["notes"]["multi_alert_extras"] == 1
    assert metrics["scanner_counts"] == {"regex": 3}   # d1, d4, d7-kept

    # drain settled; lag/latency populated from the joined alerts
    assert metrics["drain"]["settled"] is True
    assert metrics["scan_lag_ms"]["n"] == 3 and metrics["scan_lag_ms"]["p50"] == 123.0
    assert metrics["scan_latency_ms"]["n"] == 3

    # cleanup (finally-path) purged every alert incl. canary + scanner-error
    # rows, after a residual drain, and backfilled the returned metrics
    assert metrics["cleanup"] == {"alerts_purged": 6,
                                  "residual_drain_settled": True}
    assert env.alerts == []

    # config lifecycle: snapshot -> mode overrides -> safe overrides ->
    # severity pin -> restore
    assert db.snapshots_taken == 1
    assert db.applied == [SCANNER_MODES["regex"], SAFE_RUN_OVERRIDES]
    assert db.config_sets == [("dlp.severity_rules", SEVERITY_RULES_OVERRIDE)]
    assert len(db.restore_calls) == 1
    # ordering: the purge must land BEFORE the config restore
    assert db.ops.index("purge") < db.ops.index("restore")

    # artifacts on disk, results order-preserving
    run_dir = tmp_path / "run"
    for name in ("e2e_results.jsonl", "e2e_metrics.json", "run.json",
                 "config_snapshot.json"):
        assert (run_dir / name).exists()
    # metrics file re-written with the final cleanup numbers
    disk = json.loads((run_dir / "e2e_metrics.json").read_text())
    assert disk["cleanup"] == {"alerts_purged": 6, "residual_drain_settled": True}
    rows = [json.loads(line) for line in
            (run_dir / "e2e_results.jsonl").read_text().splitlines()]
    assert [r["doc_id"] for r in rows] == [d.doc_id for d in _corpus()]
    d1 = rows[0]
    assert d1["expected_alert"] is True and d1["request_id"] is not None
    assert d1["alert"]["canonical_categories"] == ["ssn"]
    assert d1["alert"]["severity"] == "major" and d1["alert"]["entities_n"] == 1
    assert d1["in_scope"] is True and d1["scan_degraded"] is False
    d2 = rows[1]
    assert d2["in_scope"] is False       # spaced SSN is regex-invisible
    d6 = rows[5]
    assert d6["alert"] is None   # scanner-error alert never joins as a real alert
    assert d6["scan_degraded"] is True


def test_run_e2e_keep_alerts_skips_purge(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    metrics = _run(env, db, tmp_path, keep_alerts=True)
    assert metrics["cleanup"]["alerts_purged"] is None
    assert db.purge_calls == [] and len(env.alerts) == 6


def test_run_e2e_degraded_dirty_doc_excluded_from_coverage(tmp_path):
    docs = [
        _doc("dd1", "Record with SSN 111-22-3333 inside.", [("ssn", "ssn.dashed")]),
        _doc("dd2", "SCANERRTRIG made this ssn doc unscannable.",
             [("ssn", "ssn.dashed")]),
    ]
    env = FakeEnv()
    db = FakeDB(env)
    metrics = _run(env, db, tmp_path, docs=docs)
    cov = metrics["coverage"]
    # dd2 was never scanned: it must not count as a miss
    assert cov["degraded_docs"] == 1
    assert cov["dirty_sent"] == 1 and cov["dirty_alerted"] == 1
    assert cov["rate"] == pytest.approx(1.0)
    assert metrics["scanner_error_alerts"] == 1
    assert "lower bound" in metrics["notes"]["scanner_error_note"]


def test_run_e2e_purges_and_restores_on_midrun_exception(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    db.fail_fetch_alerts = True
    with pytest.raises(RuntimeError, match="boom"):
        _run(env, db, tmp_path)
    # finally-path cleanup: residual drain -> purge -> restore, all AFTER the
    # failure point
    assert len(db.restore_calls) == 1 and db.snapshots_taken == 1
    assert len(db.purge_calls) == 1
    assert env.alerts == []                    # synthetic alerts did not survive
    i_fail = db.ops.index("fetch_fail")
    i_purge = db.ops.index("purge")
    i_restore = db.ops.index("restore")
    assert i_fail < i_purge < i_restore
    assert "count" in db.ops[i_fail:i_purge]   # residual drain ran before purge


def test_run_e2e_preflight_failure_still_purges_aux_ids(tmp_path):
    env = FakeEnv()
    env._scan = lambda rid, text: None     # dead worker: canary never alerts
    db = FakeDB(env)
    with pytest.raises(RuntimeError, match="no alert for a canary SSN"):
        _run(env, db, tmp_path, canary_timeout_s=0.05)
    # probe (rid 1) + canary (rid 2) were tracked as they resolved and purged
    assert db.purge_calls == [[1, 2]]
    assert len(db.restore_calls) == 1


def test_run_e2e_skip_config_never_touches_config(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    _run(env, db, tmp_path, skip_config=True)
    assert db.snapshots_taken == 0 and db.applied == [] and db.restore_calls == []
    assert db.config_sets == []


def test_run_e2e_rejects_unknown_plant_side(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    with pytest.raises(ValueError, match="plant_side"):
        _run(env, db, tmp_path, plant_side="sideways")


def test_run_e2e_echo_plant_side(tmp_path):
    env = FakeEnv()
    db = FakeDB(env)
    metrics = _run(env, db, tmp_path, plant_side="echo")
    assert metrics["run"]["plant_side"] == "echo"
    # every doc rode inside the echo instruction (prompt side carries it)
    assert env.last_content.startswith(
        "Repeat the following text exactly, with no commentary: ")
    # prompt-side payload still alerted (echo plants on BOTH sides)
    assert metrics["coverage"]["dirty_alerted"] == 2


def test_preflight_raises_when_audit_capture_off(tmp_path):
    env = FakeEnv()
    env.audit_prompts_on = False
    db = FakeDB(env)
    with pytest.raises(RuntimeError, match="audit prompt capture is OFF"):
        _run(env, db, tmp_path)
    assert len(db.restore_calls) == 1      # config restored even on preflight failure

    env2 = FakeEnv()
    env2.audit_responses_on = False
    with pytest.raises(RuntimeError, match="audit response capture is OFF"):
        _run(env2, FakeDB(env2), tmp_path, plant_side="response")
    # echo needs response capture too: the payload lands on both sides
    env3 = FakeEnv()
    env3.audit_responses_on = False
    with pytest.raises(RuntimeError, match="audit response capture is OFF"):
        _run(env3, FakeDB(env3), tmp_path, plant_side="echo")


def test_preflight_raises_when_canary_never_alerts(tmp_path):
    env = FakeEnv()
    env._scan = lambda rid, text: None     # dead worker: nothing ever alerts
    db = FakeDB(env)

    async def go():
        async with _client_for(env) as client:
            await e2e_mod._preflight_async(
                client, "http://localhost", "uk", "ak", db, "dlp-mock",
                "prompt", "regex", QUIET, poll_interval_s=0.01,
                canary_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="no alert for a canary SSN"):
        asyncio.run(go())


def test_preflight_appends_aux_ids_as_they_resolve():
    env = FakeEnv()
    env._scan = lambda rid, text: None     # canary wait will fail...
    db = FakeDB(env)
    out = []

    async def go():
        async with _client_for(env) as client:
            await e2e_mod._preflight_async(
                client, "http://localhost", "uk", "ak", db, "dlp-mock",
                "prompt", "regex", QUIET, poll_interval_s=0.01,
                canary_timeout_s=0.05, aux_ids_out=out)
    with pytest.raises(RuntimeError, match="no alert for a canary SSN"):
        asyncio.run(go())
    assert out == [1, 2]   # ...but probe + canary ids already escaped for purge


# ---------------------------------------------------------------------------
# Stream correlation preflight probe
# ---------------------------------------------------------------------------

def _preflight(env, db, stream_pct, **kw):
    async def go():
        async with _client_for(env) as client:
            return await e2e_mod._preflight_async(
                client, "http://localhost", "uk", "ak", db, "dlp-mock",
                "prompt", "regex", QUIET, poll_interval_s=0.01,
                canary_timeout_s=1.0, stream_pct=stream_pct,
                row_timeout_s=0.2, **kw)
    return asyncio.run(go())


def test_stream_preflight_probe_verifies_chunk_id_contract():
    env = FakeEnv()
    db = FakeDB(env)
    pre = _preflight(env, db, stream_pct=0.5)
    # probe + stream probe + canary all tracked for the purge
    assert len(pre["aux_request_ids"]) == 3


def test_stream_preflight_probe_skipped_when_not_streaming():
    env = FakeEnv()
    env.stream_own_ids = True              # would fail IF the probe ran
    db = FakeDB(env)
    pre = _preflight(env, db, stream_pct=0.0)
    assert len(pre["aux_request_ids"]) == 2   # probe + canary only


def test_stream_preflight_probe_rejects_foreign_chunk_ids():
    # A real backend stamps its own chatcmpl-* chunk ids, which resolve to no
    # requests row: streamed coverage joins and the alert purge would silently
    # break, so preflight must fail loudly.
    env = FakeEnv()
    env.stream_own_ids = True
    db = FakeDB(env)
    with pytest.raises(RuntimeError,
                       match="stream correlation requires the mock backend"):
        _preflight(env, db, stream_pct=0.5)


# ---------------------------------------------------------------------------
# Drain settle logic (simulated clock + count progression)
# ---------------------------------------------------------------------------

class _CountDB:
    def __init__(self, counts):
        self.counts = list(counts)
    def count_alerts_for_request_ids(self, ids, exclude_scanner_errors=True):
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]


def _drain(counts, drain_timeout_s, settle_s):
    clock = {"t": 0.0}

    async def fake_sleep(s):
        clock["t"] += s

    return asyncio.run(e2e_mod._drain_async(
        _CountDB(counts), [1, 2, 3], drain_timeout_s, settle_s,
        started_monotonic=0.0, poll_interval_s=1.0,
        _clock=lambda: clock["t"], _sleep=fake_sleep))


def test_drain_settles_after_count_stops_moving():
    seconds, settled = _drain([0, 1, 2, 2], drain_timeout_s=100.0, settle_s=4.0)
    assert settled is True
    # count last changed at t=2; settled once stable for 4s -> t=6
    assert seconds == pytest.approx(6.0)


def test_drain_gives_up_at_timeout_while_count_still_moving():
    seconds, settled = _drain(list(range(100)), drain_timeout_s=5.0, settle_s=4.0)
    assert settled is False
    assert seconds == pytest.approx(5.0)


def test_drain_with_no_request_ids_settles_immediately():
    async def go():
        return await e2e_mod._drain_async(_CountDB([0]), [], 10.0, 5.0,
                                          started_monotonic=None)
    seconds, settled = asyncio.run(go())
    assert settled is True and seconds < 1.0


# ---------------------------------------------------------------------------
# Prod guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["https://mindrouter.uidaho.edu",
                                 "http://10.0.0.5:8000"])
def test_prod_guard_run_e2e(url):
    with pytest.raises(RuntimeError, match="allow_prod"):
        e2e_mod.run_e2e([], url, "uk", "ak", db=None, out_dir=None)


def test_prod_guard_gateway_functions():
    url = "https://mindrouter.uidaho.edu"
    with pytest.raises(RuntimeError, match="allow_prod"):
        gw.ensure_group(url, "ak")
    with pytest.raises(RuntimeError, match="allow_prod"):
        gw.provision_users(url, "ak", 1, group_id=1)
    with pytest.raises(RuntimeError, match="allow_prod"):
        gw.teardown_users(url, "ak", [])
    with pytest.raises(RuntimeError, match="allow_prod"):
        asyncio.run(gw.send_chat(None, url, "k", "m", "x",
                                 stream=False, plant_side="prompt"))
    for ok_url in ("http://localhost:8000", "http://127.0.0.1", "http://[::1]:9"):
        gw.require_local(ok_url)   # must not raise
