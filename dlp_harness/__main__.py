############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/__main__.py: Operator CLI for the DLP
# evaluation harness — `python -m dlp_harness <subcommand>`.
#
# Thin argparse dispatch onto the harness modules. Heavy
# modules (httpx, pymysql, the scanner bridge) are imported
# lazily inside each handler, so `corpus`/`offline` work in
# an environment without the gateway/DB dependencies.
#
############################################################

"""Command-line interface for the DLP evaluation harness."""

import argparse
import importlib
import json
import os
import sys
from typing import Any, List, Optional


def _import(name: str):
    """Lazy module import (module-level so tests can monkeypatch it)."""
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Safety: refuse non-local gateways unless --allow-prod
# ---------------------------------------------------------------------------

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _require_local(url: str, allow_prod: bool, what: str = "gateway") -> None:
    from urllib.parse import urlparse
    host = url
    if "//" in str(url):
        host = urlparse(url).hostname or ""
    if host not in _LOCAL_HOSTS and not allow_prod:
        raise SystemExit(
            f"refusing non-local {what} {url!r} without --allow-prod "
            "(this command mutates config / creates load)")


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _int_csv(s: str) -> List[int]:
    return [int(x) for x in _csv(s)]


def _run_dir(args, kind: str) -> str:
    schemas = _import("dlp_harness.schemas")
    if getattr(args, "runs_root", None):
        schemas.RUNS_ROOT = os.path.abspath(args.runs_root)
    if getattr(args, "out", None):
        out = os.path.abspath(args.out)
        os.makedirs(out, exist_ok=True)
        return out
    return schemas.new_run_dir(kind)


def _read_corpus_docs(path: str) -> list:
    schemas = _import("dlp_harness.schemas")
    if os.path.isdir(path):
        path = os.path.join(path, "corpus.jsonl")
    return schemas.read_jsonl(path, schemas.LabeledDocument)


def _open_db(args):
    db_mod = _import("dlp_harness.db")
    allow = getattr(args, "allow_prod", False)
    # In-container (run_in_container.sh) and prod runs: the app's own
    # DATABASE_URL is the credential source of truth — the harness never
    # carries prod DB secrets. Explicit --db-* overrides always win; DB
    # stubs without the classmethod fall through to the kwargs path.
    from_url = getattr(db_mod.HarnessDB, "from_database_url", None)
    if (from_url is not None and os.environ.get("DATABASE_URL")
            and args.db_host == "127.0.0.1" and args.db_password is None):
        return from_url(allow_remote=allow)
    return db_mod.HarnessDB(
        host=args.db_host, port=args.db_port, user=args.db_user,
        password=args.db_password, database=args.db_name,
        allow_remote=allow)


def _add_db_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db-host", default="127.0.0.1")
    p.add_argument("--db-port", type=int, default=3306)
    p.add_argument("--db-user", default="mindrouter")
    p.add_argument("--db-password", default=None,
                   help="default: $MYSQL_PASSWORD or the local dev password")
    p.add_argument("--db-name", default="mindrouter")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_corpus(args) -> int:
    corpus_mod = _import("dlp_harness.corpus")
    schemas = _import("dlp_harness.schemas")
    docs = corpus_mod.generate(args.profile, args.size, args.seed,
                               dirty_rate=args.dirty_rate)
    out = _run_dir(args, "corpus")
    manifest = corpus_mod.write_corpus(docs, out)
    schemas.save_manifest(out, schemas.RunManifest(
        run_id=os.path.basename(out), kind="corpus",
        created_at=schemas.utc_now_iso(), argv=list(sys.argv[1:]),
        seed=args.seed, corpus_path=os.path.join(out, "corpus.jsonl")))
    print(f"corpus: {manifest['docs']} docs ({manifest['dirty_docs']} dirty, "
          f"{manifest['clean_docs']} clean, {manifest['entities']} entities)")
    print(out)
    return 0


def cmd_offline(args) -> int:
    oe = _import("dlp_harness.offline_eval")
    out = _run_dir(args, "offline")
    oe.run_offline(
        corpus_path=args.corpus,
        out_dir=out,
        scanners=tuple(_csv(args.scanners)),
        gliner_threshold=args.gliner_threshold,
        gliner_max_chars=args.gliner_max_chars,
        sweep=args.sweep,
        in_container=args.in_container,
        seed=args.seed,
    )
    print(out)
    return 0


def cmd_mock(args) -> int:
    mock = _import("dlp_harness.mock_backend")
    if args.mock_cmd == "serve":
        return mock.main(list(args.mock_args))
    if args.mock_cmd == "register":
        _require_local(args.base_url, args.allow_prod)
        backend = mock.register_mock_backend(args.base_url, args.admin_key,
                                             args.backend_url, name=args.name)
        print(json.dumps(backend, indent=2, default=str))
        return 0
    if args.mock_cmd == "disable":
        _require_local(args.base_url, args.allow_prod)
        mock.disable_backend(args.base_url, args.admin_key, args.backend_id)
        print(f"backend {args.backend_id} disabled")
        return 0
    raise SystemExit(f"unknown mock subcommand {args.mock_cmd!r}")


# Username prefix the gateway module uses for harness-created users; teardown
# deletes strictly by this prefix so it can never touch a real account.
HARNESS_USER_PREFIX = "_dlpharness_"


def _teardown_by_prefix(base_url: str, admin_key: str,
                        prefix: str = HARNESS_USER_PREFIX,
                        transport: Any = None) -> List[dict]:
    """Delete every user whose username starts with ``prefix``.

    Implemented inline (GET /api/admin/users pages -> DELETE each) so
    teardown works even without the gateway module.
    """
    import httpx
    deleted: List[dict] = []
    page = 200
    with httpx.Client(base_url=base_url, timeout=30.0, transport=transport,
                      headers={"Authorization": f"Bearer {admin_key}"}) as client:
        victims: List[dict] = []
        skip = 0
        while True:
            r = client.get("/api/admin/users",
                           params={"skip": skip, "limit": page, "search": prefix})
            r.raise_for_status()
            payload = r.json()
            users = payload.get("users", []) if isinstance(payload, dict) else payload
            victims.extend(u for u in users
                           if str(u.get("username", "")).startswith(prefix))
            if len(users) < page:
                break
            skip += page
        for u in victims:
            resp = client.delete(f"/api/admin/users/{u['id']}")
            if resp.status_code < 300:
                deleted.append({"id": u["id"], "username": u["username"]})
            else:
                print(f"warning: delete user {u['id']} ({u['username']}) -> "
                      f"HTTP {resp.status_code}", file=sys.stderr)
    return deleted


# Group name gateway.ensure_group creates; --teardown removes it once empty.
HARNESS_GROUP_NAME = "dlp-harness"


def _teardown_group(base_url: str, admin_key: str,
                    name: str = HARNESS_GROUP_NAME,
                    transport: Any = None) -> dict:
    """Best-effort delete of the (now empty) harness group.

    Skips with a warning if the group still has members (a group with users
    cannot — and must not — be deleted).
    """
    import httpx
    with httpx.Client(base_url=base_url, timeout=30.0, transport=transport,
                      headers={"Authorization": f"Bearer {admin_key}"}) as client:
        r = client.get("/api/admin/groups")
        r.raise_for_status()
        grp = next((g for g in r.json().get("groups", [])
                    if g.get("name") == name), None)
        if grp is None:
            return {"name": name, "deleted": False, "reason": "not found"}
        members = int(grp.get("user_count") or 0)
        if members:
            print(f"warning: group {name!r} still has {members} member(s); "
                  "not deleted", file=sys.stderr)
            return {"name": name, "deleted": False,
                    "reason": f"{members} member(s) remain"}
        resp = client.delete(f"/api/admin/groups/{grp['id']}")
        if resp.status_code < 300:
            return {"name": name, "deleted": True, "id": grp["id"]}
        print(f"warning: delete group {grp['id']} ({name}) -> "
              f"HTTP {resp.status_code}", file=sys.stderr)
        return {"name": name, "deleted": False,
                "reason": f"HTTP {resp.status_code}"}


def cmd_provision(args) -> int:
    _require_local(args.base_url, args.allow_prod)
    if args.teardown:
        deleted = _teardown_by_prefix(args.base_url, args.admin_key,
                                      prefix=args.prefix)
        try:
            group = _teardown_group(args.base_url, args.admin_key)
        except Exception as e:  # best-effort; user deletion already succeeded
            print(f"warning: group teardown failed: {e}", file=sys.stderr)
            group = {"name": HARNESS_GROUP_NAME, "deleted": False,
                     "reason": str(e)}
        print(json.dumps({"deleted": deleted, "group": group}, indent=2))
        return 0
    gw = _import("dlp_harness.gateway")
    group_id = gw.ensure_group(args.base_url, args.admin_key,
                               allow_prod=args.allow_prod)
    users = gw.provision_users(args.base_url, args.admin_key, args.users,
                               group_id, allow_prod=args.allow_prod)
    payload = {
        "group_id": group_id,
        "users": [{"username": u.username, "api_key": u.api_key} for u in users],
    }
    # Full keys go to a private file, never to stdout (terminal scrollback /
    # CI logs on the prod host) unless --show-keys asks for the old behavior.
    out = _run_dir(args, "provision")
    keys_path = os.path.join(out, "provision.json")
    fd = os.open(keys_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if args.show_keys:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({
            "group_id": group_id,
            "users": [{"username": u.username, "key_prefix": u.api_key[:12]}
                      for u in users],
            "keys_file": keys_path,
        }, indent=2))
    return 0


def cmd_e2e(args) -> int:
    _require_local(args.base_url, args.allow_prod)
    e2e = _import("dlp_harness.e2e")
    docs = _read_corpus_docs(args.corpus)
    out = _run_dir(args, "e2e")
    db = _open_db(args)
    try:
        extra = {}
        if args.drain_timeout is not None:
            extra["drain_timeout_s"] = args.drain_timeout
        if args.settle is not None:
            extra["settle_s"] = args.settle
        e2e.run_e2e(
            docs=docs,
            base_url=args.base_url,
            api_key=args.api_key,
            admin_key=args.admin_key,
            db=db,
            out_dir=out,
            scanner_mode=args.mode,
            plant_side=args.plant_side,
            stream_pct=args.stream_pct,
            concurrency=args.concurrency,
            model=args.model,
            keep_alerts=args.keep_alerts,
            allow_prod=args.allow_prod,
            seed=args.seed,
            **extra,
        )
    finally:
        db.close()
    print(out)
    return 0


def cmd_load(args) -> int:
    _require_local(args.base_url, args.allow_prod)
    load_mod = _import("dlp_harness.load")
    docs = _read_corpus_docs(args.corpus)
    out = _run_dir(args, "load")

    api_keys = _csv(args.api_keys) if args.api_keys else []
    provisioned = None
    gw = None
    if not api_keys:
        if not args.provision:
            raise SystemExit("load requires --api-keys k1,k2,... or --provision N")
        gw = _import("dlp_harness.gateway")
        group_id = gw.ensure_group(args.base_url, args.admin_key,
                                   allow_prod=args.allow_prod)
        provisioned = gw.provision_users(args.base_url, args.admin_key,
                                         args.provision, group_id,
                                         allow_prod=args.allow_prod)
        api_keys = [u.api_key for u in provisioned]

    db = _open_db(args)
    try:
        load_mod.run_load_matrix(
            base_url=args.base_url,
            api_keys=api_keys,
            admin_key=args.admin_key,
            db=db,
            out_dir=out,
            docs=docs,
            modes=_csv(args.modes),
            concurrencies=_int_csv(args.concurrencies),
            duration_s=args.duration,
            warmup_s=args.warmup,
            stream=not args.no_stream,
            model=args.model,
            max_tokens=args.max_tokens,
            dirty_rate=args.dirty_rate,
            allow_prod=args.allow_prod,
            seed=args.seed,
            compose_dir=args.compose_dir,
        )
    finally:
        if provisioned is not None and gw is not None:
            try:
                gw.teardown_users(args.base_url, args.admin_key, provisioned,
                                  allow_prod=args.allow_prod)
            except Exception as e:  # teardown is best-effort; the run succeeded
                print(f"warning: user teardown failed: {e}", file=sys.stderr)
        db.close()
    print(out)
    return 0


def cmd_report(args) -> int:
    report = _import("dlp_harness.report")
    out = _run_dir(args, "report")
    run_dirs = [os.path.abspath(p) for p in _csv(args.runs)]
    result = report.generate_report(run_dirs=run_dirs, out_dir=out, title=args.title)
    if getattr(args, "pdf", False):
        pdf_path = os.path.join(out, "report.pdf")
        try:
            report.html_to_pdf(result["html_path"], pdf_path)
            print(pdf_path)
        except RuntimeError as e:
            print(f"PDF rendering skipped: {e}", file=sys.stderr)
    print(out)
    return 0


def cmd_db_check(args) -> int:
    constants = _import("dlp_harness.constants")
    db = _open_db(args)
    try:
        present = {r["key"] for r in db.query(
            "SELECT `key` FROM app_config WHERE `key` LIKE %s", ("dlp.%",))}
        info = {
            "db_now": str(db.db_now()),
            "dlp.enabled": db.get_config("dlp.enabled"),
            "alerts_total": db.query(
                "SELECT COUNT(*) AS n FROM dlp_alerts")[0]["n"],
            "config_keys_present": sorted(present),
            "config_keys_missing": [k for k in constants.DLP_CONFIG_KEYS
                                    if k not in present],
        }
    finally:
        db.close()
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_restore_config(args) -> int:
    """Disaster recovery: replay a run dir's on-disk DLP config snapshot.

    For runs killed hard (SIGKILL, OOM, container restart) whose in-process
    ``finally:`` restore never ran.
    """
    path = os.path.abspath(args.run_dir)
    if os.path.isdir(path):
        path = os.path.join(path, "config_snapshot.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise SystemExit(f"cannot read config snapshot {path!r}: {e}")
    db = _open_db(args)
    try:
        db.restore_dlp_config(db.snapshot_from_json(data))
    finally:
        db.close()
    print(json.dumps({
        "snapshot": path,
        "restored": sorted(k for k, v in data.items() if v is not None),
        "deleted_missing_keys": sorted(k for k, v in data.items() if v is None),
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dlp_harness",
        description="MindRouter DLP evaluation harness (see dlp_harness/README.md)")
    parser.add_argument("--runs-root", default=None,
                        help="override the dlp_harness_runs/ output root")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("corpus", help="generate a labeled synthetic corpus")
    p.add_argument("--profile", required=True,
                   choices=("accuracy", "adversarial", "load", "scale"))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dirty-rate", type=float, default=None,
                   help="load profile only (default 0.2)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("offline", help="offline scanner evaluation over a corpus")
    p.add_argument("--corpus", required=True)
    p.add_argument("--scanners", default="regex", help="comma-separated: regex,gliner")
    p.add_argument("--gliner-threshold", type=float, default=0.5)
    p.add_argument("--gliner-max-chars", type=int, default=10_000)
    p.add_argument("--sweep", action="store_true",
                   help="confidence threshold sweep (scans gliner once at 0.05)")
    p.add_argument("--in-container", action="store_true",
                   help="run the scan inside the app container via docker compose")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser("mock", help="mock backend: serve / register / disable")
    mock_sub = p.add_subparsers(dest="mock_cmd", required=True)
    ps = mock_sub.add_parser("serve", help="run the mock backend "
                             "(remaining args pass through to mock_backend.main)")
    ps.add_argument("mock_args", nargs=argparse.REMAINDER)
    pr = mock_sub.add_parser("register", help="register the mock with the gateway")
    pr.add_argument("--base-url", required=True)
    pr.add_argument("--admin-key", required=True)
    pr.add_argument("--backend-url", required=True)
    pr.add_argument("--name", default="dlp-harness-mock")
    pr.add_argument("--allow-prod", action="store_true")
    pd = mock_sub.add_parser("disable", help="disable a registered backend")
    pd.add_argument("--base-url", required=True)
    pd.add_argument("--admin-key", required=True)
    pd.add_argument("--backend-id", type=int, required=True)
    pd.add_argument("--allow-prod", action="store_true")
    p.set_defaults(func=cmd_mock)

    p = sub.add_parser("provision", help="create (or tear down) harness test users")
    p.add_argument("--base-url", required=True)
    p.add_argument("--admin-key", required=True)
    p.add_argument("--users", type=int, default=4)
    p.add_argument("--teardown", action="store_true",
                   help=f"delete users whose username starts with {HARNESS_USER_PREFIX} "
                        f"(and the {HARNESS_GROUP_NAME!r} group once empty)")
    p.add_argument("--prefix", default=HARNESS_USER_PREFIX)
    p.add_argument("--show-keys", action="store_true",
                   help="print full API keys to stdout (default: usernames + "
                        "key prefixes; full keys go to provision.json, mode 0600)")
    p.add_argument("--allow-prod", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_provision)

    p = sub.add_parser("e2e", help="end-to-end detection run through the gateway")
    p.add_argument("--corpus", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--admin-key", required=True)
    p.add_argument("--mode", default="regex",
                   choices=("off", "regex", "gliner", "regex+gliner"))
    p.add_argument("--plant-side", default="prompt",
                   choices=("prompt", "response", "mixed", "echo"),
                   help="echo asks the model to repeat the planted text "
                        "(response-side testing against a real, non-mock model)")
    p.add_argument("--stream-pct", type=float, default=0.5)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model", default="dlp-mock")
    p.add_argument("--keep-alerts", action="store_true",
                   help="skip the post-run alert purge")
    p.add_argument("--drain-timeout", type=float, default=None)
    p.add_argument("--settle", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-prod", action="store_true")
    p.add_argument("--out", default=None)
    _add_db_args(p)
    p.set_defaults(func=cmd_e2e)

    p = sub.add_parser("load", help="load/overhead matrix (modes x concurrencies)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--admin-key", required=True)
    p.add_argument("--api-keys", default=None, help="comma-separated mr2_ keys")
    p.add_argument("--provision", type=int, default=0,
                   help="provision N users instead of --api-keys (torn down after)")
    p.add_argument("--modes", default="off,regex")
    p.add_argument("--concurrencies", default="1,4,16")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--warmup", type=float, default=10.0)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--model", default="dlp-mock")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--dirty-rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compose-dir", default=None,
                   help="dir containing docker-compose.yml, for CPU/queue-drop "
                        "sampling (default: repo root derived from the harness)")
    p.add_argument("--allow-prod", action="store_true")
    p.add_argument("--out", default=None)
    _add_db_args(p)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("report", help="build the HTML/JSON report from run dirs")
    p.add_argument("--pdf", action="store_true",
                   help="also print report.pdf via headless Chrome (if installed)")
    p.add_argument("--runs", required=True, help="comma-separated run directories")
    p.add_argument("--title", default="DLP Evaluation Report")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("restore-config",
                       help="disaster recovery: restore DLP config from a run "
                            "dir's config_snapshot.json after a hard-killed run")
    p.add_argument("--run-dir", required=True,
                   help="run directory containing config_snapshot.json "
                        "(or a direct path to the snapshot file)")
    p.add_argument("--allow-prod", action="store_true")
    _add_db_args(p)
    p.set_defaults(func=cmd_restore_config)

    p = sub.add_parser("db-check", help="sanity-check DB access + DLP config")
    p.add_argument("--host", dest="db_host", default="127.0.0.1")
    p.add_argument("--port", dest="db_port", type=int, default=3306)
    p.add_argument("--user", dest="db_user", default="mindrouter")
    p.add_argument("--password", dest="db_password", default=None)
    p.add_argument("--database", dest="db_name", default="mindrouter")
    p.add_argument("--allow-prod", action="store_true")
    p.set_defaults(func=cmd_db_check)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # argparse.REMAINDER cannot start with an option-like token (bpo-17050),
    # so `mock serve --port N` never reaches cmd_mock; hand the tail to
    # mock_backend.main directly. --runs-root is irrelevant to serving.
    head = argv
    if head and head[0] == "--runs-root" and len(head) >= 2:
        head = head[2:]
    elif head and head[0].startswith("--runs-root="):
        head = head[1:]
    if head[:2] == ["mock", "serve"]:
        mock = _import("dlp_harness.mock_backend")
        return mock.main(head[2:]) or 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
