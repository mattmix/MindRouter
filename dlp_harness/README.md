# DLP Evaluation Harness (`dlp_harness/`)

Measurement and validation suite for MindRouter's DLP subsystem
(`backend/app/services/dlp_scanner.py` + `dlp_worker.py`). It treats DLP as a
data-science problem: build synthetic corpora with **span-exact ground truth**,
run the **production scanner code** over them (offline and end-to-end through
the gateway), and reduce the results to precision/recall/F1, threshold sweeps,
coverage, lag, and overhead numbers with confidence intervals.

Why a harness: DLP in prod is post-hoc and silent — a clean scan writes no row,
a dropped queue item writes no row, and a disabled capture path scans nothing
at all. Without labeled ground truth and controlled runs there is no way to
distinguish "no PII" from "PII missed" from "PII never scanned".

## Architecture

```
                       +--------------------+
                       |  corpus.py         |  synthetic labeled docs
                       |  generators.py     |  (span-exact ground truth,
                       +---------+----------+   hard-negative traps)
                                 |
              corpus.jsonl       |
        +------------------------+------------------------+
        |                                                 |
        v  OFFLINE (no gateway, no DB)                    v  ONLINE (local stack)
+-------------------+                          +--------------------------+
| offline_eval.py   |                          | e2e.py / load.py         |
|  scanner_bridge --+--> dlp_scanner.py        |   |                      |
|  (local import)   |    (the real code)       |   v                      |
|  container_runner-+--> app container         | gateway :8000 --- mock_backend.py
|  (docker compose) |    (real GLiNER)         |   |  (real request path)  (scripted
+---------+---------+                          |   v                        replies)
          |                                    | dlp_worker -> dlp_alerts (MariaDB)
          v                                    |   ^                      |
  offline_metrics.json                         |   +---- db.py (config + alerts)
  offline_findings.jsonl                       +-----+--------------------+
          |                                          |
          |            e2e_results.jsonl / e2e_metrics.json
          |            load_requests.jsonl / load_phases.json
          |                                          |
          +-------------------+----------------------+
                              v
                        report.py  ->  HTML/JSON report
```

- **matching.py** — normalizes findings and matches them to ground-truth spans
  (lenient overlap + strict category), attributes false positives to traps.
- **metrics.py** — confusion math, grouped recall, scope splits, bootstrap CIs,
  threshold sweeps, latency percentiles.
- **constants.py** — canonical taxonomy, scanner scopes, the 23 `dlp.*` config
  keys, `SCANNER_MODES`, `SAFE_RUN_OVERRIDES`.
- **db.py** — direct MariaDB access (config snapshot/restore, alert queries);
  refuses non-local hosts by default.
- **mock_backend.py** — fake vLLM backend; a `<<<REPLY>>>` marker lets each
  request script its own response (and `<<<REPLY_B64>>>` plants PII in the
  *response only*, invisible to prompt-side scanning).

## Quickstart (local stack)

```bash
# 0. Local stack up (service name is `app`)
docker compose up -d --build

# 1. Generate a labeled corpus
python -m dlp_harness corpus --profile accuracy --size 500 --seed 42
# -> dlp_harness_runs/<ts>-corpus/corpus.jsonl

# 2. Offline evaluation (regex locally; add gliner via the app container)
python -m dlp_harness offline --corpus dlp_harness_runs/<ts>-corpus/corpus.jsonl
python -m dlp_harness offline --corpus ... --scanners regex,gliner --in-container --sweep

# 3. Mock backend: serve (own terminal), then register with the gateway
python -m dlp_harness mock serve --port 9101
python -m dlp_harness mock register --base-url http://localhost:8000 \
    --admin-key $ADMIN_KEY --backend-url http://host.docker.internal:9101

# 4. Provision harness users (usernames prefixed _dlpharness_)
python -m dlp_harness provision --base-url http://localhost:8000 --admin-key $ADMIN_KEY --users 4

# 5. End-to-end detection through the gateway
python -m dlp_harness e2e --corpus <corpus.jsonl> --base-url http://localhost:8000 \
    --api-key <mr2_user_key> --admin-key $ADMIN_KEY --mode regex --plant-side mixed

# 6. Load / overhead matrix (off vs regex, 1/4/16 concurrent)
python -m dlp_harness load --corpus <load-corpus.jsonl> --base-url http://localhost:8000 \
    --admin-key $ADMIN_KEY --provision 4 --modes off,regex --concurrencies 1,4,16 --duration 60

# 7. Report over any set of run dirs
python -m dlp_harness report --runs <offline-run>,<e2e-run>,<load-run>

# Cleanup: tear down harness users; disable the mock backend
python -m dlp_harness provision --base-url http://localhost:8000 --admin-key $ADMIN_KEY --teardown
python -m dlp_harness mock disable --base-url http://localhost:8000 --admin-key $ADMIN_KEY --backend-id <id>
```

Makefile shortcuts: `make dlp-corpus`, `dlp-offline`, `dlp-mock`, `dlp-e2e`,
`dlp-load`, `dlp-report` (see `make help`). `python -m dlp_harness db-check`
sanity-checks DB access and prints the live `dlp.*` config-key inventory.

## Safety model

- **Prod guard.** Every subcommand that talks to a gateway refuses a non-local
  `--base-url` without `--allow-prod`; `HarnessDB` likewise refuses non-local
  DB hosts without `allow_remote=True`. Nothing defaults to prod anywhere.
- **Config snapshot/restore.** Runs that flip scanner modes snapshot all
  `dlp.*` keys first and restore them in a `finally:` block — including keys
  that did not exist (deleted on restore, not written back as null).
- **Email always off.** `constants.SAFE_RUN_OVERRIDES` is applied on top of
  every selected scanner mode: all `dlp.email.*.mode` keys forced `"off"`,
  all recipient lists blanked, digest recipients blanked. A harness run can
  never page a human.
- **Dedup forced off** during runs (also in `SAFE_RUN_OVERRIDES`): dedup
  suppresses alert rows and would silently corrupt coverage math.
- **Alert purge.** E2E runs delete the `dlp_alerts` rows they created
  (keyed by the run's own request ids) unless `--keep-alerts` is passed.
- **Teardown by prefix.** `provision --teardown` deletes only users whose
  username starts with `_dlpharness_` — it cannot touch a real account. It
  also best-effort deletes the `dlp-harness` group once it is empty (skipped
  with a warning if members remain).

## Production runbook

Running against prod is deliberate, gated, and leaves state behind on
failure — read this whole section before the first run.

### How to run on the prod host

Use `./dlp_harness/run_in_container.sh` from the repo root on the prod host
(`/opt/mindrouter`). It copies the harness into the running `app` container
(`rm -rf` first — a bare `docker compose cp` into an existing directory
*nests* instead of replacing, silently re-running the previous version's
code), executes `python -m dlp_harness "$@"` there, and copies run artifacts
back to `./dlp_harness_runs/` on the host:

```bash
./dlp_harness/run_in_container.sh db-check --allow-prod
./dlp_harness/run_in_container.sh e2e --corpus ... --base-url http://127.0.0.1:8000 \
    --api-key ... --admin-key ... --mode regex --allow-prod
```

The container already has every harness dependency (httpx, pymysql, gliner
with a warm model cache) and host networking to the gateway and MariaDB.

### Credentials

Inside the container the harness reads the app's own `DATABASE_URL`
(`HarnessDB.from_database_url`) — the harness never carries prod DB secrets
of its own. Explicit `--db-*` flags always win over the env var; outside the
container they are the only path. Gateway credentials (`--api-key`,
`--admin-key`) still come from the operator.

### The `--allow-prod` contract

Every subcommand that mutates state refuses a non-local `--base-url` (and
`HarnessDB` a non-local DB host) unless `--allow-prod` is passed. The flag is
forwarded end-to-end: the CLI guard, the gateway provisioning helpers
(`ensure_group` / `provision_users` / `teardown_users`), and the DB layer all
receive it, so a single `--allow-prod` on the command line is sufficient —
and its absence is a hard stop at the first prod-shaped hostname, before any
HTTP or SQL is issued.

### WARNING: `off` phases disable DLP for live traffic

`scanner_mode=off` (the baseline in `e2e --mode off` and the default load
matrix `--modes off,regex`) sets `dlp.enabled=false` **globally**. DLP is a
shared, post-hoc pipeline: for the duration of an `off` phase, **all LIVE
production traffic goes unscanned**, not just harness traffic — and
`SAFE_RUN_OVERRIDES` has meanwhile blanked every alert-email recipient. Keep
`off` phases short, schedule prod runs in low-traffic windows, and prefer
dropping `off` from `--modes` entirely when a baseline is not required.

### Config snapshot / restore and alert purge

Before flipping any scanner mode, e2e and load runs snapshot all `dlp.*`
keys to `config_snapshot.json` in the run dir *and* hold the snapshot in
memory; the in-process `finally:` restores it (keys absent pre-run are
deleted, not written back as null). Synthetic alert rows are purged by the
run's own request ids unless `--keep-alerts` is passed. Both are in-process
guarantees — they run on success, on exceptions, and on Ctrl-C, but **not**
on SIGKILL / OOM / container restart.

### Disaster recovery after a hard kill

If a run died hard (config left mutated, emails still blanked), replay the
on-disk snapshot:

```bash
./dlp_harness/run_in_container.sh restore-config \
    --run-dir /tmp/dlp_harness_runs/<run-dir> --allow-prod
# or on the host, with explicit --db-* flags:
python -m dlp_harness restore-config --run-dir dlp_harness_runs/<run-dir> \
    --db-host ... --db-password ... --allow-prod
```

It prints the keys restored and the pre-run-absent keys it deleted. Then
check for leftover synthetic alerts (`db-check` prints `alerts_total`) and
leftover `_dlpharness_*` users (`provision --teardown`).

### Provisioned users and keys

`provision` writes full API keys only to `provision.json` (mode 0600) in the
run dir and prints usernames + key prefixes to stdout; `--show-keys` restores
the old full-keys-on-stdout behavior for local use. On prod, treat
`provision.json` as a secret and delete it after the run.

### Real-model response testing: `--plant-side echo`

The `response` / `mixed` plant sides rely on the mock backend's
`<<<REPLY_B64>>>` marker and only work against `dlp-mock`. To test
response-side scanning against a **real** model, use `--plant-side echo`: the
prompt asks the model to repeat the planted text back, so the PII appears in
a genuine model response. Detection then depends on the model actually
echoing faithfully — treat sub-100% coverage as a lower bound and inspect
misses before blaming the scanner.

## Interpreting the numbers

- **Lenient vs strict (span metrics).** *Lenient* counts a ground-truth entity
  as detected if any finding overlaps its span, category ignored — "would an
  alert fire on this text". *Strict* additionally requires the finding's
  canonical category to match — "would the alert say the right thing".
  Severity routing depends on category, so a large lenient/strict gap means
  alerts fire but get routed at the wrong severity.
- **In-scope vs system recall (`scope_split`).** In-scope recall is measured
  only over categories the enabled scanners are *capable* of seeing (regex:
  5 built-ins; GLiNER: its 9 default labels) — the fair per-scanner number.
  Overall/out-of-scope recall is the honest system number: it charges the
  config for categories nothing is even looking for.
- **Coverage vs drain (e2e/load).** *Coverage* = dirty requests that produced
  an alert row / dirty requests sent. *Drain* = whether the post-hoc scan
  queue settled (alert counts stopped growing) within the drain window. Low
  coverage with `settled=true` means detection misses; `settled=false` means
  you measured too early or the queue dropped items (`dlp_queue_full` in app
  logs — the per-worker asyncio queue drops silently at 10k).
- **`fp_traps`.** Clean docs carry PII lookalikes (ISBNs, tracking numbers,
  Luhn-invalid cards…). This counter attributes each false positive to the
  trap that caused it — the direct tuning to-do list for patterns/threshold.
- **Bootstrap CIs.** Percentile bootstrap resampling whole documents
  (entities within a doc are correlated). `degenerate=true` means too few
  resamples had a defined value — treat the point estimate as anecdotal.

## Known measurement caveats

- **Fullwidth digits DO match `\d`.** Python's `re` is Unicode-aware, so the
  `fullwidth_digits` obfuscation is *not* a regex bypass — those entities are
  expected hits, not expected misses. (Spaced digits and digit-words are real
  bypasses.)
- **Phone-regex span offsets.** The built-in phone pattern can anchor on
  leading context (e.g. an opening paren or preceding digit run), so finding
  spans may be offset by a character or two from ground truth. Matching is
  overlap-based (IoU threshold 0 by default) precisely to absorb this; do not
  tighten `iou_threshold` without re-checking phone recall.
- **Dedup must stay off during runs.** `dlp.dedup.enabled=true` suppresses
  repeat alert rows inside the window and destroys coverage/FP math.
  `SAFE_RUN_OVERRIDES` forces it off; don't re-enable mid-run.
- **Audit capture is a hidden hard dependency.** DLP scans what the audit
  layer stored: prompts require `AUDIT_LOG_ENABLED` + `AUDIT_LOG_PROMPTS`,
  responses additionally `AUDIT_LOG_RESPONSES`. With capture off, requests
  succeed, scans "run", and *nothing is ever scanned* — 0% coverage that has
  nothing to do with the scanners. `db-check` + a tiny e2e smoke run first.
- **Clean scans write no row.** There is no positive "scanned clean" signal
  per request; drain/settle logic infers completion from alert-count
  quiescence plus the scan-lag distribution. Treat coverage on tiny runs
  (< 50 docs) as noisy.
- **GLiNER truncation.** GLiNER sees only the first `dlp.gliner.max_scan_chars`
  (default 10k) of each text; the global cap for all scanners is 200k. The
  `scale` corpus profile plants entities at ~95% depth to measure exactly this
  blindness — don't read its recall as a detection failure.

## Artifacts

Each run writes a directory under `dlp_harness_runs/` (override with
`--runs-root` / `--out`) containing `run.json` (manifest) plus:

| Run kind | Files |
|----------|-------|
| corpus   | `corpus.jsonl`, `manifest.json` |
| offline  | `offline_metrics.json`, `offline_findings.jsonl` |
| e2e      | `e2e_results.jsonl`, `e2e_metrics.json`, `config_snapshot.json` |
| load     | `load_requests.jsonl`, `load_phases.json`, `cpu_samples.jsonl`, `config_snapshot.json` |
| report   | HTML/JSON report over any of the above |

`offline_metrics.json` top-level keys: `run`, `doc_confusion`,
`span_confusion`, `span_confusion_strict`, `scope_split`, `recall_by`
(difficulty/generator/carrier), `fp_traps`, `latency_ms`,
`latency_by_length`, `severity_accuracy`, `bootstrap` (doc_recall /
doc_precision / span_recall), `sweep` (null unless `--sweep`), `scan_errors`.
