# MindRouter DLP GPU Service

A standalone GPU microservice that serves **GLiNER PII detection** with
**dynamic GPU batching** for the MindRouter DLP subsystem.

It runs on a dedicated node (RTX A4000, 16 GB, CUDA 12.x, driver 565) and is
called over HTTP by the MindRouter DLP worker instead of loading GLiNER inside
the app process. It is fully standalone — no `backend.app.*` imports — and
`gliner`/`torch` are imported **lazily**, so the module imports and unit-tests
on a CPU-only Mac with neither installed (`DLP_DEVICE=auto` falls back to CPU).

## What it does

- Loads **R replicas** of a GLiNER model. Each replica has its own
  `asyncio.Queue`, its own batch loop, and its own single-thread executor —
  one replica maps to one CUDA stream user.
- Load-balances `/scan` requests across replicas by **least queue depth**.
- Each replica collects requests arriving within a small **time window**
  (`DLP_BATCH_WINDOW_MS`) into a batch of up to `DLP_MAX_BATCH`, groups them by
  uniform `(labels, threshold)` signature (a hard requirement of GLiNER's
  `batch_predict_entities`), and runs each group as one GPU call.
- Applies `max_chars` prefix truncation itself, mirroring `scan_gliner` in
  `backend/app/services/dlp_scanner.py`, bounded by a hard `DLP_MAX_CHARS_CAP`.
- **Never logs or persists request text.** Only counts and latencies leave the
  process.

## Wire contract

Findings use the **same field names/shape** as
`backend/app/services/dlp_scanner.py::ScanFinding`
(`scanner`/`category`/`text`/`confidence`/`start`/`end`) **except** the service
omits `scanner` — the caller sets it to `"gliner"`. `category` is the GLiNER
label, `confidence` is the model score, `start`/`end` are char offsets into the
(possibly truncated) text.

### `POST /scan`

Request headers: `X-Worker-Key: <shared secret>`, `Content-Type: application/json`

```json
{ "text": "string",
  "categories": ["email", "phone number"],   // or null -> service defaults
  "threshold": 0.5,                            // default 0.5
  "max_chars": 10000 }                         // or null
```

- `200`:
  ```json
  { "findings": [ { "category": "email", "text": "a@b.com",
                    "confidence": 0.98, "start": 10, "end": 17 } ],
    "latency_ms": 12.3, "queued_ms": 1.1, "batch_size": 4 }
  ```
  `batch_size` is the number of texts processed together in the GPU call that
  served this request. An **explicitly empty** `categories: []` returns
  `{"findings": [], ...}` with `batch_size: 0` (scan nothing, mirroring
  `scan_gliner`); `categories: null` uses the service default set.
- `503` (queue full / oversubscribed — the caller's fallback signal):
  ```json
  { "error": "oversubscribed", "queue_depth": 256, "max_queue": 256 }
  ```
- `401` (bad/missing key): `{ "error": "unauthorized" }`
- `400` (bad body): `{ "error": "<reason>" }`

### `GET /healthz`  (no auth)

```json
{ "status": "ok", "model": "urchade/gliner_multi_pii-v1",
  "device": "cuda:0", "replicas": 2, "queue_depth": 0,
  "max_queue": 512, "warm": true }
```

`warm` is `false` until the background warmup loads every replica. The batch
loop also lazy-loads on first use, so the service serves before warmup finishes.

### `GET /stats`  (`X-Worker-Key`)

```json
{ "scans_total": 1200, "batches_total": 340, "avg_batch_size": 3.5,
  "scan_p50_ms": 9.4, "scan_p95_ms": 41.2, "queue_depth": 0,
  "inflight": 0, "rejected_503": 3 }
```

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `DLP_SERVICE_KEY` | `""` | Shared secret for `X-Worker-Key`. **Empty = fail closed** (nothing authenticates). |
| `DLP_SERVICE_HOST` | `0.0.0.0` | Bind host. |
| `DLP_SERVICE_PORT` | `8710` | Bind port. |
| `DLP_MODEL` | `urchade/gliner_multi_pii-v1` | GLiNER model id. |
| `DLP_DEVICE` | `auto` | `auto` → `cuda:0` if torch sees CUDA else `cpu`; or force `cuda:0` / `cpu`. |
| `DLP_REPLICAS` | `2` | Number of model replicas (each = one CUDA stream user). |
| `DLP_MAX_BATCH` | `32` | Max texts per GPU batch. |
| `DLP_BATCH_WINDOW_MS` | `8` | Time window to coalesce arrivals into a batch. |
| `DLP_MAX_QUEUE` | `512` | Total queued requests across replicas (split evenly per replica). |
| `DLP_DEFAULT_CATEGORIES` | 8-category prod set | Comma list used when a request sends `categories: null`. |
| `DLP_MAX_CHARS_CAP` | `10000` | Default + **hard ceiling** for per-request text length. |
| `DLP_HF_HOME` | *(unset)* | HuggingFace cache dir. Put on **node-local disk**, not shared ceph. |
| `DLP_TORCH_THREADS` | `4` | `torch.set_num_threads` per process — see below. |

Default category set: `phone number, email, credit card number,
social security number, date of birth, driver license number, passport number,
bank account number` (mirrors `scan_gliner`; `person` is intentionally excluded).

### Why `DLP_TORCH_THREADS` is bounded

The in-app DLP scanner suffered a measured **~27x slowdown** from torch thread
oversubscription — many concurrent GLiNER predicts each spun up a full thread
pool and thrashed the CPU. On a dedicated GPU node the GPU does the matmuls,
but CPU-side tokenization still benefits from a small bounded thread count, so
the service sets `torch.set_num_threads(DLP_TORCH_THREADS)` once at startup.

## Running

```bash
# CPU dev / import smoke (no gliner, no torch, no GPU needed for import):
DLP_SERVICE_KEY=devkey DLP_DEVICE=cpu python -m dlp_service

# GPU host (after installing the CUDA torch build — see deploy/install.sh):
DLP_SERVICE_KEY=… DLP_DEVICE=cuda:0 DLP_REPLICAS=2 \
  DLP_HF_HOME=/data/dlp/hf_cache python -m dlp_service
```

Systemd deployment: `deploy/dlp-service.service` + `deploy/install.sh`
(fleet-standard `python3.11 -m venv` + node-local HF cache; installs and enables
the unit; idempotent). On Rocky 8 install the interpreter first with
`sudo dnf install -y python3.11 python3.11-devel`.

## Scaling out

This service is one endpoint. To add capacity, run it on additional GPU nodes
(one instance per node, or several ports per node) and register every endpoint
in MindRouter under **Admin → DLP → Off-host GLiNER service → Service URL(s)**
(one URL per line). MindRouter load-balances scans across the pool with
round-robin and automatic per-endpoint failover, and falls back to the on-host
CPU scanner only when the whole pool is unreachable. On a single node, prefer
raising `DLP_REPLICAS` / `DLP_MAX_BATCH` before adding processes — the GPU is
the shared resource, and dynamic batching uses it more efficiently than many
competing processes.

## Security model

- **Stateless / no content logging.** The service holds no DB. It never logs or
  persists request or response text — only counts and latencies. Findings are
  returned in the HTTP response and then dropped.
- **Shared-key auth.** `/scan` and `/stats` require `X-Worker-Key` matched with
  a constant-time compare against `DLP_SERVICE_KEY`. An unset/empty key **fails
  closed**: every authenticated request is rejected `401` until a key is set.
  `/healthz` is unauthenticated (liveness only, no sensitive data).
- **TLS is terminated in front of this service — two supported deployments:**
  1. **Front proxy** (nginx/caddy) on the node terminates TLS on `:443` (or the
     node's HTTPS port) and reverse-proxies to `127.0.0.1:DLP_SERVICE_PORT`.
     Bind the service to `127.0.0.1` in that case.
  2. **Behind the node's existing HTTPS sidecar port** — the same TLS-terminating
     sidecar pattern the inference nodes already run (see
     `sidecar-deploy-script.md` / `inference-node-nginx.md`). The sidecar
     forwards decrypted traffic to the local plaintext `DLP_SERVICE_PORT`.

  In both cases the plaintext service listens only on loopback (or a private
  interface), and callers reach it over TLS. Do **not** expose
  `DLP_SERVICE_PORT` on a public interface without a TLS terminator in front.

## Testing

Unit tests run with **no gliner, no torch, no GPU** (the model manager is
mocked via the `MODEL_FACTORY` hook):

```bash
python -m pytest backend/app/tests/unit/test_dlp_service.py -q
```
