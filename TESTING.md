# MindRouter — Test Manifest

> **Single source of truth** for every test in the project.
> When adding new tests, **add an entry here** so `run all tests` stays accurate.

---

## Quick Reference

| Shorthand             | Makefile target    | What it runs                                         |
|-----------------------|--------------------|------------------------------------------------------|
| `run unit tests`      | `make test-unit`   | All pytest unit tests                                |
| `run integration tests` | `make test-int` | Live-backend integration tests                       |
| `run e2e tests`       | `make test-e2e`    | E2E chat subsystem tests                             |
| `run smoke tests`     | `make test-smoke`  | API smoke test against live deployment               |
| `run stress tests`    | `make test-stress` | Multi-user load/stress test                          |
| `run matrix tests`  | `make test-matrix` | Structured output matrix tests (all API styles)    |
| `run thinking tests` | `make test-thinking` | Structured output + thinking compliance (live stack) |
| `run tool tests`    | `make test-tools`  | Live tool calling compliance (all tool-capable models) |
| `run accessibility tests` | `make test-a11y` | WCAG 2.1 accessibility tests (subset of unit)      |
| `run sidecar tests`   | `make test-sidecar`| GPU sidecar agent tests                              |
| `run all tests`       | `make test-all`    | Unit + integration + E2E + smoke + sidecar tests     |
| `run coverage`        | `make coverage`    | Unit + integration tests with coverage report        |

---

## 1. Unit Tests

**Runner:** `pytest backend/app/tests/unit/ -v`
**Makefile:** `make test-unit`
**Requirements:** No live services needed. Some tests (test_quota, test_scheduler) need pymysql.

| File | Tests | What it covers |
|------|-------|----------------|
| `backend/app/tests/unit/test_translators.py` | 12 | OpenAIIn, OllamaIn, OllamaOut, VLLMOut translator static methods |
| `backend/app/tests/unit/test_translation_roundtrip.py` | 33 | Round-trip: OpenAI ↔ Canonical ↔ Ollama parameter mapping, vision edge cases, embedding gaps, edge cases |
| `backend/app/tests/unit/test_hyperparameter_fidelity.py` | 33 | Extended sampling params (top_k, repeat_penalty, min_p), backend_options passthrough, thinking mode |
| `backend/app/tests/unit/test_completions.py` | 22 | /v1/completions and /api/generate translation, to_chat_request conversion, completion output |
| `backend/app/tests/unit/test_structured_outputs.py` | 22 | JSON schema validation, structured output for Ollama & OpenAI |
| `backend/app/tests/unit/test_streaming.py` | 21 | ndjson (Ollama) and SSE (vLLM/OpenAI) stream parsing |
| `backend/app/tests/unit/test_stream_golden.py` | 17 | Golden-stream pins for the orjson dict pipeline (think-tag gate, reasoning_content, tool deltas, usage suppression/capture) + StreamCoalescer flush boundaries |
| `backend/app/tests/unit/test_validators.py` | 28 | Input validation: request params, message schemas, constraints |
| `backend/app/tests/unit/test_cross_engine_routing.py` | 15 | Ollama ↔ vLLM routing with parameter translation |
| `backend/app/tests/unit/test_circuit_breaker.py` | 5 | CircuitBreakerState: open/half-open, failure counts, reset |
| `backend/app/tests/unit/test_latency_tracker.py` | 8 | Latency tracking, p50/p99 percentile calculations |
| `backend/app/tests/unit/test_retry_failover.py` | 12 | Retry logic, exponential backoff, backend failover |
| `backend/app/tests/unit/test_scheduler.py` | 30 | Fair-share WDRR scheduler, job queue, BackendScorer, HardConstraints |
| `backend/app/tests/unit/test_quota.py` | 24 | Quota management, token accounting, RPM/concurrent limits, group-based defaults |
| `backend/app/tests/unit/test_anthropic_translator.py` | 19 | AnthropicIn translator: Messages API request/response, multimodal, thinking, streaming format |
| `backend/app/tests/unit/test_tool_calling.py` | 33 | Tool calling: schemas, OpenAI/Ollama/Anthropic inbound, vLLM/Ollama outbound, round-trips |
| `backend/app/tests/unit/test_sidecar_client.py` | 16 | GPU sidecar client: auth, GPU info retrieval, communication |
| `backend/app/tests/unit/test_version_alignment.py` | 6 | Version alignment: pyproject.toml reading, sidecar VERSION file consistency |
| `backend/app/tests/unit/test_accessibility.py` | 126 | WCAG 2.1 Level A/AA: ARIA, semantic HTML, heading hierarchy, forms, sidebar include, Video tab + admin Video config (sidebar include, table scope/caption); admin Branding page + login logo included in the all-templates sweep |
| `backend/app/tests/unit/test_chat_mobile.py` | 37 | Chat mobile responsiveness: sidebar collapse/backdrop, thinking block collapse, compact layout CSS |
| `backend/app/tests/unit/test_rerank_translators.py` | 22 | Rerank/score translators: OpenAIIn, VLLMOut rerank & score methods, canonical schema validation |
| `backend/app/tests/unit/test_model_enrichment.py` | 28 | Model auto-enrichment: brave_web_search api_key param, LLM call helper, enrichment pipeline, CRUD helpers, config gating |
| `backend/app/tests/unit/test_voice_schema_contract.py` | 10 | Voice backend schema contract (migration 072, mirrors test_video_schema_contract.py): revision/down_revision pinned, 072 is the only migration claiming that id, the spelled-out ENUM lists are append-only AND 072's OLD_ENGINE equals 065's NEW_ENGINE (a mismatch would make `MODIFY COLUMN` drop values and rewrite rows by ordinal), only `backends` is altered (never the large `requests` table), downgrade documents the narrowing hazard, `BackendEngine` has TTS/STT, `Modality` still has TTS/STT, and discovery maps the voice engines to their modality *before* the embed/rerank name heuristics |
| `backend/app/tests/unit/test_api_key_scopes.py` | 23 | Scoped API keys (migration 073): NULL scopes = legacy so no existing key changes behaviour; scopes only REMOVE privilege, so an app-minted key owned by an administrator is still not an admin credential (the DLP-key failure mode); empty scope list permits nothing; all five group-derived admin paths in `api/auth.py` enforce scope while session-cookie branches are correctly exempt; `require_scope` is opt-in so a legacy broad key cannot drift into provisioning; migration chain + `App.created_by` detached by `delete_user`. **`inference` is now actually enforced** — it was declared but checked nowhere, which made the scope list deny-admin only and left a leaked provisioning credential able to spend its owning administrator's token budget; `authenticate_request` demands it (one place, so a new inference endpoint is covered by default) while `require_scope` and the admin dependencies sit on the pre-authorisation `authenticate_credential`, with a drift guard that no route module reaches for the latter |
| `backend/app/tests/unit/test_entra_tokens.py` | 33 | Entra id_token verification: signs REAL tokens with a throwaway RSA key and serves a REAL JWKS, because a mocked verifier cannot demonstrate forgery resistance. Accepts genuine v2.0 tokens, lowercases email, does not require a nonce (MindRouter did not issue the app's sign-in request). Rejects wrong audience **in both directions** (else any app in the tenant could provision accounts), foreign issuer, unknown key, key substitution, `alg: none`, tampered payload, mismatched `tid`, missing `oid`, expired beyond leeway and not-yet-valid; clock-skew leeway pinned as deliberate; JWKS cached not refetched per request, and an unknown `kid` cannot amplify requests against Entra. **Review fixes:** the v1.0 `sts.windows.net` issuer is now REFUSED (its access tokens carry the bare client-id GUID as audience, exactly like an id_token, making the two hard to separate); a token carrying `scp`/`scope`/`roles` is refused because an access token for the app's own API is not an assertion that the user signed in; `azp`/`appid` are validated when present; and every upstream failure — connect error, timeout, a 429 from Microsoft's rate-limited discovery endpoint, malformed JSON, a non-RSA key under our `kid` (which raises `JWKError`, *not* a `JWTError`) — leaves as `EntraTokenError` rather than an unhandled 500 |
| `backend/app/tests/unit/test_image_access_tristate.py` | 50 | Image access as a global default with per-user exceptions (migration 075): `users.image_generation_enabled` is TRI-STATE — NULL inherits `img.enabled_by_default`, True/False are explicit decisions that outrank it in both directions. The failure this file exists to prevent is that **a nullable boolean is FALSY**, so `if not user.x` (Python), `{% if user.x %}` (Jinja) and `col == True` (SQL) all read an inheriting user as denied — and before 075 there was ZERO test coverage of image gating, so every such defect would have shipped green. Covers: the six-case resolution truth table; null-aware SQL predicates for effective-access vs the exception list (a redundant override is never an exception); fails-closed on a missing user; the model declaration carries **neither** `default=` nor `server_default=` (the first writes an explicit 0 for every SSO- and app-provisioned account, the second is subtler — with a DB default present the ORM omits the column even when explicitly assigned None); video is not collaterally changed; migration 075 backfills 0→NULL but preserves explicit grants, passes `server_default=None`, seeds the global, and materialises NULLs before tightening on downgrade; the admin control is an explicit three-way set rather than the old `not None`→True negation (**the only defect here that fails OPEN**); flipping the global never rewrites user rows; the override badge tests `is none` rather than truthiness. **The load-bearing test is the drift guard**: any file outside the resolver allowlist that reads the column directly fails the build, as does any `Jinja2Templates` env missing the `image_access` global (base.html is rendered by four independent envs). **Review fixes:** the 075 downgrade passed bind params positionally to `op.execute`, whose `execution_options` is KEYWORD-ONLY — a TypeError that would have aborted a rollback *after* MariaDB had already implicitly committed the DDL, leaving a half-downgraded schema that refuses to re-run; the config DELETE now runs first so any failure commits nothing. The pre-075 backup normalizer originally sniffed for the absence of NULLs, which would have rewritten deliberate denials to inherit in any post-075 export where every user is explicitly classified — a fail-open in the disaster-recovery path — and now discriminates on the presence of the `img.enabled_by_default` config row, with edge-case coverage for empty/malformed/pre-058 exports. `exception_filter` was dead code duplicating the rule the admin page actually ran, so it is replaced by `exception_kind` and the tests execute the predicate crud applies |
| `backend/app/tests/unit/test_apps_provisioning.py` | 40 | Registered-app session endpoint (`POST /api/apps/{slug}/sessions`): identity comes only from the verified token and the request body carries no identity field at all; the credential must belong to the app it acts for (else the scope is global, not bounded); unknown and disabled apps are indistinguishable so a caller cannot enumerate registrations; the issued key is inference-only, hidden, expiring and namespaced even when its owner is an administrator; rotation reuses a key with enough life left rather than invalidating concurrent sessions, and plaintext is returned only when freshly minted (`None`, never `""`); `force_rotate` lets an app that lost its key cache recover, throttled to once a minute per user because minting runs Argon2; provisioning delegates to `find_or_create_azure_user` so the cross-provider takeover guard is not reimplemented; token rejection reasons are logged but never returned. **Review fixes:** created-ness is decided on BOTH keys the driver matches on (oid *and* email) — deciding on oid alone marked an ADOPTED local account unclassified and re-grouped it later, which for the local bootstrap admin means silent demotion; a privileged group is never overwritten by the jobTitle mapping; an admin setting a group settles the flag; reclassification resyncs the quota `rpm_limit`; a token with no email is its own 400 rather than the takeover 409; a lost provisioning race retries instead of 500ing; the audit row records the forwarded address, not the nginx peer; `_as_utc` keeps `expires_at` from being aware on mint and naive on reuse |
| `backend/app/tests/unit/test_apps_admin_panel.py` | 43 | Admin → Applications panel: disabling an app **revokes its keys** rather than only refusing new sessions (status alone leaves every minted key live for its full TTL), and enabling does not revoke; credential rotation revokes the previous credential *before* minting the next; the credential carries only `apps:provision`, is hidden and expiring, and cannot be issued for an app with no Entra registration; plaintext is rendered, never redirected into a URL; deregistration requires the slug typed exactly and revokes+detaches before deleting so the FK holds and request history survives; the Entra GUID and slug patterns are lifted from source and executed — anchored with `\Z` because `$` also matches before a trailing newline; AST guards that every mutating route requires full (not read-only) admin, that all of them are audited, and that `_err` only ever takes a literal; app-minted keys are excluded from the owner's key list (the dashboard chat/image/video pages use the first key returned) and from the group's key allowance |
| `backend/app/tests/unit/test_model_catalog_filter.py` | 19 | Model catalogs publish text models only (2.9.10): `is_catalog_model` admits chat/completion/multimodal/embedding/reranking and excludes image/video/tts/stt, fails OPEN on unknown modality so a discovery gap can never hide a working chat model; all catalog surfaces apply it (`/v1/models`, `/api/tags`, the unauthenticated `/status`, with `/anthropic/v1/models` inheriting via delegation) and a drift guard fails if a new endpoint lists backend models without a modality or engine filter; per-modality discovery exists so nothing is stranded (`/v1/images/models` selects by DIFFUSION engine, `/videos/models`), and the docs no longer point image clients at `/v1/models` |
| `backend/app/tests/unit/test_voice_router.py` | 14 | Voice backend resolution (2.9.10): `resolve_voice_backend` prefers a healthy registered `tts`/`stt` backend over the legacy `voice.*_url` app_config value, spreads across multiple backends rather than pinning to one, skips backends whose circuit is open, falls back to config when nothing is registered, keeps voice working when the registry itself raises, returns None when neither source is available, rejects unknown kinds, and labels its origin via `VoiceTarget.source` (`registry` / `config_fallback`) for rollout monitoring |
| `backend/app/tests/unit/test_voice_api.py` | 43 | Public voice API: TTSRequest validation, quota check, request recording, TTS endpoint (happy path, errors, content-type), STT endpoint (happy path, errors, timeout, language, model; 2.8.43: OpenAI model-name aliases whisper-1/whisper/etc map to configured model, non-alias passthrough, upstream-404 → actionable 400), Modality enum **2.9.10 TTS failure handling:** upstream status is checked before any response body or quota charge, so a dead TTS service yields 502 + a recorded failure instead of a billed empty HTTP 200. Covers upstream 500/timeout/unreachable each billing `token_cost=0` with an `error_message`, connection cleanup on the error path, the upstream error body never reaching logs (it can echo caller input), and the happy path billing exactly once. Mutation-verified: 5 of 6 fail against the pre-fix code. |
| `backend/app/tests/unit/test_dlp.py` | 45 | DLP scanner: regex (SSN, CC, email, keywords, custom patterns), severity classification, text extraction (messages, images, response), ScanResult/ScanFinding dataclasses; 2.9.9: scan_llm takes an injected completion callable (no api_key/base_url/httpx), `<think>` + markdown-fence stripping, parse failures never log scanned content, snippet masking, malformed custom patterns can't abort a scan, scan-text cap |
| `backend/app/tests/unit/test_config_export_redaction.py` | 39 | Role-scoped config backup export (2.9.9): `_redact_row` nulls password hashes, API key hashes, and secret app_config values while leaving `key_prefix` and ordinary settings intact; the secret matcher must NOT match bare `token` (would blank `ocr.max_tokens`, `stats.token_offset`, `vid.token_cost_per_second`); serialized-payload sentinel check; `redact_secrets` defaults off so admin restores keep working; metadata `redacted` flag; `import_config_tables` refuses a redacted backup; route redacts for non-admins, names the file distinctly, audit-logs the download; auditors are told their copy is redacted; restore no longer reflects exception text |
| `backend/app/tests/unit/test_dlp_worker.py` | 69 | DLP worker + admin surface (2.9.9, previously untested): credential removal (no raw key in app_config, no `ensure_internal_api_key`, migration 071 revokes before deleting), `_internal_chat` direct-to-backend dispatch (no Authorization header, thinking disabled on vLLM, raises when no healthy backend / circuit open), email contract (AST cross-check that every `email_service.*` call resolves — the `send_email()` that never existed), subject/body correctness, flood-guard burst + suppressed-count carry-over, config-save validation (invalid regex/JSON shape/threshold/email rejected with zero writes), fail-closed `_json_ready` guard so a dead page script can't wipe severity rules/patterns, **template block-name contract (every child `{% block %}` must exist in its parent — `dlp.html` used `scripts` where `base.html` defines `extra_js`, silently killing all page JS)**, public-documentation truth, filter-param normalization, `dlp_alerts` retention wiring |
| `backend/app/tests/unit/test_latex_normalize.py` | 29 | LaTeX normalization: $$-block preservation (v2.4.2 regression), bare command/operator wrapping, display math promotion, inline preservation, mixed content, code block immunity, \\begin/\\end environments |
| `backend/app/tests/unit/test_ocr.py` | 27 | OCR pipeline: chunking logic, fence stripping, prompt building, overlap detection, deterministic merge, image conversion, PDF fixture |
| `backend/app/tests/unit/test_responses_in.py` | 46 | ResponsesIn translator: input polymorphism (string/items/typeless), function-call round trip via call_id, flat tool re-nesting + non-function strip, text.format, reasoning.effort→think, truncation flag, format_response/build_snapshot, vLLM round trip |
| `backend/app/tests/unit/test_responses_stream.py` | 12 | Chat-SSE→Responses-SSE adapter: canonical event sequences (text/reasoning/tools), deferred terminal + usage harvesting, incomplete/failed terminals, exception hardening, drain contract |
| `backend/app/tests/unit/test_responses_api.py` | 42 | /v1/responses routes: feature flag, validation errors, OpenAI error envelopes, quota pre-flight (skip_quota_check), streaming/non-streaming dispatch, alias resolution, store persistence, previous_response_id chains, GET/DELETE/input_items/cancel, hosted web_search dispatch, count input_tokens, conversation integration |
| `backend/app/tests/unit/test_conversations_api.py` | 13 | Conversations API routes: conversation CRUD (create/seed/cap/update/delete), item endpoints (create/list/get/delete envelopes), owner-scoped 404s, flag gating |
| `backend/app/tests/unit/test_blog_email_render.py` | 8 | Blog-email HTML: [TOC] stripping, inline black table borders (alignment merge, thead guard), bordered code containers |
| `backend/app/tests/unit/test_field_validation.py` | 6 | Request-field validation (off/log/enforce): dialect fields (structured_outputs→response_format hint) + unknown fields 400 in enforce, accepted/ignored fields pass, log/off never raise, default reads setting, stream_options.include_usage parsed into canonical |
| `backend/app/tests/unit/test_install_friction_fixes.py` | 8 | Install-friction fixes from field feedback: multimodal heuristic covers qwen3.6/dots (#6), sync-script drops nonexistent supports_embeddings (#10a), vLLM streaming injects include_usage + real-usage accounting (#10b), opt-in RUN_MIGRATIONS at startup before init_registry (#1), automation-friendly admin seed (ADMIN_PASSWORD/ADMIN_API_KEY/MINT_ADMIN_KEY + single-line key) (#2a), OCR degrade-to-single-page on image-limit 400 (#3) |
| `backend/app/tests/unit/test_blog_website_publish.py` | 10 | Blog syndication contract (source-inspection): migration 064 columns, model fields, CRUD selection query (published+undeleted only), flag routes guarded + flag-only (no push), pull-model guards — website_publisher.py stays deleted, no push credentials in settings, /blog/feed.xml + /api/blog/syndicated exist, are public, read the selection query, and feed.xml registers BEFORE the /blog/{slug} catch-all (2.8.45 shadowing regression) |
| `backend/app/tests/unit/test_blog_export.py` | 9 | Institution-neutral syndication helpers (2.8.44 pull model): markdown rendering (codehilite/tables), image-path collection (dedup/order), description derivation, RSS feed renderer (links to app blog via base_url, brandable channel title, valid XML, empty feed), neutrality guard (no mindrouter.ai URL, push-era symbols stay deleted) |
| `backend/app/tests/unit/test_responses_store.py` | 20 | Responses store service: item id stamping, image offload/re-inflate + path containment, chain rebuild + item_reference, payload/row caps, persist contract, crud/migration/retention source checks |
| `backend/app/tests/unit/test_responses_websearch.py` | 11 | Hosted web_search: tool detection/synthetic tool, non-streaming loop (threading, budget, client passthrough), streaming loop (suppression, ws events, cross-round sequencing), error terminal |
| `backend/app/tests/unit/test_context_trim.py` | 6 | truncation:"auto" trimming: turn grouping, tool-pair atomicity, oldest-first drops, system/final-turn protection |
| `backend/app/tests/unit/test_video_schema_contract.py` | 9 | Video-gen v1 foundation (source-inspection + spec-load, pollution-proof): CanonicalVideoRequest defaults + text-to-video-only shape, CanonicalVideoJob OpenAI shape (in_progress status set), models.py video enums (BackendEngine.VIDEO, Modality.VIDEO_GENERATION, users.video_generation_enabled), JobModality.VIDEO_GENERATION, migration 065 ENUM widening (ALGORITHM=INSTANT) + user flag, migration 066 four tables + claim/heartbeat indexes + source_clip provisions, migration 067 config seed (vid.enabled False by default, cap 50, retention), 065→066→067 chain linear, every video_* setting has docker-compose passthrough |
| `backend/app/tests/unit/test_video_field_validation.py` | 9 | Video request-field validation dialect (spec-loaded): enforce accepts all v1 fields, rejects negative_prompt with the guidance-free hint (2.8.42: allowlisted-but-dropped fixed), rejects duration/width/height typos with hints pointing to seconds/size, rejects v1-unsupported conditioning (image/input_reference/first_frame/storyboard), rejects unknown fields, ignored fields pass, log/off never raise, VIDEO_ACCEPTED drift-guard vs CanonicalVideoRequest |
| `backend/app/tests/unit/test_video_runner.py` | 19 | VideoRunner state machine (in-memory fake repo + scriptable fake worker, spec-loaded): happy path claim→submit→poll→fetch→complete with token/duration accounting + shot rendering/rendered transitions; no-backend requeues (not fails); cancel before submit; cancel during poll (worker cancel + shot skipped); worker-reported failure fails without retry; transient submit error retries under cap / fails over cap; non-retryable submit fails immediately; tick() empty-queue False; tick() processes claimed job; run_forever reconciles then stops on cancel. **Reconciliation (no stuck jobs):** ground-truth poll of the worker for stale RENDERING jobs — completed→recover output, failed→fail, still-rendering→resume to completion, worker-lost→requeue under cap / fail over cap, wall-timeout→fail, pre-submit(no backend_job_id)→requeue, readopt-lost-to-peer→no-op |
| `backend/app/tests/unit/test_video_api.py` | 32 | /v1/videos routes (v1 text-to-video, spec-loaded with save/restore sys.modules hygiene): _job_to_dict status mapping (rendering→in_progress) + content_url gating; create_video gates — disabled 503, user-flag-off 403, missing-prompt 400, disallowed size/duration 400, bad quality 400, model-not-found 404, over-concurrency 429, over-storage-cap 507 (+under-cap proceeds, cap=0 disables); happy path returns 'queued' + persists + non-blocking; POST /videos/assets keyframe-upload gates (disabled 503, flag-off 403, non-image 400, over-cap 507); get_video 404 (no existence leak); cancel flags cancel; /videos/models capability shape (supports_image_to_video/keyframes TRUE — 2.8.42 stale-flag regression, max_shots 1); 2.8.42 seed+echo: omitted seed randomized+persisted (two submits differ), explicit seed passthrough, create + single-GET echo seed/seconds/asset ids (list stays lean); GET /content — 404 missing job, 409 not-ready, 404 file-missing, FileResponse stream (Range/206 via starlette) |
| `backend/app/tests/unit/test_diffusion_img2img.py` | 5 | img2img (reference-edit) request translation: DiffusionOutTranslator omits image/strength for txt2img, passes base64 reference list + strength for edits, image-without-strength case, empty image list stays txt2img (no key), CanonicalImageRequest image/strength default None |
| `backend/app/tests/unit/test_sso_providers.py` | 59 | Universal SSO framework (spec-loaded with stubbed db/settings, save/restore hygiene): settings gating (google/oidc/saml enabled properties, sso_enabled aggregate, SAML metadata-vs-explicit config), provider registry (labels incl. org_name inheritance, order, routes), OIDC driver (claims→profile, email_verified/hosted-domain rejection, redirect URI resolution, issuer normalization), CSRF state round-trip/tamper, shared JIT provisioning (subject lookup, email-linking keeps local password, new-user group+quota, username collision suffix), SAML eduPerson attribute mapping + NameID fallbacks + settings assembly, migration 068 contract, account_type covers generic SSO, **Azure driver parity**: keeps its own routes/azure_oid column, and its email-link guard matches the shared driver field-for-field (both refuse accounts already carrying azure_oid or sso_provider — admin-takeover regression); SAML request-id cookie must be SameSite=None+Secure (a Lax cookie is withheld on the IdP's cross-site POST, which would refuse every off-domain login), docker-compose env passthrough drift-guard, Dockerfile xmlsec/saml extra. **Security regressions (from adversarial review):** email-link refuses accounts already claimed by Azure or another provider (admin-takeover), email_verified string forms don't fail open, redirect URI from APP_BASE_URL not request.base_url, SAML handle_acs rejects unsolicited IdP-initiated POSTs (no AuthnRequest cookie) and responses whose InResponseTo doesn't echo it, with a control proving genuine SP-initiated logins still succeed (php-saml's `rejectUnsolicitedResponsesWithInResponseTo` is INERT in python3-saml — enforcement lives in the app), rejectDeprecatedAlgorithm on, metadata URL must be HTTPS, SAML host ignores X-Forwarded-Host. **2.9.6 external-deployment hardening:** JIT provisioning REFUSES when the provider's `*_DEFAULT_GROUP` names a missing group (users.group_id is NOT NULL, so passing None produced a bare HTTP 500 at the IdP callback) — no user, no quota created; id_token claims decoded as a fallback when userinfo is thin (ADFS returns only `sub`) with a total/never-raising decoder; `_email_is_verified` helper agrees with the profile rule (string 'false'/'0' must not fail open); token exchange retries with HTTP Basic client auth (Okta default) and logs response bodies; the three profile-rejection causes report distinct errors; **AST guard: no module bound to a STDLIB logger may use structlog-style kwargs** (logging.Logger.error(msg, key=...) raises TypeError at call time — caught the Azure missing-group guard crashing inside the very error path it was written to handle) plus a runtime execution of that guard's %-format log line. **2.9.7 SAML SP key pair:** no key material by default (and the two dependent security flags stay ABSENT so python3-saml cannot refuse the whole config with sp_cert_not_found_and_required); cert+key wired into the sp dict; flags enable only with a full pair and are dropped (logged) without one; PEM accepted inline, with \\n escapes, or as a file path; unreadable path and non-UTF-8 (DER/PKCS#12) files degrade instead of 500ing every SAML endpoint; **behavioral no-leak test** captures emitted log records for four non-PEM shapes incl. a base64-packed key (the earlier grep-based version missed a real `path=raw` private-key leak — verified this one fails when the leak is reintroduced); compose passthrough for all four vars |
| `backend/app/tests/unit/test_local_user_accounts.py` | 36 | Admin-created local users + account-type badges: User.account_type property (Admin > SSO > Local precedence, spec-loaded models.py with pollution-proof fresh-Base loader); POST /admin/users/create contract (full-admin gate, Argon2 hash, duplicate username/email checks, quota from group rpm_limit, user.create audit, password ≥8, username/email length caps, IntegrityError → friendly message + rollback, never reflects raw DB exception text into redirect/banner); crud.get_users account_type filter (admin-group subquery, azure_oid, groupless users); eager-load chains for api-keys/quota-request badges; top-users account_type; badge partial + filter dropdown + pagination + create modal in templates (all compile) |
| `backend/app/tests/unit/test_image_policy_edit.py` | 5 | Edit-aware content-policy judging: edit user-template carries the reference-image/anti-ambiguity note (plain template does not), evaluate_prompt forwards is_edit True/False to _call_judge, no-policy short-circuits PASS even for edits — regression for "put glasses on this man" being FAILED as ambiguous |
| `backend/app/tests/unit/test_bootstrap_paths.py` | 12 | First-boot / first-admin bootstrap (2.9.8), all new-deployment-only defects: `_run_migrations` must NOT be `@asynccontextmanager` (it is awaited in lifespan; the decorator returns a non-awaitable _AsyncGeneratorContextManager, so RUN_MIGRATIONS=1 raised TypeError and GUARANTEED the fresh-DB crash-loop it exists to prevent) with a pin proving the old shape was unawaitable, lifespan awaits it, compose passthrough present; seeder refuses an ADMIN_API_KEY lacking the `mr2_` prefix (verify_api_key rejects on prefix BEFORE lookup, so such a key would 401 forever) and imports API_KEY_PREFIX rather than hardcoding it; docs assert the SSO chicken-and-egg is written down (SSO can never create the first admin; email pre-linking is order-dependent; `*_DEFAULT_GROUP=admin` promotes EVERY user of that provider) plus fresh-DB migration ordering |
| `backend/app/tests/unit/test_session_guard.py` | 19 | Session deactivation guard (2.9.5, functional ASGI tests — module importable without db chain): signed-cookie decode (valid/tampered/wrong-secret/wrong-salt/absent), no-cookie zero-overhead passthrough, active user passes, inactive user → 302 to /login + cookie cleared (HTML) or 401 JSON + cookie cleared (API), exempt paths (/login /logout /static /auth /health) reachable while deactivated, undecodable cookie defers to route handling, non-http scopes untouched; source contracts: registered in main.py, fails OPEN on DB error, serializer salt/max-age matches dashboard routes |
| `backend/app/tests/unit/test_audit_capture.py` | 10 | Audit content-capture toggles (2.9.5, source contracts): prompt extraction + image offload inside the capture_prompts gate (audit_log_enabled AND audit_log_prompts), old unconditional extraction gone, BOTH create_response sites null content when responses capture off, settings comment states the DLP interaction, dead settings removed (CONVERSATION_RETENTION_DAYS/CLEANUP_INTERVAL, ARTIFACT_RETENTION_DAYS/MAX_SIZE_MB), startup warning when capture disabled, retention _DEFAULTS == migration-029 seeds, docs truthfulness (no dead vars documented, new admin endpoints + toggles w/ DLP note in both doc surfaces) |
| `backend/app/tests/unit/test_admin_user_mgmt.py` | 29 | Admin user-management hardening (2.9.4): revoke-key IDOR fix (crud.revoke_api_key owner/is_service scoping; self-service dashboard route passes owner_user_id + allow_service=False; admin dashboard revoke is admin-gated + audit-logged + open-redirect-guarded); admin API key endpoints (POST /api-keys/{id}/revoke, DELETE refuses referenced keys 409, reference counter covers requests/stored_responses/conversations/video_jobs); admin password reset (API + dashboard, local-accounts-only, min 8, never logs the password); **delete_user cascade completeness — schema-walk test: every FK to users.id in live metadata must be named in delete_user** (deleted or detached), preserved tables (BlogPost/EmailLog/AdminAuditLog) detached-not-deleted with nullable columns (migration 069), API delete returns 409 not 500 on IntegrityError and never echoes DB error text; manual purge (PURGE_CATEGORIES excludes admin_audit_log, server-side PURGE confirmation, shared retention lock, _detach_request_references wired into archive + no-archive request deletion); template affordances (danger zone hides self-targets, type-username delete modal, personal-key revoke buttons, purge verification modal) |

| `backend/app/tests/unit/test_runner_lease.py` | 4 | Video-runner leader lease (Redis CAS, fake redis): acquire is exclusive; only the token owner can renew; only the owner can release (then re-acquirable); unavailable Redis is a safe no-op — so only ONE runner is active across uvicorn workers/containers |
| `backend/app/tests/unit/test_redis_admission.py` | 14 | Redis-shared per-backend admission counters — fleet-wide max_concurrent at any worker count (fake redis; routing spec-imported with backend.app.db* pre-mocked): per-worker subkey `mr:adm:{backend}:{worker}` incr/decr symmetry, negative-count floor, TTL refreshed on every touch (dead worker's leaked slots self-heal), reconcile SET is absolute and deletes at zero, snapshot sums across workers with zero-fill and ignores foreign/malformed keys; fail-open (None/no-op) when Redis is unavailable or raising; route_job scores against the GLOBAL snapshot and falls back to this worker's local depths when the snapshot is None (today's semantics), slot claim and complete/fail release mirror into the shared counter (local dict always maintained); source contracts: GC eviction decrements, phantom-depth reset zeroes shared counters, 30s maintenance loop reconciles subkeys to actual local in-flight |

| `backend/app/tests/unit/test_branding.py` | 35 | UI branding service: hex validation/normalization (#abc→#aabbcc, invalid→default), color shade math (clamped), accessible-accent derivation (`_best_fg` white-unless-<3.0→black, `_accessible_ink` darken/lighten to ≥4.5:1), template-view builder (defaults, custom values, asset URL derivation, is_customized), traversal-safe on-disk asset save/resolve/delete (extension allow-lists per slot, favicon rejects webp), the email-logo slot (raster-only: rejects svg/webp; `read_email_logo` returns bytes+subtype for CID embed), and org-agnostic SSO wording (branding.org_name key/view/whitespace handling; login.html has no hardcoded institution, uses brand.org_name with generic fallbacks, "Sign in with a local account" toggle; login route SSO error org-agnostic; branding form persists org_name) |
| `backend/app/tests/unit/test_email_branding.py` | 9 | Branding applied to outgoing emails (aiosmtplib stubbed): the wrapper drops the old blue `#003DA5`, uses the gold accent rule + contrast-safe footer/link ink, embeds the raster email logo via `cid:brandlogo` (`multipart/related`) with app-name text fallback when unset; blog email title/button use ink + gold-fill-with-black-text; content with literal braces no longer crashes (old `.format()` bug); `_send_one` MIME structure verified for logo and no-logo paths |
| `backend/app/tests/unit/test_hotpath_trims.py` | 26 | Hot-path per-request trims (inference.py/policy.py spec-loaded, pure deps seeded from file — pollution-proof): /tokenize gate in cap_max_tokens — far-from-boundary requests (room ≥ 2048 and ≥ requested, or ≥ 4096 with no max_tokens) cap against the conservative tiktoken bound (est×1.3 + 16/msg + buffer) with NO backend /tokenize call; near-boundary, requested-exceeds-room, and auto_truncate still count exactly; Ollama never calls /tokenize; exact-count memoization on the request across retry attempts + force_exact invalidation; context-length-400 safety net in _proxy_with_retry (recount exact, one retry on the same backend; non-context 400s raise unchanged; recap at most once); scheduler estimate_tokens is chars//4 (no tiktoken import); Job.request_data field deleted (dataclass + policy source + built job); ollama.enforce_num_ctx 30s TTL cache (first call reads DB + caches, fresh cache skips DB, expired cache re-reads, value coerced to bool) |
| `backend/app/tests/unit/test_api_key_sha256.py` | 17 | SHA-256 fast-path API-key verification (spec-loaded api_keys.py, backend.app.db* pre-mocked): generate_api_key 4-tuple (digest matches key, Argon2 hash kept for rollback, token_urlsafe(32) entropy invariant pinned); fast path returns row on sha256 hit with NO prefix lookup and NO Argon2; belt-and-braces stored-digest mismatch rejected; Argon2 fallback verifies + backfills key_sha256 (verify-and-upgrade), wrong key rejected with no backfill; fallback bounded by Semaphore(4) + asyncio.to_thread (64 MiB/verify RSS cap); garbage keys rejected without DB/Argon2; revoked row still returned for the CALLER to reject (auth.py status/expiry/user-active checks proven to run after verify — fast path included); source contracts for migration 069, crud.get_api_key_by_sha256 (scalar_one_or_none on unique column) / create_api_key, models column, and all 3 generate_api_key callers storing both columns |

**Shared fixtures:** `backend/app/tests/conftest.py`

---

## 2. Integration Tests

**Runner:** `pytest backend/app/tests/integration/ -v`
**Makefile:** `make test-int`
**Requirements:** Live Ollama and vLLM backends (configure URLs in test constants).

| File | What it covers |
|------|----------------|
| `backend/app/tests/integration/test_live_backends.py` | Full translation pipeline against real Ollama and vLLM backends — streaming and non-streaming chat |
| `backend/app/tests/integration/test_rag_pipeline.py` | RAG pipeline: embedding, reranking, scoring endpoints through MindRouter proxy, end-to-end RAG test |
| `backend/app/tests/integration/test_structured_output_matrix.py` | Structured output matrix: all combos of API style (OpenAI/Ollama/Anthropic) × format (text/json_object/json_schema) × thinking mode × streaming across model categories |
| `backend/app/tests/integration/test_structured_outputs_live.py` | Live structured output: 5 models × 6 schema types × 3 API surfaces × 2 streaming modes + cross-engine routing (`--api-key`, `--base-url` CLI args) |

---

## 3. End-to-End Tests

**Runner:** `python tests/e2e_chat.py <args>`
**Makefile:** `make test-e2e`
**Requirements:** Live Docker stack, valid user credentials.

| File | What it covers |
|------|----------------|
| `tests/e2e_chat.py` | Chat subsystem: persistence, image preprocessing, storage, multi-turn context, vision model Q&A, cross-user isolation, CRUD operations |

**CLI arguments:**
```
--base-url       http://localhost:8000
--username       Primary user username
--password       Primary user password
--text-model     Text model ID (required, e.g. phi4:14b)
--vision-model   Vision model ID (e.g. qwen2.5-VL-32k:7b)
--username2      Second user for cross-user isolation tests
--password2      Second user password
--cookie-file    Session cookie file path
--cookie-file2   Second user cookie file
--skip-vision    Skip vision model tests
--docker-container  Container name for direct inspection
```

---

## 4. Smoke Tests (API)

**Runner:** `python test.py <args>`
**Makefile:** `make test-smoke`
**Requirements:** Live deployment, valid API key.

Exercises every API surface. Sections: `health`, `auth`, `openai`, `ollama`, `anthropic`, `cross`, `errors`, `admin`, `rerank`, `responses`.

| File | What it covers |
|------|----------------|
| `test.py` | Health endpoints, authentication, OpenAI-compatible API, Ollama-compatible API, Anthropic-compatible API, cross-engine routing, error handling, admin API, reranker (basic, top_n, return_documents), Responses API (non-streaming, typed-SSE streaming, function-call round trip, auth, store/retrieve/chain/delete, input_tokens count, Conversations lifecycle) |

**CLI arguments:**
```
--api-key        API key (required)
--base-url       http://localhost:8000
--admin-key      Admin API key (enables admin section)
--ollama-model   phi4:14b
--vllm-model     openai/gpt-oss-120b
--embedding-model EMBED/all-minilm:33m
--rerank-model   Qwen/Qwen3-Reranker-8B
--timeout        Request timeout in seconds (180)
--section        Run specific section(s) only
```

---

## 5. Stress / Load Tests

**Runner:** `python stress.py <args>`
**Makefile:** `make test-stress`
**Requirements:** Live deployment, admin API key for user provisioning.

| File | What it covers |
|------|----------------|
| `stress.py` | Multi-user concurrent load: fair-share WDRR scheduler, throughput, latency percentiles, error rates |

**CLI arguments:**
```
--api-key        Admin API key (required)
--base-url       http://localhost:8000
--duration       Test duration in seconds (300)
--concurrency    Concurrent request workers (10)
--users          Number of test users to provision (6)
--ollama-model   phi4:14b
--vllm-model     openai/gpt-oss-120b
--embedding-model EMBED/all-minilm:33m
--max-tokens     Max tokens per request (32)
--timeout        Per-request timeout in seconds (180)
--chat-only      Only send chat requests (no embeddings)
--verbose        Print individual request results
```

### Chat-capacity benchmark (`chat_bench.py`)

**Runner:** `python chat_bench.py --base-url <vllm-backend-or-gateway> --model <name> ...`
**Requirements:** Python 3.11+ with httpx; a host that can reach the target
(inference-node ports are firewalled from workstations — run it from the
mindrouter prod host's app container). No Makefile target on purpose: it is a
capacity experiment, not a CI check.

| File | What it covers |
|------|----------------|
| `chat_bench.py` | Simulated multi-turn chat users (think times, slot-templated prompts, 4-way prompt-length mixture with XL pastes, per-conversation salts) swept over user counts to find per-GPU chat capacity. Client-side TTFT/e2e/stream-tok/s/stall metrics with right-censoring at overload; per-stage vLLM /metrics deltas (queue, prefill, KV usage, spec-decode acceptance, prefix-cache hit rate) over exactly the measured window; configurable SLO gates; adaptive sweep extension + knee bisection; `--cache-adversarial` to remove cross-conversation prefix sharing (within-conversation history caching is inherent to chat and remains); `--no-think-time` for pure saturation mode. Outputs `turns.jsonl`, `server_samples.jsonl`, `summary.json`. |

Key flags: `--users 1,2,4,...` `--stage-duration 300` `--min-turns 30`
`--repeats K` `--no-adapt` `--max-users 512` `--mode direct|gateway`
`--metrics-url <vllm /metrics>` `--slo-ttft-p95 2.0` `--slo-tps-p10 15`
`--think off|on|none` (default off, matching the gateway fleet default).
See the module docstring for the full traffic-model and measurement notes.

---

## 6. Structured Output + Thinking Compliance Tests

**Runner:** `python tests/test_structured_thinking.py`
**Makefile:** `make test-thinking`
**Requirements:** Live deployment, `MINDROUTER_API_KEY` env var set.

Comprehensive matrix test covering structured output (JSON schema validation) and thinking/reasoning mode control across all 4 API surfaces, 6 models, and all reasoning modes (ON/OFF/low/medium/high/N/A). Runs 3 replicates per combination (144 total requests at 10 concurrency).

| File | What it covers |
|------|----------------|
| `tests/test_structured_thinking.py` | 12 model×reasoning combos × 4 endpoints × 3 replicates: JSON schema validation, thinking detection, thinking mode match (ON/OFF/effort levels) across OpenAI, Ollama chat, Ollama generate, and Anthropic endpoints |

**Models tested:** openai/gpt-oss-120b, gpt-oss-32k:120b, qwen/qwen3.5-400b, qwen3-32k:32b, qwen2.5-8k:7b, phi4:14b

---

## 7. Live Tool Calling Compliance Tests

**Runner:** `python tests/test_tool_calling_live.py`
**Makefile:** `make test-tools`
**Requirements:** Live deployment, `MINDROUTER_API_KEY` env var set.

Auto-discovers all tool-capable models via `/v1/models` (filtering by `capabilities.tools`), then tests each model across OpenAI, Ollama, and Anthropic API styles with both streaming and non-streaming requests. Validates that models return proper `tool_calls` with correct function names and parseable arguments.

| File | What it covers |
|------|----------------|
| `tests/test_tool_calling_live.py` | N models x 3 API styles x 2 stream modes: tool call generation, argument parsing, function name correctness, streaming tool call accumulation across OpenAI `/v1/chat/completions`, Ollama `/api/chat`, and Anthropic `/anthropic/v1/messages` |

**Models tested:** All models with `capabilities.tools == true` (auto-discovered, excludes embeddings/rerankers)

---

## 8. Accessibility Tests

**Runner:** `pytest backend/app/tests/unit/test_accessibility.py -v`
**Makefile:** `make test-a11y`
**Requirements:** None (parses template files directly).

Subset of unit tests, broken out for convenience. 117 tests validating WCAG 2.1 Level A and AA compliance across all Jinja2 HTML templates (including the Video tab and admin Video config, sidebar include, user detail, groups, API keys, and data retention).

---

## 9. GPU Sidecar Tests

**Runner:** `pytest sidecar/tests/ -v`
**Makefile:** `make test-sidecar`
**Requirements:** None (mocks pynvml).

| File | What it covers |
|------|----------------|
| `sidecar/tests/test_gpu_agent.py` | GPU agent unit tests: pynvml mocking, GPU info, auth, health |
| `sidecar/tests/test_gpu_agent_stress.py` | 60-second concurrent auth stress test for sidecar |

---

## 9b. Video Worker Service Tests

**Runner:** `cd video-worker && pytest tests/ -q`
**Requirements:** `video-worker/requirements.txt` only (fastapi/uvicorn/httpx/pytest). Runs in **mock mode — no GPU, no torch, no ltx_pipelines**; the worker's GPU deps are a separate venv on the H200 node and are NOT needed for these tests. Do not fold this into `make test-unit` (separate venv, own deps).

| File | What it covers |
|------|----------------|
| `video-worker/tests/test_worker.py` | Video worker async contract (10 tests, FastAPI TestClient, mock engine): capabilities/models/version; submit→poll→completed→fetch full lifecycle; content 409 before complete; **/health responsive (<1s) while a render occupies the executor** (off-event-loop invariant); Range request → 206 + Accept-Ranges; disallowed size/duration/missing-prompt → 400; unknown job → 404 (poll + cancel); cancel a still-queued job |

---

## 10. Security / Vulnerability Tests

> **Status:** Not yet implemented.

Planned coverage:
- [ ] SQL injection on API endpoints
- [ ] XSS in dashboard templates
- [ ] CSRF token validation
- [ ] API key brute-force rate limiting
- [ ] Header injection
- [ ] Path traversal via file upload
- [ ] Auth bypass / privilege escalation
- [ ] Dependency vulnerability scan (`pip-audit`)

---

## 11. vLLM MTP Speculative-Decoding Benchmarks (ops — on GPU nodes)

> **Not part of the repo test suite.** Manual throughput/latency benchmark run on the
> vLLM GPU hosts; harness lives under `/data/vllm/` (lynx) or `/zdata/data/vllm/`
> (aspen NFS), not in this repo. No Makefile target.

**Harness:** `vllm_bench_spec.py URL MODEL LABEL [conc_csv]` — concurrency sweep
(default `1,2,4,8,16`), temp=0, `ignore_eos`, 256 out-tokens, ~650-tok prompt; reports
median decode tok/s **per stream** (single-stream latency) and **aggregate** tok/s (server
throughput). `launch-gpu3-qwen-bench.sh <num_spec>` boots a serve on a non-prod port
(`0`=baseline, `N`=MTP depth); `run-qwen-full.sh` drives the baseline→mtp sweep with
orphan-safe GPU teardown. Enable built-in MTP via
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`.

**qwen3.6-27b MTP depth sweep** (2026-07-20, lynx-gpu3 H200, vLLM 0.25.1, gpu-mem 0.95, 16K ctx, exclusive GPU):

| depth | conc1 single-stream (tok/s) | conc1 AGG (×base) | conc16 AGG | acceptance |
|-------|-----------------------------|-------------------|-----------|------------|
| baseline | 87 | 85 (1.00×) | 1003 | — |
| mtp-1 | 131 | 127 (1.49×) | 1310 | len 1.93 / 93% |
| mtp-2 | 179 | 171 (2.01×) | 1623 | len 2.68 / 84% |
| **mtp-3** | **211** | **200 (2.35×)** | **1729** | len 3.18 / 73% |
| mtp-4 | 220 | 208 (2.45×) | 1692 | len 3.69 / 67% |

**Finding:** MTP is monotonically faster at every concurrency (1→16) through depth-3, then
plateaus (mtp-4 ≈ mtp-3, worse per-token efficiency). No regression. **mtp-3 is the deployed
optimum** on both qwen3.6-27b backends (lynx gpu1/gpu3). Caveat: an idle stray vLLM unit
sharing the GPU poisoned an earlier run — always bench on an **exclusive** GPU.

**qwen3.5-122b MTP depth sweep** (2026-07-21, aspen1-gpu0 H200, vLLM 0.25.1, 16K ctx, max-num-seqs 8):

| depth | mean accept length (tok / target-forward) | avg accept | AGG conc8 |
|-------|------------------------------------------|-----------|-----------|
| 0.23 baseline | — | — | 673 |
| mtp-1 | 1.83 | 82.8% | 921 |
| **mtp-2** | **2.43** | 71.3% | **1170** |
| mtp-3 | 1.88 | **29.4%** | 669 |

**Finding — optimal depth is per-model, always sweep it:** the 122b's MTP head is healthy at
depth 1–2 (83%/71%) but **collapses at depth 3 (29%)**, where it drafts 7,104 tokens to accept
2,086 and yields *no more* tokens/forward than mtp-1 — landing back at the 0.23 baseline. **mtp-2
is the deployed optimum** for the 122b, while qwen3.6-27b/35b hold 73–85% at depth 3 and use mtp-3.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | pytest paths, asyncio mode, markers, coverage config |
| `backend/app/tests/conftest.py` | Shared fixtures: mock settings, backends, users, API keys, streaming data |
| `Makefile` | All `make test-*` targets |

---

## Adding New Tests

When you create a new test file:

1. **Add the file** to the appropriate section in this manifest.
2. **Update the test count** in the table.
3. If it's a new *category*, add a Makefile target and a row to the Quick Reference table.
4. Ensure `conftest.py` has any new shared fixtures.
