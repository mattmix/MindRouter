# gateway-perf-4x Deploy Notes — MIGRATION MUST RUN BEFORE THE NEW CODE

> **STOP: the standard "deploy, then `alembic upgrade head`" habit (deploy/DEPLOYMENT.md
> Steps 8–9) MUST NOT be followed for this release.** Migration 069 adds
> `api_keys.key_sha256`, and the new code's ORM includes that column in **every**
> `ApiKey` SELECT — the fast path *and* the legacy prefix fallback. New code against a
> pre-069 schema fails every API-key lookup with MariaDB error 1054
> (`Unknown column 'api_keys.key_sha256' in 'field list'`), which 500s **all**
> API-key-authenticated traffic (`/v1/*`, `/api/*`, admin API, MCP, dashboard key
> paths) while `/healthz` stays green — the container looks healthy, nothing
> auto-rolls-back, and the outage lasts until the migration is applied.
>
> The reverse order is safe: 069 is a single nullable column + unique index (no FK),
> the running 2.9.x ORM neither selects nor writes it, and MariaDB unique indexes
> allow multiple NULLs, so keys minted by old code during the window cannot violate
> the index.

## Step-by-step (prod, `/opt/mindrouter` on mindrouter.uidaho.edu)

The old running container cannot apply the migration — the 2.9.x image does not
contain the 069 migration file. Run it from the **new** image via a one-off
container while the old code keeps serving.

```bash
ssh mindrouter@mindrouter.uidaho.edu
cd /opt/mindrouter

# 1. Fetch the new code and build the new image WITHOUT restarting the app.
git pull
docker compose -f docker-compose.prod.yml build app

# 2. Apply migration 069 from a one-off container running the NEW image
#    while the old code keeps serving (safe: nullable column + unique index,
#    no FK, ignored by the running old ORM).
docker compose -f docker-compose.prod.yml run --rm app alembic upgrade head

# 3. Verify the schema before swapping traffic.
docker exec mindrouter-mariadb-1 mariadb -u mindrouter -p<pw> mindrouter \
  -e "SELECT version_num FROM alembic_version; SHOW COLUMNS FROM api_keys LIKE 'key_sha256';"
# Expect: version_num = 069, and one key_sha256 VARCHAR(64) NULL row.

# 4. Only then swap to the new code.
docker compose -f docker-compose.prod.yml up -d

# 5. Verify the running version and auth (smoke key must authenticate).
docker compose -f docker-compose.prod.yml logs app | tail -20
python test.py --api-key <smoke-key> --base-url https://mindrouter.uidaho.edu
```

On a host using the dev stack (`docker-compose.yml`), the sequence is identical with
`-f docker-compose.yml` (or bare `docker compose`).

## Post-deploy: restore fleet-wide max_concurrent values

The Redis-shared admission counters make `max_concurrent` a **fleet-wide** cap.
Before this release the cap was enforced per worker (2 uvicorn workers), so
backend caps had been divided down to compensate; those divided values must be
restored to full backend capacity or the fleet runs at roughly half throughput.

Via the admin API (goes through the registry — no restart needed):

```bash
# gemma backends: restore max_concurrent=72
for id in 14 17 42; do
  curl -X PATCH "https://mindrouter.uidaho.edu/api/admin/backends/${id}" \
    -H "X-API-Key: <admin-key>" \
    -H "Content-Type: application/json" \
    -d '{"max_concurrent": 72}'
done

# All other backends: restore each one's pre-division value
# (its full capacity = ~75% of the engine's --max-num-seqs / OLLAMA_NUM_PARALLEL,
# per the standing concurrency-buffer rule). List current values first:
curl -s "https://mindrouter.uidaho.edu/api/admin/backends" -H "X-API-Key: <admin-key>"
```

## Rollback

`alembic downgrade 068` after reverting the code (never before — old order applies
symmetrically: downgrade the schema only once no running code selects the column).
The Argon2 `key_hash` column is untouched by 069, so a code rollback alone (schema
left at 069) is also safe — old code simply ignores `key_sha256`.

## Release-note flag: auth tightening (intentional)

This release also closes three pre-existing post-verify gaps by centralizing the
status/expiry/user checks (`api_key_rejection_reason` in `security/api_keys.py`):

- `/mcp/sse` now rejects **revoked and expired** keys (previously accepted).
- The admin `*_or_session` wrappers now reject **expired** admin keys.
- The dashboard `tts-voices` endpoint now rejects **expired** keys.

Long-expired integrations that silently kept working will start receiving 401s —
this is intended behavior; flag it in the release notes.
