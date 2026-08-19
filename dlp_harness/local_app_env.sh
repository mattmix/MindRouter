# Environment for running the MindRouter app NATIVELY on this Mac for DLP
# harness work (no local image builds — disk). Source before uvicorn/alembic.
# Explicit exports override the stale repo-root .env that pydantic-settings reads.
export DATABASE_URL='mysql+pymysql://mindrouter:mindrouter_password@127.0.0.1:3306/mindrouter'
export ARCHIVE_DATABASE_URL='mysql+pymysql://mindrouter_archive:archive_password@127.0.0.1:3307/mindrouter_archive'
export REDIS_URL='redis://:mindrouter_dev_redis@127.0.0.1:6379/0'
# Random local secret, generated once, kept out of version control (data/ is
# gitignored). Not the dev placeholder — the app refuses that since 2.9.18.
if [ ! -f "$PWD/data/.local_secret_key" ]; then
  python3 -c "import secrets; print(secrets.token_urlsafe(48))" > "$PWD/data/.local_secret_key"
fi
export SECRET_KEY="$(cat "$PWD/data/.local_secret_key")"
export LOG_FILE="$PWD/data/logs/app.log"
export LOG_LEVEL=INFO
export DEBUG=false
export AUDIT_LOG_ENABLED=true
export AUDIT_LOG_PROMPTS=true
export AUDIT_LOG_RESPONSES=true
export ARTIFACT_STORAGE_PATH="$PWD/data/artifacts"
export CHAT_FILES_PATH="$PWD/data/chat_files"
export VIDEO_STORAGE_PATH="$PWD/data/video"
export SESSION_COOKIE_SECURE=false
export ENABLE_API_DOCS=true
export RATE_LIMIT_LOCAL_FALLBACK=true
