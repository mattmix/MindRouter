FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    default-libmysqlclient-dev \
    libxmlsec1-dev \
    libxmlsec1-openssl \
    poppler-utils \
    libreoffice-core \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    && rm -rf /var/lib/apt/lists/*

# Create app user with explicit UID for predictable bind mount permissions
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY backend/__init__.py backend/
COPY backend/app/__init__.py backend/app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .[saml]

# TrustMark invisible image watermarking (Adobe, MIT). Installed --no-deps:
# its numpy<2 pin conflicts with this image's numpy 2.x and is empirically
# unnecessary (encode/decode roundtrip verified on numpy 2.4); its real
# runtime deps (torchvision/einops/omegaconf/lightning) are in pyproject.
# The bake downloads the Q-model weights (MD5-verified) into the image so
# nothing is fetched at runtime — and PROVES they work with an actual
# encode/decode roundtrip, because TrustMark swallows download failures
# and constructs a half-built model instead of raising: without the
# roundtrip a CDN outage would produce an image that can never watermark.
RUN pip install --no-cache-dir --no-deps trustmark==0.9.1 && \
    python -c "\
from trustmark import TrustMark; \
from PIL import Image; \
tm = TrustMark(verbose=True, model_type='Q', loadRemover=False, device='cpu'); \
assert tm.encoder is not None and tm.decoder is not None, 'TrustMark weights missing'; \
s, ok, _ = tm.decode(tm.encode(Image.new('RGB', (256, 256), (128, 64, 32)), 'BAKETEST')); \
assert ok and s == 'BAKETEST', f'TrustMark roundtrip failed: {ok} {s!r}'; \
print('TrustMark bake verified')"

# Copy application code
COPY backend/ backend/
COPY scripts/ scripts/
COPY alembic.ini ./

# Create directories
RUN mkdir -p /data/artifacts /data/chat_files /data/branding /var/log/mindrouter && \
    chown -R appuser:appuser /app /data /var/log/mindrouter

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "60"]
