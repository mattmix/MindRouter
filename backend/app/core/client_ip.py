############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# client_ip.py: Resolve the caller's address behind a proxy
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Where a request actually came from.

MindRouter runs behind nginx in production, so `request.client.host` is the
proxy — identical for every caller. Audit rows and rate limits that record it
are recording nothing. This lives in its own module rather than in
`dashboard/routes.py` so the API layer can share the one definition instead of
growing a second, drifting copy.
"""

from typing import Optional

from fastapi import Request


def get_client_ip(request: Request) -> Optional[str]:
    """Extract the real client IP from proxy headers, falling back to direct connection.

    Checks X-Forwarded-For (last entry) → X-Real-IP → request.client.host.

    The X-Forwarded-For header is a comma-separated chain that grows
    left-to-right as a request traverses proxies, and the *client* controls
    the leftmost entries by simply sending its own header. The bundled nginx
    reverse proxy appends the address it actually observed to the end of the
    chain, so the LAST entry is the closest-proxy-observed client and the only
    value not forgeable by the caller. Taking the first entry (the historical
    behaviour) let any caller spoof its recorded IP in audit rows and evade
    per-IP rate limits.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None
