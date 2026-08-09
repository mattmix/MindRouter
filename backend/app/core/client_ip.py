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

    Checks X-Forwarded-For (first entry) → X-Real-IP → request.client.host.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None
