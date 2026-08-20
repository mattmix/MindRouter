############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_service/__init__.py: Standalone GPU DLP microservice.
#
# A dedicated-node GLiNER PII scanner with dynamic GPU
# batching. Fully standalone (no backend.app.* imports);
# gliner + torch are imported lazily so it runs and unit-tests
# on a CPU-only Mac with neither installed.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Standalone GPU DLP microservice (GLiNER + dynamic batching)."""

from .config import ServiceConfig
from .server import create_app

__version__ = "0.1.0"

__all__ = ["ServiceConfig", "create_app", "__version__"]
