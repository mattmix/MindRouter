############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_service/__main__.py: uvicorn entrypoint.
#
#   python -m dlp_service
#
# Reads configuration from the environment (see config.py /
# README.md), builds the FastAPI app, and serves it with
# uvicorn on DLP_SERVICE_HOST:DLP_SERVICE_PORT.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""uvicorn entrypoint for the DLP GPU microservice."""

from __future__ import annotations

import logging

from .config import ServiceConfig
from .server import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = ServiceConfig.from_env()
    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
