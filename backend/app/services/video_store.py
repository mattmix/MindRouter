############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video_store.py: On-disk layout for rendered video artifacts.
#
# Range-capable streaming delivery is added with the content endpoint;
# for now this just owns the path layout under settings.video_storage_path.
# See docs/video-generation-plan.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Filesystem layout for generated video artifacts."""

import os

from backend.app.core.pathsafe import resolve_under


def job_output_path(storage_root: str, user_id: int, job_uuid: str) -> str:
    """Absolute path for a job's final MP4, creating the parent dir.

    Layout: <root>/<user_id>/<job_uuid>.mp4 — sharded by user so retention and
    per-user storage accounting are simple directory walks.

    Backstop: assert the constructed path is contained under ``storage_root``
    (always true for the server-generated user_id/uuid, but keeps a bad caller
    from ever composing an escaping write path). Returns the original joined
    path, not the realpath, so stored values stay stable.
    """
    user_dir = os.path.join(storage_root, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    out = os.path.join(user_dir, f"{job_uuid}.mp4")
    resolve_under(storage_root, out)  # raises PathEscapeError if it escapes root
    return out
