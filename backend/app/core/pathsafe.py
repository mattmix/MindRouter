############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# pathsafe.py: Path-traversal containment for filesystem sinks
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Path-traversal containment for filesystem sinks.

Several endpoints open, stream, or delete files whose path is built from a
value that ultimately came from a client, or from a DB row a client
influenced (a stored artifact path, a video id, a conversation item id, an
uploaded filename component). If any such value can climb out of its intended
storage root -- via ``..`` segments, an absolute path, or a symlink -- the
sink turns into an arbitrary-file read or delete.

:func:`resolve_under` is the one containment primitive: it resolves a
candidate against a storage root with ``os.path.realpath`` (which collapses
``..`` and follows symlinks) and confirms the result is the root itself or
lives beneath ``root + os.sep``. The trailing-separator test is deliberate --
a bare ``startswith(root)`` would accept a sibling like ``/data/video-evil``
for a ``/data/video`` root. On failure it raises :class:`PathEscapeError`,
which callers treat exactly like a missing file (skip / 404 / log, per site).

:func:`safe_key` is the stricter form for a value that is supposed to be a
single opaque path segment (e.g. a conversation item id): it rejects anything
carrying a separator or a traversal segment before it is ever joined into a
path.

Call sites must pass the RETURNED value -- never the raw input -- to the
open/FileResponse/os.remove.
"""

import os
import re

__all__ = ["PathEscapeError", "resolve_under", "safe_key"]


class PathEscapeError(ValueError):
    """A candidate path resolved outside its permitted storage root."""


def resolve_under(root, candidate) -> str:
    """Resolve ``candidate`` and confirm it is contained within ``root``.

    Returns the real, absolute path (symlinks resolved) when it is the root
    itself or lives beneath it; raises :class:`PathEscapeError` otherwise.

    ``candidate`` may be relative (joined onto ``root``) or absolute (taken
    as-is, then checked); either way the containment test is applied to the
    fully realpath-resolved result, so ``..`` and symlink escapes are caught.
    ``candidate`` must not be None -- callers treat a missing stored path as
    not-found before calling.
    """
    base = os.path.realpath(os.fspath(root))
    candidate = os.fspath(candidate)
    target = candidate if os.path.isabs(candidate) else os.path.join(base, candidate)
    full = os.path.realpath(target)
    if full == base or full.startswith(base + os.sep):
        return full
    raise PathEscapeError(
        f"path escapes storage root: {candidate!r} -> {full!r} not under {base!r}"
    )


_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def safe_key(segment: str) -> str:
    """Validate a single opaque path segment (no separators, no traversal).

    For a DB/user-supplied identifier that is meant to be exactly ONE path
    component before it is joined into a storage path (e.g. a conversation
    item id). Accepts only ``[A-Za-z0-9._-]`` (1..128 chars) and rejects the
    traversal segments ``.`` and ``..`` outright, so the value can neither
    introduce a path separator nor walk to a parent (or to the directory
    itself). Returns the segment unchanged on success; raises
    :class:`PathEscapeError` otherwise.
    """
    if not isinstance(segment, str) or segment in (".", "..") or not _SAFE_KEY_RE.match(segment):
        raise PathEscapeError(f"unsafe path segment: {segment!r}")
    return segment
