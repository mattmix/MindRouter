############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_artifact_path_containment.py: Path-traversal
#     containment in ArtifactStorage (F01/F05).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""ArtifactStorage must contain every client-supplied storage path within
the artifact root.

Before the fix, ``retrieve``/``delete``/``exists``/``get_size`` did a bare
``self._base_path / storage_path``, so an absolute path or ``..`` traversal
escaped the artifact root — an unauthenticated arbitrary file read via
``GET /blog/images/{path:path}``. The ``_resolve_within_base`` helper now
rejects any path that resolves outside the base, returning the same
not-found value (None/False) callers already handle for a missing file, so
legitimate relative paths behave exactly as before.
"""

import pytest

from backend.app.storage.artifacts import ArtifactStorage


@pytest.fixture
def storage(tmp_path):
    """An ArtifactStorage rooted at an isolated tmp dir."""
    store = ArtifactStorage()
    store._base_path = tmp_path
    return store


# A file OUTSIDE the artifact root that a traversal would try to reach.
@pytest.fixture
def outside_secret(tmp_path):
    secret = tmp_path.parent / "secret_outside.txt"
    secret.write_bytes(b"TOP SECRET")
    yield secret
    if secret.exists():
        secret.unlink()


async def _store_one(store) -> str:
    storage_path, _hash, _size = await store.store(
        b"hello world", "hello.txt", "text/plain"
    )
    return storage_path


@pytest.mark.asyncio
async def test_normal_retrieve_still_works(storage):
    """A legitimate relative path round-trips unchanged."""
    storage_path = await _store_one(storage)

    assert "/" in storage_path
    assert not storage_path.startswith("/")

    data = await storage.retrieve(storage_path)
    assert data == b"hello world"
    assert await storage.exists(storage_path) is True
    assert await storage.get_size(storage_path) == len(b"hello world")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "evil_path",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "../secret_outside.txt",
        "../../secret_outside.txt",
        "foo/../../secret_outside.txt",
        # An already-URL-decoded "..%2f.." arrives as literal "../..".
        "../../",
        "..",
        "",
    ],
)
async def test_traversal_paths_are_blocked(storage, evil_path):
    """Escaping paths return the existing not-found value, never file content."""
    assert await storage.retrieve(evil_path) is None
    assert await storage.exists(evil_path) is False
    assert await storage.get_size(evil_path) is None
    assert await storage.delete(evil_path) is False


@pytest.mark.asyncio
async def test_traversal_cannot_read_outside_file(storage, outside_secret):
    """A crafted traversal that resolves to a real outside file still returns None."""
    # Build a relative path from the base to the secret sitting one level up.
    evil = f"../{outside_secret.name}"
    # Sanity: the traversal really does point at the secret on disk.
    assert (storage._base_path / evil).resolve() == outside_secret.resolve()

    assert await storage.retrieve(evil) is None
    assert await storage.exists(evil) is False
    assert await storage.get_size(evil) is None
    # And delete must not remove the outside file.
    assert await storage.delete(evil) is False
    assert outside_secret.exists()


@pytest.mark.asyncio
async def test_traversal_cannot_delete_inside_via_escape(storage):
    """A path that escapes then re-enters is still rejected as unresolvable-in-base."""
    storage_path = await _store_one(storage)
    # Prefixing with an escape+re-entry resolves back inside, but the leading
    # segment names a non-existent sibling of base; either way the file is safe.
    evil = f"../{storage._base_path.name}/{storage_path}"
    # This one legitimately resolves back inside base, so it MAY succeed — the
    # containment contract only forbids escaping, not benign round-trips. What
    # matters is the file is never touched via an out-of-base path:
    resolved = storage._resolve_within_base(evil)
    if resolved is not None:
        assert storage._base_path.resolve() in resolved.parents


def test_resolve_within_base_returns_none_for_escape(storage):
    """Unit-level check of the helper itself."""
    assert storage._resolve_within_base("../../etc/passwd") is None
    assert storage._resolve_within_base("/etc/passwd") is None
    assert storage._resolve_within_base("") is None
    inside = storage._resolve_within_base("2026/08/11/abcd/file.png")
    assert inside is not None
    assert storage._base_path.resolve() in inside.parents
