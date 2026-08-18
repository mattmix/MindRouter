"""Unit tests for backend.app.core.pathsafe (path-traversal containment).

pathsafe imports only os/re (no db chain), so a direct import is fine.
"""

import os

import pytest

from backend.app.core.pathsafe import PathEscapeError, resolve_under, safe_key


# ── resolve_under ──────────────────────────────────────────────────────────

def test_accepts_file_directly_under_root(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    f = root / "a.mp4"
    f.write_bytes(b"x")
    assert resolve_under(str(root), str(f)) == os.path.realpath(str(f))


def test_accepts_nested_and_sharded_path(tmp_path):
    root = tmp_path / "chat_files"
    (root / "457").mkdir(parents=True)
    f = root / "457" / "457.jpg"
    f.write_bytes(b"x")
    assert resolve_under(str(root), str(f)) == os.path.realpath(str(f))


def test_accepts_relative_candidate_joined_under_root(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    assert resolve_under(str(root), "sub/x.txt") == os.path.join(os.path.realpath(str(root)), "sub", "x.txt")


def test_accepts_root_itself(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    assert resolve_under(str(root), str(root)) == os.path.realpath(str(root))


def test_accepts_missing_file_under_root(tmp_path):
    # realpath does not require existence; a stored path whose file was deleted
    # still resolves (the caller's open/FileResponse then 404s).
    root = tmp_path / "store"
    root.mkdir()
    assert resolve_under(str(root), str(root / "gone.mp4")) == os.path.join(os.path.realpath(str(root)), "gone.mp4")


def test_rejects_parent_traversal(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    with pytest.raises(PathEscapeError):
        resolve_under(str(root), str(root / ".." / "etc" / "passwd"))


def test_rejects_absolute_escape(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    with pytest.raises(PathEscapeError):
        resolve_under(str(root), "/etc/passwd")


def test_rejects_prefix_sibling(tmp_path):
    # A sibling dir sharing the root's name prefix must NOT be accepted
    # (/data/video-evil for a /data/video root).
    root = tmp_path / "video"
    root.mkdir()
    sibling = tmp_path / "video-evil"
    sibling.mkdir()
    (sibling / "x").write_bytes(b"x")
    with pytest.raises(PathEscapeError):
        resolve_under(str(root), str(sibling / "x"))


def test_rejects_symlink_escape(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"top secret")
    link = root / "link.txt"
    try:
        os.symlink(str(secret), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    # realpath resolves the symlink to /outside/secret.txt -> escapes.
    with pytest.raises(PathEscapeError):
        resolve_under(str(root), str(link))


def test_rejects_backslash_traversal_on_posix(tmp_path):
    # On POSIX a backslash is a normal filename char, so "..\\x" is contained
    # (a single odd filename), NOT an escape. Assert it stays under root rather
    # than escaping — the containment is by realpath, not by string.
    root = tmp_path / "store"
    root.mkdir()
    got = resolve_under(str(root), "..\\etc")  # one weird filename on posix
    assert got.startswith(os.path.realpath(str(root)) + os.sep)


def test_none_candidate_raises_typeerror(tmp_path):
    # None is a caller bug (missing stored path) — surfaces as TypeError, not
    # PathEscapeError; callers guard for a falsy path first.
    with pytest.raises(TypeError):
        resolve_under(str(tmp_path), None)


# ── safe_key ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("good", ["msg_abc123", "resp_" + "a" * 32, "item-1.v2", "A_b.C-d", "x"])
def test_safe_key_accepts_plain_segments(good):
    assert safe_key(good) == good


@pytest.mark.parametrize(
    "bad",
    [
        "../conv_victim", "a/b", "a\\b", "..", ".", "", "  ", "with space",
        "a" * 129, "x/../y", "$(whoami)", "a\x00b", "café",
    ],
)
def test_safe_key_rejects_unsafe_segments(bad):
    with pytest.raises(PathEscapeError):
        safe_key(bad)


def test_safe_key_rejects_non_str():
    for v in (None, 123, ["a"], {"a": 1}):
        with pytest.raises(PathEscapeError):
            safe_key(v)
