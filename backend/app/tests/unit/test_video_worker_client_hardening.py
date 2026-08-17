############################################################
#
# mindrouter - unit tests for video worker client hardening
#
# Covers WS9 fixes: outbound TLS verification defaults on (F20/F56) and the
# streamed artifact fetch is bounded by a cumulative byte cap (F52).
#
############################################################

"""Unit tests for HttpVideoWorkerClient TLS + fetch-cap hardening."""

import pytest

import backend.app.services.video_worker_client as vwc


class _FakeStreamResp:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, chunks, captured_kwargs):
        self._chunks = chunks
        self._captured = captured_kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        return _FakeStreamResp(self._chunks)


def _install_fake_client(monkeypatch, chunks):
    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return _FakeClient(chunks, captured)

    monkeypatch.setattr(vwc.httpx, "AsyncClient", factory)
    return captured


def test_tls_verify_defaults_true():
    # No `internal_tls_verify` setting configured -> verification ON.
    assert vwc._tls_verify() is True


def test_fetch_max_bytes_positive_default():
    assert vwc._fetch_max_bytes() > 0


@pytest.mark.asyncio
async def test_fetch_under_cap_writes_full_file(monkeypatch, tmp_path):
    monkeypatch.setattr(vwc, "_fetch_max_bytes", lambda: 1000)
    captured = _install_fake_client(monkeypatch, [b"aaaa", b"bbbb"])
    dest = tmp_path / "out.mp4"

    result = await vwc.HttpVideoWorkerClient().fetch("http://worker", "job1", str(dest))

    assert result.size_bytes == 8
    assert dest.read_bytes() == b"aaaabbbb"
    # TLS verification is enabled on the fetch client.
    assert captured.get("verify") is True


@pytest.mark.asyncio
async def test_fetch_aborts_and_cleans_up_when_over_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(vwc, "_fetch_max_bytes", lambda: 10)
    _install_fake_client(monkeypatch, [b"x" * 6, b"y" * 6])  # 12 > 10
    dest = tmp_path / "out.mp4"

    with pytest.raises(vwc.WorkerFetchError):
        await vwc.HttpVideoWorkerClient().fetch("http://worker", "job1", str(dest))

    # Partial artifact is removed rather than left on disk.
    assert not dest.exists()


@pytest.mark.asyncio
async def test_fetch_cap_disabled_allows_large_stream(monkeypatch, tmp_path):
    monkeypatch.setattr(vwc, "_fetch_max_bytes", lambda: 0)  # disabled
    _install_fake_client(monkeypatch, [b"z" * 100])
    dest = tmp_path / "out.mp4"

    result = await vwc.HttpVideoWorkerClient().fetch("http://worker", "job1", str(dest))
    assert result.size_bytes == 100
