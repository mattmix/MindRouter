############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_url_guard.py: Unit tests for the image_url SSRF guard
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the SSRF guard on user-supplied image URLs (F02/F08)."""

import socket

import pytest

from backend.app.core import url_guard
from backend.app.core.url_guard import image_url_reason, is_safe_image_url


def _fake_getaddrinfo(mapping):
    """Return a getaddrinfo stub resolving hosts per ``mapping`` (host->ip)."""

    def _inner(host, *args, **kwargs):
        if host in mapping:
            ip = mapping[host]
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 0))]
        raise socket.gaierror(f"no fake DNS for {host}")

    return _inner


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """By default, resolve example.com to a public address."""
    monkeypatch.setattr(
        url_guard.socket,
        "getaddrinfo",
        _fake_getaddrinfo({"example.com": "93.184.216.34"}),
    )


class TestImageUrlGuard:
    def test_data_uri_allowed(self):
        assert is_safe_image_url("data:image/png;base64,AAAA") is True

    def test_public_https_allowed(self):
        assert is_safe_image_url("https://example.com/cat.png") is True

    def test_public_http_allowed(self):
        assert is_safe_image_url("http://example.com/cat.png") is True

    def test_metadata_ip_blocked(self):
        ok, reason = image_url_reason("http://169.254.169.254/latest/meta-data/")
        assert ok is False
        assert reason

    def test_private_ip_literal_blocked(self):
        ok, _ = image_url_reason("http://10.0.0.5/x.png")
        assert ok is False

    def test_localhost_blocked(self, monkeypatch):
        monkeypatch.setattr(
            url_guard.socket,
            "getaddrinfo",
            _fake_getaddrinfo({"localhost": "127.0.0.1"}),
        )
        ok, _ = image_url_reason("http://localhost/x.png")
        assert ok is False

    def test_loopback_literal_blocked(self):
        assert is_safe_image_url("http://127.0.0.1/x.png") is False

    def test_ftp_scheme_blocked(self):
        ok, _ = image_url_reason("ftp://example.com/x.png")
        assert ok is False

    def test_file_scheme_blocked(self):
        assert is_safe_image_url("file:///etc/passwd") is False

    def test_empty_url_blocked(self):
        assert is_safe_image_url("") is False

    def test_dns_failure_fails_closed(self, monkeypatch):
        monkeypatch.setattr(
            url_guard.socket,
            "getaddrinfo",
            _fake_getaddrinfo({}),  # nothing resolves
        )
        ok, _ = image_url_reason("https://nonexistent.invalid/x.png")
        assert ok is False

    def test_rebinding_any_private_ip_blocked(self, monkeypatch):
        # A host that resolves to BOTH a public and a private address must be
        # rejected — the backend might connect to the private one.
        def _multi(host, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 ("10.0.0.9", 0)),
            ]

        monkeypatch.setattr(url_guard.socket, "getaddrinfo", _multi)
        ok, _ = image_url_reason("https://sneaky.example/x.png")
        assert ok is False
