"""Tests for ``mars_rec.utils.download.download_file`` against a local HTTP server (no network)."""

from __future__ import annotations

import http.server
import threading
import urllib.error

import pytest

from mars_rec.utils.download import download_file

BODY = b"user_id,item_id\n" + b"u1,i1\n" * 5000


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
        elif self.path == "/truncated":
            # Announce more bytes than are sent, then close the connection.
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY) + 100))
            self.end_headers()
            self.wfile.write(BODY)
            self.close_connection = True
        else:
            self.send_error(404)

    def log_message(self, *args):  # keep pytest output quiet
        pass


@pytest.fixture(scope="module")
def server_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir())


def test_downloads_body_and_creates_parent_dirs(tmp_path, server_url):
    out = tmp_path / "nested" / "dir" / "data.csv"
    assert download_file(f"{server_url}/ok", str(out), show_progress=False) == str(out)
    assert out.read_bytes() == BODY
    assert _leftovers(out.parent) == ["data.csv"]  # temp file was renamed, not left behind


def test_http_error_raises_and_leaves_no_file(tmp_path, server_url):
    out = tmp_path / "missing.csv"
    with pytest.raises(urllib.error.HTTPError) as exc:
        download_file(f"{server_url}/nope", str(out), show_progress=False)
    assert exc.value.code == 404
    assert _leftovers(tmp_path) == []


def test_truncated_body_raises_and_leaves_no_partial_file(tmp_path, server_url):
    out = tmp_path / "data.zip"
    with pytest.raises(ConnectionError, match="Incomplete download"):
        download_file(f"{server_url}/truncated", str(out), show_progress=False)
    assert _leftovers(tmp_path) == []


def test_existing_file_is_only_replaced_on_success(tmp_path, server_url):
    out = tmp_path / "data.csv"
    out.write_bytes(b"previous")
    with pytest.raises(urllib.error.HTTPError):
        download_file(f"{server_url}/nope", str(out), show_progress=False)
    assert out.read_bytes() == b"previous"
    download_file(f"{server_url}/ok", str(out), show_progress=False)
    assert out.read_bytes() == BODY


def test_tls_context_verifies_with_a_ca_bundle(monkeypatch):
    """HTTPS must not depend on the interpreter having a system CA store."""
    import ssl

    from mars_rec.utils import download

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    ctx = download._ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] > 0


def test_ssl_cert_file_overrides_the_bundle(monkeypatch, tmp_path):
    import certifi

    from mars_rec.utils import download

    custom = tmp_path / "ca.pem"
    custom.write_text(
        open(certifi.where()).read().split("-----END CERTIFICATE-----")[0] + "-----END CERTIFICATE-----\n"
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(custom))
    assert download._ssl_context().cert_store_stats()["x509_ca"] == 1


def test_downloaded_file_gets_regular_permissions(tmp_path, server_url):
    """mkstemp creates 0600 files; the result must look like a normal open() did it."""
    import os

    out = tmp_path / "data.csv"
    download_file(f"{server_url}/ok", str(out), show_progress=False)
    reference = tmp_path / "reference"
    reference.write_bytes(b"")
    assert oct(os.stat(out).st_mode & 0o777) == oct(os.stat(reference).st_mode & 0o777)
