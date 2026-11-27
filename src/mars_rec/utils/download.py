"""Stdlib-only file download with a progress bar.

Shared by the data-download tasks in ``mars_rec.data`` and by the
example projects, so none of them needs an HTTP client dependency.
"""

from __future__ import annotations

import http.client
import os
import shutil
import ssl
import tempfile
import urllib.request
from typing import Optional

import certifi
from tqdm import tqdm

_CHUNK_SIZE = 1024 * 1024
_USER_AGENT = "mars-rec"


def _ssl_context() -> ssl.SSLContext:
    """TLS context that verifies against certifi's CA bundle.

    Many Python builds (python.org macOS installers, uv / python-build-standalone,
    some conda setups) ship without access to a system CA store, so urllib's
    default context fails with ``CERTIFICATE_VERIFY_FAILED``. Using certifi
    matches what ``requests`` did before it was dropped. ``SSL_CERT_FILE``
    still overrides it, e.g. for a corporate CA.
    """
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def download_file(url: str, output_path: str, timeout: Optional[float] = 60.0, show_progress: bool = True) -> str:
    """Download ``url`` to ``output_path`` and return ``output_path``.

    The body is streamed to a temporary file in the destination directory
    and only renamed to ``output_path`` once it is complete, so an
    interrupted download never leaves a partial file that a later
    ``os.path.exists`` check would mistake for a finished one.

    Args:
        url: Source URL (``http(s)://``, or anything :mod:`urllib` opens).
        output_path: Destination file. Missing parent directories are created.
        timeout: Socket timeout in seconds for connecting and for each read.
        show_progress: Display a ``tqdm`` progress bar.

    Returns:
        ``output_path``.

    Raises:
        urllib.error.HTTPError: The server answered with an HTTP error status.
        urllib.error.URLError: The server could not be reached.
        ConnectionError: Fewer bytes arrived than the ``Content-Length`` header announced.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".download-", suffix=".part")
    try:
        # fdopen first so the temp file descriptor is closed even if urlopen fails.
        with (
            os.fdopen(fd, "wb") as f,
            urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response,
        ):
            total = int(response.headers.get("Content-Length") or 0)
            wrote = 0
            with tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(output_path),
                disable=not show_progress,
            ) as bar:
                try:
                    while chunk := response.read(_CHUNK_SIZE):
                        f.write(chunk)
                        wrote += len(chunk)
                        bar.update(len(chunk))
                except http.client.IncompleteRead as e:
                    # The connection closed early; report it like the size check below.
                    raise ConnectionError(f"Incomplete download from {url}: {e}") from e
        if total and wrote != total:
            raise ConnectionError(f"Incomplete download from {url}: got {wrote} of {total} bytes")
        # mkstemp creates the file as 0600; give it the permissions a plain open() would.
        os.chmod(tmp_path, 0o666 & ~_current_umask())
        shutil.move(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return output_path
