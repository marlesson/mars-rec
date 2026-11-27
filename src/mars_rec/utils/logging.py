"""Logging utilities shared across mars-rec.

Exposes:
- ``setup_logging`` to install a uniform format and silence noisy third-party
  loggers (HuggingFace ``datasets`` / ``huggingface_hub`` / ``httpx`` /
  ``urllib3`` / ``filelock``) at INFO level — these otherwise drown the
  training logs in HEAD/GET requests, including the spurious 404 against the
  legacy ``<dataset>.py`` loading-script URL that HuggingFace probes for
  every ``load_dataset`` call.
- ``banner`` to render a consistent section header (``=`` separators, fixed
  width 72) so pipeline stages are easy to scan in long logs.
- ``get_logger`` is just a thin re-export of ``logging.getLogger`` for
  callers that want to avoid importing both modules.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, Optional

_DEFAULT_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
_BANNER_WIDTH = 72

# Third-party loggers that emit INFO chatter we never want at the default
# verbosity. Each one corresponds to a real source of noise observed in
# production / Vertex logs (HF download HEAD probes, HTTP retries, file
# locks acquired during cache writes, Lightning startup tips).
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "datasets",
    "urllib3",
    "filelock",
    "fsspec",
    "pytorch_lightning",
    "lightning.pytorch",
    "lightning_fabric",
)

_configured = False


def setup_logging(
    level: int = logging.INFO,
    *,
    fmt: str = _DEFAULT_FORMAT,
    silence: Iterable[str] = _NOISY_LOGGERS,
    silence_level: int = logging.WARNING,
    force: bool = False,
) -> None:
    """Install root logging config.

    Idempotent unless ``force=True``. The first caller wins so libraries
    that import this module don't clobber a user-supplied configuration.

    Args:
        level: root level for the project's own loggers.
        fmt: format string applied to the root handler.
        silence: third-party logger names to clamp to ``silence_level``.
        silence_level: maximum level allowed for ``silence`` loggers.
        force: rebuild the root handler even if logging was already
            configured (useful for tests / re-entrant entry points).
    """
    global _configured
    if _configured and not force:
        return

    # ``basicConfig`` is a no-op once a handler exists, so when ``force``
    # is set we strip existing root handlers first.
    if force:
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    logging.basicConfig(format=fmt, level=level, stream=sys.stdout, force=force)

    for name in silence:
        logging.getLogger(name).setLevel(silence_level)

    # Allow callers to override silencing via env (comma-separated names that
    # should stay verbose, e.g. when debugging a specific HF download).
    keep_verbose = os.environ.get("MARS_REC_LOG_VERBOSE", "")
    for name in (n.strip() for n in keep_verbose.split(",") if n.strip()):
        logging.getLogger(name).setLevel(level)

    _configured = True


def banner(
    title: str,
    *,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
    char: str = "=",
    width: int = _BANNER_WIDTH,
) -> None:
    """Log a 3-line section header — top rule, centred title, bottom rule.

    Use for high-level stage transitions ("Loading data", "Evaluating",
    "Generating submission") so they stand out in the stream of regular
    INFO lines.
    """
    log = logger or logging.getLogger()
    rule = char * width
    title_line = f" {title} ".center(width, char)
    log.log(level, rule)
    log.log(level, title_line)
    log.log(level, rule)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper around ``logging.getLogger``."""
    return logging.getLogger(name)
