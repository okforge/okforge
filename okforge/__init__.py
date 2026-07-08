"""okforge: local LLM knowledge-base engine (import package; hard fork of OpenKB)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("okforge")
except PackageNotFoundError:
    # Fallback for environments still carrying the pre-fork distribution.
    try:
        __version__ = _version("openkb")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
