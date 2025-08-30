__version__ = '2.5.2'

# Ensure required data folders are present. If missing/empty, download from Drive.
import os
import warnings
from typing import Iterable


def _is_nonempty_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def _ensure_required_data():
    pkg_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
    required_dirs: Iterable[str] = ("forward_mod", "PHO_STELLAR_MODEL")

    missing = [d for d in required_dirs if not _is_nonempty_dir(os.path.join(pkg_dir, d))]
    if not missing:
        return

    # Attempt to download the full folder from Google Drive via gdown
    drive_folder_id = "1CQutXQ8Ki59TB9Dndo61sktwS3uOM7qZ"
    try:
        import gdown  # type: ignore
    except Exception as e:  # pragma: no cover - import-time environment dependent
        raise RuntimeError(
            "Required data folders missing: "
            + ", ".join(missing)
            + ". Please install 'gdown' (pip install gdown) and retry, or manually "
            "download 'forward_mod' and 'PHO_STELLAR_MODEL' from the project's README link."
        ) from e

    try:
        # Inform the user that a download is starting.
        print(
            "ExoReL: Required data folders missing: "
            + ", ".join(missing)
            + ". Downloading from Google Drive... This may take a few minutes.",
            flush=True,
        )

        # Download the entire Drive folder into the package directory
        # use_cookies=False avoids interactive confirmation for public files.
        gdown.download_folder(id=drive_folder_id, output=pkg_dir, use_cookies=False, quiet=True)
    except Exception as e:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to download required data folders from Google Drive. "
            "Please check your internet connection or download them manually from: "
            "https://drive.google.com/drive/folders/1CQutXQ8Ki59TB9Dndo61sktwS3uOM7qZ"
        ) from e

    # Validate again after download
    still_missing = [d for d in required_dirs if not _is_nonempty_dir(os.path.join(pkg_dir, d))]
    if still_missing:
        raise RuntimeError(
            "Downloaded data appears incomplete. Missing: "
            + ", ".join(still_missing)
            + ". Please try again or download manually from the provided link."
        )


def ensure_required_data() -> None:
    """Public helper to ensure data folders exist (and download if needed).

    This mirrors the internal behavior but provides a stable public API
    callers can invoke explicitly.
    """
    _ensure_required_data()


# Only auto-download at import if explicitly enabled by env var.
if os.getenv("EXOREL_AUTO_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}:
    _ensure_required_data()
else:
    # If data is missing, emit a guidance warning instead of downloading.
    pkg_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
    required_dirs: Iterable[str] = ("forward_mod", "PHO_STELLAR_MODEL")
    missing = [d for d in required_dirs if not _is_nonempty_dir(os.path.join(pkg_dir, d))]
    if missing:
        warnings.warn(
            "ExoReL data folders missing: "
            + ", ".join(missing)
            + ". Set EXOREL_AUTO_DOWNLOAD=1 to auto-download at import, "
              "or install 'gdown' and call exorel.ensure_required_data(), "
              "or download manually via the README link.",
            RuntimeWarning,
        )

from .__main__ import *
