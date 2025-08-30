__version__ = '2.5.2'

import os
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
    drive_forward_mod = "1NB_u8N2w2_kcuMSKTm6Wh-F9JvfxDt6I"
    drive_PHO_STELLAR_MODEL = "1ypxxofMwHYeHEx1eFKVWWWVEaaoNmdho"
    try:
        import gdown  # type: ignore
        import zipfile
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
            + ". Downloading from Google Drive... This may take a minute.",
            flush=True,
        )

        # Download the entire Drive folder into the package directory
        # use_cookies=False avoids interactive confirmation for public files.
        for i in missing:
            if i == "forward_mod":
                gdown.download(id=drive_forward_mod, output=pkg_dir)
                with zipfile.ZipFile(pkg_dir + i + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
                os.remove(pkg_dir + "forward_mod.zip")
            elif i == "PHO_STELLAR_MODEL":
                gdown.download(id=drive_PHO_STELLAR_MODEL, output=pkg_dir)
                with zipfile.ZipFile(pkg_dir + i + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
                os.remove(pkg_dir + "PHO_STELLAR_MODEL.zip")

        os.system("rm -rf " + pkg_dir + "__MACOSX")

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
            + ". Please try again or download manually from the provided link in the README.md file."
        )
    else:
        print("Success: ExoReL is ready!")


def ensure_required_data() -> None:
    """Public helper to ensure data folders exist (and download if needed).

    This mirrors the internal behavior but provides a stable public API
    callers can invoke explicitly.
    """
    _ensure_required_data()


_ensure_required_data()

from .__main__ import *
