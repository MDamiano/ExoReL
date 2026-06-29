__version__ = '3.0.1'
__fmod_version__ = '3.0'

import os
import shutil
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
    forward_mod_path = os.path.join(pkg_dir, "forward_mod")
    version_marker_path = os.path.join(forward_mod_path, "__fmod_version__")
    version_match = False
    if os.path.isdir(forward_mod_path):
        try:
            with open(version_marker_path, "r", encoding="utf-8") as marker:
                version_match = marker.read().strip() == __fmod_version__
        except OSError:
            version_match = False
        if not version_match:
            shutil.rmtree(forward_mod_path, ignore_errors=True)

    missing = [d for d in required_dirs if not _is_nonempty_dir(os.path.join(pkg_dir, d))]
    if not missing:
        return

    # Attempt to download the full folder from Google Drive via gdown
    drive_forward_mod = "16eKd2Dlefi4Hclzkp5k5A2fos5ou0VY8"
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
                downloaded_zip = gdown.download(id=drive_forward_mod, output=pkg_dir, use_cookies=False)
                if not downloaded_zip:
                    raise RuntimeError("Download failed for forward_mod archive.")
                with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
                with os.scandir(pkg_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.startswith("forward_mod") and entry.name.endswith(".zip"):
                            os.remove(entry.path)
            elif i == "PHO_STELLAR_MODEL":
                downloaded_zip = gdown.download(id=drive_PHO_STELLAR_MODEL, output=pkg_dir, use_cookies=False)
                if not downloaded_zip:
                    raise RuntimeError("Download failed for PHO_STELLAR_MODEL archive.")
                with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(pkg_dir)
                if os.path.isfile(downloaded_zip):
                    os.remove(downloaded_zip)
        
        if not os.path.isdir(forward_mod_path):
            with os.scandir(pkg_dir) as entries:
                for entry in entries:
                    if not entry.is_dir() or not entry.name.startswith("forward_mod"):
                        continue
                    if entry.name != "forward_mod":
                        os.rename(entry.path, forward_mod_path)
                    break

        os.system("rm -rf " + pkg_dir + "__MACOSX")

    except Exception as e:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download required data folders from Google Drive: {e}"
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


_ensure_required_data()

from .__main__ import *
