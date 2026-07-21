__version__ = '3.0.3'
__fmod_version__ = '3.0'

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from typing import Iterable


# ``pheonix`` is the spelling used in the official STScI archive filename.
_STSYNPHOT_PHOENIX_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/"
    "hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar"
)


def _is_nonempty_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def _phoenix_atlas_path(data_root: str) -> str:
    return os.path.join(data_root, "grid", "phoenix")


def _is_complete_phoenix_directory(phoenix_path: str) -> bool:
    if not os.path.isfile(os.path.join(phoenix_path, "catalog.fits")):
        return False
    for _root, _dirs, files in os.walk(phoenix_path):
        if any(
            filename != "catalog.fits"
            and filename.lower().endswith((".fits", ".fits.gz"))
            for filename in files
        ):
            return True
    return False


def _is_complete_phoenix_atlas(data_root: str) -> bool:
    return _is_complete_phoenix_directory(_phoenix_atlas_path(data_root))


def _set_stsynphot_data_root(data_root: str) -> str:
    data_root = os.path.abspath(os.path.expanduser(data_root))
    os.environ["PYSYN_CDBS"] = data_root

    # stsynphot reads PYSYN_CDBS when imported. Keep an already-imported module
    # synchronized as well (for notebooks that import stsynphot first).
    stsynphot_module = sys.modules.get("stsynphot")
    if stsynphot_module is not None and hasattr(stsynphot_module, "conf"):
        stsynphot_module.conf.rootdir = data_root
    return data_root


def _safe_extract_tar(archive_path: str, destination: str) -> None:
    destination = os.path.realpath(destination)
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = os.path.realpath(os.path.join(destination, member.name))
            if os.path.commonpath((destination, member_path)) != destination:
                raise RuntimeError(
                    f"Unsafe path in stsynphot PHOENIX archive: {member.name}"
                )
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Unsupported link in stsynphot PHOENIX archive: {member.name}"
                )
        archive.extractall(destination)


def _find_extracted_phoenix(extraction_root: str) -> str:
    for root, _dirs, files in os.walk(extraction_root):
        if os.path.basename(root) == "phoenix" and "catalog.fits" in files:
            return root
    raise RuntimeError(
        "The downloaded archive does not contain a grid/phoenix/catalog.fits atlas."
    )


def setup_stsynphot_data(data_root=None, force: bool = False) -> str:
    """Download and configure the local stsynphot PHOENIX reference atlas.

    The one-time download is approximately 1.8 GB. By default the atlas is
    installed under ``ExoReL/stsynphot_data/grid/phoenix``. A complete existing
    installation is reused unless ``force=True`` is requested.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if data_root is None:
        data_root = os.path.join(pkg_dir, "stsynphot_data")
    data_root = os.path.abspath(os.path.expanduser(os.fspath(data_root)))

    if _is_complete_phoenix_atlas(data_root) and not force:
        return _set_stsynphot_data_root(data_root)

    # Keep the large temporary archive inside the ignored data directory so an
    # interrupted setup never leaves a multi-gigabyte untracked repository file.
    os.makedirs(data_root, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=".stsynphot-phoenix-", dir=data_root)
    archive_path = os.path.join(work_dir, "phoenix.tar")
    extraction_root = os.path.join(work_dir, "extracted")
    os.makedirs(extraction_root)
    staging_path = None

    try:
        print(
            "ExoReL: Downloading the stsynphot PHOENIX reference atlas "
            "(~1.8 GB) from STScI. This is a one-time setup...",
            flush=True,
        )

        last_reported = [-10]

        def _report_progress(block_count, block_size, total_size):
            if total_size <= 0:
                return
            percent = min(100, int(block_count * block_size * 100 / total_size))
            if percent >= last_reported[0] + 10:
                last_reported[0] = percent
                print(f"ExoReL: PHOENIX download {percent}%", flush=True)

        urllib.request.urlretrieve(
            _STSYNPHOT_PHOENIX_URL,
            archive_path,
            reporthook=_report_progress,
        )
        _safe_extract_tar(archive_path, extraction_root)
        extracted_phoenix = _find_extracted_phoenix(extraction_root)

        # Build and validate a staging atlas before replacing a partial/old one.
        grid_path = os.path.join(data_root, "grid")
        os.makedirs(grid_path, exist_ok=True)
        staging_path = tempfile.mkdtemp(prefix=".phoenix-", dir=grid_path)
        os.rmdir(staging_path)
        shutil.move(extracted_phoenix, staging_path)
        if not _is_complete_phoenix_directory(staging_path):
            raise RuntimeError("The extracted stsynphot PHOENIX atlas is incomplete.")

        phoenix_path = _phoenix_atlas_path(data_root)
        if os.path.exists(phoenix_path):
            shutil.rmtree(phoenix_path)
        os.replace(staging_path, phoenix_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to install the stsynphot PHOENIX atlas from STScI. "
            f"Retry the setup/download. Details: {exc}"
        ) from exc
    finally:
        if staging_path and os.path.isdir(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)

    if not _is_complete_phoenix_atlas(data_root):
        raise RuntimeError("The installed stsynphot PHOENIX atlas is incomplete.")

    print(f"ExoReL: stsynphot PHOENIX atlas installed in {data_root}", flush=True)
    return _set_stsynphot_data_root(data_root)


def _ensure_stsynphot_data(pkg_dir: str) -> str:
    local_root = os.path.join(pkg_dir, "stsynphot_data")
    if _is_complete_phoenix_atlas(local_root):
        return _set_stsynphot_data_root(local_root)

    configured_root = os.environ.get("PYSYN_CDBS")
    if configured_root and _is_complete_phoenix_atlas(
        os.path.abspath(os.path.expanduser(configured_root))
    ):
        return _set_stsynphot_data_root(configured_root)

    return setup_stsynphot_data(local_root)


def _ensure_required_data():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    required_dirs: Iterable[str] = ("forward_mod",)
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
    if missing:
        # Attempt to download the full folder from Google Drive via gdown
        drive_forward_mod = "16eKd2Dlefi4Hclzkp5k5A2fos5ou0VY8"
        try:
            import gdown  # type: ignore
            import zipfile
        except Exception as e:  # pragma: no cover - import-time environment dependent
            raise RuntimeError(
                "Required data folders missing: "
                + ", ".join(missing)
                + ". Please install 'gdown' (pip install gdown) and retry, or manually "
                "download 'forward_mod' from the project's README link."
            ) from e

        try:
            # Inform the user that a download is starting.
            print(
                "ExoReL: Required data folders missing: "
                + ", ".join(missing)
                + ". Downloading from Google Drive... This may take a minute.",
                flush=True,
            )

            # Download the archive to an explicit filename. Passing ``pkg_dir``
            # directly is ambiguous to gdown: versions that do not see a trailing
            # separator can treat the directory itself as the output filename and
            # return it, causing ZipFile to receive a directory path.
            # use_cookies=False avoids interactive confirmation for public files.
            for i in missing:
                if i == "forward_mod":
                    with tempfile.TemporaryDirectory(
                        prefix=".forward-mod-", dir=pkg_dir
                    ) as download_dir:
                        archive_path = os.path.join(download_dir, "forward_mod.zip")
                        downloaded_zip = gdown.download(
                            id=drive_forward_mod,
                            output=archive_path,
                            use_cookies=False,
                        )
                        if not downloaded_zip or not os.path.isfile(archive_path):
                            raise RuntimeError("Download failed for forward_mod archive.")
                        if not zipfile.is_zipfile(archive_path):
                            raise RuntimeError(
                                "Downloaded forward_mod file is not a valid ZIP archive."
                            )
                        with zipfile.ZipFile(archive_path, "r") as zip_ref:
                            zip_ref.extractall(pkg_dir)

            if not os.path.isdir(forward_mod_path):
                with os.scandir(pkg_dir) as entries:
                    for entry in entries:
                        if not entry.is_dir() or not entry.name.startswith("forward_mod"):
                            continue
                        if entry.name != "forward_mod":
                            os.rename(entry.path, forward_mod_path)
                        break

            shutil.rmtree(os.path.join(pkg_dir, "__MACOSX"), ignore_errors=True)

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
    elif missing:
        print("Success: ExoReL forward-model data are ready!")

    _ensure_stsynphot_data(pkg_dir)


_ensure_required_data()

from .__main__ import *
