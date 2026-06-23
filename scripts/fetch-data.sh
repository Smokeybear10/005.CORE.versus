#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 - "${TARGET_DIR_OVERRIDE:-$PROJECT_DIR/data}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request


REPO = "Smokeybear10/802-DATA.MMA"
TAG = "versus-runtime-2026-11-05-v1"
ASSET = f"{TAG}.tar.gz"
ARCHIVE_SHA256 = "f20cfa3a8afee060dd95cdb86c41d4f43843e2e4bae597529b5d99396370e6e2"
ARCHIVE_BYTES = 18494946
FILES = {
    "data/evaluation/career-supplement-v1/LICENSE": "f1a9c3dbae3031fba5d02c3c05077dd78cb56fce834a36124426ceb1b8453cbc",
    "data/evaluation/career-supplement-v1/bouts.parquet": "bb49064b37b363b369cf0d624bbcbb92ca819c403daf138bc19fb9162962d6a3",
    "data/evaluation/career-supplement-v1/crosswalk.csv": "c824b0ab51b00e60c855b436645dafcbd53af00fc347dfe3966f1b546bf6ac9b",
    "data/evaluation/career-supplement-v1/exclusions.parquet": "8fc4dd7dccc723f31db2846259021a5c609c32c26b51a3ce9a084ec1ec9374dd",
    "data/evaluation/career-supplement-v1/fighter_bouts.parquet": "e6f463e494745628c1ef276bab7054e021aa5d7f78ea6f0d80f6a97e23292d3a",
    "data/evaluation/career-supplement-v1/identity-proof.json": "69d0f556503fc4155a6585a86b25c6d471f6ad7ebd5fea7af94731c69fefa9f4",
    "data/evaluation/career-supplement-v1/manifest.json": "40c62cd008b45b8d81ddb2c39adcd5732e8fea3459b3a52296be6938a314718e",
    "data/evaluation/career-supplement-v1/overlap-audit.parquet": "9eb7ffbef6e2b8d662807ff8abbac10013e572f25bf75f37af43fe0e4fcef76c",
    "data/evaluation/identity-repair-v1/candidate_contract.json": "eb92479bf7986df6451919df2f4698e3e5643b3a74f3ffe178eefb030cc0c445",
    "data/evaluation/identity-repair-v1/evaluation.json": "2a3609d2f809eea922d41af5f24ec9c0bbc2b02ea5a05f6c1093beb20d1ee1ac",
    "data/evaluation/identity-repair-v1/inference/fighters.parquet": "bddd469986ec3e126f77262f7083b23595573f37d4f48459170487321c8c25ed",
    "data/evaluation/identity-repair-v1/inference/fights.parquet": "5fb0336d957c79f27357b0e9e0594d522d1246f10d1ab4d46985a8c43db8873a",
    "data/evaluation/identity-repair-v1/inference/manifest.json": "ea21687f2dad98f247168cfd0bbc99a059f6ab5c80c6a0143afe4744797fd3ad",
    "data/evaluation/identity-repair-v1/inference/rounds.parquet": "7713ec93e1d71af344fbfb1abfc8d9067caef14fc2a95e3fb9d619800f3d268e",
    "data/evaluation/identity-repair-v1/models/development_and_calibration_features.parquet": "39d191e73b1b0ff903114dac125e245afbe372db1670097f16dcc392e1a51c98",
    "data/evaluation/identity-repair-v1/models/provisional_challenger.joblib": "6fde3bc72e4788c944efa65ad8a911c2e18d840404a337e2cd09559a8530a924",
    "data/evaluation/identity-repair-v1/research-data/fighters.parquet": "bddd469986ec3e126f77262f7083b23595573f37d4f48459170487321c8c25ed",
    "data/evaluation/identity-repair-v1/research-data/fights.parquet": "5fb0336d957c79f27357b0e9e0594d522d1246f10d1ab4d46985a8c43db8873a",
    "data/evaluation/identity-repair-v1/research-data/manifest.json": "027e79362e4631dcc47b2f52c797893bc7431d9c31ec1803422c9d5e745b0839",
    "data/evaluation/identity-repair-v1/research-data/rounds.parquet": "7713ec93e1d71af344fbfb1abfc8d9067caef14fc2a95e3fb9d619800f3d268e",
    "data/events.parquet": "a063e8d5e8efc6ca75dbd23567d9499e4cfcb8e2e603270e65869c057ce44e1d",
    "data/fighter_snapshots.parquet": "b800f96f0833bb9a3da01d6e7a9654cc057813dd72df8f20124ea3507770b66c",
    "data/fights.parquet": "c388f2a65c27a3c6739f2e3802f4e5d7c602acf0b3cb26b3915949cfd92e3bd9",
    "data/model_v2.pkl": "f1a5c3b2defb843c04eff0b8b68fb442e5350c389bd276fcc63ea08d7ee5191f",
    "data/ufc-master.csv": "0235c126af82436498fa4be7a8218b4bb6aca7938c058fe3d79013d4a4052681"
}


def runtime_path(directory: Path, name: str) -> Path:
    path = directory
    for part in ("", *PurePosixPath(name).relative_to("data").parts):
        path /= part
        if path.is_symlink():
            raise ValueError(f"Runtime destination cannot contain a symlink: {path}")
    return path


def verified(directory: Path) -> bool:
    for name, expected in FILES.items():
        path = runtime_path(directory, name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False
    return True


class AssetRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        destination = urllib.parse.urlsplit(url)
        if destination.scheme != "https":
            raise ValueError("Refusing a non-HTTPS asset redirect")
        redirected = super().redirect_request(request, response, code, message, headers, url)
        if redirected is not None and destination.netloc != urllib.parse.urlsplit(request.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def main() -> None:
    directory = Path(sys.argv[1]).expanduser().absolute()
    if verified(directory):
        print(f"Verified {len(FILES)} runtime files for {TAG}; download not needed.")
        return

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token and shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            token = result.stdout.strip()
    if not token:
        raise ValueError(f"Set GITHUB_TOKEN or sign into gh to fetch the private {REPO} runtime.")

    opener = urllib.request.build_opener(AssetRedirectHandler())
    api = f"https://api.github.com/repos/{REPO}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    print(f"Fetching pinned runtime {TAG}...")
    with opener.open(urllib.request.Request(f"{api}/releases/tags/{TAG}", headers=headers), timeout=60) as response:
        release = json.load(response)
    assets = [asset for asset in release.get("assets", []) if asset.get("name") == ASSET]
    if len(assets) != 1 or type(assets[0].get("id")) is not int or assets[0]["id"] <= 0:
        raise ValueError(f"Release {TAG} must contain exactly one {ASSET} asset.")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".versus-runtime-", dir=directory.parent) as temporary:
        archive_path = Path(temporary) / ASSET
        request = urllib.request.Request(f"{api}/releases/assets/{assets[0]['id']}", headers={**headers, "Accept": "application/octet-stream"})
        with opener.open(request, timeout=60) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                if output.tell() > ARCHIVE_BYTES:
                    raise ValueError("Runtime archive exceeds its pinned size.")
        if archive_path.stat().st_size != ARCHIVE_BYTES or hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Runtime archive checksum does not match the pinned release.")

        staged_data = Path(temporary) / "data"
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(FILES) or {member.name for member in members} != set(FILES) or any(not member.isfile() for member in members):
                raise ValueError("Runtime archive contains missing, duplicate, or unexpected entries.")
            for member in members:
                destination = runtime_path(staged_data, member.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if not verified(staged_data):
            raise ValueError("Extracted runtime files do not match the pinned checksums.")

        destinations = {name: runtime_path(directory, name) for name in FILES}
        for name, destination in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            runtime_path(staged_data, name).replace(destination)
        if not verified(directory):
            raise ValueError("Installed runtime verification failed.")
    print(f"Verified and installed {len(FILES)} runtime files for {TAG}.")


try:
    main()
except (OSError, ValueError, tarfile.TarError) as error:
    raise SystemExit(f"ERROR: {error}") from None
PY
