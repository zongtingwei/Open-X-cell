"""Authenticated GitHub Release downloads, verified before installation."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from open_x_cell.bundle import BUNDLE_FILES, verify_bundle


DEFAULT_REPO = "zongtingwei/Open-X-cell"
DEFAULT_RELEASE = "v0.2.0"
ASSET_NAME = "open-x-cell-mini-orion.tar.gz"


def extract_bundle(archive: str | Path, output: str | Path) -> Path:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    allowed = BUNDLE_FILES | {"manifest.json"}
    with tempfile.TemporaryDirectory(prefix=".xcell-extract-", dir=target.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != allowed:
                raise ValueError("archive must contain exactly the portable bundle files")
            for member in members:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    raise ValueError("unsafe archive member")
                destination = stage / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError("unreadable archive member")
                with source, destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        verify_bundle(stage)
        stage.rename(target)
    return target


def download_bundle(output: str | Path, *, repo: str = DEFAULT_REPO,
                    release: str = DEFAULT_RELEASE) -> Path:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    if not shutil.which("gh"):
        raise RuntimeError("Install the GitHub CLI (gh) and run 'gh auth login' to download the release")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xcell-download-", dir=target.parent) as tmp:
        subprocess.run(["gh", "release", "download", release, "--repo", repo,
                        "--pattern", ASSET_NAME, "--dir", tmp], check=True)
        return extract_bundle(Path(tmp) / ASSET_NAME, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--archive", help="Install an already downloaded archive without network access.")
    args = parser.parse_args(argv)
    path = (extract_bundle(args.archive, args.output) if args.archive else
            download_bundle(args.output, repo=args.repo, release=args.release))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
