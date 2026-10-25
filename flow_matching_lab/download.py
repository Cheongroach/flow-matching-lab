"""Download SHA256-verified release checkpoints into the repository root."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request


RELEASE = "https://github.com/Cheongroach/flow-matching-lab/releases/download/v0.2.0"
ASSETS = {
    "demo": "mnist28-demo.tar.gz",
    "original": "mnist14-experiments.tar.gz",
    "full": "mnist28-experiments.tar.gz",
}


def extract_checked(archive, destination):
    """Release archives may contain regular files under flow_matching_lab only."""
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or not path.parts or path.parts[0] != "flow_matching_lab"):
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = destination / path
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Archive destination escapes root: {member.name}")
            source = bundle.extractfile(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as stream:
                while chunk := source.read(1024 * 1024):
                    stream.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=ASSETS, default="demo")
    parser.add_argument("--destination", type=Path, default=Path("."))
    args = parser.parse_args()
    name = ASSETS[args.asset]
    with urllib.request.urlopen(RELEASE + "/SHA256SUMS", timeout=30) as response:
        checksums = dict(line.split()[::-1] for line in response.read().decode().splitlines())
    expected = checksums[name]
    with tempfile.TemporaryDirectory(prefix="fm-download-") as directory:
        archive = Path(directory) / name
        request = urllib.request.Request(RELEASE + "/" + name, headers={"User-Agent": "FlowMatchingLab/0.2"})
        digest = hashlib.sha256()
        total = 0
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                total += len(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Release archive SHA256 mismatch")
        extract_checked(archive, args.destination)
    print(json.dumps({"asset": name, "bytes": total, "sha256": expected,
                      "destination": str(args.destination.resolve())}), flush=True)


if __name__ == "__main__":
    main()
