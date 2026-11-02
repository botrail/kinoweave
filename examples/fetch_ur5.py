"""Fetch the pinned UR5 demo assets, checking their original Git blob hashes."""

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ASSET_ROOT = Path(__file__).parent / "assets/ur5"


def fetch(directory=ASSET_ROOT / "source"):
    lock = json.loads((ASSET_ROOT / "source-lock.json").read_text())
    directory = Path(directory)

    def one(record):
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid locked asset path")
        target = directory / relative
        if target.is_file():
            data = target.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{relative}"
            print(f"Fetching {relative}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if digest != record["git_sha1"]:
            raise ValueError(f"Asset changed or hash mismatch: {relative}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return {
            "path": str(relative),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, lock["files"]))
    return {"repository": lock["repository"], "commit": lock["commit"], "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ASSET_ROOT / "source")
    args = parser.parse_args()
    print(json.dumps(fetch(args.directory), indent=2))


if __name__ == "__main__":
    main()
