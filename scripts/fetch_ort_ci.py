"""CI-only ONNX Runtime fetch for Windows builds of fl_caption.

The pinned v1.31.0 release can no longer be downloaded from GitHub
(404 / "release not found"), so this script asks the GitHub API which of the
candidate releases is available, downloads its Windows CPU and CUDA 12
archives and installs the same files scripts/fetch_onnxruntime.py would:

- runtime/ort-cpu/: onnxruntime.dll, onnxruntime_providers_shared.dll
- runtime/ort/:     + onnxruntime_providers_cuda.dll (CUDA 12 build)
- LICENSE-onnxruntime.txt, ThirdPartyNotices-onnxruntime.txt in both

Needs GH_TOKEN in the environment (the workflow's github.token).
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "microsoft/onnxruntime"
# Closest to the pinned 1.31.0 first; newer runtimes keep the older C API.
CANDIDATES = ["v1.31.0", "v1.31.1", "v1.32.0"]
TOKEN = os.environ.get("GH_TOKEN", "").strip()

CPU_MEMBERS = ("/lib/onnxruntime.dll", "/lib/onnxruntime_providers_shared.dll")
GPU_MEMBERS = CPU_MEMBERS + ("/lib/onnxruntime_providers_cuda.dll",)


def request(url: str, accept: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "fl_caption-ci"})
    if TOKEN:
        # Not forwarded on redirect: the asset host rejects a second auth.
        req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=900) as response:
        return response.read()


def api(path: str):
    return json.loads(request(f"https://api.github.com/repos/{REPO}{path}", "application/vnd.github+json"))


def release(tag: str):
    try:
        return api(f"/releases/tags/{tag}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def pick():
    for tag in CANDIDATES:
        rel = release(tag)
        if rel is None:
            print(f"[ort] {tag}: no release")
            continue
        version = tag.lstrip("v")
        assets = {a["name"]: a for a in rel.get("assets", [])}
        cpu = assets.get(f"onnxruntime-win-x64-{version}.zip")
        gpu = assets.get(f"onnxruntime-win-x64-gpu_cuda12-{version}.zip")
        if cpu and gpu:
            print(f"[ort] using {tag}")
            return cpu, gpu
        print(f"[ort] {tag}: Windows archives missing; assets: {sorted(assets)}")
    recent = [r["tag_name"] for r in api("/releases?per_page=20")]
    print(f"[ort] recent releases: {recent}")
    raise SystemExit("[ort] none of the candidate releases has the Windows archives")


def install(asset: dict, dest: Path, members: tuple[str, ...]) -> None:
    print(f"[ort] downloading {asset['name']} ({asset['size'] / 1e6:.0f} MB)")
    data = request(asset["url"], "application/octet-stream")
    dest.mkdir(parents=True, exist_ok=True)
    found = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            base = name.rsplit("/", 1)[-1]
            if any(name.endswith(m) for m in members):
                (dest / base).write_bytes(archive.read(name))
                found.add(base)
                print(f"[ort] {dest / base}")
            elif name.count("/") == 1 and base in ("LICENSE", "ThirdPartyNotices.txt"):
                (dest / f"{Path(base).stem}-onnxruntime.txt").write_bytes(archive.read(name))
    missing = [m for m in members if m.rsplit("/", 1)[-1] not in found]
    if missing:
        raise SystemExit(f"[ort] {asset['name']} lacks {missing}")


def main() -> int:
    if not TOKEN:
        print("[ort] warning: GH_TOKEN not set, API rate limits apply", file=sys.stderr)
    cpu, gpu = pick()
    install(cpu, ROOT / "runtime" / "ort-cpu", CPU_MEMBERS)
    install(gpu, ROOT / "runtime" / "ort", GPU_MEMBERS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
