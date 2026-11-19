"""Fetches ONNX Runtime for fl_caption on Windows from PyPI.

github.com/microsoft/onnxruntime (and its release archives) has returned 404
since 2026-11-17, so scripts/fetch_onnxruntime.py can no longer download the
pinned 1.31.0 builds. The same libraries ship in the PyPI wheels, which this
script downloads (pinned, SHA-256 checked) and unpacks.

CPU runtime, used by the CI build (program package):

    python scripts/fetch_ort_pypi.py
    -> <repo>/runtime/ort-cpu/

GPU runtime, run on the user's PC next to an unpacked fl_caption build:

    python fetch_ort_pypi.py --gpu --dest D:/fl_caption --mirror https://pypi.tuna.tsinghua.edu.cn
    -> D:/fl_caption/runtime/ort/

The PyPI onnxruntime-gpu 1.31 wheel is a CUDA 13 build (it imports
cublas64_13.dll), not the CUDA 12 build of the GitHub archives, so the GPU
runtime here carries the CUDA 13 runtime, cuBLAS 13 and cuDNN 9 for CUDA 13.
CUDA 13 needs an NVIDIA driver of the R580 series or newer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPI_FILES = "https://files.pythonhosted.org"

# (path under files.pythonhosted.org, sha256, member prefix, file suffixes)
ORT_CPU = (
    "/packages/45/b2/793b4051d8dc6acb379ead9b72050f0ffa283ca49a6904add2e218b83e2e/"
    "onnxruntime-1.31.1-cp312-cp312-win_amd64.whl",
    "ae8a9a351efda685cc59a66226cdefaa37a126c1d66872ca23a2de9425326130",
    "onnxruntime/capi/",
    ("onnxruntime.dll", "onnxruntime_providers_shared.dll"),
)
ORT_GPU = (
    "/packages/05/ec/fe99397c6d6582fe26502b608f132a1c7748381fa72523c8b7a0686b9b14/"
    "onnxruntime_gpu-1.31.1-cp312-cp312-win_amd64.whl",
    "2bb1e9373baba88f6415257f5b6a589d23e23c16004f7d89adf4344eada5749e",
    "onnxruntime/capi/",
    ("onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"),
)
CUDA = [
    (
        "/packages/86/00/d5436004268f049214193659ebc36550b5ef3925c3d13b4cc980e13be6f5/"
        "nvidia_cuda_runtime-13.4.92-py3-none-win_amd64.whl",
        "08dca5e4aba480c2fd5b55075c0fa71b84ef9dcf0521f2d58baa14a803a7311c",
        "nvidia/cu13/bin/x86_64/",
        ("cudart64_13.dll",),
    ),
    (
        "/packages/a3/df/f1246959833e2c437db8be3e5b477f66b87f8817821ed40de6c7561c9a36/"
        "nvidia_cublas-13.8.0.4-py3-none-win_amd64.whl",
        "8c5494423bb8a46822cb6b0cb95d7fa4be2d7b96a31155dff083839ec8297910",
        "nvidia/cu13/bin/x86_64/",
        ("cublas64_13.dll", "cublasLt64_13.dll"),
    ),
    (
        "/packages/87/6a/e55ff0ac26a5c6e2b21f41c9d04ad096b4ed6da593fba7e25845c61b0532/"
        "nvidia_cudnn_cu13-9.27.0.42-py3-none-win_amd64.whl",
        "7d96f634adafd55c72231eb0500ca77ab109ec8ebff7b33000b76e081bc4558e",
        "nvidia/cudnn/bin/",
        ("64_9.dll",),
    ),
]

NOTICE = """\
GPU runtime for fl_caption, assembled from PyPI wheels by fetch_ort_pypi.py:
- ONNX Runtime 1.31.1, CUDA 13 build (onnxruntime-gpu, MIT):
  LICENSE-onnxruntime.txt, ThirdPartyNotices-onnxruntime.txt
- CUDA runtime 13.4 and cuBLAS 13.8 (NVIDIA CUDA EULA):
  https://docs.nvidia.com/cuda/eula/
- cuDNN 9.27 for CUDA 13 (NVIDIA cuDNN license):
  https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html
"""


def fetch(url_path: str, sha256: str, mirror: str | None, cache: Path) -> zipfile.ZipFile:
    name = url_path.rsplit("/", 1)[1]
    cached = cache / name
    if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == sha256:
        print(f"[ort] cached {name}")
        return zipfile.ZipFile(io.BytesIO(cached.read_bytes()))
    url = (mirror.rstrip("/") if mirror else PYPI_FILES) + url_path
    print(f"[ort] downloading {url}", flush=True)
    buffer = io.BytesIO()
    with urllib.request.urlopen(url, timeout=1800) as response:
        total = int(response.headers.get("Content-Length") or 0)
        step = 50 << 20
        mark = step
        while chunk := response.read(1 << 20):
            buffer.write(chunk)
            if buffer.tell() >= mark:
                done = buffer.tell() >> 20
                print(f"[ort]   {done} MB" + (f" / {total >> 20} MB" if total else ""), flush=True)
                mark += step
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        sys.exit(f"[ort] SHA-256 mismatch for {name}: {digest} != {sha256}")
    cache.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return zipfile.ZipFile(io.BytesIO(data))


def unpack(archive: zipfile.ZipFile, prefix: str, suffixes: tuple[str, ...], dest: Path, licenses: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    found = []
    for member in archive.namelist():
        base = member.rsplit("/", 1)[-1]
        if member.startswith(prefix) and member.count("/") == prefix.count("/") and any(base.endswith(s) for s in suffixes):
            (dest / base).write_bytes(archive.read(member))
            found.append(base)
            print(f"[ort] {dest / base}")
        elif licenses and member in ("onnxruntime/LICENSE", "onnxruntime/ThirdPartyNotices.txt"):
            (dest / f"{Path(base).stem}-onnxruntime.txt").write_bytes(archive.read(member))
    missing = [s for s in suffixes if not any(f.endswith(s) for f in found)]
    if missing:
        sys.exit(f"[ort] {missing} not found under {prefix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true", help="the GPU runtime (runtime/ort) instead of the CPU one")
    parser.add_argument("--dest", type=Path, default=ROOT, help="the fl_caption folder (default: this repository)")
    parser.add_argument("--mirror", default=None, help="a PyPI mirror serving /packages/..., e.g. https://pypi.tuna.tsinghua.edu.cn")
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "fl_caption-wheels")
    args = parser.parse_args()
    dest = args.dest.resolve()
    if args.gpu and not (dest / "fl_caption.exe").is_file():
        print(f"[ort] warning: no fl_caption.exe in {dest}; is --dest the unpacked program folder?", file=sys.stderr)
    if not args.gpu:
        unpack(fetch(*ORT_CPU[:2], args.mirror, args.cache), *ORT_CPU[2:], dest / "runtime" / "ort-cpu", licenses=True)
        return 0
    ort = dest / "runtime" / "ort"
    unpack(fetch(*ORT_GPU[:2], args.mirror, args.cache), *ORT_GPU[2:], ort, licenses=True)
    for url_path, sha256, prefix, suffixes in CUDA:
        unpack(fetch(url_path, sha256, args.mirror, args.cache), prefix, suffixes, ort)
    (ort / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    print(f"[ort] GPU runtime ready in {ort}")
    print(f"[ort] downloaded wheels are cached in {args.cache}; delete it when done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
