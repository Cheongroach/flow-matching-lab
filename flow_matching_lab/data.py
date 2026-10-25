"""Download and verify the original MNIST IDX files, without torchvision."""

import gzip
import hashlib
import os
from pathlib import Path
import struct
import urllib.request

import numpy as np
import torch
from torch.nn import functional as F


FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load_mnist(cache=None, image_size=14):
    if image_size not in {14, 28}:
        raise ValueError("Use image_size=14 or 28.")
    cache = Path(cache or os.environ.get("FLOW_MATCHING_CACHE", "/tmp/flow_matching_mnist"))
    cache.mkdir(parents=True, exist_ok=True)
    arrays, manifest = {}, {}
    for name, expected in FILES.items():
        path = cache / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "FlowMatchingLab/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if hashlib.md5(payload).hexdigest() != expected:
                raise ValueError(f"MNIST checksum mismatch: {name}")
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(payload)
            temporary.replace(path)
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != expected:
            raise ValueError(f"Corrupted cached MNIST file: {path}")
        manifest[name] = {
            "url": BASE_URL + name,
            "md5": expected,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "compressed_bytes": len(payload),
        }
        raw = gzip.decompress(payload)
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            height, width = struct.unpack(">II", raw[8:16])
            if (height, width) != (28, 28) or len(raw) != 16 + count * height * width:
                raise ValueError("Unexpected MNIST image layout.")
            images = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(count, 1, height, width)
            ).float() / 255
            if image_size == 14:
                images = F.avg_pool2d(images, 2)
            arrays[name] = images.flatten(1) * 2 - 1
        elif magic == 2049 and len(raw) == 8 + count:
            arrays[name] = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()
        else:
            raise ValueError("Unexpected IDX header.")
    # Fixed split, independent of the training seed. The official test set stays untouched.
    permutation = torch.randperm(60000, generator=torch.Generator().manual_seed(314159))
    images = arrays["train-images-idx3-ubyte.gz"]
    labels = arrays["train-labels-idx1-ubyte.gz"]
    return {
        "train": (images[permutation[:55000]], labels[permutation[:55000]]),
        "validation": (images[permutation[55000:]], labels[permutation[55000:]]),
        "test": (arrays["t10k-images-idx3-ubyte.gz"], arrays["t10k-labels-idx1-ubyte.gz"]),
        "manifest": manifest,
    }
