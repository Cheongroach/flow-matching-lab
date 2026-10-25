"""Check the editor's HTTP contract against an actual trained checkpoint."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            origin + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()

    with tempfile.TemporaryDirectory(prefix="fm-http-check-") as directory:
        with open(Path(directory) / "server.log", "w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "flow_matching_lab.draw", "--run", str(args.run),
                 "--port", str(port), "--output", directory],
                stdout=log, stderr=log,
            )
            try:
                for _ in range(80):
                    try:
                        code, page = request("/")
                        break
                    except OSError:
                        if process.poll() is not None:
                            raise RuntimeError("Server failed to start")
                        time.sleep(0.25)
                else:
                    raise TimeoutError("Server did not become ready")
                assert code == 200 and "__SIDE__" not in page and 'id="mode"' in page
                code, text = request("/example?digit=3")
                assert code == 200
                known = json.loads(text)["image"]
                side = int(len(known) ** 0.5)
                noise = np.random.default_rng(42).standard_normal(side * side).tolist()
                mask = [int(i % side < side // 2) for i in range(side * side)]
                payload = dict(noise=noise, known=known, mask=mask, digit=3, steps=32, cfg=1)

                def generate(settings, path="/generate"):
                    code, text = request(path, settings)
                    assert code == 200, text
                    data = json.loads(text)
                    values = np.array(data["samples"])
                    assert np.isfinite(values).all()
                    return values, data

                a, first = generate(payload)
                repeated, _ = generate(payload)
                assert np.array_equal(a, repeated)
                observed = np.array(mask).astype(bool)
                # Server inputs are converted to Float32 before inference.
                assert np.array_equal(a[observed], np.array(known, dtype=np.float32)[observed])
                assert first["known_max_error"] == 0
                shifted = np.roll(np.array(known).reshape(side, side), 2, axis=0).ravel().tolist()
                b, _ = generate(payload | {"known": shifted})
                unknown_rms = float(np.square(a[~observed] - b[~observed]).mean() ** 0.5)
                assert unknown_rms > 1e-5
                changed_noise = np.array(noise).reshape(side, side)
                changed_noise[10:18, 10:18] += 1
                c, _ = generate(payload | {"noise": changed_noise.ravel().tolist()})
                noise_rms = float(np.square(a[~observed] - c[~observed]).mean() ** 0.5)
                assert noise_rms > 1e-5
                full, _ = generate(payload | {"mask": [1] * (side * side)})
                assert np.array_equal(full, np.array(known, dtype=np.float32))

                invalid = [
                    payload | {"mask": [0.5] * (side * side)},
                    payload | {"noise": [0]},
                    payload | {"noise": [float("nan")] * (side * side)},
                    payload | {"digit": 10},
                    payload | {"steps": 0},
                    payload | {"cfg": float("inf")},
                ]
                for item in invalid:
                    assert request("/generate", item)[0] == 400
                assert request("/missing")[0] == 404
                saved, info = generate(payload, "/save")
                assert np.array_equal(saved, a)
                folder = Path(info["path"])
                assert all((folder / name).is_file() for name in
                           ("input.png", "output.png", "mask.png", "settings.json"))
                metadata = json.loads((folder / "settings.json").read_text())
                assert metadata["samples"] == a.tolist() and metadata["noise"] == noise
                summary = {
                    "run": args.run.name, "page_and_example_served": True,
                    "deterministic_repeat": True, "known_max_error": 0.0,
                    "changed_observation_unknown_rms": unknown_rms,
                    "changed_noise_unknown_rms": noise_rms,
                    "fully_observed_image_exact": True, "invalid_requests_rejected": len(invalid),
                    "save_round_trip": True,
                    "scope": "Trained inference and HTTP API; does not verify browser pointer/upload behavior.",
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(summary), flush=True)
            finally:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
