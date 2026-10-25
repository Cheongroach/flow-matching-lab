"""Optional Playwright check of the real editor (install Playwright separately)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fm-browser-verification.json"))
    parser.add_argument("--screenshot", type=Path, default=Path("/tmp/fm-editor-browser.png"))
    args = parser.parse_args()
    browser_config = tempfile.TemporaryDirectory(prefix="fm-chromium-config-")
    browser_env = dict(os.environ)
    browser_env["XDG_CONFIG_HOME"] = browser_config.name
    process = subprocess.Popen(
        [sys.executable, "-m", "flow_matching_lab.draw", "--run", str(args.run), "--port", "7867"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen("http://127.0.0.1:7867", timeout=1).close()
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("Editor failed to start; check port 7867")
                time.sleep(0.25)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=os.environ.get("FM_CHROMIUM_PATH"),
                env=browser_env,
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:7867/")
            page.wait_for_function("!busy && !pending && output !== null")
            original = page.locator("#result").evaluate("(c)=>c.toDataURL()")
            box = page.locator("#input").bounding_box()
            page.mouse.move(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.25)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.5, steps=10)
            page.mouse.up()
            page.wait_for_function("!busy && !pending")
            assert page.locator("#result").evaluate("(c)=>c.toDataURL()") != original
            page.locator("#reset").click()
            page.wait_for_function("!busy && !pending")
            assert page.locator("#result").evaluate("(c)=>c.toDataURL()") == original
            page.locator("#mode").select_option("noise")
            page.wait_for_function("!busy && !pending")
            assert "初始噪声" in page.locator("#input-title").inner_text()
            page.locator("#mode").select_option("completion")
            page.wait_for_function("!busy && !pending")
            page.locator("#erase").click()
            page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
            page.wait_for_function("!busy && !pending")
            assert page.locator("#coverage").inner_text() != "已知区域：50.0%"
            import numpy as np
            from PIL import Image
            examples = json.loads((Path(__file__).parent / "improved/examples.json").read_text())
            pixels = ((np.array(examples["images"]["8"]).reshape(28, 28) + 1) * 127.5).astype("uint8")
            with tempfile.TemporaryDirectory(prefix="fm-browser-") as directory:
                fixture = Path(directory) / "upload.png"
                Image.fromarray(pixels).save(fixture)
                page.locator("#file").set_input_files(str(fixture))
                page.wait_for_function("!busy && !pending && document.querySelector('#coverage').textContent==='已知区域：50.0%'")
            assert "已更新" in page.locator("#status").inner_text()
            page.screenshot(path=str(args.screenshot), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
            assert not errors, errors
            summary = {
                "canvas_edit_changes_output": True, "reset_restores_output": True,
                "noise_mode": True, "erase_changes_mask": True, "image_upload": True,
                "mobile_no_overflow": True, "page_errors": errors, "weights": str(args.run),
            }
            args.output.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)
        browser_config.cleanup()


if __name__ == "__main__":
    main()
