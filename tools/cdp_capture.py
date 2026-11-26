"""Screenshot a local web page with headless Edge/Chrome over the DevTools protocol.

Unlike ``--screenshot``, this can wait for a single-page app, click through its UI and hide side panels
before capturing. Only used for localhost pages (the MLflow UI).
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp

# Switches the MLflow UI to "Model training" mode (the classic runs and registry views).
PREPARE_MLFLOW = r"""
(() => {
  const byText = (t) => [...document.querySelectorAll('button, [role=radio], label, span')]
      .find(e => e.textContent.trim() === t);
  const training = byText('Model training');
  if (training) training.click();
  return Boolean(training);
})()
"""


ASSISTANT_PANEL_SHARE = 0.262  # MLflow 3.18 shows an assistant side panel (about 26% of the width); cropped off


def crop_right(png: Path, pixels: int) -> None:
    from PIL import Image

    img = Image.open(png)
    img.crop((0, 0, img.width - pixels, img.height)).save(png)


class Page:
    def __init__(self, ws):
        self.ws = ws
        self.next_id = 0

    async def call(self, method: str, **params):
        self.next_id += 1
        my_id = self.next_id
        await self.ws.send_str(json.dumps({"id": my_id, "method": method, "params": params}))
        while True:
            msg = json.loads((await self.ws.receive()).data)
            if msg.get("id") == my_id:
                return msg.get("result", {})


async def _capture(port: int, shots: list[tuple[str, Path, int, int, str | None]]) -> None:
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())  # noqa: S310
    ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")
    async with aiohttp.ClientSession() as session, session.ws_connect(ws_url, max_msg_size=0) as ws:
        page = Page(ws)
        await page.call("Page.enable")
        for url, png, width, height, script in shots:
            await page.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1,
                            mobile=False)
            await page.call("Page.navigate", url=url)
            await asyncio.sleep(6)
            for _ in range(2):  # the app re-renders after the first change
                await page.call("Runtime.evaluate", expression=PREPARE_MLFLOW)
                await asyncio.sleep(2)
            if script:
                await page.call("Runtime.evaluate", expression=script)
                await asyncio.sleep(2)
            shot = await page.call("Page.captureScreenshot", format="png")
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(base64.b64decode(shot["data"]))
            crop_right(png, round(width * ASSISTANT_PANEL_SHARE))
            print("screenshot:", png.name)


def capture_many(browser: str, profile: Path, shots: list[tuple[str, Path, int, int, str | None]],
                 port: int = 9333) -> None:
    proc = subprocess.Popen(  # noqa: S603 - local browser, localhost pages only
        [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run", "--disable-extensions", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2)  # noqa: S310
                break
            except OSError:
                time.sleep(0.5)
        asyncio.run(_capture(port, shots))
    finally:
        proc.terminate()
