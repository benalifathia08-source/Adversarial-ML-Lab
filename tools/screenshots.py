"""Make the README screenshots: real command output rendered as terminal windows, and the MLflow UI.

Uses headless Microsoft Edge (or Chrome) to capture PNGs into docs/screenshots/. Run after all stages:
    python tools/screenshots.py
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
WORK = ROOT / "artifacts" / "work" / "shots"
PY = sys.executable
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "msedge", "google-chrome", "chromium",
]
PORT = 5055


def browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    raise SystemExit("no Edge/Chrome found for screenshots")


def capture(url: str, png: Path, width: int = 1400, height: int = 900, wait_ms: int = 4000) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - fixed arguments, local files and localhost only
        [browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={width},{height}",
         f"--user-data-dir={WORK / 'browser-profile'}", "--no-first-run", "--disable-extensions",
         f"--virtual-time-budget={wait_ms}", f"--screenshot={png}", url],
        check=True, capture_output=True, timeout=90,
    )
    print("screenshot:", png.relative_to(ROOT))


def run(cmd: list[str]) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_COLOR="1", MLFLOW_DISABLE_AGENT_HINT="1")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",  # noqa: S603
                            errors="replace", env=env)
    text = (result.stdout + result.stderr).strip()
    # show repo-relative paths only (no local user folders in public screenshots)
    for root in (str(ROOT) + "\\", str(ROOT) + "/", str(ROOT)):
        text = text.replace(root, "")
    return text


TERMINAL = """<!doctype html><html><head><meta charset="utf-8"><style>
body {{ margin:0; background:#f6f8fa; font-family: 'Cascadia Mono', Consolas, monospace; }}
.win {{ margin:24px; border-radius:10px; overflow:hidden; box-shadow:0 8px 30px rgba(0,0,0,.18); background:#0d1117; }}
.bar {{ background:#21262d; color:#c9d1d9; padding:10px 14px; font: 13px 'Segoe UI', sans-serif; }}
.dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; }}
pre {{ margin:0; padding:18px 20px; color:#e6edf3; font-size:14px; line-height:1.45; white-space:pre-wrap; }}
.cmd {{ color:#7ee787; }} .bad {{ color:#ff7b72; font-weight:bold; }} .good {{ color:#3fb950; font-weight:bold; }}
.warn {{ color:#d29922; }}
</style></head><body><div class="win"><div class="bar">
<span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span>
<span class="dot" style="background:#28c840"></span>&nbsp; {title}</div><pre>{body}</pre></div></body></html>"""


def colorize(text: str) -> str:
    out = []
    for line in html.escape(text).splitlines():
        if any(w in line for w in ("REFUSED", "REJECTED", "CRITICAL", "FAILED", "ran code")):
            line = f'<span class="bad">{line}</span>'
        elif any(w in line for w in ("[ok]", "ACCEPTED", "passed", "No issues")):
            line = f'<span class="good">{line}</span>'
        elif line.startswith("$ "):
            line = f'<span class="cmd">{line}</span>'
        out.append(line)
    return "\n".join(out)


def terminal(name: str, title: str, sections: list[tuple[str, str]], height: int) -> None:
    body = "\n\n".join(f"$ {cmd}\n{output}" for cmd, output in sections)
    page = WORK / f"{name}.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(TERMINAL.format(title=html.escape(title), body=colorize(body)), encoding="utf-8")
    capture(page.as_uri(), OUT / f"{name}.png", width=1250, height=height, wait_ms=500)
    crop_background(OUT / f"{name}.png")


def crop_background(png: Path, margin: int = 24) -> None:
    """Cut the empty page area below the terminal window."""
    from PIL import Image

    img = Image.open(png).convert("RGB")
    bg = img.getpixel((img.width - 1, img.height - 1))
    rows = [y for y in range(img.height) if any(img.getpixel((x, y)) != bg for x in range(0, img.width, 7))]
    if rows:
        img.crop((0, 0, img.width, min(img.height, rows[-1] + margin))).save(png)


def shorten_scan(output: str) -> str:
    keep = [ln for ln in output.splitlines() if ln.strip() and not ln.startswith("WARNING")]
    return "\n".join(keep[:40])


def command_shots() -> None:
    sc = ROOT / "artifacts" / "work" / "supply_chain"
    evil = sc / "evil_checkpoint.pt"
    modelscan = str(Path(PY).with_name("modelscan.exe" if os.name == "nt" else "modelscan"))
    demo = json.loads((ROOT / "results" / "m6_supply_chain.json").read_text())["unsafe_pickle_demo"]
    terminal("01_malicious_pickle_modelscan", "Unsafe model format: pickle checkpoint with a hidden payload", [
        ("python -c \"torch.load('evil_checkpoint.pt', weights_only=False)\"",
         f"code ran while loading the model: {demo['torch_load_weights_only_false_ran_code']}  (marker file written)"),
        ("python -c \"torch.load('evil_checkpoint.pt', weights_only=True)\"",
         f"{demo['torch_load_weights_only_true']}; code ran: {demo['weights_only_true_ran_code']}"),
        ("modelscan -p evil_checkpoint.pt", shorten_scan(run([modelscan, "-p", str(evil)]))),
    ], height=900)
    terminal("02_verify_release", "Signed release: verify before loading", [
        ("python -m advml.verify models/release --gate", run([PY, "-m", "advml.verify", "models/release", "--gate"])),
    ], height=560)
    terminal("03_tamper_and_gate", "Tampered and weak models are refused", [
        ("python -m advml.verify artifacts/.../tampered_weights",
         run([PY, "-m", "advml.verify", str(sc / "tampered_weights")])),
        ("python -m advml.verify artifacts/.../extra_pickle",
         run([PY, "-m", "advml.verify", str(sc / "extra_pickle")])),
        ("python -m advml.verify artifacts/.../wrong_key", run([PY, "-m", "advml.verify", str(sc / "wrong_key")])),
        ("python -m advml.verify artifacts/.../baseline --gate",
         run([PY, "-m", "advml.verify", str(sc / "baseline"), "--gate"])),
    ], height=760)
    tests = run([PY, "-m", "pytest", "-p", "no:cacheprovider"])
    lint = run([PY, "-m", "ruff", "check", "src", "tests"])
    terminal("04_tests", "Tests and lint", [("ruff check src tests", lint), ("pytest -q", tests[-600:])], height=380)


def mlflow_shots() -> None:
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1")
    uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    server = subprocess.Popen(  # noqa: S603
        [PY, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(90):
            try:
                urllib.request.urlopen(base, timeout=2)  # noqa: S310 - localhost
                break
            except OSError:
                time.sleep(1)
        import mlflow

        mlflow.set_tracking_uri(uri)
        experiments = {e.name: e.experiment_id for e in mlflow.search_experiments()}
        rejected = mlflow.search_runs(experiment_names=["registry"], filter_string="tags.gate = 'rejected'")
        baseline_rejected = rejected[rejected["tags.mlflow.runName"] == "candidate_baseline"].iloc[0]["run_id"]
        sys.path.insert(0, str(Path(__file__).parent))
        from cdp_capture import capture_many

        capture_many(browser(), WORK / "cdp-profile", [
            (f"{base}/#/experiments/{experiments['baseline']}/runs?compareRunsMode=CHART",
             OUT / "05_mlflow_baseline_runs.png", 1800, 820, None),
            (f"{base}/#/experiments/{experiments['registry']}/runs/{baseline_rejected}",
             OUT / "06_mlflow_gate_rejected.png", 1600, 880, None),
            (f"{base}/#/models/unsw-nids-mlp", OUT / "07_mlflow_registry_champion.png", 1700, 480, None),
        ])
    finally:
        server.terminate()


if __name__ == "__main__":
    command_shots()
    mlflow_shots()
