"""`agis setup` and `agis doctor`."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.prompt import Confirm
from rich.table import Table

from agis import __version__
from agis import config as config_mod
from agis.errors import AgisError

DEFAULT_HOST = "http://127.0.0.1:11434"
OLLAMA_APP = Path("/Applications/Ollama.app")


@dataclass
class Hardware:
    chip: str
    memory_gb: float
    apple_silicon: bool


@dataclass
class ModelChoice:
    name: str
    download_gb: float
    reason: str


def detect_hardware() -> Hardware:
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    memory = _sysctl("hw.memsize")
    if memory:
        memory_gb = int(memory) / 1024**3
    else:
        try:
            memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError):
            memory_gb = 0.0
    return Hardware(chip=chip, memory_gb=memory_gb, apple_silicon=apple_silicon)


def recommend_model(_hw: Hardware) -> ModelChoice:
    """The model setup offers. Hardware is still reported, but it doesn't change the choice."""
    return ModelChoice(config_mod.DEFAULT_MODEL, 4.7,
                       "best balance of speed and planning accuracy")


def ollama_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    bundled = OLLAMA_APP / "Contents" / "Resources" / "ollama"
    return str(bundled) if bundled.exists() else None


def ollama_version(host: str | None) -> str | None:
    """Server version, or None if Ollama isn't reachable."""
    try:
        response = httpx.get(f"{_host(host)}/api/version", timeout=2)
        response.raise_for_status()
        return response.json().get("version", "unknown")
    except (httpx.HTTPError, ValueError):
        return None


def installed_models(host: str | None) -> list[str]:
    return [m.model for m in ollama.Client(host=_host(host)).list().models]


def is_installed(model: str, installed: list[str]) -> bool:
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in installed


def run_setup(console: Console, model: str | None = None, host: str | None = None,
              assume_yes: bool = False) -> None:
    hw = detect_hardware()
    console.print(f"[bold]This Mac:[/bold] {escape(hw.chip)}, {hw.memory_gb:.0f} GB memory")
    if not hw.apple_silicon:
        console.print("[yellow]No Apple Silicon GPU detected. agis will work, but models run on "
                      "the CPU and planning can take much longer.[/yellow]")
    elif hw.memory_gb < 8:
        console.print("[yellow]Less than 8 GB of memory: expect slow responses.[/yellow]")

    if ollama_binary() is None:
        console.print(
            "\n[red]Ollama isn't installed.[/red] agis uses it to run models locally.\n"
            "Install it with [bold]brew install ollama[/bold] or download the app from "
            "https://ollama.com/download, then run [bold]agis setup[/bold] again."
        )
        raise AgisError("Ollama is required")

    version = ollama_version(host)
    if version is None:
        version = _start_ollama(console, host, assume_yes)
    console.print(f"[green]Ollama {escape(version)} is running[/green] at {_host(host)}")

    choice = ModelChoice(model, 0, "chosen with --model") if model else recommend_model(hw)
    console.print(f"[bold]Model:[/bold] {choice.name} ({choice.reason})")

    installed = installed_models(host)
    if is_installed(choice.name, installed):
        console.print(f"{choice.name} is already downloaded.")
    else:
        size = f" (about {choice.download_gb:g} GB)" if choice.download_gb else ""
        if not assume_yes and not _ask(f"Download {choice.name}{size}?", console):
            raise AgisError("setup cancelled; no model downloaded")
        _pull(console, choice.name, host)

    path = config_mod.save(choice.name, host)
    console.print(f"Saved settings to {path}")
    console.print('\n[bold green]Ready.[/bold green] In a folder with GIS files, try:\n'
                  "  agis\n"
                  '  agis "what is in this folder?"\n'
                  '  agis "reproject roads.shp to web mercator"')


def run_doctor(console: Console, host: str | None = None) -> bool:
    import geopandas
    import pyogrio
    import pyproj
    import rasterio
    import shapely

    cfg = config_mod.load()
    host = host or cfg.host
    hw = detect_hardware()
    rows: list[tuple[str, str, str]] = []

    def add(name: str, value: str, status: str = "ok"):
        rows.append((name, value, status))

    add("agis", __version__)
    add("python", f"{platform.python_version()} ({sys.executable})")
    mac = platform.mac_ver()[0]
    add("os", f"macOS {mac}" if mac else platform.platform())
    add("hardware", f"{hw.chip}, {hw.memory_gb:.0f} GB",
        "ok" if hw.apple_silicon else "warn")
    add("gdal", rasterio.__gdal_version__)
    add("proj", pyproj.proj_version_str)
    add("geos", shapely.geos_version_string)
    add("libraries", f"geopandas {geopandas.__version__}, rasterio {rasterio.__version__}, "
        f"pyogrio {pyogrio.__version__}, pyproj {pyproj.__version__}")

    path = config_mod.config_path()
    add("config", str(path) if cfg.from_file else f"{path} (not created; run agis setup)",
        "ok" if cfg.from_file else "warn")
    add("model", cfg.model)

    binary = ollama_binary()
    add("ollama binary", binary or "not found (brew install ollama)", "ok" if binary else "fail")
    version = ollama_version(host)
    add("ollama server", f"{version} at {_host(host)}" if version else
        f"not reachable at {_host(host)} (start with `ollama serve` or open Ollama.app)",
        "ok" if version else "fail")
    if version:
        installed = installed_models(host)
        if is_installed(cfg.model, installed):
            add("model installed", "yes")
        else:
            add("model installed", f"no (run `ollama pull {cfg.model}` or `agis setup`); "
                f"installed: {', '.join(installed) or 'none'}", "fail")

    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    marks = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]fail[/red]"}
    for name, value, status in rows:
        table.add_row(marks[status], name, escape(value))
    console.print(table)
    return all(status != "fail" for _, _, status in rows)


def _start_ollama(console: Console, host: str | None, assume_yes: bool) -> str:
    console.print("Ollama is installed but not running.")
    if not assume_yes and not _ask("Start it now?", console):
        raise AgisError("start Ollama with `ollama serve` or open Ollama.app, then rerun setup")
    if OLLAMA_APP.exists():
        subprocess.run(["open", "-a", "Ollama"], check=False)
    else:
        subprocess.Popen([ollama_binary(), "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    with console.status("Waiting for Ollama to start..."):
        for _ in range(30):
            time.sleep(0.5)
            version = ollama_version(host)
            if version:
                return version
    raise AgisError("Ollama didn't start; try running `ollama serve` in another terminal")


def _pull(console: Console, model: str, host: str | None) -> None:
    client = ollama.Client(host=_host(host))
    columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(),
               TransferSpeedColumn())
    try:
        with Progress(*columns, console=console) as progress:
            task = progress.add_task(f"Downloading {model}", total=None)
            for update in client.pull(model, stream=True):
                if update.total:
                    progress.update(task, total=update.total, completed=update.completed or 0)
                if update.status and not update.status.startswith("pulling"):
                    progress.update(task, description=update.status)
    except ollama.ResponseError as exc:
        raise AgisError(f"couldn't download {model}: {exc.error}") from exc


def _ask(question: str, console: Console) -> bool:
    if not sys.stdin.isatty():
        raise AgisError(f"{question} (not a terminal; rerun with --yes)")
    return Confirm.ask(question, default=True, console=console)


def _host(host: str | None) -> str:
    host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    return host if host.startswith("http") else f"http://{host}"


def _sysctl(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", name], check=False, capture_output=True,
                             text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None
