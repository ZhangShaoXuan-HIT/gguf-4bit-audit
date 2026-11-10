"""Download GGUF quantization levels and register them with Ollama.

Why this exists: Ollama's default tag for a model is a single quantization
(usually Q4_K_M). To study how precision affects behaviour you need several
levels of *the same weights*, which means pulling the GGUFs yourself and
importing them.

Usage
-----
    python scripts/pull_quants.py --list
    python scripts/pull_quants.py --models qwen2.5-0.5b --quants Q4_K_M Q8_0
    python scripts/pull_quants.py --models qwen2.5-0.5b qwen2.5-3b   # all quants

Downloads land in ``--models-dir`` (default F:/bio_llm/models if it exists, else
./models) and are deleted after Ollama imports them, because ``ollama create``
copies the blob into its own store.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def default_models_dir() -> Path:
    f = Path("F:/bio_llm/models")
    return f if Path("F:/").exists() else (config.ROOT / "models")


def candidates(prefix: str, quant: str) -> list[str]:
    """Filename patterns seen in the wild for bartowski-style GGUF repos."""
    return [
        f"{prefix}-{quant}.gguf",
        f"{prefix}.{quant}.gguf",
        f"{prefix.lower()}-{quant}.gguf",
    ]


def try_download(repo: str, prefix: str, quant: str, dest: Path) -> Path | None:
    """Download one GGUF, resuming partial files and retrying transient stalls.

    A plain ``urlopen`` with a long socket timeout is not robust against this
    mirror. Measured here, Llama-3.2-3B Q8_0 aborted at 250 MB of 3.1 GB, and a
    truncated file would otherwise be picked up by the cache check and imported
    as if complete. So: discard junk fragments, resume with an HTTP Range
    request, use a short per-read timeout so a stall surfaces quickly, and retry.
    """
    for fname in candidates(prefix, quant):
        url = f"{config.HF_MIRROR}/{repo}/resolve/main/{fname}"
        tmp = dest / fname

        if tmp.exists() and tmp.stat().st_size < 50 * 1024 * 1024:
            tmp.unlink(missing_ok=True)

        for attempt in range(6):
            pos = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if pos:
                headers["Range"] = f"bytes={pos}-"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as r:
                    status = getattr(r, "status", 200)
                    if status == 200 and pos:
                        pos = 0          # server ignored Range; restart cleanly
                        mode = "wb"
                    else:
                        mode = "ab" if pos else "wb"
                    total = int(r.headers.get("Content-Length", 0) or 0) + pos
                    t0 = time.perf_counter()
                    got = pos
                    last = 0.0
                    with tmp.open(mode) as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            got += len(chunk)
                            now = time.perf_counter()
                            if total and now - last > 5.0:
                                pct = 100 * got / total
                                mbps = (got - pos) / max(now - t0, 1e-6) / 1024**2
                                print(f"    {fname}: {pct:5.1f}%  {got/1024**2:7.0f}/"
                                      f"{total/1024**2:.0f} MB  {mbps:.1f} MB/s", flush=True)
                                last = now
                final = tmp.stat().st_size if tmp.exists() else 0
                if total and final >= total - 1024:
                    print(f"    done: {fname}  ({final/1024**2:.0f} MB)")
                    return tmp
                print(f"    incomplete ({final/1024**2:.0f}/{total/1024**2:.0f} MB), resuming")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break            # wrong filename; try the next candidate
                if e.code == 416 and tmp.exists():
                    print(f"    done: {fname} (server reports complete)")
                    return tmp
                print(f"    HTTP {e.code} (attempt {attempt + 1}/6), retrying")
                time.sleep(2 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                have = tmp.stat().st_size / 1024**2 if tmp.exists() else 0
                print(f"    {type(e).__name__} (attempt {attempt + 1}/6), resuming from {have:.0f} MB")
                time.sleep(2 * (attempt + 1))
    return None


def ollama_exists(name: str) -> bool:
    """True if the model is already registered.

    The ``encoding='utf-8', errors='replace'`` is load-bearing on Windows:
    Ollama draws a Unicode spinner (U+2800 block) even when stdout is captured,
    and the default GBK locale codec raises inside subprocess's reader thread,
    leaving ``stdout`` unusable. That silently made every existence check return
    False, so a re-run would re-download every GGUF.
    """
    try:
        out = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        ).stdout or ""
    except Exception:
        return False
    return any(line.split()[0] == name for line in out.splitlines()[1:] if line.strip())


def create_model(name: str, gguf: Path) -> bool:
    mf = gguf.with_suffix(".Modelfile")
    mf.write_text(
        f"FROM {gguf.as_posix()}\nPARAMETER temperature 0\nPARAMETER seed 42\n",
        encoding="ascii",
    )
    print(f"    ollama create {name} ...")
    r = subprocess.run(
        ["ollama", "create", name, "-f", str(mf)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode != 0:
        print(f"    create FAILED: {(r.stderr or '').strip()[:400]}")
        mf.unlink(missing_ok=True)
        return False
    mf.unlink(missing_ok=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--quants", nargs="*", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--keep-gguf", action="store_true", help="do not delete the downloaded GGUF")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    if a.list:
        print("available model keys:")
        for k, v in config.MODELS.items():
            print(f"  {k:<16} repo={v['repo']}")
        print("\nquants:", " ".join(config.QUANTS))
        return

    models = a.models or list(config.MODELS)
    quants = a.quants or config.QUANTS
    dest = Path(a.models_dir) if a.models_dir else default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"models dir: {dest}")
    free = shutil.disk_usage(dest).free / 1024**3
    print(f"free space: {free:.1f} GB\n")

    if not shutil.which("ollama"):
        print("!! `ollama` not on PATH. Start Ollama first.")
        return

    made, skipped, failed = 0, 0, 0
    for mk in models:
        if mk not in config.MODELS:
            print(f"!! unknown model {mk!r}")
            continue
        spec = config.MODELS[mk]
        for q in quants:
            name = f"{mk}:{q}"
            print(f"\n[{name}]")
            if ollama_exists(name):
                print("    already registered, skipping")
                skipped += 1
                continue
            gguf = try_download(spec["repo"], spec["prefix"], q, dest)
            if gguf is None:
                print(f"    could not find a file for {q}. Check the repo file list manually:")
                print(f"    {config.HF_MIRROR}/{spec['repo']}/tree/main")
                failed += 1
                continue
            if create_model(name, gguf):
                made += 1
                if not a.keep_gguf:
                    gguf.unlink(missing_ok=True)
            else:
                failed += 1

    print(f"\n=== done: {made} created, {skipped} already present, {failed} failed ===")
    print("verify with:  ollama list")


if __name__ == "__main__":
    main()

