#!/usr/bin/env python3
"""Download a public reaction-SMILES corpus (USPTO-50K, falling back to USPTO-MIT) for chemistry experiments."""

import argparse
import urllib.request
from pathlib import Path

SOURCES = {
    "USPTO_50K": "https://raw.githubusercontent.com/wengong-jin/nips2018/master/data/USPTO_50K.txt",
    "USPTO_MIT": "https://raw.githubusercontent.com/wengong-jin/nips2018/master/data/USPTO_MIT.txt",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="data/external")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        try:
            text = urllib.request.urlopen(url, timeout=60).read().decode()
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: download failed ({exc})")
            continue
        path = out / f"{name}.txt"
        path.write_text(text)
        lines = text.strip().splitlines()
        print(f"{name}: {len(lines)} reactions → {path}\nsample: {lines[0][:100]}")
        return
    raise SystemExit("all sources failed")


if __name__ == "__main__":
    main()
