#!/usr/bin/env python3
"""
Download GloVe 6B pre-trained word vectors (Stanford NLP).

Files downloaded to data/glove/:
  glove.6B.50d.txt   ~69 MB
  glove.6B.100d.txt  ~128 MB   ← default used by training
  glove.6B.200d.txt  ~256 MB
  glove.6B.300d.txt  ~376 MB

Usage:
    python scripts/download_glove.py              # 100-dim only (fastest)
    python scripts/download_glove.py --all        # all four dims
"""
import argparse
import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
OUT_DIR   = "data/glove"


def download_with_progress(url: str, dest: str) -> None:
    def _progress(count, block_size, total_size):
        pct = count * block_size / total_size * 100
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Extract all four embedding sizes instead of just 100d")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    zip_path = os.path.join(args.out_dir, "glove.6B.zip")

    if not os.path.exists(zip_path):
        download_with_progress(GLOVE_URL, zip_path)
    else:
        print(f"Zip already downloaded: {zip_path}")

    print("Extracting …")
    keep = None if args.all else {"glove.6B.100d.txt"}

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in members:
            if keep is None or member in keep:
                dest = os.path.join(args.out_dir, member)
                if not os.path.exists(dest):
                    print(f"  {member}")
                    zf.extract(member, args.out_dir)
                else:
                    print(f"  {member} (already extracted)")

    print(f"\nDone. Vectors saved to {args.out_dir}/")
    print("Train with GloVe:")
    print("  python scripts/train.py --glove-path data/glove/glove.6B.100d.txt")


if __name__ == "__main__":
    main()
