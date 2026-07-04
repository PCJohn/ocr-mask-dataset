"""One script to download every dataset this repo knows how to fetch
automatically -- no login/manual steps for any of them.

Usage:
    python -m src.download_all
    python -m src.download_all --datasets cord naf          # subset
    python -m src.download_all --skip-textocr                # textocr's zip is ~6.5GB, skip if you're short on space/time
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cord", "naf", "textocr", "synslides"],
                    choices=["cord", "naf", "textocr", "synslides", "bstd", "doclaynet"],
                    help="publaynet is streamed on demand, nothing to pre-download. "
                         "bstd (multilingual Indian scene text, ~17GB) and doclaynet (custom loading "
                         "script, unverified from this environment) aren't in the default set -- pass explicitly")
    args = p.parse_args()

    if "cord" in args.datasets:
        print("\n=== CORD ===")
        from src.datasets import cord
        cord.download()

    if "naf" in args.datasets:
        print("\n=== NAF ===")
        from src.datasets import naf
        naf.download()

    if "textocr" in args.datasets:
        print("\n=== TextOCR (this one's a big zip, ~6.5GB, be patient) ===")
        from src.datasets import textocr
        textocr.download()

    if "synslides" in args.datasets:
        print("\n=== SynSlides (synthetic lecture slides, ~544MB) ===")
        from src.datasets import synslides
        synslides.download()

    if "bstd" in args.datasets:
        print("\n=== BSTD (multilingual Indian scene text, ~17GB, be very patient) ===")
        from src.datasets import bstd
        bstd.download()

    if "doclaynet" in args.datasets:
        print("\n=== DocLayNet (real documents, line-level boxes) ===")
        from src.datasets import doclaynet
        list(doclaynet.iter_samples())  # caches via HF `datasets` on first load

    print("\nAll requested datasets downloaded. Now run:")
    print("  python -m src.build_dataset --datasets cord naf textocr synslides publaynet [bstd] [doclaynet]")


if __name__ == "__main__":
    main()
