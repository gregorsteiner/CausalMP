"""Reproduce every figure and number in the paper by running each example script in turn.

Usage (from the repository root):
    python run_all.py               # everything
    python run_all.py --skip-r      # skip simulated_examples.illustration (needs R + npcausal + rpy2)
    python run_all.py --skip-slow   # skip simulated_examples.coverage (the long coverage experiment)
"""

import argparse
import os
import subprocess
import sys
import time

MODULES = [
    "real_examples.zinc",
    "real_examples.vitamin_a",
    "real_examples.lalonde",
    "simulated_examples.motivating",
    "simulated_examples.illustration",
    "simulated_examples.coverage",
    "simulated_examples.iv",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-r", action="store_true", help="skip the npcausal benchmark (simulated_examples.illustration)")
    parser.add_argument("--skip-slow", action="store_true", help="skip the coverage experiment (simulated_examples.coverage)")
    args = parser.parse_args()

    skip = set()
    if args.skip_r:
        skip.add("simulated_examples.illustration")
    if args.skip_slow:
        skip.add("simulated_examples.coverage")

    root = os.path.dirname(os.path.abspath(__file__))
    failed = []
    for module in MODULES:
        if module in skip:
            print(f"=== skipping {module}")
            continue
        print(f"=== running {module}", flush=True)
        start = time.time()
        ret = subprocess.run([sys.executable, "-m", module], cwd=root).returncode
        print(f"=== {module} finished in {time.time() - start:.0f}s (exit code {ret})", flush=True)
        if ret != 0:
            failed.append(module)

    if failed:
        print("Failed scripts:", ", ".join(failed))
        sys.exit(1)
    print("All scripts finished.")


if __name__ == "__main__":
    main()
