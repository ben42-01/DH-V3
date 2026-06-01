#!/usr/bin/env python3
"""Entry point for TraceReality experiments.

Usage:
    python main.py --n-states 80 --n-clusters 8 --n-observers 40 --n-steps 100000 --seeds 1 2 3 --prefix my_exp
"""

import sys
from pathlib import Path

# Add src to path so we can import tracereality
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracereality.experiments.main_core import main

if __name__ == "__main__":
    main()