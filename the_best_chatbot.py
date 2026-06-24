"""Backward-compatible entry point. Prefer: python main.py chat"""

import sys
from main import main

if __name__ == "__main__":
    sys.exit(main(["chat"] + sys.argv[1:]))
