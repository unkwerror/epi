"""Точка входа для python -m epi."""

import sys

from .cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
