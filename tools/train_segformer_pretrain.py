#!/usr/bin/env python3
"""Thin Stage-1 entry reusing DDPS's AMP/runtime-audited MMCV train path."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from train_diffusion import main


if __name__ == '__main__':
    main()
