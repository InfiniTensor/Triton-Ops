# -*- coding: utf-8 -*-
"""Let tests/ find the kernels/ package (adds the project root to the module search path)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reference_precision import configure_reference

REFERENCE_PRECISION = configure_reference()


def pytest_report_header(config):
    return f"Reference precision: {REFERENCE_PRECISION}"
