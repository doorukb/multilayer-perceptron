"""
Make `src/` importable as a package root for tests so we can write
`from mlp import ...` without installing the project.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
