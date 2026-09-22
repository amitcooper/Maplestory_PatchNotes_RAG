"""Makes src/ modules importable the same way they import each other.

The scripts in src/ use bare sibling imports (e.g. retrieve.py does
`from chunk import KNOWN_ENTITIES`), which only resolve because running
`python src/retrieve.py` puts src/ itself on sys.path. Tests need the same
setup rather than a `src.` package import, or retrieve.py's own internal
import would break.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
