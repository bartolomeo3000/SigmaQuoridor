"""Put the repo root on ``sys.path`` for the verification scripts in this directory.

These are plain argparse CLIs run directly (``python tests/test_cpp_parity.py --games 50``),
not pytest-discovered tests -- there is no pytest config in this repo. They import the
project's modules (``game``, ``dual_network``, ...) and the compiled ``quoridor_cpp``
extension, all of which sit one level up at the repo root.

Running ``python tests/test_x.py`` puts *this* directory on ``sys.path`` -- not the repo
root, and not the current working directory -- so those imports would fail without this.
Import it before any project import:

    import _bootstrap  # noqa: F401  (repo root -> sys.path)
    from game import State

Mirrors ``tools/_bootstrap.py``; kept as its own copy so each directory stays
self-contained (importing one from the other would need the very path fix it performs).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
