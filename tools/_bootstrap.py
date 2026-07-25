"""Put the repo root on ``sys.path`` for the one-off scripts in this directory.

These scripts live in ``tools/`` but import the project's modules (``game``,
``dual_network``, ``mcts``, ``train``, ...) and the compiled ``quoridor_cpp``
extension, all of which sit one level up at the repo root.

Running ``python tools/_foo.py`` puts *this* directory on ``sys.path`` -- not
the repo root, and not the current working directory -- so those imports would
fail without this. Import it before any project import:

    import _bootstrap  # noqa: F401  (repo root -> sys.path)
    from game import State

Keeping ``tools/`` itself first on ``sys.path`` (Python's default for a script's
own directory) is what lets these scripts still import each other directly,
e.g. ``from _replay_game import draw_board``.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
