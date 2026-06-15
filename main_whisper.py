"""Speech-enabled entry point for VL-GRiP3.

This is a thin wrapper around ``main.py`` that enables the Whisper voice prompt,
so the pipeline logic lives in a single place. It is equivalent to:

    python main.py --voice

Any extra CLI flags (``--no-capture``, ``--no-robot``, ``--scene-dir``, ...)
are forwarded to ``main``.
"""
import sys

from main import main

if __name__ == "__main__":
    main(["--voice", *sys.argv[1:]])
