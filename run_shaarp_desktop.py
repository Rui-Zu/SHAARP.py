"""Entry point for the SHAARP.py desktop app (PyInstaller target).

Build the .exe with build_desktop_exe.bat (PyInstaller one-folder, windowed).
"""

import sys

from shaarp.desktop_app import main

if __name__ == "__main__":
    sys.exit(main())
