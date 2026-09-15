"""Back-compat shim: use contracts/gate.py (Global Execution Contract System).

Equivalent to:
    python contracts/gate.py --contract contracts/smart_router_reference.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--contract", "contracts/smart_router_reference.md"]
    sys.exit(main())
