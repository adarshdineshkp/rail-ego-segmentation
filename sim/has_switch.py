"""Exit code 0 if the run folder's scene has a switch (used by the .bat scripts)."""
import json
import sys
from pathlib import Path

try:
    sys.exit(0 if json.loads((Path(sys.argv[1]) / "meta.json").read_text()).get("switch_s_m") else 1)
except Exception:
    sys.exit(1)
