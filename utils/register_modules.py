"""
Register custom ARF-YOLO modules with the Ultralytics framework.

This module patches the Ultralytics task module registry so that
AGRH, AMFF, FSRA, and MPFA can be referenced in YAML model configs.
"""

import sys
from pathlib import Path

# Ensure project root is importable
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.agrh import AGRH, MPFA
from models.amff import AMFF, FAUS, FRS, AFFS
from models.fsra import FSRA, ChannelAssigner, PatchAssigner, FrameAssigner


def register_custom_modules():
    """
    Register ARF-YOLO custom modules into the Ultralytics module registry.

    After calling this function, module names like 'AGRH', 'AMFF', 'FSRA'
    can be used directly in YAML model configuration files, and
    Ultralytics will instantiate the correct classes.
    """
    try:
        # Import the Ultralytics task map
        import ultralytics.nn.modules as unn
        import ultralytics.nn.tasks as tasks

        # Register each custom module
        custom_modules = {
            "AGRH": AGRH,
            "MPFA": MPFA,
            "AMFF": AMFF,
            "FAUS": FAUS,
            "FRS": FRS,
            "AFFS": AFFS,
            "FSRA": FSRA,
            "ChannelAssigner": ChannelAssigner,
            "PatchAssigner": PatchAssigner,
            "FrameAssigner": FrameAssigner,
        }

        for name, cls in custom_modules.items():
            # Add to ultralytics.nn.modules namespace
            setattr(unn, name, cls)

            # Add to the tasks module parse table if it exists
            if hasattr(tasks, "parse_model"):
                # Ensure the module class is accessible during model parsing
                if not hasattr(tasks, name):
                    setattr(tasks, name, cls)

        print(f"[ARF-YOLO] Registered {len(custom_modules)} custom modules: "
              f"{', '.join(custom_modules.keys())}")

    except ImportError as e:
        print(f"[ARF-YOLO] Warning: Could not register modules with Ultralytics: {e}")
        print("[ARF-YOLO] Custom modules are still available for standalone use.")

    return True
