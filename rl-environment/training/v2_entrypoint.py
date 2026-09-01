"""Safe default container entrypoint: initialize v2 but never start legacy evolution."""
from __future__ import annotations

import json

from v2_runtime import initialize_v2_runtime


def main() -> None:
    paths = initialize_v2_runtime()
    print(json.dumps({
        "status": "initialized",
        "message": "TFM RL v2 is isolated. Run teacher collection, annotation import, pretraining, then self-play explicitly.",
        "paths": paths,
    }, indent=2))


if __name__ == "__main__":
    main()
