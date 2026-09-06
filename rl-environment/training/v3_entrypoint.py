"""Safe V3 container entrypoint; initialization never starts training."""
from __future__ import annotations

import json
import os

from v2_runtime import initialize_v2_runtime


def main() -> None:
    os.environ["TFM_RL_V3"] = "1"
    paths = initialize_v2_runtime()
    print(json.dumps({
        "status": "initialized",
        "message": "TFM RL V3 is isolated. Warm-start it explicitly, then launch v3_self_play.",
        "paths": paths,
    }, indent=2))


if __name__ == "__main__":
    main()
