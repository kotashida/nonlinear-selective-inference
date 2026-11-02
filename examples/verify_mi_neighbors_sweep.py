"""Verify the completed combined-AIS follow-up against the baseline manifest."""
import json
from pathlib import Path
import sys


def verify_variant(output_base, neighbors):
    root = Path(output_base) / f"neighbors_{neighbors}"
    expected = json.loads(Path(__file__).with_name("mi_neighbors_sweep_expected.json").read_text())
    expected["mi_neighbors"] = neighbors
    shards = list(root.glob("shard_*/validation_settings.json"))
    if len(shards) != 32:
        raise ValueError(f"Expected 32 completed shards, found {len(shards)}")
    for i in range(32):
        settings = json.loads((root / f"shard_{i}/validation_settings.json").read_text())
        required = {**expected, "seed": 20260908 + i,
                    "n_null_iters": 7 if i < 8 else 6,
                    "n_power_iters": 7 if i < 8 else 6}
        for key, value in required.items():
            if settings.get(key) != value:
                raise ValueError(f"shard {i}: unexpected {key}: {settings.get(key)!r}")
    pooled = json.loads((root / "pooled_summary/validation_settings.json").read_text())
    for key, value in {**expected, "n_shards": 32, "n_null_iters": 200,
                       "n_power_iters": 200,
                       "shard_root_seeds": list(range(20260908, 20260940))}.items():
        if pooled.get(key) != value:
            raise ValueError(f"Pooled settings: unexpected {key}")
    for name in ("calibration_decisions.csv", "power_decisions.csv"):
        path = root / "pooled_summary" / name
        if not path.is_file():
            raise ValueError(f"Missing {path}")
        print(path.read_text())
    print(f"Verified MI neighbors={neighbors}; 32 shards; 200 trials per condition; combined AIS.")


if __name__ == "__main__":
    for neighbors in (12, 16, 20, 25, 30):
        verify_variant(sys.argv[1], neighbors)
