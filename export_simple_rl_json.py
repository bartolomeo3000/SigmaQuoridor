"""Export tabular simple-RL agents to browser-readable JSON.

The static docs site cannot load Python joblib/pickle files, so this converts
the greedy policy data into compact JSON tables keyed by the exact integer
state hash used by rl_models.BaseAgent._state_to_key().
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib


SRC_DIR = Path("models_7x7_with_5_walls/best")
OUT_DIR = Path("docs/models/simple_rl")

DISPLAY_NAMES = {
    "QLearningAgent": "Q-Learning",
    "SarsaAgent": "SARSA",
    "ExpectedSarsaAgent": "Expected SARSA",
    "DoubleQLearningAgent": "Double Q-Learning",
    "DoubleSarsaAgent": "Double SARSA",
    "DoubleExpectedSarsaAgent": "Double Expected SARSA",
}


def _table(keys, indices, values):
    rows = []
    for state_key, idxs, vals in zip(keys, indices, values):
        pairs = [(int(idx), float(val)) for idx, val in zip(idxs, vals) if float(val) != 0.0]
        if not pairs:
            continue
        rows.append([str(int(state_key)), [idx for idx, _ in pairs], [val for _, val in pairs]])
    return rows


def _combined_double_table(data):
    by_state = {}
    for prefix in ("qa", "qb"):
        for state_key, idxs, vals in zip(data[f"{prefix}_keys"], data[f"{prefix}_indices"], data[f"{prefix}_values"]):
            key = str(int(state_key))
            state_vals = by_state.setdefault(key, {})
            for idx, val in zip(idxs, vals):
                if float(val) == 0.0:
                    continue
                i = int(idx)
                state_vals[i] = state_vals.get(i, 0.0) + float(val)
    return [
        [key, [i for i in sorted(state_vals) if state_vals[i] != 0.0], [state_vals[i] for i in sorted(state_vals) if state_vals[i] != 0.0]]
        for key, state_vals in sorted(by_state.items(), key=lambda item: int(item[0]))
        if any(value != 0.0 for value in state_vals.values())
    ]


def export_one(path: Path) -> None:
    data = joblib.load(path)
    class_name = path.stem.removesuffix("_compressed")
    if "q_keys" in data:
        table = _table(data["q_keys"], data["q_indices"], data["q_values"])
    else:
        table = _combined_double_table(data)

    out = {
        "id": class_name,
        "name": DISPLAY_NAMES.get(class_name, class_name),
        "boardsize": int(data["boardsize"]),
        "optimistic_init": float(data.get("optimistic_init", 0.0)),
        "table": table,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{class_name}.json"
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{dst} ({dst.stat().st_size / 1e6:.2f} MB, {len(table)} states)")


def main() -> None:
    for path in sorted(SRC_DIR.glob("*.pkl")):
        export_one(path)


if __name__ == "__main__":
    main()