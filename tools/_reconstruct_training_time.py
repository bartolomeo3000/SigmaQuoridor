"""One-off: retroactively fill self-play time into a lineage's training_stats.csv.

Background: cpp_train_loop.py runs self-play (selfplay_cpp.py) and training
(train.py --train-only) as *separate* subprocesses, and only train.py writes the
stats CSV. So historically `cycle_time_s` recorded training time only -- all the
self-play time (usually the bulk of a cycle) was never captured, and
`cumulative_time_s` badly undercounts total compute.

This script recovers the missing self-play time from the self-play logs. Every
selfplay_cpp.py run tees to logs/selfplay_<ts>.log, and each such log records
both its duration and the cycle it produced, e.g.:

    done: 2000 games in 251.3s (28648 games/hour)
      saved data_9x9_scratch\\cycle_0019.npz

We parse {cycle -> self-play seconds} from those logs (gated to the lineage's
data dir), then rewrite the CSV so that going backwards it matches the new
schema train.py now writes going forwards:

    cycle_time_s      = selfplay_time_s + train_time_s   (true per-cycle total)
    selfplay_time_s   = recovered from the self-play log (or estimated)
    train_time_s      = the OLD cycle_time_s value (training was all it measured)
    cumulative_time_s = running sum of the corrected cycle_time_s

The original CSV is backed up to <path>.bak before writing.

Usage:
    python _reconstruct_training_time.py                       # scratch lineage
    python _reconstruct_training_time.py --model-dir models_9x9_scratch \
        --data-dir-name data_9x9_scratch --logs logs --apply
    (omit --apply for a dry run that only prints the summary)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import statistics

_PAT_CYCLE = re.compile(r"cycle_(\d+)\.npz")
_PAT_DONE = re.compile(r"done:\s+\d+\s+games in ([\d.]+)s")


def parse_selfplay_logs(logs_dir: str, data_dir_name: str) -> dict[int, float]:
    """Map cycle number -> self-play seconds, from every selfplay log that
    produced a cycle_*.npz in ``data_dir_name``. If several logs map to the same
    cycle (a retried/re-run cycle), the last one in filename (chronological)
    order wins -- that's the run whose .npz actually survived on disk."""
    out: dict[int, float] = {}
    for path in sorted(glob.glob(os.path.join(logs_dir, "selfplay_*.log"))):
        try:
            txt = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if data_dir_name not in txt:          # not this lineage
            continue
        md = _PAT_DONE.search(txt)
        mc = _PAT_CYCLE.search(txt)
        if md and mc:
            out[int(mc.group(1))] = float(md.group(1))  # later files overwrite
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", default="runs/models_9x9_scratch",
                   help="lineage dir holding training_stats.csv")
    p.add_argument("--data-dir-name", default="runs/data_9x9_scratch",
                   help="the self-play data dir string as it appears in the logs "
                        "(used to gate logs to this lineage)")
    p.add_argument("--logs", default="runs/logs", help="directory of selfplay_*.log files")
    p.add_argument("--apply", action="store_true",
                   help="write the corrected CSV (otherwise dry-run: print only)")
    args = p.parse_args()

    stats_path = os.path.join(args.model_dir, "training_stats.csv")
    if not os.path.exists(stats_path):
        raise SystemExit(f"not found: {stats_path}")

    with open(stats_path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if "cycle_time_s" not in header:
        raise SystemExit(f"{stats_path} has no cycle_time_s column; nothing to reconstruct")
    if "selfplay_time_s" in header:
        raise SystemExit(f"{stats_path} already has selfplay_time_s -- looks already "
                         f"reconstructed. Aborting to avoid double-counting.")

    sp = parse_selfplay_logs(args.logs, args.data_dir_name)
    if not sp:
        raise SystemExit(f"no self-play logs matched {args.data_dir_name!r} in {args.logs}/")
    # Fallback for cycles with no recoverable log: median of the last 10 known
    # cycles (self-play time drifts as the net strengthens, so recent is a better
    # estimate than the global median).
    recent = [sp[c] for c in sorted(sp)[-10:]]
    fallback = statistics.median(recent)

    # New header: insert the breakdown columns right after cycle_time_s, matching
    # train.py's _STATS_COLUMNS order (cycle_time_s, selfplay_time_s, train_time_s,
    # cumulative_time_s, ...).
    new_header = list(header)
    i = new_header.index("cycle_time_s")
    new_header[i + 1:i + 1] = ["selfplay_time_s", "train_time_s"]

    cumulative = 0.0
    est_cycles: list[int] = []
    old_train_total = 0.0
    sp_total = 0.0
    for row in rows:
        cycle = int(row["cycle"])
        train_time = float(row["cycle_time_s"]) if row.get("cycle_time_s") else 0.0
        if cycle in sp:
            selfplay_time = sp[cycle]
        else:
            selfplay_time = fallback
            est_cycles.append(cycle)
        total = train_time + selfplay_time
        cumulative += total
        old_train_total += train_time
        sp_total += selfplay_time
        row["train_time_s"] = f"{train_time:.1f}"
        row["selfplay_time_s"] = f"{selfplay_time:.1f}"
        row["cycle_time_s"] = f"{total:.1f}"
        row["cumulative_time_s"] = f"{cumulative:.1f}"

    def hrs(s: float) -> str:
        return f"{s:,.0f}s ({s/3600:.1f}h)"

    print(f"lineage           : {args.model_dir}")
    print(f"cycles in CSV     : {len(rows)}")
    print(f"cycles matched    : {len(rows) - len(est_cycles)}  "
          f"(estimated via fallback: {len(est_cycles)} -> {est_cycles or '[]'}, "
          f"each {fallback:.0f}s)")
    print(f"training time      (old cycle_time_s sum): {hrs(old_train_total)}")
    print(f"self-play time     (recovered)           : {hrs(sp_total)}")
    print(f"TOTAL compute      (corrected cumulative): {hrs(cumulative)}")
    if not args.apply:
        print("\n[dry run] re-run with --apply to write the corrected CSV.")
        return

    backup = stats_path + ".bak"
    shutil.copy2(stats_path, backup)
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {stats_path}  (backup: {backup})")


if __name__ == "__main__":
    main()
