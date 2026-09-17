"""Optional local dashboard over runs, checkpoints, evaluations, the candidate registry and benchmark evidence.

Start with ``forgeline dashboard``; export a static JSON snapshot with ``forgeline dashboard --export``. Training,
evaluation, serving and the registry do not depend on it.
"""

from forgeline.dashboard.data import (
    DashboardSources, benchmark_records, discover_runs, inspect_checkpoint, list_checkpoints, list_evaluations,
    registry_view, run_detail, snapshot,
)

__all__ = ["DashboardSources", "benchmark_records", "discover_runs", "inspect_checkpoint", "list_checkpoints",
           "list_evaluations", "registry_view", "run_detail", "snapshot"]
