"""Range-based user-id shard planning for stage 02_rev_import."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_user_id_shard_manifest(
    *,
    shard_manifest_parquet: str | Path,
    n_shards: int,
    user_id_min: int | None = None,
    user_id_max: int | None = None,
    overwrite: bool = False,
) -> dict[str, int | str]:
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")

    manifest_path = Path(shard_manifest_parquet)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not overwrite:
        df = pd.read_parquet(manifest_path)
    else:
        if user_id_min is None or user_id_max is None:
            raise ValueError("user_id_min and user_id_max are required when building a new shard manifest.")
        if user_id_min <= 0 or user_id_max <= 0:
            raise ValueError(
                f"user_id_min and user_id_max must be positive, got {user_id_min=} {user_id_max=}"
            )
        if user_id_max < user_id_min:
            raise ValueError(f"user_id_max must be >= user_id_min, got {user_id_min=} {user_id_max=}")

        total_user_id_span = int(user_id_max) - int(user_id_min) + 1
        effective_n_shards = min(int(n_shards), total_user_id_span)
        base_width = total_user_id_span // effective_n_shards
        remainder = total_user_id_span % effective_n_shards

        rows: list[dict[str, int]] = []
        next_lower = int(user_id_min)
        for shard_id in range(effective_n_shards):
            shard_width = base_width + (1 if shard_id < remainder else 0)
            lower_bound = next_lower
            upper_bound = lower_bound + shard_width - 1
            rows.append(
                {
                    "shard_id": shard_id,
                    "user_id_lower_bound": lower_bound,
                    "user_id_upper_bound": upper_bound,
                    "estimated_first_user_id": lower_bound,
                    "estimated_last_user_id": upper_bound,
                    "estimated_n_user_ids": shard_width,
                }
            )
            next_lower = upper_bound + 1
        df = pd.DataFrame(rows)
        df.to_parquet(manifest_path, index=False)

    return {
        "shard_manifest_parquet": str(manifest_path),
        "configured_n_shards": int(n_shards),
        "n_shards": int(df.shape[0]),
        "resolved_user_id_min": int(df["user_id_lower_bound"].min()),
        "resolved_user_id_max": int(df["user_id_upper_bound"].max()),
        "estimated_total_user_ids": int(df["estimated_n_user_ids"].fillna(0).sum()),
        "estimated_max_shard_size": int(df["estimated_n_user_ids"].fillna(0).max()),
    }
