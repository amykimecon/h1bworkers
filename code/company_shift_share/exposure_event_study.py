"""Exposure-based firm-level event study around the 2016 OPT policy change.

This module keeps the legacy exposure measures used in ``company_shift_share``
and adds a model-based ``opt_probability_index`` exposure. The index is trained
on pre-2016 firm features built by ``revelio_company_features.py``.

Exposure versions:
  1. ``opt_hire_rate``: pre-period OPT hires / pre-period new hires.
  2. ``school_opt_share``: share of hires from OPT-intensive schools.
  3. ``opt_probability_index``: modeled probability of post-2016 OPT use.

Modeling options for ``opt_probability_index``:
  - ``logit``: unpenalized logistic regression with optional class weights.
    Supports ``ntiles`` or ``continuous``.
  - ``lpm``: linear probability model. Supports ``ntiles`` or ``continuous``.
  - ``lasso``: L1-penalized logistic regression. Supports ``ntiles`` or
    ``continuous``.
  - ``random_forest``: binary classifier only. Event-study exposure is the
    hard predicted class, with ``ntiles=2`` and ``continuous`` disallowed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable, Optional, Sequence

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Flush stdout/stderr immediately for clean progress logging in interactive sessions.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    PanelOLS = None  # type: ignore[assignment,misc]

try:
    import statsmodels.api as sm
except ImportError:
    sm = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError:
    RandomForestClassifier = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    LogisticRegressionCV = None  # type: ignore[assignment]
    brier_score_loss = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    train_test_split = None  # type: ignore[assignment]

try:
    from company_shift_share.config_loader import (
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
        testing_enabled,
    )
    from company_shift_share.revelio_company_features import (
        load_or_build_company_features,
        validate_feature_window,
    )
    from company_shift_share.source_exposure_data import load_or_build_source_analysis_panel
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_PATH,
        apply_testing_output_suffix,
        get_cfg_section,
        load_config,
        testing_enabled,
    )
    from company_shift_share.revelio_company_features import (  # type: ignore[no-redef]
        load_or_build_company_features,
        validate_feature_window,
    )
    from company_shift_share.source_exposure_data import (  # type: ignore[no-redef]
        load_or_build_source_analysis_panel,
    )


LEGACY_EXPOSURE_VERSIONS = ("opt_hire_rate", "school_opt_share")
INDEX_EXPOSURE_VERSION = "opt_probability_index"
MODEL_METHODS = ("logit", "lpm", "lasso", "random_forest")
ENTRY_MODES = ("ntiles", "continuous")
PREDICTION_META_SUFFIX = ".meta.json"
EVENT_STUDY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "company_shift_share_exposure_event_study.yaml"
_CFG_BRACE_PATTERN = re.compile(r"\{([^{}]+)\}")
COEFFICIENT_MODEL_METHODS = {"logit", "lpm", "lasso"}
INTERACTION_MODEL_METHODS = {"lasso", "random_forest"}
MODEL_LABELS = {
    "logit": "Logit",
    "lpm": "LPM",
    "lasso": "LASSO",
    "random_forest": "RF",
}
INTERACTION_NUMERIC_ANCHORS = (
    "masters_opt_hire_rate_annual_pre_level",
    "school_opt_share_new_hire_masters_annual_pre_level",
    "school_opt_share_tenured_masters_annual_pre_level",
)
INTERACTION_CONTEXT_NUMERICS = (
    "company_n_users_log1p",
    "company_age_feature",
    "firm_size_annual_pre_level",
    "n_new_hires_annual_pre_level",
    "nonus_educ_share_annual_pre_level",
    "race_share_api_annual_pre_level",
    "occ_share_computing_math_annual_pre_level",
    "occ_share_engineering_annual_pre_level",
)
INTERACTION_ANCHOR_PAIRS = (
    ("masters_opt_hire_rate_annual_pre_level", "school_opt_share_new_hire_masters_annual_pre_level"),
    ("masters_opt_hire_rate_annual_pre_level", "school_opt_share_tenured_masters_annual_pre_level"),
)
INTERACTION_CATEGORY_SLOPES = {
    "naics2": (
        "masters_opt_hire_rate_annual_pre_level",
        "school_opt_share_new_hire_masters_annual_pre_level",
    ),
    "company_hq_region": (
        "masters_opt_hire_rate_annual_pre_level",
        "school_opt_share_new_hire_masters_annual_pre_level",
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _escape(path: Path) -> str:
    """Escape single quotes in a path string for use in DuckDB SQL literals."""
    return str(path).replace("'", "''")


def _resolve_path(paths_cfg: dict, key: str, *, allow_missing: bool = False) -> Path:
    """Resolve a path from config, substituting {root} with the repo root."""
    value = paths_cfg.get(key)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError(f"Config paths.{key} must be set.")
    root = str(Path(__file__).resolve().parents[2])
    path = Path(str(value).replace("{root}", root))
    if not allow_missing and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def _lookup_cfg_var(cfg: dict, key: str) -> object:
    current: object = cfg
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_optional_path_value(value: object, *, cfg: Optional[dict] = None) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    root = str(Path(__file__).resolve().parents[2])
    text = text.replace("{root}", root)
    if cfg is not None:
        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key == "root":
                return root
            resolved = _lookup_cfg_var(cfg, key)
            return str(resolved) if resolved is not None else match.group(0)

        text = _CFG_BRACE_PATTERN.sub(_sub, text)
    return Path(text).expanduser()


def _ensure_cfg_section(cfg: dict, name: str) -> dict:
    section = cfg.get(name)
    if section is None:
        cfg[name] = {}
        return cfg[name]
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section


def _resolve_main_config_path(config_path: Optional[Path]) -> Optional[Path]:
    if config_path is not None:
        return config_path
    env_config = os.getenv("EXPOSURE_EVENT_STUDY_CONFIG", "").strip()
    if env_config:
        return Path(env_config)
    if EVENT_STUDY_CONFIG_PATH.exists():
        return EVENT_STUDY_CONFIG_PATH
    return None


def _configure_legacy_cache_mode(flag: Optional[bool], config_default: Optional[bool] = None) -> bool:
    if flag is True:
        os.environ["EXPOSURE_EVENT_STUDY_LEGACY_CACHE"] = "1"
        return True
    if flag is False:
        os.environ.pop("EXPOSURE_EVENT_STUDY_LEGACY_CACHE", None)
        return False
    if config_default is not None:
        if config_default:
            os.environ["EXPOSURE_EVENT_STUDY_LEGACY_CACHE"] = "1"
        else:
            os.environ.pop("EXPOSURE_EVENT_STUDY_LEGACY_CACHE", None)
        return bool(config_default)
    return os.getenv("EXPOSURE_EVENT_STUDY_LEGACY_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configure_legacy_cache_ignore_keys(value: object) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        raw_values = [str(v).strip() for v in value]
    else:
        raw_values = [p.strip() for p in str(value).split(",")]
    keys = [k for k in raw_values if k]
    if keys:
        os.environ["EXPOSURE_EVENT_STUDY_LEGACY_CACHE_IGNORE_KEYS"] = ",".join(keys)
    else:
        os.environ.pop("EXPOSURE_EVENT_STUDY_LEGACY_CACHE_IGNORE_KEYS", None)


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self._streams[0], "encoding", None)


@contextmanager
def _tee_output(log_path: Optional[Path]):
    if log_path is None:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _TeeStream(original_stdout, handle)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, handle)  # type: ignore[assignment]
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + PREDICTION_META_SUFFIX)


def _write_metadata(path: Path, metadata: dict) -> None:
    meta_path = _metadata_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _savefig(name: str, slides_out: Optional[Path], oc_tag: str, version_tag: str) -> None:
    """Save current matplotlib figure with outcome + version tags appended to filename."""
    if slides_out is None:
        return
    stem, _, ext = name.rpartition(".")
    tagged = f"{stem}_{oc_tag}_{version_tag}.{ext}"
    plt.savefig(slides_out / tagged, dpi=150, bbox_inches="tight")
    print(f"[info] Saved {tagged}")


def _ensure_derived_outcome(df: pd.DataFrame, col: str, x_source_col: str) -> None:
    """Derive binary or log-transformed outcome columns in-place if not already present."""
    if col in df.columns:
        return
    if col.startswith("log1p_"):
        base = col[len("log1p_"):]
        if base in df.columns:
            df[col] = np.where(df[base].notna() & (df[base] >= 0), np.log1p(df[base]), np.nan)
        return
    if x_source_col not in df.columns:
        return
    if col == "x_bin_any_nonzero":
        df[col] = (df[x_source_col].fillna(0) != 0).astype("int8")


def _parse_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(float(text))


def _parse_float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _resolve_raw_plot_ntiles(exp_cfg: dict, event_study_ntiles: int) -> int:
    """Return the ntile count used for raw plots.

    By default, raw figures use at least quartiles even when the event-study
    regression is estimated on a coarser split such as two tiles.
    """
    raw_plot_ntiles = exp_cfg.get("raw_plot_ntiles")
    if raw_plot_ntiles is None:
        return max(int(event_study_ntiles), 4)
    raw_plot_ntiles = int(raw_plot_ntiles)
    if raw_plot_ntiles < 2:
        raise ValueError("raw_plot_ntiles must be at least 2.")
    return raw_plot_ntiles


def _describe_ntile_partition(ntiles: int) -> str:
    if ntiles == 4:
        return "quartile"
    if ntiles == 10:
        return "decile"
    return f"{ntiles}-tile"


def _select_index_analysis_firms(
    pred_df: pd.DataFrame,
    *,
    exclude_outside_negatives: bool,
) -> pd.DataFrame:
    cols = ["c", "predicted_index", "predicted_class", "preferred_rcid_source", "outside_negative_candidate"]
    sample_pred = (
        pred_df.loc[pred_df["event_study_sample"].eq(1), cols]
        .drop_duplicates(subset=["c"])
        .copy()
    )
    if sample_pred.empty:
        return sample_pred

    n_total = int(sample_pred["c"].nunique())
    n_preferred = int(sample_pred["preferred_rcid_source"].fillna(0).eq(1).sum())
    n_outside = int(sample_pred["outside_negative_candidate"].fillna(0).eq(1).sum())
    print(
        "[opt_probability_index] Held-out analysis firms before optional filtering: "
        f"{n_total:,} total | {n_preferred:,} preferred-source | {n_outside:,} outside negatives"
    )

    if exclude_outside_negatives:
        sample_pred = sample_pred[sample_pred["outside_negative_candidate"].fillna(0).ne(1)].copy()
        print(
            "[opt_probability_index] Excluding outside-negative firms from event-study analysis and raw plots: "
            f"{len(sample_pred):,} preferred-source firms remain"
        )

    return sample_pred


def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> Optional[float]:
    if roc_auc_score is None or y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_brier(y_true: pd.Series, y_score: pd.Series) -> Optional[float]:
    if brier_score_loss is None:
        return None
    return float(brier_score_loss(y_true, y_score))


def _load_feature_frame_for_window(
    cfg_full: dict,
    exp_cfg: dict,
    *,
    config_path: Optional[Path],
    year_min: int,
    year_max: int,
) -> tuple[pd.DataFrame, dict]:
    return load_or_build_company_features(
        config_path=config_path,
        cfg=cfg_full,
        feature_year_min=year_min,
        feature_year_max=year_max,
        force_rebuild=bool(exp_cfg.get("force_rebuild_company_features", False)),
    )


def _legacy_exposure_from_feature_frame(
    feature_df: pd.DataFrame,
    *,
    version: str,
    year_min: int,
    year_max: int,
) -> pd.Series:
    col_map = {
        "opt_hire_rate": "opt_hire_rate_annual_pre_level",
        "school_opt_share": "school_opt_share_new_hire_annual_pre_level",
    }
    if version not in col_map:
        raise ValueError(f"Unsupported legacy exposure version: {version!r}")
    exposure_col = col_map[version]
    if exposure_col not in feature_df.columns:
        raise ValueError(
            f"Feature frame is missing legacy exposure column '{exposure_col}'.\n"
            f"Available columns: {list(feature_df.columns)}"
        )

    analysis_only = feature_df[feature_df["in_analysis_universe"].fillna(0).eq(1)].copy()
    exposure = analysis_only[["c", exposure_col]].dropna(subset=[exposure_col]).copy()
    exposure[exposure_col] = pd.to_numeric(exposure[exposure_col], errors="coerce")
    exposure = exposure.dropna(subset=[exposure_col])
    print(
        f"[legacy exposure {version}] Using source-built features from {year_min}–{year_max}: "
        f"{len(exposure):,} firms with non-null exposure out of "
        f"{int(analysis_only['c'].nunique()):,} analysis-panel firms"
    )
    print(
        f"[legacy exposure {version}] Distribution:\n"
        f"{exposure[exposure_col].describe().to_string()}"
    )
    return exposure.drop_duplicates(subset=["c"], keep="first").set_index("c")[exposure_col]


# ---------------------------------------------------------------------------
# Exposure measure computation
# ---------------------------------------------------------------------------

def _compute_exposure_opt_hire_rate(
    panel: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.Series:
    """Aggregate OPT hire rate exposure over the configured legacy exposure window."""
    for col in ("masters_opt_hires_correction_aware", "y_new_hires_lag0"):
        if col not in panel.columns:
            raise ValueError(
                f"Analysis panel missing required column for opt_hire_rate exposure: '{col}'.\n"
                f"Available columns: {list(panel.columns)}"
            )

    win = panel[panel["t"].between(year_min, year_max)].copy()
    print(
        f"[exposure opt_hire_rate] Firm-year obs in [{year_min}, {year_max}]: {len(win)}, "
        f"firms: {win['c'].nunique()}"
    )

    agg = (
        win.groupby("c", as_index=False)
        .agg(
            opt_hires_sum=("masters_opt_hires_correction_aware", "sum"),
            new_hires_sum=("y_new_hires_lag0", "sum"),
        )
    )
    n_total = len(agg)
    agg = agg[agg["new_hires_sum"].notna() & (agg["new_hires_sum"] > 0)]
    n_dropped = n_total - len(agg)
    if n_dropped:
        print(f"[exposure opt_hire_rate] Dropped {n_dropped} firms with zero/null new-hire denominator.")
    print(f"[exposure opt_hire_rate] Firms with valid exposure: {len(agg)}")

    agg["exposure"] = agg["opt_hires_sum"] / agg["new_hires_sum"]
    print(f"[exposure opt_hire_rate] Distribution:\n{agg['exposure'].describe().to_string()}")
    return agg.set_index("c")["exposure"]


def _compute_exposure_school_opt_share(
    components: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.Series:
    """Share of hires from OPT-intensive schools, with schools classified inside the window."""
    for col in ("k", "t", "g_kt", "n_transitions_full", "total_new_hires_full"):
        if col not in components.columns:
            raise ValueError(
                f"instrument_components missing required column for school_opt_share: '{col}'.\n"
                f"Available: {list(components.columns)}"
            )

    win = components[components["t"].between(year_min, year_max)].copy()
    print(f"[exposure school_opt_share] Component rows in [{year_min}, {year_max}]: {len(win)}")

    school_rate = (
        win.groupby("k", as_index=False)["g_kt"]
        .mean()
        .rename(columns={"g_kt": "school_opt_rate"})
    )
    if school_rate.empty:
        raise ValueError("No school-level OPT rates available in the exposure window.")

    median_rate = school_rate["school_opt_rate"].median()
    school_rate["opt_intensive"] = school_rate["school_opt_rate"] > median_rate
    print(f"[exposure school_opt_share] Schools with g_kt in window: {len(school_rate)}")
    print(f"[exposure school_opt_share] Median school OPT rate: {median_rate:.4f}")
    print(
        f"[exposure school_opt_share] OPT-intensive schools: "
        f"{int(school_rate['opt_intensive'].sum())} / {len(school_rate)}"
    )

    comp_ck = (
        components.groupby(["c", "k"], as_index=False)
        .agg(
            n_transitions_full=("n_transitions_full", "first"),
            total_new_hires_full=("total_new_hires_full", "first"),
        )
    )
    comp_ck = comp_ck.merge(school_rate[["k", "opt_intensive"]], on="k", how="left")
    comp_ck["opt_intensive"] = comp_ck["opt_intensive"].fillna(False)

    intensive = (
        comp_ck[comp_ck["opt_intensive"]]
        .groupby("c", as_index=False)["n_transitions_full"]
        .sum()
        .rename(columns={"n_transitions_full": "intensive_transitions"})
    )
    firm_totals = comp_ck.groupby("c", as_index=False)["total_new_hires_full"].first()
    firm_exp = firm_totals.merge(intensive, on="c", how="left")
    firm_exp["intensive_transitions"] = firm_exp["intensive_transitions"].fillna(0)

    n_total = len(firm_exp)
    firm_exp = firm_exp[
        firm_exp["total_new_hires_full"].notna() & (firm_exp["total_new_hires_full"] > 0)
    ]
    n_dropped = n_total - len(firm_exp)
    if n_dropped:
        print(f"[exposure school_opt_share] Dropped {n_dropped} firms with zero/null denominator.")
    print(f"[exposure school_opt_share] Firms with valid exposure: {len(firm_exp)}")

    firm_exp["exposure"] = firm_exp["intensive_transitions"] / firm_exp["total_new_hires_full"]
    print(f"[exposure school_opt_share] Distribution:\n{firm_exp['exposure'].describe().to_string()}")
    return firm_exp.set_index("c")["exposure"]


# ---------------------------------------------------------------------------
# Exposure assignment / grouping
# ---------------------------------------------------------------------------

def _assign_ntiles(exposure: pd.Series, ntiles: int, zero_separate: bool = False) -> pd.DataFrame:
    """Assign firms to exposure ntile groups."""
    valid = exposure.dropna()
    n_dropped = len(exposure) - len(valid)
    if n_dropped:
        print(f"[ntiles] Dropped {n_dropped} firms with null exposure.")
    if valid.empty:
        raise ValueError("No firms remain after exposure filtering.")

    parts: list[pd.DataFrame] = []

    if zero_separate:
        zero_mask = valid <= 0
        n_zero = int(zero_mask.sum())
        pos = valid[~zero_mask]
        print(f"[ntiles] Zero-exposure firms: {n_zero} | Positive-exposure firms: {len(pos)}")

        if n_zero > 0:
            parts.append(pd.DataFrame({"c": valid.index[zero_mask], "ntile": 1}))

        n_pos_groups = ntiles - 1 if n_zero > 0 else ntiles
        offset = 2 if n_zero > 0 else 1
        if len(pos) > 0 and n_pos_groups > 0:
            codes, bins = pd.qcut(pos, n_pos_groups, labels=False, retbins=True, duplicates="drop")
            parts.append(pd.DataFrame({"c": pos.index, "ntile": (codes + offset).astype(int)}))
            for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:]), start=offset):
                print(f"  Q{i}: ({lo:.4g}, {hi:.4g}]")
        has_zero_group = n_zero > 0
    else:
        print(f"[ntiles] Firms: {len(valid)}")
        ranks = valid.rank(method="first")
        codes = pd.cut(ranks, bins=ntiles, labels=False, include_lowest=True)
        parts.append(pd.DataFrame({"c": valid.index, "ntile": (codes + 1).astype(int)}))
        tmp = pd.DataFrame({"exposure": valid.values, "ntile": (codes + 1).values})
        for q in sorted(tmp["ntile"].unique()):
            sub = tmp[tmp["ntile"] == q]["exposure"]
            print(f"  Q{q}: [{sub.min():.4g}, {sub.max():.4g}]  (n={len(sub)})")
        has_zero_group = False

    df = pd.concat(parts, ignore_index=True)
    actual_ntiles = int(df["ntile"].max())

    def _label(q: int, n: int) -> str:
        if has_zero_group and q == 1:
            return "Q1 (no OPT hires)"
        if q == n:
            return f"Q{n} (highest)"
        if q == (2 if has_zero_group else 1):
            return f"Q{q} (lowest positive)" if has_zero_group else "Q1 (lowest)"
        return f"Q{q}"

    df["ntile_label"] = df["ntile"].apply(lambda q: _label(q, actual_ntiles))
    print("[ntiles] Firm counts per ntile:")
    print(df.groupby(["ntile", "ntile_label"])["c"].count().to_string())
    return df


def _rf_group_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    valid = pred_df[["c", "predicted_class"]].dropna().copy()
    valid["predicted_class"] = valid["predicted_class"].astype(int)
    out = valid.rename(columns={"predicted_class": "rf_group"})
    out["ntile"] = out["rf_group"] + 1
    label_map = {0: "Predicted No OPT", 1: "Predicted OPT"}
    out["ntile_label"] = out["rf_group"].map(label_map)
    return out[["c", "ntile", "ntile_label"]]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_NTILE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def _ntile_color_map(ntile_order: Sequence[str]) -> dict[str, str]:
    return {label: _NTILE_COLORS[i % len(_NTILE_COLORS)] for i, label in enumerate(ntile_order)}


def _plot_raw_means(
    panel: pd.DataFrame,
    outcome_col: str,
    event_year: int,
    ref_year: int,
    ntile_order: Sequence[str],
    ntile_colors: dict[str, str],
    slides_out: Optional[Path],
    oc_tag: str,
    version_tag: str,
) -> None:
    """Plot mean outcome by calendar year, stratified by ntile group."""
    grp = panel.groupby(["ntile_label", "t"])[outcome_col]
    means = grp.mean().reset_index(name="mean_outcome")
    counts = grp.count().reset_index(name="n_obs")
    stds = grp.std().reset_index(name="std_outcome")
    stats = means.merge(counts, on=["ntile_label", "t"]).merge(stds, on=["ntile_label", "t"])
    stats["se"] = stats["std_outcome"] / np.sqrt(stats["n_obs"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label in ntile_order:
        s = stats[stats["ntile_label"] == label].sort_values("t")
        if s.empty:
            continue
        n_firms = panel[panel["ntile_label"] == label]["c"].nunique()
        ax.errorbar(
            s["t"],
            s["mean_outcome"],
            yerr=1.96 * s["se"].fillna(0),
            fmt="o-",
            capsize=3,
            color=ntile_colors.get(label),
            label=f"{label} (n={n_firms})",
        )

    ax.axvline(event_year, color="black", linestyle="--", linewidth=1, label=f"Event ({event_year})")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Mean {outcome_col}")
    ax.set_title(f"Outcome by OPT exposure group — {version_tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("exposure_es_raw.png", slides_out, oc_tag, version_tag)
    plt.show()
    plt.close(fig)


def _plot_ntile_regression(
    coef_df: pd.DataFrame,
    outcome_col: str,
    event_year: int,
    ref_year: int,
    non_ref_ntiles: Sequence[str],
    ntile_colors: dict[str, str],
    slides_out: Optional[Path],
    oc_tag: str,
    version_tag: str,
) -> None:
    """Plot year × group interaction coefficients on a single axes."""
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for ntile_label in non_ref_ntiles:
        s = coef_df[coef_df["ntile_label"] == ntile_label].sort_values("year")
        if s.empty:
            continue
        ax.errorbar(
            s["year"],
            s["coef"],
            yerr=1.96 * s["se"].fillna(0),
            fmt="o-",
            capsize=3,
            color=ntile_colors.get(ntile_label),
            label=ntile_label,
        )

    ax.axvline(event_year, color="black", linestyle="--", linewidth=1, label=f"Event ({event_year})")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Year × group coef vs reference in {ref_year} (± 1.96 SE)")
    ax.set_title(f"Year × OPT-exposure group interactions — {version_tag}\nOutcome: {outcome_col}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("exposure_es_reg.png", slides_out, oc_tag, version_tag)
    plt.show()
    plt.close(fig)


def _plot_continuous_regression(
    coef_df: pd.DataFrame,
    outcome_col: str,
    event_year: int,
    ref_year: int,
    slides_out: Optional[Path],
    oc_tag: str,
    version_tag: str,
) -> None:
    """Plot year × standardized num_opt_hires exposure coefficients."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    s = coef_df.sort_values("year")
    ax.errorbar(
        s["year"],
        s["coef"],
        yerr=1.96 * s["se"].fillna(0),
        fmt="o-",
        capsize=3,
        color="tab:blue",
        label="Per 1 SD higher num_opt_hires",
    )
    ax.axvline(event_year, color="black", linestyle="--", linewidth=1, label=f"Event ({event_year})")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Year × num_opt_hires coef vs {ref_year} (± 1.96 SE)")
    ax.set_title(f"Year × num_opt_hires interactions — {version_tag}\nOutcome: {outcome_col}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("exposure_es_continuous_reg.png", slides_out, oc_tag, version_tag)
    plt.show()
    plt.close(fig)


def _summarize_numeric_features(feature_df: pd.DataFrame, numeric_cols: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for col in numeric_cols:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        nonnull = series.dropna()
        if nonnull.empty:
            records.append(
                {
                    "feature": col,
                    "n_nonnull": 0,
                    "missing_pct": 100.0,
                    "zero_pct": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "p10": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n_unique": 0,
                }
            )
            continue
        records.append(
            {
                "feature": col,
                "n_nonnull": int(nonnull.notna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "zero_pct": float((nonnull == 0).mean() * 100),
                "mean": float(nonnull.mean()),
                "std": float(nonnull.std(ddof=0)) if len(nonnull) > 1 else 0.0,
                "p10": float(nonnull.quantile(0.10)),
                "p50": float(nonnull.quantile(0.50)),
                "p90": float(nonnull.quantile(0.90)),
                "min": float(nonnull.min()),
                "max": float(nonnull.max()),
                "n_unique": int(nonnull.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(records)


def _summarize_categorical_features(
    feature_df: pd.DataFrame,
    categorical_cols: Sequence[str],
    *,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    records: list[dict[str, object]] = []
    top_counts: dict[str, pd.Series] = {}
    for col in categorical_cols:
        values = (
            feature_df[col]
            .astype("string")
            .fillna("__MISSING__")
            .replace({"": "__MISSING__"})
        )
        counts = values.value_counts(dropna=False)
        top_counts[col] = counts.head(top_n)
        top_value = str(counts.index[0]) if len(counts) else ""
        top_share = float(counts.iloc[0] / counts.sum() * 100) if len(counts) else np.nan
        records.append(
            {
                "feature": col,
                "n_unique": int(values.nunique(dropna=False)),
                "missing_pct": float(values.eq("__MISSING__").mean() * 100),
                "top_value": top_value,
                "top_value_pct": top_share,
            }
        )
    return pd.DataFrame(records), top_counts


def _predictive_power_table(
    pred_df: pd.DataFrame,
    evaluation_mask: pd.Series,
    *,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_df = pred_df.loc[evaluation_mask, ["predicted_prob", "predicted_class", "post2016_any_opt", "target_source"]].copy()
    eval_df["predicted_prob"] = pd.to_numeric(eval_df["predicted_prob"], errors="coerce")
    eval_df["post2016_any_opt"] = pd.to_numeric(eval_df["post2016_any_opt"], errors="coerce")
    eval_df["predicted_class"] = pd.to_numeric(eval_df["predicted_class"], errors="coerce")
    eval_df = eval_df.dropna(subset=["predicted_prob", "post2016_any_opt", "predicted_class"]).copy()
    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), eval_df

    n_eval_bins = max(1, min(int(n_bins), int(eval_df["predicted_prob"].nunique())))
    if n_eval_bins == 1:
        eval_df["score_bin"] = 1
    else:
        eval_df["score_bin"] = pd.qcut(
            eval_df["predicted_prob"].rank(method="first"),
            q=n_eval_bins,
            labels=False,
            duplicates="drop",
        ) + 1

    bin_table = (
        eval_df.groupby("score_bin", as_index=False)
        .agg(
            n_obs=("post2016_any_opt", "size"),
            actual_rate=("post2016_any_opt", "mean"),
            predicted_rate=("predicted_prob", "mean"),
            predicted_class_share=("predicted_class", "mean"),
            score_min=("predicted_prob", "min"),
            score_max=("predicted_prob", "max"),
        )
    )
    conf = pd.crosstab(
        eval_df["post2016_any_opt"].astype(int),
        eval_df["predicted_class"].astype(int),
        dropna=False,
    )
    conf.index.name = "actual"
    conf.columns.name = "predicted"
    return bin_table, conf, eval_df


def _report_model_diagnostics(
    pred_df: pd.DataFrame,
    diagnostics: dict,
    artifacts: dict[str, object],
    *,
    detailed: bool,
    top_n: int,
    predictive_n_bins: int,
) -> None:
    feature_cols = [str(col) for col in artifacts.get("feature_columns_raw", [])]
    numeric_cols = [str(col) for col in artifacts.get("numeric_feature_columns_raw", [])]
    categorical_cols = [str(col) for col in artifacts.get("categorical_feature_columns_raw", [])]
    weight_series = artifacts.get("weight_series")
    evaluation_mask = artifacts.get("evaluation_mask")

    print("\n[model diagnostics] Sample composition:")
    sample_summary = (
        pred_df.groupby(["target_source", "post2016_any_opt"], dropna=False)
        .agg(
            n_firms=("c", "nunique"),
            in_training=("train_sample", "sum"),
            in_event_study=("event_study_sample", "sum"),
        )
        .reset_index()
    )
    print(sample_summary.to_string(index=False))

    if feature_cols:
        print(
            "\n[model diagnostics] Raw feature counts: "
            f"{len(feature_cols)} total | {len(numeric_cols)} numeric | {len(categorical_cols)} categorical"
        )
        interaction_count = int(diagnostics.get("n_interaction_columns_added", 0) or 0)
        if interaction_count > 0:
            print(
                "[model diagnostics] Interaction columns added: "
                f"{interaction_count} total | "
                f"{int(diagnostics.get('n_numeric_interaction_columns_added', 0) or 0)} numeric | "
                f"{int(diagnostics.get('n_category_slope_interaction_columns_added', 0) or 0)} category-slope"
            )
            skipped_sources = [str(col) for col in diagnostics.get("skipped_interaction_source_columns", [])]
            if skipped_sources:
                print(
                    "[model diagnostics] Skipped interaction sources: "
                    + ", ".join(skipped_sources)
                )
            if detailed:
                interaction_cols = [str(col) for col in artifacts.get("interaction_column_names", [])]
                if interaction_cols:
                    print("\n[model diagnostics] Active interaction columns:")
                    print(pd.Series(interaction_cols, name="interaction_feature").to_string(index=False))
        numeric_summary = _summarize_numeric_features(pred_df, numeric_cols) if numeric_cols else pd.DataFrame()
        categorical_summary, categorical_top = _summarize_categorical_features(
            pred_df,
            categorical_cols,
            top_n=top_n,
        ) if categorical_cols else (pd.DataFrame(), {})

        if not numeric_summary.empty:
            if detailed:
                print("\n[model diagnostics] Numeric feature summary:")
                print(numeric_summary.to_string(index=False))
            else:
                compact_numeric = numeric_summary.sort_values(
                    ["missing_pct", "std"],
                    ascending=[False, False],
                ).head(top_n)
                print("\n[model diagnostics] Numeric feature overview:")
                print(compact_numeric.to_string(index=False))

        if not categorical_summary.empty:
            print("\n[model diagnostics] Categorical feature overview:")
            display_cat = categorical_summary if detailed else categorical_summary.head(top_n)
            print(display_cat.to_string(index=False))
            if detailed:
                for col in categorical_cols:
                    counts = categorical_top.get(col)
                    if counts is None or counts.empty:
                        continue
                    print(f"\n[model diagnostics] Top levels for {col}:")
                    print(counts.to_string())

    if isinstance(weight_series, pd.Series) and not weight_series.empty:
        weight_summary = pd.Series(
            {
                "n_weights": int(weight_series.shape[0]),
                "mean": float(weight_series.mean()),
                "std": float(weight_series.std(ddof=0)) if len(weight_series) > 1 else 0.0,
                "min": float(weight_series.min()),
                "p10": float(weight_series.quantile(0.10)),
                "p50": float(weight_series.quantile(0.50)),
                "p90": float(weight_series.quantile(0.90)),
                "max": float(weight_series.max()),
            }
        )
        print("\n[model diagnostics] Model weight summary:")
        print(weight_summary.to_string())
        if str(diagnostics.get("model_method")) in COEFFICIENT_MODEL_METHODS:
            print("\n[model diagnostics] Largest positive coefficients:")
            print(weight_series.sort_values(ascending=False).head(top_n).to_string())
            print("\n[model diagnostics] Largest negative coefficients:")
            print(weight_series.sort_values(ascending=True).head(top_n).to_string())
        else:
            print("\n[model diagnostics] Largest feature importances:")
            print(weight_series.sort_values(ascending=False).head(top_n).to_string())

    if isinstance(evaluation_mask, pd.Series):
        bin_table, conf, eval_df = _predictive_power_table(
            pred_df,
            evaluation_mask,
            n_bins=predictive_n_bins,
        )
        print("\n[model diagnostics] Predictive power summary:")
        summary_metrics = pd.Series(
            {
                "evaluation_n": diagnostics.get("evaluation_n"),
                "evaluation_target_mean": diagnostics.get("evaluation_target_mean"),
                "evaluation_auc": diagnostics.get("evaluation_auc"),
                "evaluation_brier": diagnostics.get("evaluation_brier"),
                "evaluation_class_1_share": diagnostics.get("evaluation_class_1_share"),
            }
        )
        print(summary_metrics.to_string())
        if not conf.empty:
            print("\n[model diagnostics] Confusion matrix at 0.5 threshold:")
            print(conf.to_string())
        if not bin_table.empty:
            print("\n[model diagnostics] Score-bin predictive power:")
            print(bin_table.to_string(index=False))
        if detailed and not eval_df.empty:
            source_perf = (
                eval_df.groupby("target_source", as_index=False)
                .agg(
                    n_obs=("post2016_any_opt", "size"),
                    actual_rate=("post2016_any_opt", "mean"),
                    predicted_rate=("predicted_prob", "mean"),
                    predicted_class_share=("predicted_class", "mean"),
                )
            )
            print("\n[model diagnostics] Predictive power by source:")
            print(source_perf.to_string(index=False))


def _plot_predictive_power(
    pred_df: pd.DataFrame,
    diagnostics: dict,
    artifacts: dict[str, object],
    *,
    slides_out: Optional[Path],
    version_tag: str,
    predictive_n_bins: int,
) -> None:
    evaluation_mask = artifacts.get("evaluation_mask")
    if not isinstance(evaluation_mask, pd.Series):
        return
    bin_table, _, eval_df = _predictive_power_table(
        pred_df,
        evaluation_mask,
        n_bins=predictive_n_bins,
    )
    if eval_df.empty:
        print("[model diagnostics] No evaluation sample available for predictive-power plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax_left, ax_right = axes

    if not bin_table.empty:
        ax_left.plot([0, 1], [0, 1], linestyle=":", color="grey", linewidth=1)
        ax_left.plot(
            bin_table["predicted_rate"],
            bin_table["actual_rate"],
            marker="o",
            color="tab:blue",
        )
        for _, row in bin_table.iterrows():
            ax_left.annotate(str(int(row["score_bin"])), (row["predicted_rate"], row["actual_rate"]), fontsize=8)
        ax_left.set_xlabel("Mean predicted probability")
        ax_left.set_ylabel("Observed OPT rate")
        ax_left.set_title("Calibration by score bin")

    pos = eval_df[eval_df["post2016_any_opt"].eq(1)]["predicted_prob"]
    neg = eval_df[eval_df["post2016_any_opt"].eq(0)]["predicted_prob"]
    bins = np.linspace(0, 1, 11)
    if not neg.empty:
        ax_right.hist(neg, bins=bins, alpha=0.6, label="Actual 0", color="tab:orange")
    if not pos.empty:
        ax_right.hist(pos, bins=bins, alpha=0.6, label="Actual 1", color="tab:green")
    ax_right.set_xlabel("Predicted probability")
    ax_right.set_ylabel("Count")
    auc_value = diagnostics.get("evaluation_auc")
    title = "Score distribution by class"
    if auc_value is not None:
        title += f"\nAUC={float(auc_value):.3f}"
    ax_right.set_title(title)
    ax_right.legend(fontsize=8)

    fig.tight_layout()
    _savefig("opt_index_predictive_power.png", slides_out, "model", version_tag)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _run_ntile_regression(
    panel: pd.DataFrame,
    outcome_col: str,
    ref_year: int,
    data_min_t: int,
    data_max_t: int,
) -> Optional[pd.DataFrame]:
    """Panel regression with firm FE, year FE, and year × group interactions."""
    if PanelOLS is None:
        print("[regression] linearmodels not installed. Install with: pip install linearmodels")
        return None

    work = (
        panel[panel["t"].between(data_min_t, data_max_t)]
        .dropna(subset=[outcome_col])
        .drop_duplicates(subset=["c", "t"])
        .reset_index(drop=True)
    )
    work[outcome_col] = work[outcome_col].astype(float)

    print(
        f"\n[regression] Outcome: {outcome_col} | N obs: {len(work)} | "
        f"N firms: {work['c'].nunique()} | N years: {work['t'].nunique()}"
    )
    if len(work) < 10:
        print("[regression] Too few observations. Skipping.")
        return None

    ntile_order_df = work[["ntile", "ntile_label"]].drop_duplicates().sort_values("ntile")
    ntile_order_all = ntile_order_df["ntile_label"].tolist()
    ntile_ref = ntile_order_all[0]
    non_ref_ntiles = ntile_order_all[1:]

    years = sorted(work["t"].unique())
    non_ref_years = [y for y in years if y != ref_year]
    if not non_ref_years or not non_ref_ntiles:
        print("[regression] Not enough year or group variation to estimate interactions.")
        return None

    year_dummies = pd.get_dummies(work["t"], prefix="yr", dtype=float)
    ref_yr_col = f"yr_{ref_year}"
    if ref_yr_col in year_dummies.columns:
        year_dummies = year_dummies.drop(columns=[ref_yr_col])

    interact_cols: dict[str, pd.Series] = {}
    for yr in non_ref_years:
        for ntile_label in non_ref_ntiles:
            safe = ntile_label.replace(" ", "_").replace("(", "").replace(")", "")
            col_name = f"yr{yr}_x_{safe}"
            interact_cols[col_name] = (
                (work["t"] == yr).astype(float) * (work["ntile_label"] == ntile_label).astype(float)
            )

    exog = pd.concat([year_dummies, pd.DataFrame(interact_cols, index=work.index)], axis=1)
    zero_var = exog.columns[exog.std() == 0].tolist()
    if zero_var:
        exog = exog.drop(columns=zero_var)

    idx = pd.MultiIndex.from_arrays([work["c"].values, work["t"].values], names=["c", "t"])
    dep = pd.Series(work[outcome_col].values, index=idx, name=outcome_col)
    exog_indexed = exog.set_index(idx)

    try:
        model = PanelOLS(dependent=dep, exog=exog_indexed, entity_effects=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        print(f"[regression] PanelOLS failed: {exc}")
        return None

    records: list[dict] = []
    for yr in non_ref_years:
        for ntile_label in non_ref_ntiles:
            safe = ntile_label.replace(" ", "_").replace("(", "").replace(")", "")
            col_name = f"yr{yr}_x_{safe}"
            if col_name not in result.params.index:
                continue
            records.append(
                {
                    "year": yr,
                    "ntile_label": ntile_label,
                    "coef": float(result.params[col_name]),
                    "se": float(result.std_errors[col_name]),
                    "tstat": float(result.tstats[col_name]),
                    "pval": float(result.pvalues[col_name]),
                }
            )
    if not records:
        print("[regression] No interaction coefficients extracted.")
        return None

    coef_df = pd.DataFrame(records)
    print(f"\n[regression] Interaction coefficients (year × group vs {ntile_ref} in {ref_year}):")
    print(coef_df.to_string(index=False))
    return coef_df


def _run_continuous_regression(
    panel: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    ref_year: int,
    data_min_t: int,
    data_max_t: int,
) -> Optional[pd.DataFrame]:
    """Panel regression with firm FE, year FE, and year × num_opt_hires interactions."""
    if PanelOLS is None:
        print("[regression] linearmodels not installed. Install with: pip install linearmodels")
        return None

    work = (
        panel[panel["t"].between(data_min_t, data_max_t)]
        .dropna(subset=[outcome_col, exposure_col])
        .drop_duplicates(subset=["c", "t"])
        .reset_index(drop=True)
    )
    work[outcome_col] = work[outcome_col].astype(float)
    work[exposure_col] = work[exposure_col].astype(float)

    print(
        f"\n[regression num_opt_hires] Outcome: {outcome_col} | N obs: {len(work)} | "
        f"N firms: {work['c'].nunique()} | N years: {work['t'].nunique()}"
    )
    if len(work) < 10:
        print("[regression num_opt_hires] Too few observations. Skipping.")
        return None

    years = sorted(work["t"].unique())
    non_ref_years = [y for y in years if y != ref_year]
    if not non_ref_years:
        print("[regression num_opt_hires] No non-reference years in sample.")
        return None

    year_dummies = pd.get_dummies(work["t"], prefix="yr", dtype=float)
    ref_yr_col = f"yr_{ref_year}"
    if ref_yr_col in year_dummies.columns:
        year_dummies = year_dummies.drop(columns=[ref_yr_col])

    interact_cols = {
        f"yr{yr}_x_index": (work["t"] == yr).astype(float) * work[exposure_col]
        for yr in non_ref_years
    }
    exog = pd.concat([year_dummies, pd.DataFrame(interact_cols, index=work.index)], axis=1)
    zero_var = exog.columns[exog.std() == 0].tolist()
    if zero_var:
        exog = exog.drop(columns=zero_var)

    idx = pd.MultiIndex.from_arrays([work["c"].values, work["t"].values], names=["c", "t"])
    dep = pd.Series(work[outcome_col].values, index=idx, name=outcome_col)
    exog_indexed = exog.set_index(idx)

    try:
        model = PanelOLS(dependent=dep, exog=exog_indexed, entity_effects=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        print(f"[regression num_opt_hires] PanelOLS failed: {exc}")
        return None

    records: list[dict] = []
    for yr in non_ref_years:
        col_name = f"yr{yr}_x_index"
        if col_name not in result.params.index:
            continue
        records.append(
            {
                "year": yr,
                "coef": float(result.params[col_name]),
                "se": float(result.std_errors[col_name]),
                "tstat": float(result.tstats[col_name]),
                "pval": float(result.pvalues[col_name]),
            }
        )
    if not records:
        print("[regression num_opt_hires] No interaction coefficients extracted.")
        return None

    coef_df = pd.DataFrame(records)
    print(f"\n[regression num_opt_hires] Interaction coefficients (year × num_opt_hires vs {ref_year}):")
    print(coef_df.to_string(index=False))
    return coef_df


# ---------------------------------------------------------------------------
# OPT probability index helpers
# ---------------------------------------------------------------------------

def validate_opt_probability_config(
    *,
    model_method: str,
    entry_mode: str,
    ntiles: int,
    feature_year_min: int,
    feature_year_max: int,
    leaveout_enabled: bool,
    leaveout_share: float,
) -> None:
    """Validate model-specific constraints for the opt_probability_index workflow."""
    validate_feature_window(feature_year_min, feature_year_max)
    if model_method not in MODEL_METHODS:
        raise ValueError(f"index_model_method must be one of {MODEL_METHODS}, got {model_method!r}.")
    if entry_mode not in ENTRY_MODES:
        raise ValueError(f"index_entry_mode must be one of {ENTRY_MODES}, got {entry_mode!r}.")
    if model_method == "random_forest" and entry_mode == "continuous":
        raise ValueError("random_forest requires discrete exposure. Set index_entry_mode='ntiles'.")
    if model_method == "random_forest" and int(ntiles) != 2:
        raise ValueError("random_forest requires ntiles=2 because event-study exposure is binary.")
    if leaveout_enabled and not (0 < float(leaveout_share) < 1):
        raise ValueError("leaveout_share must lie strictly between 0 and 1 when leaveout_enabled=true.")


def _build_post2016_target(
    panel: pd.DataFrame,
    *,
    x_source_col: str,
    target_year_min: int,
    target_year_max: int,
) -> pd.DataFrame:
    """Build the firm-level binary target: any OPT use in the post period."""
    if x_source_col not in panel.columns:
        raise ValueError(f"Analysis panel is missing target source column '{x_source_col}'.")
    target = panel[panel["t"].between(target_year_min, target_year_max)][["c", x_source_col]].copy()
    target[x_source_col] = pd.to_numeric(target[x_source_col], errors="coerce").fillna(0)
    target = (
        target.groupby("c", as_index=False)[x_source_col]
        .max()
        .rename(columns={x_source_col: "max_opt_post"})
    )
    target["post2016_any_opt"] = (target["max_opt_post"] > 0).astype(int)
    all_analysis_firms = panel[["c"]].drop_duplicates().copy()
    out = all_analysis_firms.merge(target[["c", "post2016_any_opt"]], on="c", how="left")
    out["post2016_any_opt"] = out["post2016_any_opt"].fillna(0).astype(int)
    return out


INDEX_FEATURE_FAMILIES = ("all_features", "matching_fundamentals")


def _feature_family_excludes(column: str, feature_family: str) -> bool:
    if feature_family == "all_features":
        return False
    if feature_family == "matching_fundamentals":
        return (
            column.startswith("school_")
            or column.startswith("n_schools_")
            or column.startswith("opt_")
            or "_opt_" in column
        )
    raise ValueError(
        f"Unsupported feature_family={feature_family!r}. "
        f"Expected one of {INDEX_FEATURE_FAMILIES}."
    )


def _select_index_feature_columns(
    feature_df: pd.DataFrame,
    *,
    feature_family: str = "all_features",
) -> list[str]:
    exclude = {
        "c",
        "in_analysis_universe",
        "preferred_rcid_source",
        "outside_negative_candidate",
        "post2016_any_opt",
        "target_source",
        "train_sample",
        "event_study_sample",
        "leaveout_training_firm",
        "predicted_prob",
        "predicted_class",
        "predicted_index",
        "exposure_value",
        "model_method",
        "index_entry_mode",
    }
    feature_cols = [
        col
        for col in feature_df.columns
        if col not in exclude and not _feature_family_excludes(str(col), feature_family)
    ]
    return sorted(feature_cols)


def _infer_feature_types(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
    for col in feature_cols:
        series = feature_df[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    return categorical_cols, numeric_cols


def _model_label(model_method: str) -> str:
    return MODEL_LABELS.get(model_method, model_method.replace("_", " ").upper())


def _build_interaction_features(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    model_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    meta: dict[str, object] = {
        "interaction_columns": [],
        "numeric_interaction_columns": [],
        "category_slope_interaction_columns": [],
        "skipped_interaction_source_columns": [],
    }
    if model_method not in INTERACTION_MODEL_METHODS:
        return x_train, x_all, meta

    raw_feature_set = set(feature_cols)
    train_col_set = set(x_train.columns)
    skipped_sources: set[str] = set()
    numeric_specs: list[tuple[str, str, str]] = []
    category_slope_specs: list[tuple[str, str, str]] = []

    def _require_numeric_source(col: str) -> bool:
        if col not in raw_feature_set or col not in train_col_set:
            skipped_sources.add(col)
            return False
        return True

    for anchor in INTERACTION_NUMERIC_ANCHORS:
        for context in INTERACTION_CONTEXT_NUMERICS:
            if not _require_numeric_source(anchor) or not _require_numeric_source(context):
                continue
            name = f"ix__{anchor}__x__{context}"
            numeric_specs.append((name, anchor, context))

    for left, right in INTERACTION_ANCHOR_PAIRS:
        if not _require_numeric_source(left) or not _require_numeric_source(right):
            continue
        name = f"ix__{left}__x__{right}"
        numeric_specs.append((name, left, right))

    for category_col, anchors in INTERACTION_CATEGORY_SLOPES.items():
        if category_col not in raw_feature_set:
            skipped_sources.add(category_col)
            continue
        dummy_cols = [col for col in x_train.columns if col.startswith(f"{category_col}_")]
        if not dummy_cols:
            skipped_sources.add(category_col)
            continue
        for anchor in anchors:
            if not _require_numeric_source(anchor):
                continue
            for dummy_col in dummy_cols:
                name = f"ix__{dummy_col}__x__{anchor}"
                category_slope_specs.append((name, dummy_col, anchor))

    interaction_train: dict[str, pd.Series] = {}
    interaction_all: dict[str, pd.Series] = {}
    for name, left, right in numeric_specs:
        interaction_train[name] = x_train[left].astype(float) * x_train[right].astype(float)
        interaction_all[name] = x_all[left].astype(float) * x_all[right].astype(float)
    for name, dummy_col, anchor in category_slope_specs:
        interaction_train[name] = x_train[dummy_col].astype(float) * x_train[anchor].astype(float)
        interaction_all[name] = x_all[dummy_col].astype(float) * x_all[anchor].astype(float)

    meta["interaction_columns"] = list(interaction_train.keys()) + list(
        name for name in interaction_all.keys() if name not in interaction_train
    )
    meta["numeric_interaction_columns"] = [name for name, _, _ in numeric_specs]
    meta["category_slope_interaction_columns"] = [name for name, _, _ in category_slope_specs]
    meta["skipped_interaction_source_columns"] = sorted(skipped_sources)
    if not interaction_train:
        return x_train, x_all, meta

    x_train_aug = pd.concat([x_train, pd.DataFrame(interaction_train, index=x_train.index)], axis=1)
    x_all_aug = pd.concat([x_all, pd.DataFrame(interaction_all, index=x_all.index)], axis=1)
    return x_train_aug, x_all_aug, meta


def _finalize_interaction_meta(
    interaction_meta: dict[str, object],
    *,
    active_cols: Sequence[str],
) -> dict[str, object]:
    active_col_set = set(active_cols)
    interaction_cols = [
        str(col) for col in interaction_meta.get("interaction_columns", [])
        if str(col) in active_col_set
    ]
    numeric_interaction_cols = [
        str(col) for col in interaction_meta.get("numeric_interaction_columns", [])
        if str(col) in active_col_set
    ]
    category_slope_interaction_cols = [
        str(col) for col in interaction_meta.get("category_slope_interaction_columns", [])
        if str(col) in active_col_set
    ]
    return {
        "interaction_column_names": interaction_cols,
        "numeric_interaction_column_names": numeric_interaction_cols,
        "category_slope_interaction_column_names": category_slope_interaction_cols,
        "n_interaction_columns_added": int(len(interaction_cols)),
        "n_numeric_interaction_columns_added": int(len(numeric_interaction_cols)),
        "n_category_slope_interaction_columns_added": int(len(category_slope_interaction_cols)),
        "skipped_interaction_source_columns": [
            str(col) for col in interaction_meta.get("skipped_interaction_source_columns", [])
        ],
    }


def _is_binary_active_column(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(values) == 0:
        return False
    return set(np.round(values.astype(float), 12).tolist()).issubset({0.0, 1.0})


def _standardize_nonbinary_features(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train_std = x_train.copy()
    x_all_std = x_all.copy()
    standardized_cols: list[str] = []
    for col in x_train.columns:
        if _is_binary_active_column(x_train[col]):
            continue
        mean = float(pd.to_numeric(x_train[col], errors="coerce").mean())
        std = float(pd.to_numeric(x_train[col], errors="coerce").std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            continue
        x_train_std[col] = (pd.to_numeric(x_train[col], errors="coerce") - mean) / std
        x_all_std[col] = (pd.to_numeric(x_all[col], errors="coerce") - mean) / std
        standardized_cols.append(col)
    return x_train_std, x_all_std, standardized_cols


def _resolve_lasso_cv_folds(y_train: pd.Series, requested_folds: Optional[int] = None) -> int:
    class_counts = y_train.value_counts()
    if class_counts.empty or int(class_counts.min()) < 2:
        raise ValueError("lasso requires at least 2 observations in each training class for cross-validation.")
    max_available_folds = int(class_counts.min())
    if requested_folds is None:
        return max(2, min(5, max_available_folds))
    requested = int(requested_folds)
    if requested < 2:
        raise ValueError("lasso_cv_folds must be at least 2 when provided.")
    return min(requested, max_available_folds)


def _resolve_lasso_cs(n_cs: Optional[int] = None) -> np.ndarray:
    if n_cs is None:
        return np.logspace(-4, 4, 25)
    resolved = int(n_cs)
    if resolved < 2:
        raise ValueError("lasso_n_cs must be at least 2 when provided.")
    return np.logspace(-4, 4, resolved)


def _resolve_binary_class_weight(
    requested: object,
    *,
    default: Optional[str] = None,
) -> Optional[str]:
    if requested is None:
        return default
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized in {"", "none", "null"}:
            return None
        if normalized == "balanced":
            return "balanced"
    raise ValueError("Binary class weight must be one of None/'none' or 'balanced'.")


def _fit_lpm_least_squares(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    y_train: pd.Series,
    *,
    active_cols: Sequence[str],
    output_index: pd.Index,
) -> tuple[pd.Series, pd.Series, Optional[float], pd.Series]:
    """Fast dense least-squares fit for the LPM path."""
    x_train_arr = x_train.to_numpy(dtype=float, copy=False)
    x_all_arr = x_all.to_numpy(dtype=float, copy=False)
    y_train_arr = y_train.to_numpy(dtype=float, copy=False)

    x_train_fit = np.empty((x_train_arr.shape[0], x_train_arr.shape[1] + 1), dtype=float)
    x_train_fit[:, 0] = 1.0
    x_train_fit[:, 1:] = x_train_arr

    coef, _, _, _ = np.linalg.lstsq(x_train_fit, y_train_arr, rcond=None)

    x_all_fit = np.empty((x_all_arr.shape[0], x_all_arr.shape[1] + 1), dtype=float)
    x_all_fit[:, 0] = 1.0
    x_all_fit[:, 1:] = x_all_arr

    predicted_prob = pd.Series(x_all_fit @ coef, index=output_index).clip(0, 1)
    predicted_class = (predicted_prob >= 0.5).astype(int)
    intercept_value = float(coef[0]) if len(coef) else None
    weight_series = pd.Series(coef[1:], index=list(active_cols)).sort_values(
        key=lambda s: s.abs(),
        ascending=False,
    )
    return predicted_prob, predicted_class, intercept_value, weight_series


def _build_design_matrices(
    feature_df: pd.DataFrame,
    train_mask: pd.Series,
    *,
    model_method: str,
    feature_family: str = "all_features",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, float], dict[str, list[str]], dict[str, object]]:
    feature_cols = _select_index_feature_columns(feature_df, feature_family=feature_family)
    categorical_cols, numeric_cols = _infer_feature_types(feature_df, feature_cols)

    train_df = feature_df.loc[train_mask, feature_cols].copy()
    medians: dict[str, float] = {}
    for col in numeric_cols:
        values = pd.to_numeric(train_df[col], errors="coerce")
        median = values.median()
        medians[col] = float(0.0 if pd.isna(median) else median)

    categories: dict[str, list[str]] = {}
    for col in categorical_cols:
        vals = (
            train_df[col]
            .astype("string")
            .fillna("__MISSING__")
            .replace({"": "__MISSING__"})
        )
        uniq = sorted(vals.dropna().unique().tolist())
        categories[col] = uniq or ["__MISSING__"]

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if numeric_cols:
            num = pd.DataFrame(
                {
                    col: pd.to_numeric(df[col], errors="coerce").fillna(medians[col]).astype(float)
                    for col in numeric_cols
                },
                index=df.index,
            )
            parts.append(num)

        for col in categorical_cols:
            vals = (
                df[col]
                .astype("string")
                .fillna("__MISSING__")
                .replace({"": "__MISSING__"})
            )
            cat = pd.Categorical(vals, categories=categories[col])
            dummies = pd.get_dummies(cat, prefix=col, dtype=float)
            dummies.index = df.index
            parts.append(dummies)

        if not parts:
            raise ValueError("No feature columns available for OPT probability index model.")
        return pd.concat(parts, axis=1)

    x_train = _transform(feature_df.loc[train_mask, feature_cols].copy())
    x_all = _transform(feature_df[feature_cols].copy())
    x_train, x_all, raw_interaction_meta = _build_interaction_features(
        x_train,
        x_all,
        feature_cols=feature_cols,
        model_method=model_method,
    )
    non_constant = x_train.columns[x_train.nunique(dropna=False) > 1].tolist()
    if not non_constant:
        raise ValueError("No non-constant features available for OPT probability index model.")
    return (
        x_train[non_constant],
        x_all[non_constant],
        feature_cols,
        non_constant,
        medians,
        categories,
        raw_interaction_meta,
    )


def _limit_model_features(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    active_cols: list[str],
    *,
    n_train_obs: int,
    max_active_features: Optional[int],
    max_feature_to_train_ratio: Optional[float],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    effective_limit: Optional[int] = None
    if max_active_features is not None:
        effective_limit = int(max_active_features)
    if max_feature_to_train_ratio is not None:
        ratio_limit = max(1, int(np.floor(float(max_feature_to_train_ratio) * max(1, int(n_train_obs)))))
        effective_limit = ratio_limit if effective_limit is None else min(effective_limit, ratio_limit)

    if effective_limit is None or effective_limit <= 0 or len(active_cols) <= effective_limit:
        return x_train, x_all, active_cols, {
            "feature_downsampled": False,
            "n_active_features_before_sampling": int(len(active_cols)),
            "n_active_features_after_sampling": int(len(active_cols)),
            "max_active_features": None if max_active_features is None else int(max_active_features),
            "max_feature_to_train_ratio": max_feature_to_train_ratio,
            "selected_feature_seed": int(random_seed),
        }

    rng = np.random.default_rng(int(random_seed))
    chosen = sorted(rng.choice(np.array(active_cols, dtype=object), size=int(effective_limit), replace=False).tolist())
    return (
        x_train[chosen].copy(),
        x_all[chosen].copy(),
        chosen,
        {
            "feature_downsampled": True,
            "n_active_features_before_sampling": int(len(active_cols)),
            "n_active_features_after_sampling": int(len(chosen)),
            "max_active_features": None if max_active_features is None else int(max_active_features),
            "max_feature_to_train_ratio": max_feature_to_train_ratio,
            "selected_feature_seed": int(random_seed),
        },
    )


def _sample_leaveout_training_ids(
    source_df: pd.DataFrame,
    *,
    n_train: int,
    random_seed: int,
    stratify_values: Optional[pd.Series] = None,
) -> list[int]:
    if source_df.empty or int(n_train) <= 0:
        return []

    source_ids = pd.to_numeric(source_df["c"], errors="coerce").dropna().astype(int)
    unique_ids = pd.Index(source_ids.unique())
    if len(unique_ids) <= int(n_train):
        return unique_ids.astype(int).tolist()

    stratify = None
    if stratify_values is not None:
        candidate = pd.Series(stratify_values, index=source_df.index).reindex(source_df.index)
        candidate = candidate.fillna("__MISSING__").astype(str)
        n_groups = int(candidate.nunique())
        if (
            n_groups > 1
            and int(candidate.value_counts().min()) >= 2
            and int(n_train) >= n_groups
            and (len(source_df) - int(n_train)) >= n_groups
        ):
            stratify = candidate

    train_ids, _ = train_test_split(
        source_df["c"],
        train_size=int(n_train),
        random_state=int(random_seed),
        stratify=stratify,
    )
    return pd.Series(train_ids).dropna().astype(int).tolist()


def _build_leaveout_masks(
    model_df: pd.DataFrame,
    *,
    leaveout_enabled: bool,
    leaveout_share: float,
    leaveout_seed: int,
) -> pd.DataFrame:
    work = model_df.copy()
    analysis = work[work["in_analysis_universe"].eq(1)].copy()
    if analysis.empty:
        raise ValueError("No analysis-universe firms available for opt_probability_index.")

    work["leaveout_training_firm"] = 0
    work["event_study_sample"] = work["in_analysis_universe"].fillna(0).astype(int)

    if not leaveout_enabled:
        work.loc[work["in_analysis_universe"].eq(1), "leaveout_training_firm"] = 1
    else:
        if train_test_split is None:
            raise ImportError("scikit-learn is required for leaveout_enabled=true.")
        preferred = analysis[analysis["preferred_rcid_source"].fillna(0).eq(1)].copy()
        outside = analysis[analysis["outside_negative_candidate"].fillna(0).eq(1)].copy()

        if not preferred.empty and not outside.empty:
            requested_total = max(1, int(round(float(leaveout_share) * len(analysis))))
            requested_per_source = max(1, int(round(requested_total / 2.0)))
            max_preferred_train = max(0, len(preferred) - 1)
            max_outside_train = max(0, len(outside) - 1)
            n_train_per_source = min(requested_per_source, max_preferred_train, max_outside_train)
            train_ids: list[int]
            if n_train_per_source > 0:
                preferred_train_ids = _sample_leaveout_training_ids(
                    preferred,
                    n_train=n_train_per_source,
                    random_seed=int(leaveout_seed),
                    stratify_values=preferred["post2016_any_opt"],
                )
                outside_train_ids = _sample_leaveout_training_ids(
                    outside,
                    n_train=n_train_per_source,
                    random_seed=int(leaveout_seed) + 10_000,
                )
                train_ids = preferred_train_ids + outside_train_ids
            else:
                train_ids = []
        else:
            stratify = None
            candidate_strata = (
                analysis["preferred_rcid_source"].fillna(0).astype(int).astype(str)
                + "_"
                + analysis["post2016_any_opt"].fillna(0).astype(int).astype(str)
            )
            if candidate_strata.nunique() > 1 and candidate_strata.value_counts().min() >= 2:
                stratify = candidate_strata
            elif (
                analysis["post2016_any_opt"].nunique() > 1
                and analysis["post2016_any_opt"].value_counts().min() >= 2
            ):
                stratify = analysis["post2016_any_opt"]
            train_ids, _ = train_test_split(
                analysis["c"],
                train_size=float(leaveout_share),
                random_state=int(leaveout_seed),
                stratify=stratify,
            )
            train_ids = pd.Series(train_ids).dropna().astype(int).tolist()

        work.loc[work["c"].isin(pd.Series(train_ids).astype(int)), "leaveout_training_firm"] = 1
        work.loc[work["leaveout_training_firm"].eq(1), "event_study_sample"] = 0

    work["train_sample"] = work["leaveout_training_firm"].fillna(0).astype(int)
    return work


def fit_opt_probability_index(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    model_method: str,
    entry_mode: str,
    ntiles: int,
    feature_year_min: int,
    feature_year_max: int,
    leaveout_enabled: bool,
    leaveout_share: float,
    leaveout_seed: int,
    rf_n_estimators: int = 500,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_leaf: int = 10,
    rf_min_samples_split: int = 20,
    logit_class_weight: Optional[str] = "balanced",
    lasso_cv_folds: Optional[int] = None,
    lasso_n_cs: Optional[int] = None,
    max_active_features: Optional[int] = None,
    max_feature_to_train_ratio: Optional[float] = None,
    feature_sample_seed: int = 42,
    feature_family: str = "all_features",
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, dict[str, object]]:
    """Train the OPT probability index model and score all firms in the feature frame."""
    validate_opt_probability_config(
        model_method=model_method,
        entry_mode=entry_mode,
        ntiles=ntiles,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
    )

    model_df = feature_df.merge(target_df[["c", "post2016_any_opt"]], on="c", how="left")
    model_df["post2016_any_opt"] = np.where(
        model_df["preferred_rcid_source"].fillna(0).eq(1),
        model_df["post2016_any_opt"].fillna(0),
        0,
    ).astype(int)
    model_df["target_source"] = np.where(
        model_df["preferred_rcid_source"].fillna(0).eq(1),
        "preferred_rcid_source",
        np.where(
            model_df["outside_negative_candidate"].fillna(0).eq(1),
            "outside_negative",
            "other_source",
        ),
    )
    model_df = _build_leaveout_masks(
        model_df,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
        leaveout_seed=leaveout_seed,
    )

    train_mask = model_df["train_sample"].eq(1)
    y_train = model_df.loc[train_mask, "post2016_any_opt"].astype(int)
    if y_train.nunique() < 2:
        raise ValueError("Exposure model training sample has only one outcome class.")

    feature_cols = _select_index_feature_columns(model_df, feature_family=feature_family)
    categorical_cols, numeric_cols = _infer_feature_types(model_df, feature_cols)
    x_train, x_all, _, active_cols, _, _, interaction_meta = _build_design_matrices(
        model_df,
        train_mask,
        model_method=model_method,
        feature_family=feature_family,
    )
    x_train, x_all, active_cols, feature_limit_meta = _limit_model_features(
        x_train,
        x_all,
        active_cols,
        n_train_obs=int(train_mask.sum()),
        max_active_features=max_active_features,
        max_feature_to_train_ratio=max_feature_to_train_ratio,
        random_seed=feature_sample_seed,
    )
    interaction_meta = _finalize_interaction_meta(interaction_meta, active_cols=active_cols)
    diagnostics: dict[str, object] = {
        "model_method": model_method,
        "index_entry_mode": entry_mode,
        "n_train_obs": int(train_mask.sum()),
        "n_event_study_firms": int(model_df["event_study_sample"].sum()),
        "n_train_preferred_source": int(
            model_df.loc[
                train_mask & model_df["preferred_rcid_source"].fillna(0).eq(1),
                "c",
            ].nunique()
        ),
        "n_train_outside_negative": int(
            model_df.loc[
                train_mask & model_df["outside_negative_candidate"].fillna(0).eq(1),
                "c",
            ].nunique()
        ),
        "n_event_study_preferred_source": int(
            model_df.loc[
                model_df["event_study_sample"].eq(1) & model_df["preferred_rcid_source"].fillna(0).eq(1),
                "c",
            ].nunique()
        ),
        "n_event_study_outside_negative": int(
            model_df.loc[
                model_df["event_study_sample"].eq(1) & model_df["outside_negative_candidate"].fillna(0).eq(1),
                "c",
            ].nunique()
        ),
        "n_active_features": int(len(active_cols)),
        "train_target_mean": float(y_train.mean()),
        "n_feature_columns_raw": int(len(feature_cols)),
        "n_numeric_feature_columns_raw": int(len(numeric_cols)),
        "n_categorical_feature_columns_raw": int(len(categorical_cols)),
        "feature_family": feature_family,
        **interaction_meta,
        **feature_limit_meta,
    }
    weight_series: pd.Series
    intercept_value: Optional[float] = None
    standardized_feature_columns: list[str] = []

    if model_method == "logit":
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for index_model_method='logit'.")
        resolved_logit_class_weight = _resolve_binary_class_weight(
            logit_class_weight,
            default="balanced",
        )
        x_train_fit, x_all_fit, standardized_feature_columns = _standardize_nonbinary_features(x_train, x_all)
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            class_weight=resolved_logit_class_weight,
            max_iter=5000,
            n_jobs=-1,
        )
        model.fit(x_train_fit, y_train)
        class_order = list(model.classes_)
        prob_matrix = model.predict_proba(x_all_fit)
        class_one_idx = class_order.index(1) if 1 in class_order else 0
        predicted_prob = pd.Series(prob_matrix[:, class_one_idx], index=model_df.index)
        predicted_class = (predicted_prob >= 0.5).astype(int)
        intercept_value = float(model.intercept_[0]) if len(model.intercept_) else None
        weight_series = pd.Series(model.coef_.ravel(), index=active_cols).sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
        top_loadings = weight_series.head(15)
        diagnostics["top_coefficients"] = {str(k): float(v) for k, v in top_loadings.items()}
        diagnostics["logit_class_weight"] = resolved_logit_class_weight
        diagnostics["n_standardized_features"] = int(len(standardized_feature_columns))
    elif model_method == "lpm":
        predicted_prob, predicted_class, intercept_value, weight_series = _fit_lpm_least_squares(
            x_train,
            x_all,
            y_train.astype(float),
            active_cols=active_cols,
            output_index=model_df.index,
        )
        top_loadings = weight_series.head(15)
        diagnostics["top_coefficients"] = {str(k): float(v) for k, v in top_loadings.items()}
    elif model_method == "lasso":
        if LogisticRegressionCV is None:
            raise ImportError("scikit-learn is required for index_model_method='lasso'.")
        cv_folds = _resolve_lasso_cv_folds(y_train, requested_folds=lasso_cv_folds)
        lasso_cs = _resolve_lasso_cs(lasso_n_cs)
        x_train_fit, x_all_fit, standardized_feature_columns = _standardize_nonbinary_features(x_train, x_all)
        print(
            "[opt_probability_index] LASSO fit setup: "
            f"{cv_folds}-fold CV | {len(lasso_cs)} C values | "
            f"{len(active_cols):,} active features | {int(train_mask.sum()):,} training firms"
        )
        model = LogisticRegressionCV(
            Cs=lasso_cs,
            cv=cv_folds,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            scoring="neg_log_loss",
            max_iter=5000,
            random_state=int(leaveout_seed),
            n_jobs=-1,
        )
        model.fit(x_train_fit, y_train)
        class_order = list(model.classes_)
        prob_matrix = model.predict_proba(x_all_fit)
        class_one_idx = class_order.index(1) if 1 in class_order else 0
        predicted_prob = pd.Series(prob_matrix[:, class_one_idx], index=model_df.index)
        predicted_class = (predicted_prob >= 0.5).astype(int)
        intercept_value = float(model.intercept_[0]) if len(model.intercept_) else None
        weight_series = pd.Series(model.coef_.ravel(), index=active_cols).sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
        top_loadings = weight_series.head(15)
        diagnostics["top_coefficients"] = {str(k): float(v) for k, v in top_loadings.items()}
        diagnostics["lasso_cv_folds"] = int(cv_folds)
        diagnostics["lasso_n_cs"] = int(len(lasso_cs))
        diagnostics["lasso_selected_c"] = float(np.ravel(model.C_)[0])
        diagnostics["n_standardized_features"] = int(len(standardized_feature_columns))
    else:
        if RandomForestClassifier is None:
            raise ImportError("scikit-learn is required for index_model_method='random_forest'.")
        model = RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            max_depth=rf_max_depth,
            min_samples_leaf=int(rf_min_samples_leaf),
            min_samples_split=int(rf_min_samples_split),
            class_weight="balanced_subsample",
            random_state=int(leaveout_seed),
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        prob_matrix = model.predict_proba(x_all)
        class_order = list(model.classes_)
        class_one_idx = class_order.index(1) if 1 in class_order else 0
        predicted_prob = pd.Series(prob_matrix[:, class_one_idx], index=model_df.index)
        predicted_class = pd.Series(model.predict(x_all), index=model_df.index).astype(int)
        weight_series = pd.Series(model.feature_importances_, index=active_cols).sort_values(ascending=False)
        importances = weight_series.head(15)
        diagnostics["top_feature_importances"] = {str(k): float(v) for k, v in importances.items()}

    model_df["predicted_prob"] = pd.to_numeric(predicted_prob, errors="coerce").clip(0, 1)
    model_df["predicted_class"] = pd.to_numeric(predicted_class, errors="coerce").fillna(0).astype(int)
    model_df["predicted_index"] = model_df["predicted_prob"]
    model_df["exposure_value"] = np.where(
        model_method == "random_forest",
        model_df["predicted_class"],
        model_df["predicted_index"],
    )
    model_df["model_method"] = model_method
    model_df["index_entry_mode"] = entry_mode

    eval_mask = model_df["event_study_sample"].eq(1)
    if not eval_mask.any():
        eval_mask = model_df["in_analysis_universe"].eq(1)
    y_eval = model_df.loc[eval_mask, "post2016_any_opt"].astype(int)
    p_eval = model_df.loc[eval_mask, "predicted_prob"].astype(float)
    diagnostics["evaluation_auc"] = _safe_auc(y_eval, p_eval)
    diagnostics["evaluation_brier"] = _safe_brier(y_eval, p_eval)
    diagnostics["evaluation_target_mean"] = float(y_eval.mean()) if len(y_eval) else None
    diagnostics["evaluation_n"] = int(len(y_eval))
    diagnostics["evaluation_class_1_share"] = float((model_df.loc[eval_mask, "predicted_class"] == 1).mean())

    if not return_artifacts:
        return model_df, diagnostics

    artifacts: dict[str, object] = {
        "feature_columns_raw": feature_cols,
        "numeric_feature_columns_raw": numeric_cols,
        "categorical_feature_columns_raw": categorical_cols,
        "active_feature_columns": active_cols,
        "interaction_column_names": interaction_meta["interaction_column_names"],
        "numeric_interaction_column_names": interaction_meta["numeric_interaction_column_names"],
        "category_slope_interaction_column_names": interaction_meta["category_slope_interaction_column_names"],
        "skipped_interaction_source_columns": interaction_meta["skipped_interaction_source_columns"],
        "standardized_feature_columns": standardized_feature_columns,
        "weight_series": weight_series,
        "intercept": intercept_value,
        "evaluation_mask": eval_mask.copy(),
    }
    return model_df, diagnostics, artifacts


def _save_opt_probability_predictions(
    pred_df: pd.DataFrame,
    diagnostics: dict,
    out_path: Path,
    *,
    feature_year_min: int,
    feature_year_max: int,
    target_year_min: int,
    target_year_max: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [
        "c",
        "in_analysis_universe",
        "preferred_rcid_source",
        "outside_negative_candidate",
        "post2016_any_opt",
        "target_source",
        "leaveout_training_firm",
        "train_sample",
        "event_study_sample",
        "predicted_prob",
        "predicted_class",
        "predicted_index",
        "exposure_value",
        "model_method",
        "index_entry_mode",
    ]
    pred_df[save_cols].to_parquet(out_path, index=False)
    metadata = dict(diagnostics)
    metadata.update(
        {
            "feature_year_min": int(feature_year_min),
            "feature_year_max": int(feature_year_max),
            "target_year_min": int(target_year_min),
            "target_year_max": int(target_year_max),
        }
    )
    _write_metadata(out_path, metadata)
    print(f"[opt_probability_index] Wrote predictions to {out_path}")


def _build_opt_probability_index(
    panel: pd.DataFrame,
    cfg_full: dict,
    exp_cfg: dict,
    *,
    config_path: Optional[Path],
    x_source_col: str,
    preloaded_feature_frame: Optional[tuple[pd.DataFrame, dict]] = None,
) -> tuple[pd.DataFrame, dict, dict[str, object]]:
    paths_cfg = get_cfg_section(cfg_full, "paths")
    testing_cfg = get_cfg_section(cfg_full, "testing")
    feature_year_min = int(exp_cfg.get("feature_year_min", 2010))
    feature_year_max = int(exp_cfg.get("feature_year_max", 2015))
    target_year_min = int(exp_cfg.get("target_year_min", 2016))
    target_year_max = int(exp_cfg.get("target_year_max", 2022))
    model_method = str(exp_cfg.get("index_model_method", "logit"))
    entry_mode = str(exp_cfg.get("index_entry_mode", "ntiles"))
    feature_family = str(exp_cfg.get("feature_family", "all_features")).strip() or "all_features"
    ntiles = int(exp_cfg.get("ntiles", 4))
    leaveout_enabled = bool(exp_cfg.get("leaveout_enabled", False))
    leaveout_share = float(exp_cfg.get("leaveout_share", 0.25))
    leaveout_seed = int(exp_cfg.get("leaveout_seed", 42))
    logit_class_weight = exp_cfg.get("logit_class_weight", "balanced")
    lasso_cv_folds = _parse_int_or_none(
        testing_cfg.get("lasso_cv_folds")
        if testing_enabled(cfg_full)
        else exp_cfg.get("lasso_cv_folds")
    )
    lasso_n_cs = _parse_int_or_none(
        testing_cfg.get("lasso_n_cs")
        if testing_enabled(cfg_full)
        else exp_cfg.get("lasso_n_cs")
    )
    max_active_features = _parse_int_or_none(
        testing_cfg.get("model_max_active_features")
        if testing_enabled(cfg_full)
        else exp_cfg.get("model_max_active_features")
    )
    max_feature_to_train_ratio = _parse_float_or_none(
        testing_cfg.get("model_max_feature_to_train_ratio")
        if testing_enabled(cfg_full)
        else exp_cfg.get("model_max_feature_to_train_ratio")
    )
    feature_sample_seed = int(testing_cfg.get("feature_sample_seed", testing_cfg.get("random_seed", leaveout_seed)))

    validate_opt_probability_config(
        model_method=model_method,
        entry_mode=entry_mode,
        ntiles=ntiles,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
    )

    if preloaded_feature_frame is not None:
        feature_df, feature_meta = preloaded_feature_frame
    else:
        feature_df, feature_meta = _load_feature_frame_for_window(
            cfg_full,
            exp_cfg,
            config_path=config_path,
            year_min=feature_year_min,
            year_max=feature_year_max,
        )
    print(
        f"[opt_probability_index] Feature frame: {len(feature_df):,} firms | "
        f"{int(feature_df['in_analysis_universe'].fillna(0).sum()):,} analysis-panel firms | "
        f"{int(feature_df.get('preferred_rcid_source', pd.Series(dtype=float)).fillna(0).sum()):,} preferred-source | "
        f"{int(feature_df['outside_negative_candidate'].fillna(0).sum()):,} outside negatives"
    )
    print(f"[opt_probability_index] Feature metadata: {feature_meta}")

    target_df = _build_post2016_target(
        panel,
        x_source_col=x_source_col,
        target_year_min=target_year_min,
        target_year_max=target_year_max,
    )
    pred_df, diagnostics, artifacts = fit_opt_probability_index(
        feature_df,
        target_df,
        model_method=model_method,
        entry_mode=entry_mode,
        ntiles=ntiles,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        leaveout_enabled=leaveout_enabled,
        leaveout_share=leaveout_share,
        leaveout_seed=leaveout_seed,
        rf_n_estimators=int(exp_cfg.get("rf_n_estimators", 500)),
        rf_max_depth=_parse_int_or_none(exp_cfg.get("rf_max_depth")),
        rf_min_samples_leaf=int(exp_cfg.get("rf_min_samples_leaf", 10)),
        rf_min_samples_split=int(exp_cfg.get("rf_min_samples_split", 20)),
        logit_class_weight=logit_class_weight,
        lasso_cv_folds=lasso_cv_folds,
        lasso_n_cs=lasso_n_cs,
        max_active_features=max_active_features,
        max_feature_to_train_ratio=max_feature_to_train_ratio,
        feature_sample_seed=feature_sample_seed,
        feature_family=feature_family,
        return_artifacts=True,
    )
    print(f"[opt_probability_index] Diagnostics: {json.dumps(diagnostics, indent=2, sort_keys=True)}")

    out_path = apply_testing_output_suffix(
        _resolve_path(paths_cfg, "opt_probability_index_out", allow_missing=True),
        cfg_full,
    )
    _save_opt_probability_predictions(
        pred_df,
        diagnostics,
        out_path,
        feature_year_min=feature_year_min,
        feature_year_max=feature_year_max,
        target_year_min=target_year_min,
        target_year_max=target_year_max,
    )
    return pred_df, diagnostics, artifacts


def _diagnostic_top_n(cfg_full: dict, exp_cfg: dict, *, detailed: bool) -> int:
    testing_cfg = get_cfg_section(cfg_full, "testing")
    default_value = 20 if detailed else 10
    if detailed:
        return int(testing_cfg.get("diagnostics_top_n", exp_cfg.get("diagnostics_top_n", default_value)))
    return int(exp_cfg.get("diagnostics_top_n", default_value))


def _predictive_n_bins(cfg_full: dict, exp_cfg: dict, *, detailed: bool) -> int:
    testing_cfg = get_cfg_section(cfg_full, "testing")
    default_value = 8 if detailed else 10
    if detailed:
        return int(testing_cfg.get("predictive_n_bins", exp_cfg.get("predictive_n_bins", default_value)))
    return int(exp_cfg.get("predictive_n_bins", default_value))


def _testing_verbose(cfg_full: dict) -> bool:
    testing_cfg = get_cfg_section(cfg_full, "testing")
    return bool(testing_cfg.get("verbose", True))


def _resolve_run_log_path(
    cfg_full: dict,
    exp_cfg: dict,
    *,
    cli_log_file: Optional[Path],
) -> Optional[Path]:
    if cli_log_file is not None:
        path = cli_log_file.expanduser()
        return path if path.is_absolute() else (Path.cwd() / path)
    path = _resolve_optional_path_value(exp_cfg.get("log_out_path"), cfg=cfg_full)
    if path is None:
        return None
    return apply_testing_output_suffix(path, cfg_full)


def _get_or_build_index_result(
    panel: pd.DataFrame,
    cfg_full: dict,
    exp_cfg: dict,
    *,
    config_path: Optional[Path],
    feature_cache: Optional[dict[tuple[int, int], tuple[pd.DataFrame, dict]]] = None,
    index_cache: Optional[dict[tuple[object, ...], tuple[pd.DataFrame, dict, dict[str, object]]]] = None,
    report_diagnostics: bool = False,
    detailed_diagnostics: bool = False,
    force_report_diagnostics: bool = False,
    do_plot: bool = False,
    slides_out: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict, dict[str, object]]:
    feature_year_min = int(exp_cfg.get("feature_year_min", 2010))
    feature_year_max = int(exp_cfg.get("feature_year_max", 2015))
    target_year_min = int(exp_cfg.get("target_year_min", 2016))
    target_year_max = int(exp_cfg.get("target_year_max", 2022))
    model_method = str(exp_cfg.get("index_model_method", "logit"))
    entry_mode = str(exp_cfg.get("index_entry_mode", "ntiles"))
    x_source_col = str(exp_cfg.get("x_source_col", "any_opt_hires_correction_aware"))
    testing_cfg = get_cfg_section(cfg_full, "testing")

    feature_key = (feature_year_min, feature_year_max)
    if feature_cache is not None and feature_key in feature_cache:
        preloaded_feature_frame = feature_cache[feature_key]
    else:
        preloaded_feature_frame = _load_feature_frame_for_window(
            cfg_full,
            exp_cfg,
            config_path=config_path,
            year_min=feature_year_min,
            year_max=feature_year_max,
        )
        if feature_cache is not None:
            feature_cache[feature_key] = preloaded_feature_frame

    index_key = (
        feature_year_min,
        feature_year_max,
        target_year_min,
        target_year_max,
        model_method,
        entry_mode,
        int(exp_cfg.get("ntiles", 4)),
        bool(exp_cfg.get("leaveout_enabled", False)),
        float(exp_cfg.get("leaveout_share", 0.25)),
        int(exp_cfg.get("leaveout_seed", 42)),
        int(exp_cfg.get("rf_n_estimators", 500)),
        _parse_int_or_none(exp_cfg.get("rf_max_depth")),
        int(exp_cfg.get("rf_min_samples_leaf", 10)),
        int(exp_cfg.get("rf_min_samples_split", 20)),
        str(exp_cfg.get("logit_class_weight", "balanced")).strip().lower(),
        _parse_int_or_none(testing_cfg.get("lasso_cv_folds"))
        if bool(testing_cfg.get("enabled", False))
        else _parse_int_or_none(exp_cfg.get("lasso_cv_folds")),
        _parse_int_or_none(testing_cfg.get("lasso_n_cs"))
        if bool(testing_cfg.get("enabled", False))
        else _parse_int_or_none(exp_cfg.get("lasso_n_cs")),
        str(x_source_col),
        bool(testing_cfg.get("enabled", False)),
        int(testing_cfg.get("analysis_sample_n", 50)),
        int(testing_cfg.get("random_seed", 42)),
        _parse_int_or_none(testing_cfg.get("model_max_active_features")),
        _parse_float_or_none(testing_cfg.get("model_max_feature_to_train_ratio")),
    )

    built_now = False
    if index_cache is not None and index_key in index_cache:
        pred_df, diagnostics, artifacts = index_cache[index_key]
    else:
        pred_df, diagnostics, artifacts = _build_opt_probability_index(
            panel,
            cfg_full,
            exp_cfg,
            config_path=config_path,
            x_source_col=x_source_col,
            preloaded_feature_frame=preloaded_feature_frame,
        )
        built_now = True
        if index_cache is not None:
            index_cache[index_key] = (pred_df, diagnostics, artifacts)

    if report_diagnostics and (built_now or detailed_diagnostics or force_report_diagnostics):
        top_n = _diagnostic_top_n(cfg_full, exp_cfg, detailed=detailed_diagnostics)
        predictive_bins = _predictive_n_bins(cfg_full, exp_cfg, detailed=detailed_diagnostics)
        _report_model_diagnostics(
            pred_df,
            diagnostics,
            artifacts,
            detailed=detailed_diagnostics,
            top_n=top_n,
            predictive_n_bins=predictive_bins,
        )
        if do_plot:
            version_tag = f"index-{diagnostics['model_method']}-{diagnostics['index_entry_mode']}"
            if testing_enabled(cfg_full):
                version_tag = f"{version_tag}-testing"
            _plot_predictive_power(
                pred_df,
                diagnostics,
                artifacts,
                slides_out=slides_out,
                version_tag=version_tag,
                predictive_n_bins=predictive_bins,
            )

    return pred_df, diagnostics, artifacts


# ---------------------------------------------------------------------------
# Per-version / per-outcome pipeline
# ---------------------------------------------------------------------------

def _run_one_version(
    panel: pd.DataFrame,
    outcome_col: str,
    version: str,
    exp_cfg: dict,
    cfg_full: dict,
    slides_out: Optional[Path],
    oc_tag: str,
    do_plot: bool,
    *,
    config_path: Optional[Path],
    feature_cache: Optional[dict[tuple[int, int], tuple[pd.DataFrame, dict]]] = None,
    index_cache: Optional[dict[tuple[object, ...], tuple[pd.DataFrame, dict, dict[str, object]]]] = None,
) -> None:
    """Run the full exposure event study for one exposure version and one outcome column."""
    legacy_year_min = int(exp_cfg.get("exposure_year_min", 2010))
    legacy_year_max = int(exp_cfg.get("exposure_year_max", 2015))
    ntiles = int(exp_cfg.get("ntiles", 4))
    event_year = int(exp_cfg.get("event_year", 2016))
    ref_year = int(exp_cfg.get("ref_year", 2015))
    data_min_t = int(exp_cfg.get("data_min_t", 2010))
    data_max_t = int(exp_cfg.get("data_max_t", 2022))
    x_source_col = str(exp_cfg.get("x_source_col", "any_opt_hires_correction_aware"))
    version_tag = version.replace("_", "-")

    print(f"\n{'=' * 72}")
    print(f" Version: {version_tag}  |  Outcome: {outcome_col}")
    print(f"{'=' * 72}")

    _ensure_derived_outcome(panel, outcome_col, x_source_col)
    if outcome_col not in panel.columns:
        print(f"[warn] Outcome column '{outcome_col}' not available. Skipping.")
        return

    if version in LEGACY_EXPOSURE_VERSIONS:
        print(f"\n--- Computing exposure ({version}) ---")
        feature_key = (legacy_year_min, legacy_year_max)
        if feature_cache is not None and feature_key in feature_cache:
            feature_df, feature_meta = feature_cache[feature_key]
        else:
            feature_df, feature_meta = _load_feature_frame_for_window(
                cfg_full,
                exp_cfg,
                config_path=config_path,
                year_min=legacy_year_min,
                year_max=legacy_year_max,
            )
            if feature_cache is not None:
                feature_cache[feature_key] = (feature_df, feature_meta)
        print(f"[legacy exposure {version}] Feature metadata: {feature_meta}")
        exposure = _legacy_exposure_from_feature_frame(
            feature_df,
            version=version,
            year_min=legacy_year_min,
            year_max=legacy_year_max,
        )
        ntile_df = _assign_ntiles(exposure, ntiles, zero_separate=(version == "opt_hire_rate"))
        panel_with_group = panel.merge(ntile_df[["c", "ntile", "ntile_label"]], on="c", how="inner")
        print(
            f"\n[merge] Panel rows with group assignment: {len(panel_with_group)} | "
            f"firms: {panel_with_group['c'].nunique()} | years: "
            f"{panel_with_group['t'].min()}–{panel_with_group['t'].max()}"
        )
        ntile_order = (
            panel_with_group[["ntile", "ntile_label"]].drop_duplicates().sort_values("ntile")["ntile_label"].tolist()
        )
        ntile_colors = _ntile_color_map(ntile_order)
        non_ref_ntiles = ntile_order[1:]
        print(f"\n[summary] Mean {outcome_col} by group (full panel):")
        print(panel_with_group.groupby("ntile_label")[outcome_col].agg(["mean", "std", "count"]).to_string())

        if do_plot:
            print("\n--- Plot 1: raw means by group over time ---")
            _plot_raw_means(
                panel_with_group,
                outcome_col,
                event_year,
                ref_year,
                ntile_order,
                ntile_colors,
                slides_out,
                oc_tag,
                version_tag,
            )

        print("\n--- Regression: year × group interactions + firm FE ---")
        coef_df = _run_ntile_regression(panel_with_group, outcome_col, ref_year, data_min_t, data_max_t)
        if coef_df is not None and not coef_df.empty:
            ref_rows = pd.DataFrame(
                [
                    {"year": ref_year, "ntile_label": lbl, "coef": 0.0, "se": 0.0, "tstat": np.nan, "pval": np.nan}
                    for lbl in non_ref_ntiles
                ]
            )
            coef_df = pd.concat([ref_rows, coef_df], ignore_index=True)
        if do_plot and coef_df is not None and not coef_df.empty:
            print("\n--- Plot 2: regression interaction coefficients ---")
            _plot_ntile_regression(
                coef_df,
                outcome_col,
                event_year,
                ref_year,
                non_ref_ntiles,
                ntile_colors,
                slides_out,
                oc_tag,
                version_tag,
            )
        return

    if version != INDEX_EXPOSURE_VERSION:
        raise ValueError(
            f"Unknown exposure version: {version!r}. "
            f"Choose one of {LEGACY_EXPOSURE_VERSIONS + (INDEX_EXPOSURE_VERSION, 'both', 'all')}."
        )

    print("\n--- Building OPT probability index ---")
    pred_df, diagnostics, _ = _get_or_build_index_result(
        panel,
        cfg_full,
        exp_cfg,
        config_path=config_path,
        feature_cache=feature_cache,
        index_cache=index_cache,
    )
    model_method = str(diagnostics["model_method"])
    model_label = _model_label(model_method)
    entry_mode = str(diagnostics["index_entry_mode"])
    exclude_outside_negatives = bool(exp_cfg.get("event_study_exclude_outside_negatives", False))
    sample_pred = _select_index_analysis_firms(
        pred_df,
        exclude_outside_negatives=exclude_outside_negatives,
    )
    sample_pred = sample_pred[["c", "predicted_index", "predicted_class"]].copy()
    if sample_pred.empty:
        print("[opt_probability_index] No event-study firms remain after leave-out split. Skipping.")
        return

    if model_method == "random_forest":
        ntile_df = _rf_group_df(sample_pred)
        panel_with_group = panel.merge(ntile_df, on="c", how="inner")
        print(
            f"[opt_probability_index] Panel rows in estimation sample: {len(panel_with_group):,} | "
            f"firms: {panel_with_group['c'].nunique():,}"
        )
        ntile_order = (
            panel_with_group[["ntile", "ntile_label"]].drop_duplicates().sort_values("ntile")["ntile_label"].tolist()
        )
        ntile_colors = _ntile_color_map(ntile_order)
        non_ref_ntiles = ntile_order[1:]
        print(f"\n[summary] Mean {outcome_col} by RF group:")
        print(panel_with_group.groupby("ntile_label")[outcome_col].agg(["mean", "std", "count"]).to_string())
        if do_plot:
            print("\n--- Plot 1: raw means by RF group over time ---")
            _plot_raw_means(
                panel_with_group,
                outcome_col,
                event_year,
                ref_year,
                ntile_order,
                ntile_colors,
                slides_out,
                oc_tag,
                version_tag,
            )
        print("\n--- Regression: year × RF group interactions + firm FE ---")
        coef_df = _run_ntile_regression(panel_with_group, outcome_col, ref_year, data_min_t, data_max_t)
        if coef_df is not None and not coef_df.empty:
            ref_rows = pd.DataFrame(
                [
                    {"year": ref_year, "ntile_label": lbl, "coef": 0.0, "se": 0.0, "tstat": np.nan, "pval": np.nan}
                    for lbl in non_ref_ntiles
                ]
            )
            coef_df = pd.concat([ref_rows, coef_df], ignore_index=True)
        if do_plot and coef_df is not None and not coef_df.empty:
            print("\n--- Plot 2: RF interaction coefficients ---")
            _plot_ntile_regression(
                coef_df,
                outcome_col,
                event_year,
                ref_year,
                non_ref_ntiles,
                ntile_colors,
                slides_out,
                oc_tag,
                version_tag,
            )
        return

    if entry_mode == "continuous":
        firm_index = sample_pred[["c", "predicted_index"]].drop_duplicates().copy()
        index_mean = float(firm_index["predicted_index"].mean())
        index_sd = float(firm_index["predicted_index"].std(ddof=0))
        if not np.isfinite(index_sd) or index_sd <= 0:
            raise ValueError("Predicted index has zero variance in the event-study sample.")
        firm_index["predicted_index_z"] = (firm_index["predicted_index"] - index_mean) / index_sd
        panel_with_index = panel.merge(firm_index[["c", "predicted_index_z"]], on="c", how="inner")
        print(
            f"[opt_probability_index] Panel rows in estimation sample: {len(panel_with_index):,} | "
            f"firms: {panel_with_index['c'].nunique():,}"
        )
        print(
            f"[opt_probability_index] num_opt_hires summary:\n"
            f"{firm_index['predicted_index'].describe().to_string()}"
        )
        print("\n--- Regression: year × standardized num_opt_hires interactions + firm FE ---")
        coef_df = _run_continuous_regression(
            panel_with_index,
            outcome_col,
            "predicted_index_z",
            ref_year,
            data_min_t,
            data_max_t,
        )
        if coef_df is not None and not coef_df.empty:
            ref_row = pd.DataFrame(
                [{"year": ref_year, "coef": 0.0, "se": 0.0, "tstat": np.nan, "pval": np.nan}]
            )
            coef_df = pd.concat([ref_row, coef_df], ignore_index=True)
        if do_plot and coef_df is not None and not coef_df.empty:
            print("\n--- Plot: num_opt_hires interaction coefficients ---")
            _plot_continuous_regression(
                coef_df,
                outcome_col,
                event_year,
                ref_year,
                slides_out,
                oc_tag,
                version_tag,
            )
        return

    exposure = sample_pred.set_index("c")["predicted_index"]
    ntile_df = _assign_ntiles(exposure, ntiles, zero_separate=False)
    panel_with_group = panel.merge(ntile_df[["c", "ntile", "ntile_label"]], on="c", how="inner")
    print(
        f"[opt_probability_index] Panel rows in estimation sample: {len(panel_with_group):,} | "
        f"firms: {panel_with_group['c'].nunique():,}"
    )
    ntile_order = (
        panel_with_group[["ntile", "ntile_label"]].drop_duplicates().sort_values("ntile")["ntile_label"].tolist()
    )
    ntile_colors = _ntile_color_map(ntile_order)
    non_ref_ntiles = ntile_order[1:]
    print(f"\n[summary] Mean {outcome_col} by {model_label} index ntile:")
    print(panel_with_group.groupby("ntile_label")[outcome_col].agg(["mean", "std", "count"]).to_string())
    if do_plot:
        raw_plot_ntiles = _resolve_raw_plot_ntiles(exp_cfg, ntiles)
        raw_ntile_df = _assign_ntiles(exposure, raw_plot_ntiles, zero_separate=False)
        raw_panel_with_group = panel.merge(raw_ntile_df[["c", "ntile", "ntile_label"]], on="c", how="inner")
        raw_ntile_order = (
            raw_panel_with_group[["ntile", "ntile_label"]]
            .drop_duplicates()
            .sort_values("ntile")["ntile_label"]
            .tolist()
        )
        raw_ntile_colors = _ntile_color_map(raw_ntile_order)
        print(f"\n[summary] Mean {outcome_col} by {model_label} raw-plot ntile:")
        print(raw_panel_with_group.groupby("ntile_label")[outcome_col].agg(["mean", "std", "count"]).to_string())
        print(
            f"\n--- Plot 1: raw means by {model_label} index "
            f"({_describe_ntile_partition(raw_plot_ntiles)}) group over time ---"
        )
        _plot_raw_means(
            raw_panel_with_group,
            outcome_col,
            event_year,
            ref_year,
            raw_ntile_order,
            raw_ntile_colors,
            slides_out,
            oc_tag,
            version_tag,
        )
    print("\n--- Regression: year × index ntile interactions + firm FE ---")
    coef_df = _run_ntile_regression(panel_with_group, outcome_col, ref_year, data_min_t, data_max_t)
    if coef_df is not None and not coef_df.empty:
        ref_rows = pd.DataFrame(
            [
                {"year": ref_year, "ntile_label": lbl, "coef": 0.0, "se": 0.0, "tstat": np.nan, "pval": np.nan}
                for lbl in non_ref_ntiles
            ]
        )
        coef_df = pd.concat([ref_rows, coef_df], ignore_index=True)
    if do_plot and coef_df is not None and not coef_df.empty:
        print(f"\n--- Plot 2: {model_label} interaction coefficients ---")
        _plot_ntile_regression(
            coef_df,
            outcome_col,
            event_year,
            ref_year,
            non_ref_ntiles,
            ntile_colors,
            slides_out,
            oc_tag,
            version_tag,
        )


# ---------------------------------------------------------------------------
# Testing-mode execution
# ---------------------------------------------------------------------------

def _run_testing_mode(
    panel: pd.DataFrame,
    cfg_full: dict,
    exp_cfg: dict,
    *,
    config_path: Optional[Path],
    feature_cache: Optional[dict[tuple[int, int], tuple[pd.DataFrame, dict]]] = None,
    index_cache: Optional[dict[tuple[object, ...], tuple[pd.DataFrame, dict, dict[str, object]]]] = None,
    slides_out: Optional[Path] = None,
    do_plot: bool = False,
) -> None:
    version_raw = str(exp_cfg.get("exposure_version", INDEX_EXPOSURE_VERSION))
    verbose = _testing_verbose(cfg_full)
    if version_raw != INDEX_EXPOSURE_VERSION:
        print(
            f"[testing] Ignoring exposure_version={version_raw!r}; testing mode always runs "
            f"{INDEX_EXPOSURE_VERSION!r} because it exercises the feature/model pipeline only."
        )

    print("\n" + "=" * 72)
    print(" Testing Mode: feature/model diagnostics only")
    print("=" * 72)
    print("[testing] Event-study regressions and exposure-by-outcome plots are skipped by design.")
    print(f"[testing] Diagnostic verbosity: {'verbose' if verbose else 'basic'}")

    pred_df, diagnostics, _ = _get_or_build_index_result(
        panel,
        cfg_full,
        exp_cfg,
        config_path=config_path,
        feature_cache=feature_cache,
        index_cache=index_cache,
        report_diagnostics=True,
        detailed_diagnostics=verbose,
        force_report_diagnostics=True,
        do_plot=do_plot,
        slides_out=slides_out,
    )
    print(
        "[testing] Completed sample-model build: "
        f"{len(pred_df):,} firms | "
        f"{int(pred_df['train_sample'].sum()):,} training firms | "
        f"{int(pred_df['event_study_sample'].sum()):,} evaluation firms"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exposure-based firm-level event study around 2016 OPT policy.",
        allow_abbrev=False,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to config YAML "
            f"(default: {EVENT_STUDY_CONFIG_PATH if EVENT_STUDY_CONFIG_PATH.exists() else DEFAULT_CONFIG_PATH})."
        ),
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional run log path. All stdout/stderr is tee'd into this file.",
    )
    p.add_argument(
        "--outcome-col",
        type=str,
        default=None,
        help="Comma-separated outcome column(s). Overrides exposure_event_study.outcome_cols.",
    )
    p.add_argument(
        "--exposure-version",
        type=str,
        default=None,
        choices=("opt_hire_rate", "school_opt_share", "opt_probability_index", "both", "all"),
        help="Exposure measure to use. Overrides exposure_event_study.exposure_version.",
    )
    p.add_argument("--ntiles", type=int, default=None, help="Number of ntile groups.")
    p.add_argument("--feature-year-min", type=int, default=None, help="Override exposure_event_study.feature_year_min.")
    p.add_argument("--feature-year-max", type=int, default=None, help="Override exposure_event_study.feature_year_max.")
    p.add_argument(
        "--index-model-method",
        type=str,
        default=None,
        choices=MODEL_METHODS,
        help="Override exposure_event_study.index_model_method.",
    )
    p.add_argument(
        "--index-entry-mode",
        type=str,
        default=None,
        choices=ENTRY_MODES,
        help="Override exposure_event_study.index_entry_mode.",
    )
    p.add_argument("--target-year-min", type=int, default=None, help="Override exposure_event_study.target_year_min.")
    p.add_argument("--target-year-max", type=int, default=None, help="Override exposure_event_study.target_year_max.")
    p.add_argument(
        "--leaveout-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override exposure_event_study.leaveout_enabled.",
    )
    p.add_argument("--leaveout-share", type=float, default=None, help="Override exposure_event_study.leaveout_share.")
    p.add_argument("--leaveout-seed", type=int, default=None, help="Override exposure_event_study.leaveout_seed.")
    testing_group = p.add_mutually_exclusive_group()
    testing_group.add_argument("--testing", dest="testing", action="store_true")
    testing_group.add_argument("--no-testing", dest="testing", action="store_false")
    p.set_defaults(testing=None)
    legacy_cache_group = p.add_mutually_exclusive_group()
    legacy_cache_group.add_argument(
        "--legacy-cache",
        dest="legacy_cache",
        action="store_true",
        help="Enable compatibility with legacy cache metadata.",
    )
    legacy_cache_group.add_argument(
        "--no-legacy-cache",
        dest="legacy_cache",
        action="store_false",
        help="Disable legacy cache metadata compatibility.",
    )
    p.set_defaults(legacy_cache=None)
    p.add_argument("--no-plot", action="store_true", default=False, help="Skip all plot generation.")

    if args is None:
        args = tuple(sys.argv[1:])
    cli_args = [str(a) for a in args]
    cli_args = [arg for arg in cli_args if not (arg == "--f" or arg == "-f" or arg.startswith("--f=") or arg.startswith("-f="))]

    if args is None:
        argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
        has_kernel_argv = (
            len(sys.argv) >= 3
            and sys.argv[1] == "-f"
            and str(sys.argv[2]).lower().endswith(".json")
        )
        in_ipython_ctx = (
            "IPython" in sys.modules
            or "ipykernel" in sys.modules
            or "ipykernel_launcher" in argv0
            or has_kernel_argv
        )
        if in_ipython_ctx:
            parsed, unknown = p.parse_known_args(cli_args)
            if unknown:
                print(f"[info] Ignoring unknown IPython args: {unknown}")
            return parsed
    return p.parse_args(cli_args)


def _resolve_versions(version_raw: str) -> list[str]:
    if version_raw == "both":
        return list(LEGACY_EXPOSURE_VERSIONS)
    if version_raw == "all":
        return list(LEGACY_EXPOSURE_VERSIONS) + [INDEX_EXPOSURE_VERSION]
    return [version_raw]


def _validate_main_config(exp_cfg: dict, versions: list[str]) -> None:
    if any(version in LEGACY_EXPOSURE_VERSIONS for version in versions):
        validate_feature_window(
            int(exp_cfg.get("exposure_year_min", 2010)),
            int(exp_cfg.get("exposure_year_max", 2015)),
        )
    if INDEX_EXPOSURE_VERSION in versions:
        validate_opt_probability_config(
            model_method=str(exp_cfg.get("index_model_method", "logit")),
            entry_mode=str(exp_cfg.get("index_entry_mode", "ntiles")),
            ntiles=int(exp_cfg.get("ntiles", 4)),
            feature_year_min=int(exp_cfg.get("feature_year_min", 2010)),
            feature_year_max=int(exp_cfg.get("feature_year_max", 2015)),
            leaveout_enabled=bool(exp_cfg.get("leaveout_enabled", False)),
            leaveout_share=float(exp_cfg.get("leaveout_share", 0.25)),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cli_args: Optional[Iterable[str]] = None) -> None:
    """Run the exposure event study. Pass cli_args=[] to run from IPython with config defaults."""
    import time

    t0 = time.time()
    args = _parse_args(cli_args)
    resolved_config_path = _resolve_main_config_path(args.config)
    cfg_full = load_config(resolved_config_path)
    exp_cfg = _ensure_cfg_section(cfg_full, "exposure_event_study")
    legacy_cache = _configure_legacy_cache_mode(
        args.legacy_cache,
        config_default=bool(exp_cfg.get("legacy_cache", False)),
    )
    if legacy_cache:
        _configure_legacy_cache_ignore_keys(exp_cfg.get("legacy_cache_ignore_keys"))
    else:
        os.environ.pop("EXPOSURE_EVENT_STUDY_LEGACY_CACHE_IGNORE_KEYS", None)
    testing_cfg = _ensure_cfg_section(cfg_full, "testing")

    if args.ntiles is not None:
        exp_cfg["ntiles"] = args.ntiles
    if args.exposure_version is not None:
        exp_cfg["exposure_version"] = args.exposure_version
    if args.feature_year_min is not None:
        exp_cfg["feature_year_min"] = args.feature_year_min
    if args.feature_year_max is not None:
        exp_cfg["feature_year_max"] = args.feature_year_max
    if args.index_model_method is not None:
        exp_cfg["index_model_method"] = args.index_model_method
    if args.index_entry_mode is not None:
        exp_cfg["index_entry_mode"] = args.index_entry_mode
    if args.target_year_min is not None:
        exp_cfg["target_year_min"] = args.target_year_min
    if args.target_year_max is not None:
        exp_cfg["target_year_max"] = args.target_year_max
    if args.leaveout_enabled is not None:
        exp_cfg["leaveout_enabled"] = args.leaveout_enabled
    if args.leaveout_share is not None:
        exp_cfg["leaveout_share"] = args.leaveout_share
    if args.leaveout_seed is not None:
        exp_cfg["leaveout_seed"] = args.leaveout_seed
    if args.testing is not None:
        testing_cfg["enabled"] = args.testing

    effective_testing = testing_enabled(cfg_full)
    log_path = _resolve_run_log_path(cfg_full, exp_cfg, cli_log_file=args.log_file)

    with _tee_output(log_path):
        print(
            "[main] Config path:       "
            f"{resolved_config_path if resolved_config_path is not None else DEFAULT_CONFIG_PATH}"
        )
        print(f"[main] Legacy cache mode:  {'on' if legacy_cache else 'off'}")
        if log_path is not None:
            print(f"[main] Log output:        {log_path}")

        slides_raw = str(exp_cfg.get("slides_out_dir", "")).strip()
        if slides_raw and slides_raw.lower() not in {"none", "null", ""}:
            root = str(Path(__file__).resolve().parents[2])
            slides_out: Optional[Path] = Path(slides_raw.replace("{root}", root))
            slides_out.mkdir(parents=True, exist_ok=True)
            print(f"[main] Slides output:     {slides_out}")
        else:
            slides_out = None
            print("[main] slides_out_dir not set — plots will display only, not saved.")

        version_raw = str(exp_cfg.get("exposure_version", "opt_hire_rate"))
        versions = _resolve_versions(version_raw)
        _validate_main_config(exp_cfg, [INDEX_EXPOSURE_VERSION] if effective_testing else versions)

        outcome_raw = args.outcome_col or exp_cfg.get(
            "outcome_cols",
            [
                "x_bin_any_nonzero",
                "log1p_y_cst_lag0",
                "log1p_y_cst_foreign_lag0",
                "log1p_y_cst_native_lag0",
                "log1p_y_new_hires_lag0",
                "log1p_y_new_hires_foreign_lag0",
                "log1p_y_new_hires_native_lag0",
            ],
        )
        if isinstance(outcome_raw, str):
            outcome_cols = [v.strip() for v in outcome_raw.split(",") if v.strip()]
        else:
            outcome_cols = [str(v).strip() for v in outcome_raw if str(v).strip()]
        do_plot = not args.no_plot

        print(f"[main] Testing mode:      {effective_testing}")
        print(f"[main] Exposure versions: {versions}")
        print(f"[main] Outcome columns:   {outcome_cols}")
        print(
            f"[main] Legacy exposure window: {exp_cfg.get('exposure_year_min', 2010)}–"
            f"{exp_cfg.get('exposure_year_max', 2015)} | "
            f"Index feature window: {exp_cfg.get('feature_year_min', 2010)}–"
            f"{exp_cfg.get('feature_year_max', 2015)} | "
            f"Event year: {exp_cfg.get('event_year', 2016)}"
        )

        panel, panel_meta = load_or_build_source_analysis_panel(
            config_path=resolved_config_path,
            cfg=cfg_full,
            data_min_t=int(exp_cfg.get("data_min_t", 2010)),
            data_max_t=int(exp_cfg.get("data_max_t", 2022)),
            force_rebuild=bool(exp_cfg.get("force_rebuild_source_analysis_panel", False)),
        )
        panel["c"] = pd.to_numeric(panel["c"], errors="coerce")
        panel["t"] = pd.to_numeric(panel["t"], errors="coerce")
        panel = panel.dropna(subset=["c", "t"]).copy()
        panel["c"] = panel["c"].astype(int)
        panel["t"] = panel["t"].astype(int)
        print(
            f"[main] Source analysis panel: {len(panel):,} rows | {panel['c'].nunique():,} firms | "
            f"years {int(panel['t'].min())}–{int(panel['t'].max())}"
        )
        if "preferred_rcid_source" in panel.columns and "outside_negative_candidate" in panel.columns:
            panel_firms = panel[["c", "preferred_rcid_source", "outside_negative_candidate"]].drop_duplicates()
            print(
                f"[main] Analysis-panel firms: "
                f"{int(panel_firms['preferred_rcid_source'].fillna(0).sum()):,} preferred-source + "
                f"{int(panel_firms['outside_negative_candidate'].fillna(0).sum()):,} outside negatives"
            )
        print(f"[main] Source analysis panel metadata: {panel_meta}")

        feature_cache: dict[tuple[int, int], tuple[pd.DataFrame, dict]] = {}
        index_cache: dict[tuple[object, ...], tuple[pd.DataFrame, dict, dict[str, object]]] = {}

        if effective_testing:
            _run_testing_mode(
                panel,
                cfg_full,
                exp_cfg,
                config_path=resolved_config_path,
                feature_cache=feature_cache,
                index_cache=index_cache,
                slides_out=slides_out,
                do_plot=do_plot,
            )
            elapsed = time.time() - t0
            print(f"\n[main] Done. Total time: {elapsed:.1f}s")
            return

        if INDEX_EXPOSURE_VERSION in versions:
            _get_or_build_index_result(
                panel,
                cfg_full,
                exp_cfg,
                config_path=resolved_config_path,
                feature_cache=feature_cache,
                index_cache=index_cache,
                report_diagnostics=True,
                detailed_diagnostics=False,
                do_plot=do_plot,
                slides_out=slides_out,
            )

        total_passes = len(versions) * len(outcome_cols)
        pass_n = 0
        for version in versions:
            for outcome_col in outcome_cols:
                pass_n += 1
                print(f"\n[main] Pass {pass_n}/{total_passes}: version={version}, outcome={outcome_col}")
                oc_tag = outcome_col.replace("/", "_").replace(" ", "_")
                _run_one_version(
                    panel,
                    outcome_col,
                    version,
                    exp_cfg,
                    cfg_full,
                    slides_out,
                    oc_tag,
                    do_plot,
                    config_path=resolved_config_path,
                    feature_cache=feature_cache,
                    index_cache=index_cache,
                )

        elapsed = time.time() - t0
        print(f"\n[main] Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
