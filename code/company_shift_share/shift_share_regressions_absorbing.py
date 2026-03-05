"""Shift-share regressions with absorbing-event treatment assignment.

This script creates an absorbing treatment cohort `g` per unit by selecting, in
the event window, the largest positive instrument shock that is locally
isolated: for candidate event year e with shock s, there must be no other shock
with absolute value > 0.5 * |s| within [e-3, e+3]. Event assignment always uses
the raw (non-binary) instrument with a configurable shock metric
(percent-change or first-difference).

Then it estimates first-stage and reduced-form outcomes with three specs:
1) TWFE (feols with unit and year FE),
2) DID2S (Gardner two-step, via pyfixest event_study),
3) Saturated event-study (Sun-Abraham-style interactions) aggregated to a
   post-treatment ATT.

In event-time mode, all three estimators produce dynamic profiles over:
    y_ct = alpha_c + lambda_t + sum_{r != r0} beta_r * 1[t - g_c = r]
with one coefficient per event time in a configured window.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.did.saturated_twfe import _compute_lincomb_stats, compute_period_weights

# Ensure logs are flushed promptly when running via nohup/redirection.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

try:
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config
except ModuleNotFoundError:
    # Allow direct execution when repo root is not already on PYTHONPATH.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from company_shift_share.config_loader import DEFAULT_CONFIG_PATH, get_cfg_section, load_config


_DEFAULT_INSTRUMENT_COL = "z_ct_full"
_DEFAULT_TREATMENT_COL = "x_bin_any_nonzero"
_DID2S_STATE_FE_FALLBACK_WARNED = False
_X_LAG_SUFFIX_MAP = {
    "masters_opt_hires_correction_aware": "",
    "masters_opt_hires": "",
    "valid_masters_opt_hires": "",
    "x_bin_any_nonzero": "_bin_any_nonzero",
    "x_bin_above_year_median": "_bin_above_year_median",
    "x_bin_topbot_quartile": "_bin_topbot_quartile",
}


def _parse_lag_token(token: str) -> int:
    return -int(token[1:]) if token.startswith("m") else int(token)


def _parse_prefixed_lag(colname: str, prefix: str) -> int | None:
    m = re.fullmatch(rf"{re.escape(prefix)}(m?\d+)", colname)
    if not m:
        return None
    return _parse_lag_token(m.group(1))


def _parse_x_lag_col(colname: str, suffix: str) -> int | None:
    m = re.fullmatch(rf"x_cst_lag(m?\d+){re.escape(suffix)}", colname)
    if not m:
        return None
    return _parse_lag_token(m.group(1))


def _enforce_balanced_panel(
    df: pd.DataFrame,
    years: np.ndarray,
    id_col: str = "c",
    time_col: str = "t",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.asarray(years, dtype=np.int64)
    years = np.unique(years)
    if years.size == 0 or df.empty:
        return df.copy(), pd.DataFrame(columns=[id_col, "n_rows", "n_years"])

    panel = df.loc[df[time_col].isin(years)].copy()
    n_years_expected = int(years.size)
    unit_stats = (
        panel.groupby(id_col, as_index=False)
        .agg(
            n_rows=(time_col, "size"),
            n_years=(time_col, "nunique"),
        )
    )
    balanced_ids = unit_stats.loc[
        (unit_stats["n_rows"] == n_years_expected)
        & (unit_stats["n_years"] == n_years_expected),
        id_col,
    ]
    balanced = panel.loc[panel[id_col].isin(set(balanced_ids))].copy()
    return balanced, unit_stats


def _ensure_derived_outcome_col(df: pd.DataFrame, col: str, x_source_col: str) -> None:
    if col in df.columns:
        return

    if col == "log1p_y_cst_lag0" and "y_cst_lag0" in df.columns:
        df[col] = np.where(
            df["y_cst_lag0"].notna() & (df["y_cst_lag0"] >= 0),
            np.log1p(df["y_cst_lag0"]),
            np.nan,
        )
        return

    if col.startswith("log1p_"):
        base_col = col[len("log1p_"):]
        if base_col in df.columns:
            df[col] = np.where(
                df[base_col].notna() & (df[base_col] >= 0),
                np.log1p(df[base_col]),
                np.nan,
            )
            return

    if x_source_col not in df.columns:
        return

    if col == "x_bin_any_nonzero":
        df[col] = (df[x_source_col].fillna(0) != 0).astype("int8")
        return

    if col == "x_bin_above_year_median":
        med = df.groupby("t")[x_source_col].transform("median")
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[x_source_col].notna() & med.notna()
        if mask.any():
            idx = df.index[mask]
            df.loc[idx, col] = (df.loc[idx, x_source_col] > med.loc[idx]).astype("int8")
        return

    if col == "x_bin_topbot_quartile":
        p25 = df.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.25))
        p75 = df.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.75))
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[x_source_col].notna() & p25.notna() & p75.notna()
        if mask.any():
            idx = df.index[mask]
            xv = df.loc[idx, x_source_col]
            p25v = p25.loc[idx]
            p75v = p75.loc[idx]
            df.loc[idx[xv <= p25v], col] = 0
            df.loc[idx[xv >= p75v], col] = 1
        return

    m = re.fullmatch(r"(x_cst_lagm?\d+)_bin_any_nonzero", col)
    if m and m.group(1) in df.columns:
        df[col] = (df[m.group(1)].fillna(0) != 0).astype("int8")
        return

    m = re.fullmatch(r"(x_cst_lagm?\d+)_bin_above_year_median", col)
    if m and m.group(1) in df.columns:
        base = m.group(1)
        med = df.groupby("t")[base].transform("median")
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[base].notna() & med.notna()
        if mask.any():
            idx = df.index[mask]
            df.loc[idx, col] = (df.loc[idx, base] > med.loc[idx]).astype("int8")
        return

    m = re.fullmatch(r"(x_cst_lagm?\d+)_bin_topbot_quartile", col)
    if m and m.group(1) in df.columns:
        base = m.group(1)
        p25 = df.groupby("t")[base].transform(lambda s: s.quantile(0.25))
        p75 = df.groupby("t")[base].transform(lambda s: s.quantile(0.75))
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[base].notna() & p25.notna() & p75.notna()
        if mask.any():
            idx = df.index[mask]
            xv = df.loc[idx, base]
            p25v = p25.loc[idx]
            p75v = p75.loc[idx]
            df.loc[idx[xv <= p25v], col] = 0
            df.loc[idx[xv >= p75v], col] = 1


def _build_x_lag_bin_cols(df: pd.DataFrame) -> None:
    base_x_lag_cols = [c for c in df.columns if re.fullmatch(r"x_cst_lag(m?\d+)", c)]
    for col in base_x_lag_cols:
        any_col = f"{col}_bin_any_nonzero"
        med_col = f"{col}_bin_above_year_median"
        qb_col = f"{col}_bin_topbot_quartile"
        _ensure_derived_outcome_col(df, any_col, x_source_col=col)
        _ensure_derived_outcome_col(df, med_col, x_source_col=col)
        _ensure_derived_outcome_col(df, qb_col, x_source_col=col)


def _build_required_x_lag_variant_cols(df: pd.DataFrame, suffix: str) -> None:
    if suffix == "":
        return
    if suffix not in {
        "_bin_any_nonzero",
        "_bin_above_year_median",
        "_bin_topbot_quartile",
    }:
        return

    base_x_lag_cols = [c for c in df.columns if re.fullmatch(r"x_cst_lag(m?\d+)", c)]
    if not base_x_lag_cols:
        return

    t_vals = pd.to_numeric(df["t"], errors="coerce")
    for col in base_x_lag_cols:
        out_col = f"{col}{suffix}"
        if out_col in df.columns:
            continue

        if suffix == "_bin_any_nonzero":
            df[out_col] = (df[col].fillna(0) != 0).astype("int8")
            continue

        if suffix == "_bin_above_year_median":
            med_by_t = df.groupby("t", sort=False)[col].median()
            med = t_vals.map(med_by_t)
            out = pd.Series(pd.NA, index=df.index, dtype="Int8")
            mask = df[col].notna() & med.notna()
            if mask.any():
                out.loc[mask] = (df.loc[mask, col] > med.loc[mask]).astype("int8")
            df[out_col] = out
            continue

        # suffix == "_bin_topbot_quartile"
        p25_by_t = df.groupby("t", sort=False)[col].quantile(0.25)
        p75_by_t = df.groupby("t", sort=False)[col].quantile(0.75)
        p25 = t_vals.map(p25_by_t)
        p75 = t_vals.map(p75_by_t)
        out = pd.Series(pd.NA, index=df.index, dtype="Int8")
        mask = df[col].notna() & p25.notna() & p75.notna()
        if mask.any():
            out.loc[mask & (df[col] <= p25)] = 0
            out.loc[mask & (df[col] >= p75)] = 1
        df[out_col] = out


def _identify_absorbing_events(
    df: pd.DataFrame,
    instrument_col_raw: str,
    event_min_t: int,
    event_max_t: int,
    local_window: int = 3,
    neighbor_shock_ratio: float = 0.5,
    use_pct_change: bool = True,
    min_shock_abs: float | None = None,
    min_shock_quantile: float | None = None,
) -> pd.DataFrame:
    if instrument_col_raw not in df.columns:
        raise ValueError(
            f"Raw instrument column '{instrument_col_raw}' not found in panel for event assignment."
        )
    if local_window < 0:
        raise ValueError("local_window must be >= 0.")
    if not (0 < neighbor_shock_ratio <= 1.0):
        raise ValueError("neighbor_shock_ratio must be in (0, 1].")
    if min_shock_quantile is not None and not (0 <= float(min_shock_quantile) <= 1):
        raise ValueError("min_shock_quantile must be in [0, 1].")

    z_df = (
        df[["c", "t", instrument_col_raw]]
        .dropna(subset=["c", "t"])
        .sort_values(["c", "t"])
        .copy()
    )
    z_df["z_lag1"] = z_df.groupby("c")[instrument_col_raw].shift(1)
    if use_pct_change:
        z_df["shock_raw"] = np.where(
            z_df["z_lag1"].notna() & (z_df["z_lag1"] != 0),
            (z_df[instrument_col_raw] - z_df["z_lag1"]) / z_df["z_lag1"],
            np.nan,
        )
    else:
        z_df["shock_raw"] = np.where(
            z_df["z_lag1"].notna(),
            z_df[instrument_col_raw] - z_df["z_lag1"],
            np.nan,
        )
    z_df = z_df.dropna(subset=["shock_raw"]).copy()
    if z_df.empty:
        return pd.DataFrame(columns=["c", "g", "event_shock", "event_abs_shock"])

    # Optional global floor to avoid selecting tiny positive blips as events.
    positive_in_window = z_df.loc[
        z_df["t"].between(event_min_t, event_max_t) & (z_df["shock_raw"] > 0),
        "shock_raw",
    ].to_numpy(dtype=np.float64)
    shock_floor_candidates: list[float] = [0.0]
    if min_shock_abs is not None:
        shock_floor_candidates.append(float(min_shock_abs))
    if min_shock_quantile is not None and positive_in_window.size > 0:
        shock_floor_candidates.append(
            float(np.nanquantile(positive_in_window, float(min_shock_quantile)))
        )
    min_required_positive_shock = max(shock_floor_candidates)

    picked_rows: list[dict[str, float | int]] = []
    # Loop by firm, but evaluate each firm's candidate set in vectorized numpy blocks.
    for firm, gdf in z_df.groupby("c", sort=False):
        t_all = gdf["t"].to_numpy(dtype=np.int64)
        shock_all = gdf["shock_raw"].to_numpy(dtype=np.float64)
        abs_all = np.abs(shock_all)

        cand_mask = (
            (t_all >= event_min_t)
            & (t_all <= event_max_t)
            & (shock_all > 0)
            & (shock_all >= min_required_positive_shock)
        )
        if not cand_mask.any():
            continue
        cand_idx = np.flatnonzero(cand_mask)
        # Rank candidates by raw positive shock, largest first.
        cand_idx = cand_idx[np.argsort(-shock_all[cand_idx], kind="mergesort")]

        cand_t = t_all[cand_idx][:, None]
        cand_abs = abs_all[cand_idx][:, None]
        threshold = neighbor_shock_ratio * cand_abs

        # Neighbor check remains absolute-value based.
        dt = np.abs(t_all[None, :] - cand_t)
        is_neighbor = (dt <= local_window) & (dt > 0)
        has_large_neighbor = np.any(is_neighbor & (abs_all[None, :] > threshold), axis=1)
        valid_pos = np.flatnonzero(~has_large_neighbor)
        if valid_pos.size == 0:
            continue

        chosen_idx = cand_idx[int(valid_pos[0])]
        chosen_shock = float(shock_all[chosen_idx])
        picked_rows.append(
            {
                "c": firm,
                "g": int(t_all[chosen_idx]),
                "event_shock": chosen_shock,
                "event_abs_shock": abs(chosen_shock),
            }
        )

    if not picked_rows:
        return pd.DataFrame(columns=["c", "g", "event_shock", "event_abs_shock"])
    return pd.DataFrame(picked_rows).drop_duplicates(subset=["c"]).copy()


def _prepare_base_did_sample(df: pd.DataFrame, include_state_year_fe: bool = False) -> pd.DataFrame:
    base_cols = ["c", "t", "g"]
    if include_state_year_fe and "state_year_fe" in df.columns:
        base_cols.append("state_year_fe")
    base = df[base_cols].copy()
    base["c"] = pd.to_numeric(base["c"], errors="coerce")
    base["t"] = pd.to_numeric(base["t"], errors="coerce")
    base["g"] = pd.to_numeric(base["g"], errors="coerce")
    base = base.dropna(subset=["c", "t", "g"])
    base["c"] = base["c"].astype("int64")
    base["t"] = base["t"].astype("int64")
    base["g"] = base["g"].astype("int64")
    base["is_treated"] = ((base["g"] > 0) & (base["t"] >= base["g"])).astype("int8")
    return base


def _prepare_did_sample_from_base(
    base_sample: pd.DataFrame,
    df: pd.DataFrame,
    outcome_col: str,
) -> pd.DataFrame:
    sample = base_sample.join(df[[outcome_col]], how="inner")
    sample = sample.dropna(subset=[outcome_col]).copy()
    return sample


def _event_study_xfml(
    sample: pd.DataFrame,
    use_state_year_fe: bool,
    extra_terms: list[str] | None = None,
) -> str | None:
    terms: list[str] = []
    if extra_terms:
        for term in extra_terms:
            if term in sample.columns:
                terms.append(term)

    if use_state_year_fe and "state_year_fe" in sample.columns and sample["state_year_fe"].notna().any():
        terms.append("C(state_year_fe)")

    if not terms:
        return None
    return " + ".join(terms)


def _fit_twfe_att(
    sample: pd.DataFrame,
    outcome_col: str,
    use_state_year_fe: bool = False,
) -> tuple[float, float]:
    fe_rhs = "c + t + state_year_fe" if (use_state_year_fe and "state_year_fe" in sample.columns) else "c + t"
    fit = pf.feols(
        f"{outcome_col} ~ is_treated | {fe_rhs}",
        data=sample,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    tidy = fit.tidy()
    return float(tidy.loc["is_treated", "Estimate"]), float(tidy.loc["is_treated", "Std. Error"])


def _fit_did2s_att(
    sample: pd.DataFrame,
    outcome_col: str,
    use_state_year_fe: bool = False,
) -> tuple[float, float]:
    # pyfixest did2s can fail when a calendar year has no not-yet-treated rows
    # (common after dropping never-treated units). Restrict to supported years.
    untreated_years = sample.loc[sample["is_treated"] == 0, "t"].unique()
    if untreated_years.size == 0:
        raise ValueError("No not-yet-treated observations available for DID2S first stage.")
    did2s_sample = sample.loc[sample["t"].isin(untreated_years)].copy()
    xfml = _event_study_xfml(did2s_sample, use_state_year_fe=use_state_year_fe)

    try:
        fit = pf.event_study(
            did2s_sample,
            yname=outcome_col,
            idname="c",
            tname="t",
            gname="g",
            xfml=xfml,
            estimator="did2s",
            cluster="c",
            att=True,
        )
    except KeyError as exc:
        # pyfixest did2s can fail with high-dimensional categorical xfml terms.
        if xfml is None:
            raise
        global _DID2S_STATE_FE_FALLBACK_WARNED
        if not _DID2S_STATE_FE_FALLBACK_WARNED:
            print(
                "[warn] DID2S with state-year FE covariates failed in pyfixest; "
                "falling back to DID2S without state-year FE."
            )
            _DID2S_STATE_FE_FALLBACK_WARNED = True
        fit = pf.event_study(
            did2s_sample,
            yname=outcome_col,
            idname="c",
            tname="t",
            gname="g",
            xfml=None,
            estimator="did2s",
            cluster="c",
            att=True,
        )
    tidy = fit.tidy()
    return float(tidy.loc["is_treated", "Estimate"]), float(tidy.loc["is_treated", "Std. Error"])


def _fit_saturated_post_att(
    sample: pd.DataFrame,
    outcome_col: str,
    use_state_year_fe: bool = False,
) -> tuple[float, float]:
    xfml = _event_study_xfml(sample, use_state_year_fe=use_state_year_fe)
    fit = pf.event_study(
        sample,
        yname=outcome_col,
        idname="c",
        tname="t",
        gname="g",
        xfml=xfml,
        estimator="saturated",
        cluster="c",
        att=True,
    )

    data_for_weights = fit._data.copy()
    post_period_counts = (
        data_for_weights.loc[data_for_weights["is_treated"] == 1, "rel_time"]
        .value_counts()
        .sort_index()
    )
    post_period_counts = post_period_counts.loc[post_period_counts.index >= 0]
    if post_period_counts.empty:
        raise ValueError("No post-treatment periods available for saturated ATT aggregation.")
    period_weight = (post_period_counts / post_period_counts.sum()).to_dict()
    period_weight = {float(k): float(v) for k, v in period_weight.items()}

    cell_w_df = compute_period_weights(
        data=data_for_weights,
        cohort=fit._gname,
        period="rel_time",
        treatment="is_treated",
    )
    cell_w = cell_w_df.rename(
        columns={fit._gname: "cohort", "rel_time": "rel_t", "weight": "cell_w"}
    )[["cohort", "rel_t", "cell_w"]]

    coefnames = pd.Series([str(x) for x in fit._coefnames], name="coefname")
    parsed = coefnames.str.extract(r"\[(?:T\.)?(-?\d+(?:\.\d+)?)\]:cohort_dummy_(\d+)$")
    coef_map = pd.DataFrame(
        {
            "coef_idx": np.arange(len(coefnames), dtype=int),
            "rel_t": pd.to_numeric(parsed[0], errors="coerce"),
            "cohort": pd.to_numeric(parsed[1], errors="coerce"),
        }
    ).dropna(subset=["rel_t", "cohort"])
    coef_map["cohort"] = coef_map["cohort"].astype("int64")
    coef_map = coef_map.loc[coef_map["rel_t"] >= 0].copy()
    if coef_map.empty:
        raise ValueError("No post-treatment saturated coefficients available for aggregation.")

    coef_map["period_w"] = coef_map["rel_t"].map(period_weight).fillna(0.0)
    coef_map = coef_map.merge(cell_w, on=["cohort", "rel_t"], how="left")
    coef_map["cell_w"] = coef_map["cell_w"].fillna(0.0)
    coef_map["agg_w"] = coef_map["period_w"] * coef_map["cell_w"]

    r_vec = np.zeros(len(coefnames), dtype=float)
    if not coef_map.empty:
        r_vec[coef_map["coef_idx"].to_numpy(dtype=int)] = coef_map["agg_w"].to_numpy(dtype=float)

    if not np.any(r_vec):
        raise ValueError("Failed to build non-zero aggregation weights for saturated ATT.")
    stats = _compute_lincomb_stats(R=r_vec, coefs=fit._beta_hat, vcov=fit._vcov)
    return float(stats["Estimate"]), float(stats["Std. Error"])


def _run_three_specs(
    df: pd.DataFrame,
    outcome_col: str,
    label: str,
    base_sample: pd.DataFrame | None = None,
    include_state_year_fe: bool = False,
) -> pd.DataFrame:
    if base_sample is None:
        base_sample = _prepare_base_did_sample(df, include_state_year_fe=include_state_year_fe)
    sample = _prepare_did_sample_from_base(base_sample=base_sample, df=df, outcome_col=outcome_col)
    out_rows: list[dict[str, object]] = []
    n_obs = int(len(sample))
    n_units = int(sample["c"].nunique()) if not sample.empty else 0
    n_treated_units = int(sample.loc[sample["g"] > 0, "c"].nunique()) if not sample.empty else 0

    if sample.empty or n_treated_units == 0:
        for est in ("twfe", "did2s", "saturated"):
            out_rows.append(
                {
                    "label": label,
                    "outcome_col": outcome_col,
                    "estimator": est,
                    "coef": np.nan,
                    "se": np.nan,
                    "n_obs": n_obs,
                    "n_units": n_units,
                    "n_treated_units": n_treated_units,
                    "status": "no_treated_or_empty",
                }
            )
        return pd.DataFrame(out_rows)

    for est in ("twfe", "did2s", "saturated"):
        try:
            if est == "twfe":
                coef, se = _fit_twfe_att(sample, outcome_col, use_state_year_fe=include_state_year_fe)
            elif est == "did2s":
                coef, se = _fit_did2s_att(sample, outcome_col, use_state_year_fe=include_state_year_fe)
            else:
                coef, se = _fit_saturated_post_att(sample, outcome_col, use_state_year_fe=include_state_year_fe)
            status = "ok"
        except Exception as exc:  # pragma: no cover - handled for robustness in exploratory runs
            print(f"[warn] {est} failed for {label} ({outcome_col}): {exc}")
            coef = np.nan
            se = np.nan
            status = f"error:{type(exc).__name__}"

        out_rows.append(
            {
                "label": label,
                "outcome_col": outcome_col,
                "estimator": est,
                "coef": coef,
                "se": se,
                "n_obs": n_obs,
                "n_units": n_units,
                "n_treated_units": n_treated_units,
                "status": status,
            }
        )
    return pd.DataFrame(out_rows)


def _collect_lag_results(
    df: pd.DataFrame,
    lag_cols: list[str],
    lag_parser,
    label_prefix: str,
    base_sample: pd.DataFrame | None = None,
    include_state_year_fe: bool = False,
) -> pd.DataFrame:
    if base_sample is None:
        base_sample = _prepare_base_did_sample(df, include_state_year_fe=include_state_year_fe)
    rows: list[pd.DataFrame] = []
    for col in lag_cols:
        lag = lag_parser(col)
        if lag is None:
            continue
        sub = _run_three_specs(
            df,
            col,
            label=f"{label_prefix}_lag_{lag}",
            base_sample=base_sample,
            include_state_year_fe=include_state_year_fe,
        )
        sub["lag"] = lag
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["estimator", "lag"])


def _plot_lag_results(df: pd.DataFrame, title: str, ylabel: str) -> None:
    if df.empty:
        print(f"[warn] No results to plot for '{title}'.")
        return

    colors = {"twfe": "tab:blue", "did2s": "tab:orange", "saturated": "tab:green"}
    plt.figure(figsize=(9, 5))
    for est in ("twfe", "did2s", "saturated"):
        sub = df.loc[df["estimator"] == est].sort_values("lag")
        if sub.empty:
            continue
        x = sub["lag"].to_numpy(dtype=float)
        y = sub["coef"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)

        plt.plot(x, y, marker="o", color=colors[est], label=est.upper())
        valid = np.isfinite(y) & np.isfinite(se)
        if valid.any():
            lo = y - 1.96 * se
            hi = y + 1.96 * se
            plt.fill_between(x[valid], lo[valid], hi[valid], color=colors[est], alpha=0.18)

    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Lag")
    plt.ylabel(ylabel)
    plt.title(title)
    xticks = sorted(int(v) for v in pd.to_numeric(df["lag"], errors="coerce").dropna().unique())
    if xticks:
        plt.xticks(xticks)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _event_dummy_col_name(event_time: int) -> str:
    return f"evt_m{abs(event_time)}" if event_time < 0 else f"evt_p{event_time}"


def _fit_twfe_event_time_profile(
    df: pd.DataFrame,
    outcome_col: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
    ref_event_time: int = -1,
    use_state_year_fe: bool = False,
    shock_rhs_col: str | None = None,
) -> pd.DataFrame:
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' is missing for event-time TWFE.")

    cols = ["c", "t", "g", outcome_col]
    if use_state_year_fe and "state_year_fe" in df.columns:
        cols.append("state_year_fe")
    if shock_rhs_col:
        if shock_rhs_col not in df.columns:
            raise ValueError(f"shock_rhs_col '{shock_rhs_col}' is missing for event-time TWFE.")
        cols.append(shock_rhs_col)
    sample = df[cols].copy()
    sample["c"] = pd.to_numeric(sample["c"], errors="coerce")
    sample["t"] = pd.to_numeric(sample["t"], errors="coerce")
    sample["g"] = pd.to_numeric(sample["g"], errors="coerce")
    drop_cols = ["c", "t", "g", outcome_col]
    if shock_rhs_col:
        sample[shock_rhs_col] = pd.to_numeric(sample[shock_rhs_col], errors="coerce")
        drop_cols.append(shock_rhs_col)
    sample = sample.dropna(subset=drop_cols).copy()
    if sample.empty:
        return pd.DataFrame()

    sample["c"] = sample["c"].astype("int64")
    sample["t"] = sample["t"].astype("int64")
    sample["g"] = sample["g"].astype("int64")
    sample["event_time"] = sample["t"] - sample["g"]
    sample = sample.loc[
        (sample["event_time"] >= event_time_min) & (sample["event_time"] <= event_time_max)
    ].copy()
    if sample.empty:
        return pd.DataFrame()

    available_event_times = sorted(int(v) for v in sample["event_time"].dropna().unique())
    if not available_event_times:
        return pd.DataFrame()

    ref_use = int(ref_event_time)
    if ref_use not in available_event_times:
        ref_use = -1 if -1 in available_event_times else int(min(available_event_times))

    rhs_terms = []
    event_time_term_map: list[tuple[int, str]] = []
    for r in available_event_times:
        if r == ref_use:
            continue
        col = _event_dummy_col_name(r)
        sample[col] = (sample["event_time"] == r).astype("int8")
        rhs_terms.append(col)
        event_time_term_map.append((r, col))

    if not rhs_terms:
        return pd.DataFrame(
            [{"event_time": ref_use, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0}]
        )

    fe_rhs = "c + t + state_year_fe" if (use_state_year_fe and "state_year_fe" in sample.columns) else "c + t"
    rhs_formula = " + ".join(rhs_terms)
    if shock_rhs_col:
        rhs_formula = f"{shock_rhs_col} + {rhs_formula}"

    fit = pf.feols(
        f"{outcome_col} ~ {rhs_formula} | {fe_rhs}",
        data=sample,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    tidy = fit.tidy()

    rows: list[dict[str, float | int]] = []
    for r, term in event_time_term_map:
        if term in tidy.index:
            est = float(tidy.loc[term, "Estimate"])
            se = float(tidy.loc[term, "Std. Error"])
            rows.append(
                {
                    "event_time": int(r),
                    "coef": est,
                    "se": se,
                    "lo": est - 1.96 * se,
                    "hi": est + 1.96 * se,
                }
            )
        else:
            rows.append(
                {
                    "event_time": int(r),
                    "coef": np.nan,
                    "se": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                }
            )

    rows.append({"event_time": int(ref_use), "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0})
    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    out["outcome_col"] = outcome_col
    out["estimator"] = "twfe_event_time"
    out["ref_event_time"] = int(ref_use)
    out["n_obs"] = int(len(sample))
    out["n_units"] = int(sample["c"].nunique())
    return out


def _fit_did2s_event_time_profile(
    df: pd.DataFrame,
    outcome_col: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
    ref_event_time: int = -1,
    use_state_year_fe: bool = False,
    shock_rhs_col: str | None = None,
) -> pd.DataFrame:
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' is missing for DID2S event-time profile.")

    cols = ["c", "t", "g", outcome_col]
    if use_state_year_fe and "state_year_fe" in df.columns:
        cols.append("state_year_fe")
    if shock_rhs_col:
        if shock_rhs_col not in df.columns:
            raise ValueError(f"shock_rhs_col '{shock_rhs_col}' is missing for DID2S event-time profile.")
        cols.append(shock_rhs_col)
    sample = df[cols].copy()
    sample["c"] = pd.to_numeric(sample["c"], errors="coerce")
    sample["t"] = pd.to_numeric(sample["t"], errors="coerce")
    sample["g"] = pd.to_numeric(sample["g"], errors="coerce")
    drop_cols = ["c", "t", "g", outcome_col]
    if shock_rhs_col:
        sample[shock_rhs_col] = pd.to_numeric(sample[shock_rhs_col], errors="coerce")
        drop_cols.append(shock_rhs_col)
    sample = sample.dropna(subset=drop_cols).copy()
    if sample.empty:
        return pd.DataFrame()

    sample["c"] = sample["c"].astype("int64")
    sample["t"] = sample["t"].astype("int64")
    sample["g"] = sample["g"].astype("int64")
    sample["event_time"] = sample["t"] - sample["g"]
    sample = sample.loc[
        (sample["event_time"] >= event_time_min) & (sample["event_time"] <= event_time_max)
    ].copy()
    if sample.empty:
        return pd.DataFrame()

    available_event_times = sorted(int(v) for v in sample["event_time"].dropna().unique())
    if not available_event_times:
        return pd.DataFrame()

    ref_use = int(ref_event_time)
    if ref_use not in available_event_times:
        ref_use = -1 if -1 in available_event_times else int(min(available_event_times))

    sample["is_untreated"] = (sample["t"] < sample["g"]).astype("int8")
    untreated_years = sample.loc[sample["is_untreated"] == 1, "t"].unique()
    if untreated_years.size == 0:
        raise ValueError("No untreated observations available for DID2S first stage.")
    sample = sample.loc[sample["t"].isin(untreated_years)].copy()
    if sample.empty:
        raise ValueError("No rows remain after restricting to untreated-supported years for DID2S.")

    # First stage: residualize y on FE using untreated observations only.
    fe_rhs = "c + t + state_year_fe" if (use_state_year_fe and "state_year_fe" in sample.columns) else "c + t"
    first_stage = pf.feols(
        f"{outcome_col} ~ 1 | {fe_rhs}",
        data=sample.loc[sample["is_untreated"] == 1].copy(),
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    sample["y_tilde"] = sample[outcome_col].to_numpy(dtype=float) - first_stage.predict(sample)
    sample = sample.loc[sample["y_tilde"].notna()].copy()
    if sample.empty:
        return pd.DataFrame()

    rhs_terms = []
    event_time_term_map: list[tuple[int, str]] = []
    for r in available_event_times:
        if r == ref_use:
            continue
        col = f"did2s_{_event_dummy_col_name(r)}"
        sample[col] = (sample["event_time"] == r).astype("int8")
        rhs_terms.append(col)
        event_time_term_map.append((r, col))

    if not rhs_terms:
        return pd.DataFrame(
            [{"event_time": ref_use, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0}]
        )

    rhs_formula = " + ".join(rhs_terms)
    if shock_rhs_col:
        rhs_formula = f"{shock_rhs_col} + {rhs_formula}"

    second_stage = pf.feols(
        f"y_tilde ~ {rhs_formula}",
        data=sample,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    tidy = second_stage.tidy()

    rows: list[dict[str, float | int]] = []
    for r, term in event_time_term_map:
        if term in tidy.index:
            est = float(tidy.loc[term, "Estimate"])
            se = float(tidy.loc[term, "Std. Error"])
            rows.append(
                {
                    "event_time": int(r),
                    "coef": est,
                    "se": se,
                    "lo": est - 1.96 * se,
                    "hi": est + 1.96 * se,
                }
            )
        else:
            rows.append(
                {
                    "event_time": int(r),
                    "coef": np.nan,
                    "se": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                }
            )

    rows.append({"event_time": int(ref_use), "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0})
    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    out["outcome_col"] = outcome_col
    out["estimator"] = "did2s_event_time"
    out["ref_event_time"] = int(ref_use)
    out["n_obs"] = int(len(sample))
    out["n_units"] = int(sample["c"].nunique())
    return out


def _fit_saturated_event_time_profile(
    df: pd.DataFrame,
    outcome_col: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
    ref_event_time: int = -1,
    use_state_year_fe: bool = False,
    shock_rhs_col: str | None = None,
) -> pd.DataFrame:
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' is missing for saturated event-time profile.")

    cols = ["c", "t", "g", outcome_col]
    if use_state_year_fe and "state_year_fe" in df.columns:
        cols.append("state_year_fe")
    if shock_rhs_col:
        if shock_rhs_col not in df.columns:
            raise ValueError(f"shock_rhs_col '{shock_rhs_col}' is missing for saturated event-time profile.")
        cols.append(shock_rhs_col)
    sample = df[cols].copy()
    sample["c"] = pd.to_numeric(sample["c"], errors="coerce")
    sample["t"] = pd.to_numeric(sample["t"], errors="coerce")
    sample["g"] = pd.to_numeric(sample["g"], errors="coerce")
    drop_cols = ["c", "t", "g", outcome_col]
    if shock_rhs_col:
        sample[shock_rhs_col] = pd.to_numeric(sample[shock_rhs_col], errors="coerce")
        drop_cols.append(shock_rhs_col)
    sample = sample.dropna(subset=drop_cols).copy()
    if sample.empty:
        return pd.DataFrame()

    sample["c"] = sample["c"].astype("int64")
    sample["t"] = sample["t"].astype("int64")
    sample["g"] = sample["g"].astype("int64")
    sample["event_time"] = sample["t"] - sample["g"]
    sample = sample.loc[
        (sample["event_time"] >= event_time_min) & (sample["event_time"] <= event_time_max)
    ].copy()
    if sample.empty:
        return pd.DataFrame()

    available_event_times = sorted(int(v) for v in sample["event_time"].dropna().unique())
    if not available_event_times:
        return pd.DataFrame()
    ref_use = int(ref_event_time)
    if ref_use not in available_event_times:
        ref_use = -1 if -1 in available_event_times else int(min(available_event_times))

    extra_terms = [shock_rhs_col] if shock_rhs_col else None
    xfml = _event_study_xfml(sample, use_state_year_fe=use_state_year_fe, extra_terms=extra_terms)
    fit = pf.event_study(
        sample,
        yname=outcome_col,
        idname="c",
        tname="t",
        gname="g",
        xfml=xfml,
        estimator="saturated",
        cluster="c",
        att=False,
    )

    data_for_weights = fit._data.copy()
    data_for_weights["cohort"] = pd.to_numeric(data_for_weights[fit._gname], errors="coerce")
    data_for_weights["rel_t"] = pd.to_numeric(data_for_weights["rel_time"], errors="coerce")
    data_for_weights = data_for_weights.dropna(subset=["cohort", "rel_t"]).copy()
    data_for_weights["cohort"] = data_for_weights["cohort"].astype("int64")
    data_for_weights = data_for_weights.loc[
        (data_for_weights["rel_t"] >= event_time_min) & (data_for_weights["rel_t"] <= event_time_max)
    ].copy()

    cell_counts = (
        data_for_weights.groupby(["cohort", "rel_t"], as_index=False)
        .agg(n_cell=("c", "size"))
        if not data_for_weights.empty
        else pd.DataFrame(columns=["cohort", "rel_t", "n_cell"])
    )

    coefnames = pd.Series([str(x) for x in fit._coefnames], name="coefname")
    parsed = coefnames.str.extract(r"\[(?:T\.)?(-?\d+(?:\.\d+)?)\]:cohort_dummy_(\d+)$")
    coef_map = pd.DataFrame(
        {
            "coef_idx": np.arange(len(coefnames), dtype=int),
            "rel_t": pd.to_numeric(parsed[0], errors="coerce"),
            "cohort": pd.to_numeric(parsed[1], errors="coerce"),
        }
    ).dropna(subset=["rel_t", "cohort"])
    coef_map["cohort"] = coef_map["cohort"].astype("int64")
    coef_map = coef_map.loc[
        (coef_map["rel_t"] >= event_time_min) & (coef_map["rel_t"] <= event_time_max)
    ].copy()

    rows: list[dict[str, float | int]] = []
    for r in available_event_times:
        if r == ref_use:
            rows.append({"event_time": int(r), "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0})
            continue
        rel_coef = coef_map.loc[coef_map["rel_t"] == float(r)].copy()
        if rel_coef.empty:
            rows.append({"event_time": int(r), "coef": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        rel_coef = rel_coef.merge(cell_counts, on=["cohort", "rel_t"], how="left")
        rel_coef["n_cell"] = rel_coef["n_cell"].fillna(0.0)
        if float(rel_coef["n_cell"].sum()) <= 0:
            rows.append({"event_time": int(r), "coef": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        rel_coef["w"] = rel_coef["n_cell"] / rel_coef["n_cell"].sum()
        r_vec = np.zeros(len(coefnames), dtype=float)
        r_vec[rel_coef["coef_idx"].to_numpy(dtype=int)] = rel_coef["w"].to_numpy(dtype=float)
        stats = _compute_lincomb_stats(R=r_vec, coefs=fit._beta_hat, vcov=fit._vcov)
        est = float(stats["Estimate"])
        se = float(stats["Std. Error"])
        rows.append(
            {
                "event_time": int(r),
                "coef": est,
                "se": se,
                "lo": est - 1.96 * se,
                "hi": est + 1.96 * se,
            }
        )

    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    out["outcome_col"] = outcome_col
    out["estimator"] = "saturated_event_time"
    out["ref_event_time"] = int(ref_use)
    out["n_obs"] = int(len(sample))
    out["n_units"] = int(sample["c"].nunique())
    return out


def _collect_event_time_profiles(
    df: pd.DataFrame,
    outcome_col: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
    ref_event_time: int = -1,
    use_state_year_fe: bool = False,
    shock_rhs_col: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    spec_fns = [
        ("twfe_event_time", _fit_twfe_event_time_profile),
        ("did2s_event_time", _fit_did2s_event_time_profile),
        ("saturated_event_time", _fit_saturated_event_time_profile),
    ]
    for est_name, fn in spec_fns:
        try:
            out = fn(
                df,
                outcome_col=outcome_col,
                event_time_min=event_time_min,
                event_time_max=event_time_max,
                ref_event_time=ref_event_time,
                use_state_year_fe=use_state_year_fe,
                shock_rhs_col=shock_rhs_col,
            )
            if out is None or out.empty:
                continue
            rows.append(out)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] {est_name} failed for event-time profile ({outcome_col}): {exc}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["estimator", "event_time"])


def _plot_event_time_profiles(df: pd.DataFrame, title: str, ylabel: str) -> None:
    if df.empty:
        print(f"[warn] No event-time coefficients to plot for '{title}'.")
        return
    colors = {
        "twfe_event_time": "tab:blue",
        "did2s_event_time": "tab:orange",
        "saturated_event_time": "tab:green",
    }
    labels = {
        "twfe_event_time": "TWFE",
        "did2s_event_time": "DID2S",
        "saturated_event_time": "SATURATED",
    }
    plt.figure(figsize=(9, 5))
    for est in ("twfe_event_time", "did2s_event_time", "saturated_event_time"):
        sub = df.loc[df["estimator"] == est].sort_values("event_time")
        if sub.empty:
            continue
        x = sub["event_time"].to_numpy(dtype=float)
        y = sub["coef"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", color=colors[est], label=labels[est])
        valid = np.isfinite(y) & np.isfinite(se)
        if valid.any():
            lo = y - 1.96 * se
            hi = y + 1.96 * se
            plt.fill_between(x[valid], lo[valid], hi[valid], color=colors[est], alpha=0.15)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time (t - g)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(int(v) for v in pd.to_numeric(df["event_time"], errors="coerce").dropna().unique()))
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_event_time_profile(df: pd.DataFrame, title: str, ylabel: str) -> None:
    if df.empty:
        print(f"[warn] No event-time coefficients to plot for '{title}'.")
        return
    plt.figure(figsize=(8, 4.5))
    yerr = 1.96 * df["se"].fillna(0)
    plt.errorbar(
        df["event_time"],
        df["coef"],
        yerr=yerr,
        fmt="o-",
        capsize=4,
        color="tab:blue",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time (t - g)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(int(v) for v in df["event_time"].unique()))
    plt.tight_layout()
    plt.show()


def _plot_raw_means_by_event_time(
    df: pd.DataFrame,
    outcome_col: str,
    title: str,
    ylabel: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
) -> None:
    if outcome_col not in df.columns:
        print(f"[warn] Column '{outcome_col}' is not available for raw means plot.")
        return

    plot_df = df[["t", "g", outcome_col]].copy()
    plot_df = plot_df.dropna(subset=["t", "g", outcome_col])
    if plot_df.empty:
        print(f"[warn] No rows available for raw means plot ({outcome_col}).")
        return

    plot_df["event_time"] = plot_df["t"].astype("int64") - plot_df["g"].astype("int64")
    plot_df = plot_df.loc[
        (plot_df["event_time"] >= event_time_min) & (plot_df["event_time"] <= event_time_max)
    ].copy()
    if plot_df.empty:
        print(
            f"[warn] No rows in event-time window [{event_time_min}, {event_time_max}] "
            f"for raw means plot ({outcome_col})."
        )
        return

    mean_df = (
        plot_df.groupby(["g", "event_time"], as_index=False)
        .agg(
            mean_value=(outcome_col, "mean"),
            n_obs=(outcome_col, "size"),
        )
        .sort_values(["g", "event_time"])
    )
    print(f"\n[raw_means_by_event_time::{outcome_col}]")
    print(mean_df)

    cohorts = sorted(int(v) for v in mean_df["g"].dropna().unique())
    if not cohorts:
        print(f"[warn] No treatment cohorts available for raw means plot ({outcome_col}).")
        return
    cmap = plt.cm.get_cmap("viridis", len(cohorts))

    plt.figure(figsize=(9, 5))
    for i, cohort in enumerate(cohorts):
        sub = mean_df.loc[mean_df["g"] == cohort].sort_values("event_time")
        if sub.empty:
            continue
        plt.plot(
            sub["event_time"],
            sub["mean_value"],
            marker="o",
            linewidth=1.5,
            color=cmap(i),
            label=f"g={cohort}",
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time (t - g)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(list(range(event_time_min, event_time_max + 1)))
    plt.legend(title="Treatment year", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def _plot_mean_instrument_by_event_time(
    df: pd.DataFrame,
    instrument_col_raw: str,
    event_time_min: int = -3,
    event_time_max: int = 3,
) -> None:
    if instrument_col_raw not in df.columns:
        print(f"[warn] Instrument column '{instrument_col_raw}' not available for event-time means plot.")
        return

    plot_df = df[["t", "g", instrument_col_raw]].copy()
    plot_df = plot_df.dropna(subset=["t", "g", instrument_col_raw])
    if plot_df.empty:
        print("[warn] No rows available for instrument-by-event-time plot.")
        return

    plot_df["event_time"] = plot_df["t"].astype("int64") - plot_df["g"].astype("int64")
    plot_df = plot_df.loc[
        (plot_df["event_time"] >= event_time_min) & (plot_df["event_time"] <= event_time_max)
    ].copy()
    if plot_df.empty:
        print(
            f"[warn] No rows in event-time window [{event_time_min}, {event_time_max}] "
            f"for instrument '{instrument_col_raw}'."
        )
        return

    mean_df = (
        plot_df.groupby("event_time", as_index=False)
        .agg(
            mean_instrument=(instrument_col_raw, "mean"),
            se_instrument=(instrument_col_raw, "sem"),
            n_obs=(instrument_col_raw, "size"),
        )
        .sort_values("event_time")
    )
    print("\n[mean_instrument_by_event_time]")
    print(mean_df)

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        mean_df["event_time"],
        mean_df["mean_instrument"],
        yerr=1.96 * mean_df["se_instrument"].fillna(0),
        fmt="o-",
        capsize=4,
        color="tab:purple",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Event time")
    plt.ylabel(f"Mean {instrument_col_raw}")
    plt.title(f"Mean {instrument_col_raw} by Event Time")
    plt.xticks(list(range(event_time_min, event_time_max + 1)))
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    paths_cfg = get_cfg_section(cfg, "paths")
    build_cfg = get_cfg_section(cfg, "build_company_shift_share")
    reg_cfg = get_cfg_section(cfg, "shift_share_regressions")
    include_bachelors_sample = bool(reg_cfg.get("include_bachelors_sample", False))
    opt_shifts = bool(build_cfg.get("opt_shifts", False))
    opt_shifts_degree_scope = str(build_cfg.get("opt_shifts_degree_scope", "bachelors_masters"))

    outcome_prefix = str(reg_cfg.get("outcome_prefix", "y_cst_lag"))
    treatment_col = str(
        reg_cfg.get("treatment_col", reg_cfg.get("dependent", _DEFAULT_TREATMENT_COL))
    )
    x_source_col = str(reg_cfg.get("alt_event_x_source_col", "masters_opt_hires_correction_aware"))
    instrument_col_raw = str(reg_cfg.get("instrument", _DEFAULT_INSTRUMENT_COL))
    use_log_outcome_for_reduced_form = bool(reg_cfg.get("use_log_outcome_for_reduced_form", False))
    absorbing_plot_raw_means_by_event_time = bool(
        reg_cfg.get("absorbing_plot_raw_means_by_event_time", False)
    )
    absorbing_binary_instrument = bool(reg_cfg.get("absorbing_binary_instrument", False))
    use_state_year_fe = bool(reg_cfg.get("use_state_year_fe", False))
    absorbing_use_event_time_twfe = bool(reg_cfg.get("absorbing_use_event_time_twfe", True))
    absorbing_event_time_ref = int(reg_cfg.get("absorbing_event_time_ref", -1))

    data_min_t = int(reg_cfg.get("alt_event_data_min_t", 2008))
    data_max_t = int(reg_cfg.get("alt_event_data_max_t", 2022))
    event_min_t = int(reg_cfg.get("alt_event_event_min_t", 2012))
    event_max_t = int(reg_cfg.get("alt_event_event_max_t", 2018))
    # Restrict lag/event-time graphs to [-3, +3] by design.
    lag_plot_min = -3
    lag_plot_max = 3
    event_local_window = int(reg_cfg.get("absorbing_event_local_window", 3))
    neighbor_shock_ratio = float(reg_cfg.get("absorbing_neighbor_shock_ratio", 0.5))

    # Event-assignment defaults are stricter/more stable when opt_shifts are used:
    # first-differences (not pct changes) + minimum shock floor at a high quantile.
    shock_metric_cfg = str(reg_cfg.get("absorbing_event_shock_metric", "auto")).strip().lower()
    if shock_metric_cfg == "auto":
        use_pct_change_for_event_assignment = not opt_shifts
    elif shock_metric_cfg in {"pct_change", "percent_change", "pct"}:
        use_pct_change_for_event_assignment = True
    elif shock_metric_cfg in {"first_difference", "difference", "diff", "level_change"}:
        use_pct_change_for_event_assignment = False
    else:
        raise ValueError(
            "absorbing_event_shock_metric must be one of: "
            "auto, pct_change, first_difference."
        )

    min_shock_quantile_cfg = reg_cfg.get("absorbing_event_min_shock_quantile", None)
    if isinstance(min_shock_quantile_cfg, str) and min_shock_quantile_cfg.strip().lower() in {"", "none", "null"}:
        min_shock_quantile_cfg = None
    min_shock_quantile = (
        float(min_shock_quantile_cfg)
        if min_shock_quantile_cfg is not None
        else (0.90 if opt_shifts else None)
    )

    min_shock_abs_cfg = reg_cfg.get("absorbing_event_min_shock_abs", None)
    if isinstance(min_shock_abs_cfg, str) and min_shock_abs_cfg.strip().lower() in {"", "none", "null"}:
        min_shock_abs_cfg = None
    min_shock_abs = float(min_shock_abs_cfg) if min_shock_abs_cfg is not None else None

    if data_min_t > data_max_t:
        raise ValueError("Require data_min_t <= data_max_t.")
    if event_min_t > event_max_t:
        raise ValueError("Require event_min_t <= event_max_t.")

    panel_key = "analysis_panel"
    if include_bachelors_sample:
        if str(paths_cfg.get("analysis_panel_ma_ba", "")).strip():
            panel_key = "analysis_panel_ma_ba"
        else:
            print(
                "[warn] include_bachelors_sample=true but paths.analysis_panel_ma_ba is unset; "
                "falling back to paths.analysis_panel."
            )
    con = ddb.connect()
    panel_path = Path(paths_cfg[panel_key])
    if not panel_path.is_absolute():
        panel_path = Path.cwd() / panel_path
    panel_path_esc = str(panel_path).replace("'", "''")
    con.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{panel_path_esc}')")
    print(f"[info] Using analysis panel source: paths.{panel_key} -> {panel_path}")

    panel = con.sql(
        f"""
        SELECT *
        FROM analysis_panel
        WHERE t BETWEEN {data_min_t} AND {data_max_t}
        """
    ).df()
    if panel.empty:
        raise ValueError("Panel is empty after data-window filter.")

    # Build only derived columns actually needed by this run.
    _ensure_derived_outcome_col(panel, treatment_col, x_source_col=x_source_col)
    if treatment_col not in panel.columns:
        raise ValueError(
            f"Treatment column '{treatment_col}' is unavailable after deriving helper columns."
        )
    if treatment_col not in _X_LAG_SUFFIX_MAP:
        print(
            f"[warn] treatment_col='{treatment_col}' has no explicit x-lag mapping; "
            "defaulting to x_cst_lag* columns."
        )
    x_suffix = _X_LAG_SUFFIX_MAP.get(treatment_col, "")
    _build_required_x_lag_variant_cols(panel, suffix=x_suffix)

    base_rf_col = f"{outcome_prefix}0"
    if base_rf_col not in panel.columns:
        raise ValueError(
            f"Reduced-form base outcome '{base_rf_col}' not found. "
            "Check shift_share_regressions.outcome_prefix."
        )
    if use_log_outcome_for_reduced_form:
        reduced_form_outcome_col = f"log1p_{base_rf_col}"
        _ensure_derived_outcome_col(panel, reduced_form_outcome_col, x_source_col=x_source_col)
    else:
        reduced_form_outcome_col = base_rf_col

    if reduced_form_outcome_col not in panel.columns:
        raise ValueError(f"Reduced-form outcome column '{reduced_form_outcome_col}' is unavailable.")

    if instrument_col_raw.startswith("log_"):
        raise ValueError(
            f"Configured instrument '{instrument_col_raw}' appears to be logged. "
            "Use a non-log instrument column for absorbing-event assignment."
        )

    if instrument_col_raw not in panel.columns:
        raise ValueError(f"Instrument column '{instrument_col_raw}' not found in panel.")

    # Event assignment always uses the raw instrument. Shock metric defaults can
    # switch based on opt_shifts unless overridden in config.
    instrument_col_for_events = instrument_col_raw
    if opt_shifts and shock_metric_cfg == "auto":
        print(
            "[info] opt_shifts=true: using first-difference shocks for absorbing "
            "event assignment and a default min-shock quantile floor (0.90)."
        )
    print(
        f"[info] absorbing event assignment config: metric="
        f"{'pct_change' if use_pct_change_for_event_assignment else 'first_difference'}, "
        f"local_window={event_local_window}, neighbor_shock_ratio={neighbor_shock_ratio}, "
        f"min_shock_abs={min_shock_abs}, min_shock_quantile={min_shock_quantile}"
    )

    # Optional binary-z is created for estimation/diagnostics only (not for event assignment).
    instrument_col_for_estimation = instrument_col_raw
    if absorbing_binary_instrument:
        instrument_col_for_estimation = f"{instrument_col_raw}_bin01"
        z_raw_num = pd.to_numeric(panel[instrument_col_raw], errors="coerce")
        panel[instrument_col_for_estimation] = np.where(
            z_raw_num.notna(),
            (z_raw_num != 0).astype("int8"),
            np.nan,
        )
        print(
            f"[info] absorbing_binary_instrument=true: using raw '{instrument_col_for_events}' "
            f"for event assignment, and '{instrument_col_for_estimation}' for estimation/diagnostics."
        )

    event_map = _identify_absorbing_events(
        panel,
        instrument_col_raw=instrument_col_for_events,
        event_min_t=event_min_t,
        event_max_t=event_max_t,
        local_window=event_local_window,
        neighbor_shock_ratio=neighbor_shock_ratio,
        use_pct_change=use_pct_change_for_event_assignment,
        min_shock_abs=min_shock_abs,
        min_shock_quantile=min_shock_quantile,
    )
    did_panel = panel.merge(event_map, on="c", how="left")
    did_panel["g"] = pd.to_numeric(did_panel["g"], errors="coerce")
    # Drop never-treated units from the estimation sample.
    did_panel = did_panel.loc[did_panel["g"].notna() & (did_panel["g"] > 0)].copy()
    did_panel["g"] = did_panel["g"].astype("int64")
    did_panel["is_treated"] = ((did_panel["g"] > 0) & (did_panel["t"] >= did_panel["g"])).astype("int8")
    n_units_treated_pre_balance = int(did_panel["c"].nunique())
    n_rows_treated_pre_balance = int(len(did_panel))

    # Enforce strict balance over the estimation window.
    expected_years = np.arange(data_min_t, data_max_t + 1, dtype=np.int64)
    did_panel, _unit_stats_pre_balance = _enforce_balanced_panel(
        did_panel,
        years=expected_years,
        id_col="c",
        time_col="t",
    )
    did_panel["is_treated"] = ((did_panel["g"] > 0) & (did_panel["t"] >= did_panel["g"])).astype("int8")
    if use_state_year_fe:
        if "company_state" not in did_panel.columns:
            print(
                "[warn] use_state_year_fe=true but 'company_state' is missing in analysis panel; "
                "state-year FE will be skipped."
            )
            use_state_year_fe = False
        else:
            _state = did_panel["company_state"].astype("string").str.strip().str.upper()
            _state = _state.mask(_state.isna() | (_state == ""), "UNK")
            did_panel["company_state"] = _state
            _t_int = pd.to_numeric(did_panel["t"], errors="coerce").astype("Int64")
            did_panel["state_year_fe"] = _state.astype(str) + "_" + _t_int.astype(str)
            print(
                "[info] use_state_year_fe=true: added state_year_fe "
                f"(n_states={did_panel['company_state'].nunique(dropna=True)})."
            )

    n_units_all = int(panel["c"].nunique())
    n_units = int(did_panel["c"].nunique())
    n_treated_units = int(did_panel.loc[did_panel["g"] > 0, "c"].nunique())
    n_units_dropped_never_treated = n_units_all - n_units_treated_pre_balance
    n_units_dropped_unbalanced = n_units_treated_pre_balance - n_units

    event_counts_by_year = (
        did_panel[["c", "g"]]
        .drop_duplicates(subset=["c", "g"])
        .groupby("g", as_index=False)
        .agg(n_events=("c", "nunique"))
        .sort_values("g")
        if not did_panel.empty
        else pd.DataFrame(columns=["g", "n_events"])
    )

    print("\n[absorbing_event_counts_by_year]")
    print(event_counts_by_year)

    print("\n[absorbing_event_assignment_summary]")
    print(
        pd.DataFrame(
            [
                {
                    "n_rows": len(did_panel),
                    "n_units_all": n_units_all,
                    "n_units_total": n_units,
                    "n_units_treated": n_treated_units,
                    "n_units_dropped_never_treated": n_units_dropped_never_treated,
                    "n_units_treated_pre_balance": n_units_treated_pre_balance,
                    "n_units_dropped_unbalanced": n_units_dropped_unbalanced,
                    "n_rows_treated_pre_balance": n_rows_treated_pre_balance,
                    "n_rows_dropped_unbalanced": n_rows_treated_pre_balance - len(did_panel),
                    "balanced_year_start": int(expected_years.min()),
                    "balanced_year_end": int(expected_years.max()),
                    "balanced_n_years": int(expected_years.size),
                    "share_treated_units": (n_treated_units / n_units) if n_units else np.nan,
                    "event_min_t": event_min_t,
                    "event_max_t": event_max_t,
                    "local_window": event_local_window,
                    "neighbor_shock_ratio": neighbor_shock_ratio,
                    "min_shock_abs": min_shock_abs,
                    "min_shock_quantile": min_shock_quantile,
                    "event_shock_metric": (
                        "pct_change" if use_pct_change_for_event_assignment else "first_difference"
                    ),
                    "instrument_for_events": instrument_col_for_events,
                    "instrument_for_estimation": instrument_col_for_estimation,
                    "opt_shifts": opt_shifts,
                    "opt_shifts_degree_scope": opt_shifts_degree_scope,
                    "absorbing_plot_raw_means_by_event_time": absorbing_plot_raw_means_by_event_time,
                    "absorbing_binary_instrument": absorbing_binary_instrument,
                    "use_state_year_fe": use_state_year_fe,
                    "absorbing_use_event_time_twfe": absorbing_use_event_time_twfe,
                    "absorbing_event_time_ref": absorbing_event_time_ref,
                }
            ]
        )
    )

    _plot_mean_instrument_by_event_time(
        did_panel,
        instrument_col_raw=instrument_col_for_estimation,
        event_time_min=lag_plot_min,
        event_time_max=lag_plot_max,
    )

    # Core first-stage / reduced-form comparisons.
    base_sample_main = _prepare_base_did_sample(
        did_panel,
        include_state_year_fe=use_state_year_fe,
    )
    fs_res = _run_three_specs(
        did_panel,
        treatment_col,
        label="first_stage",
        base_sample=base_sample_main,
        include_state_year_fe=use_state_year_fe,
    )
    rf_res = _run_three_specs(
        did_panel,
        reduced_form_outcome_col,
        label="reduced_form",
        base_sample=base_sample_main,
        include_state_year_fe=use_state_year_fe,
    )
    main_res = pd.concat([fs_res, rf_res], ignore_index=True)
    main_res["lo"] = main_res["coef"] - 1.96 * main_res["se"]
    main_res["hi"] = main_res["coef"] + 1.96 * main_res["se"]
    print("\n[absorbing_first_stage_and_reduced_form]")
    print(main_res)

    # In continuous-z mode, include z on the event-time RHS (not interacted with event-time dummies).
    event_time_shock_rhs_col: str | None = None
    if not absorbing_binary_instrument:
        event_time_shock_rhs_col = instrument_col_for_estimation
        print(
            f"[info] absorbing_binary_instrument=false: event-time specs include "
            f"plain RHS shock '{event_time_shock_rhs_col}' (no z x event_time interactions)."
        )

    if absorbing_plot_raw_means_by_event_time:
        x_mean_col = treatment_col if treatment_col in did_panel.columns else None
        y_mean_col = (
            reduced_form_outcome_col
            if reduced_form_outcome_col in did_panel.columns
            else (base_rf_col if base_rf_col in did_panel.columns else None)
        )
        print(
            f"\n[info] absorbing_plot_raw_means_by_event_time=true: "
            f"x_mean_col={x_mean_col}, y_mean_col={y_mean_col}"
        )
        if x_mean_col is None:
            print("[warn] Unable to locate treatment column for raw means plot.")
        else:
            _plot_raw_means_by_event_time(
                did_panel,
                outcome_col=x_mean_col,
                title="Raw mean X by event time (colored by treatment year)",
                ylabel=f"Mean {x_mean_col}",
                event_time_min=lag_plot_min,
                event_time_max=lag_plot_max,
            )
        if y_mean_col is None:
            print("[warn] Unable to locate reduced-form outcome column for raw means plot.")
        else:
            _plot_raw_means_by_event_time(
                did_panel,
                outcome_col=y_mean_col,
                title="Raw mean Y by event time (colored by treatment year)",
                ylabel=f"Mean {y_mean_col}",
                event_time_min=lag_plot_min,
                event_time_max=lag_plot_max,
            )

    if absorbing_use_event_time_twfe:
        try:
            fs_evt = _collect_event_time_profiles(
                did_panel,
                outcome_col=treatment_col,
                event_time_min=lag_plot_min,
                event_time_max=lag_plot_max,
                ref_event_time=absorbing_event_time_ref,
                use_state_year_fe=use_state_year_fe,
                shock_rhs_col=event_time_shock_rhs_col,
            )
            print("\n[absorbing_event_time_first_stage]")
            print(fs_evt)
            _plot_event_time_profiles(
                fs_evt,
                title="Absorbing-event event-time profile (first stage)",
                ylabel=f"Coefficient on 1[event_time=r] ({treatment_col})",
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Event-time profile failed for first stage ({treatment_col}): {exc}")

        try:
            rf_evt = _collect_event_time_profiles(
                did_panel,
                outcome_col=reduced_form_outcome_col,
                event_time_min=lag_plot_min,
                event_time_max=lag_plot_max,
                ref_event_time=absorbing_event_time_ref,
                use_state_year_fe=use_state_year_fe,
                shock_rhs_col=event_time_shock_rhs_col,
            )
            print("\n[absorbing_event_time_reduced_form]")
            print(rf_evt)
            _plot_event_time_profiles(
                rf_evt,
                title="Absorbing-event event-time profile (reduced form)",
                ylabel=f"Coefficient on 1[event_time=r] ({reduced_form_outcome_col})",
            )
        except Exception as exc:  # pragma: no cover
            print(
                f"[warn] Event-time profile failed for reduced form "
                f"({reduced_form_outcome_col}): {exc}"
            )
    else:
        # Legacy outcome-lag profile mode.
        x_lag_cols = [
            c
            for c in did_panel.columns
            if (_parse_x_lag_col(c, x_suffix) is not None)
            and (lag_plot_min <= int(_parse_x_lag_col(c, x_suffix)) <= lag_plot_max)
        ]
        x_lag_cols = sorted(x_lag_cols, key=lambda c: int(_parse_x_lag_col(c, x_suffix)))
        y_lag_cols = [
            c
            for c in did_panel.columns
            if (_parse_prefixed_lag(c, outcome_prefix) is not None)
            and (lag_plot_min <= int(_parse_prefixed_lag(c, outcome_prefix)) <= lag_plot_max)
        ]
        y_lag_cols = sorted(y_lag_cols, key=lambda c: int(_parse_prefixed_lag(c, outcome_prefix)))

        if not x_lag_cols:
            print(
                f"\n[warn] No x-lag columns found for treatment_col='{treatment_col}' "
                f"(suffix='{x_suffix}', lag window=[{lag_plot_min}, {lag_plot_max}])."
            )
        else:
            x_lag_res = _collect_lag_results(
                did_panel,
                lag_cols=x_lag_cols,
                lag_parser=lambda c: _parse_x_lag_col(c, x_suffix),
                label_prefix="x",
                base_sample=base_sample_main,
                include_state_year_fe=use_state_year_fe,
            )
            x_lag_res["lo"] = x_lag_res["coef"] - 1.96 * x_lag_res["se"]
            x_lag_res["hi"] = x_lag_res["coef"] + 1.96 * x_lag_res["se"]
            print("\n[absorbing_coefficients_by_xlag]")
            print(x_lag_res)
            _plot_lag_results(
                x_lag_res,
                title="Absorbing-event coefficients by x lag (TWFE vs DID2S vs Saturated)",
                ylabel="Coefficient on treatment exposure",
            )

        if not y_lag_cols:
            print(
                f"\n[warn] No y-lag columns found matching prefix '{outcome_prefix}' "
                f"in lag window=[{lag_plot_min}, {lag_plot_max}]."
            )
        else:
            y_lag_res = _collect_lag_results(
                did_panel,
                lag_cols=y_lag_cols,
                lag_parser=lambda c: _parse_prefixed_lag(c, outcome_prefix),
                label_prefix="y",
                base_sample=base_sample_main,
                include_state_year_fe=use_state_year_fe,
            )
            y_lag_res["lo"] = y_lag_res["coef"] - 1.96 * y_lag_res["se"]
            y_lag_res["hi"] = y_lag_res["coef"] + 1.96 * y_lag_res["se"]
            print("\n[absorbing_coefficients_by_ylag]")
            print(y_lag_res)
            _plot_lag_results(
                y_lag_res,
                title="Absorbing-event coefficients by y lag (TWFE vs DID2S vs Saturated)",
                ylabel="Coefficient on treatment exposure",
            )


if __name__ == "__main__":
    main()
