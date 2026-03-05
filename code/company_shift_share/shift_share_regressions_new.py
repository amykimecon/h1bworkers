"""Build the main regression panel in DuckDB."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import seaborn as sns

# WHICH SECTIONS TO RUN
regs = True #True # regressions, table output
coef_plots_bycat = True #True # coefficient plots by year and by lagged firm size ventile (requires a lag-1 outcome column)
zct_hists = True # histogram of z_ct_full and averages by year and by lagged firm size decile
binscatters_for_regs = True #True # y on z and y on x binscatters
coefs_by_xlag = True #True # coefficients of x_cst_lag[num] ~ instrument (no FE), separately by year
_INSTRUMENT_COL = "z_ct_full" 
#_INSTRUMENT_COL = "z_bin_topbot_quartile"
#_TREATMENT_COL = "masters_opt_hires_correction_aware" 
_TREATMENT_COL = "x_bin_any_nonzero"#"masters_opt_hires_correction_aware" #"x_bin_any_nonzero" #"x_bin_any_nonzero" #"masters_opt_hires_correction_aware"
#_TREATMENT_COL = "x_bin_topbot_quartile"
_USE_INSTRUMENT_X_FIRM_SIZE_VENTILE = False 
_ALT_EVENT_STUDY= False
_ALT_EVENT_MATCH_CONTROLS_ON_Y_CST_LAGM3 = False
_USE_LOG_OUTCOME_FOR_REDUCED_FORM = True

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


cfg = load_config(DEFAULT_CONFIG_PATH)
paths_cfg = get_cfg_section(cfg, "paths")
reg_cfg = get_cfg_section(cfg, "shift_share_regressions")
outcome_prefix = str(reg_cfg.get("outcome_prefix", "y_cst_lag"))
include_bachelors_sample = bool(reg_cfg.get("include_bachelors_sample", False))

instrument_col = str(reg_cfg.get("instrument", _INSTRUMENT_COL))
if instrument_col not in {"z_ct", "z_ct_full"}:
    raise ValueError(f"Unsupported instrument column '{instrument_col}'. Use 'z_ct' or 'z_ct_full'.")
use_log_instrument = bool(reg_cfg.get("use_log_instrument", False))
instrument_interact_post2016 = bool(reg_cfg.get("instrument_interact_post2016", False))
instrument_interact_post_year = int(reg_cfg.get("instrument_interact_post_year", 2016))
treatment_col = _TREATMENT_COL
# str(
#     reg_cfg.get("treatment_col", reg_cfg.get("dependent", "masters_opt_hires_correction_aware"))
# )
plot_x_lag_by_year = bool(reg_cfg.get("plot_x_lag_by_year", False))
x_lag_by_year_years_cfg = reg_cfg.get("x_lag_by_year_years")
use_instrument_x_firm_size_ventile = _USE_INSTRUMENT_X_FIRM_SIZE_VENTILE #bool(reg_cfg.get("use_instrument_x_firm_size_ventile", False))
use_state_year_fe = bool(reg_cfg.get("use_state_year_fe", False))
firm_size_ventile_n = int(reg_cfg.get("firm_size_ventile_n", 3))
n_universities_ventile_n = int(reg_cfg.get("n_universities_ventile_n", firm_size_ventile_n))
alt_event_study = bool(reg_cfg.get("alt_event_study", _ALT_EVENT_STUDY))
alt_event_instrument_col = str(reg_cfg.get("alt_event_instrument_col", "z_ct"))
alt_event_treat_pctile = float(reg_cfg.get("alt_event_treat_pctile", 90.0))
alt_event_control_pctile = float(reg_cfg.get("alt_event_control_pctile", 50.0))
alt_event_time_min = int(reg_cfg.get("alt_event_time_min", -3))
alt_event_time_max = int(reg_cfg.get("alt_event_time_max", 3))
alt_event_seed = int(reg_cfg.get("alt_event_seed", 42))
alt_event_plot_years_cfg = reg_cfg.get("alt_event_plot_years", [2012, 2014, 2016])
alt_event_data_min_t = int(reg_cfg.get("alt_event_data_min_t", 2008))
alt_event_data_max_t = int(reg_cfg.get("alt_event_data_max_t", 2022))
alt_event_event_min_t = int(reg_cfg.get("alt_event_event_min_t", 2012))
alt_event_event_max_t = int(reg_cfg.get("alt_event_event_max_t", 2018))
alt_event_match_controls_on_y_cst_lagm3 = _ALT_EVENT_MATCH_CONTROLS_ON_Y_CST_LAGM3
# bool(
#     reg_cfg.get("alt_event_match_controls_on_y_cst_lagm3", False)
# )
y_cst_lagm3_min = reg_cfg.get("y_cst_lagm3_min", None)
y_cst_lagm3_max = reg_cfg.get("y_cst_lagm3_max", None)
use_log_outcome_for_reduced_form = _USE_LOG_OUTCOME_FOR_REDUCED_FORM #bool(reg_cfg.get("use_log_outcome_for_reduced_form", False))

_slides_out_raw = str(reg_cfg.get("slides_out_dir", "")).strip()
if _slides_out_raw:
    from config import root as _root  # noqa: E402
    slides_out = Path(_slides_out_raw.replace("{root}", str(_root)))
else:
    slides_out = None
if slides_out is not None:
    slides_out.mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving figures to {slides_out}")

def _savefig(name: str) -> None:
    """Save current figure to slides_out if configured, then show."""
    if slides_out is not None:
        plt.savefig(slides_out / name, dpi=150, bbox_inches="tight")
        print(f"[info] Saved {name}")

con = ddb.connect()
panel_key = "analysis_panel"
if include_bachelors_sample:
    if str(paths_cfg.get("analysis_panel_ma_ba", "")).strip():
        panel_key = "analysis_panel_ma_ba"
    else:
        print(
            "[warn] include_bachelors_sample=true but paths.analysis_panel_ma_ba is unset; "
            "falling back to paths.analysis_panel."
        )
panel_path = Path(paths_cfg[panel_key])
if not panel_path.is_absolute():
    panel_path = Path.cwd() / panel_path
panel_path_esc = str(panel_path).replace("'", "''")
con.sql(f"CREATE OR REPLACE VIEW analysis_panel AS SELECT * FROM read_parquet('{panel_path_esc}')")
print(f"[info] Using analysis panel source: paths.{panel_key} -> {panel_path}")

con.sql(
    f"""
    CREATE OR REPLACE TABLE regression_panel AS
    WITH base AS (SELECT * FROM analysis_panel WHERE t BETWEEN 2012 AND 2018),
    keep_z AS (SELECT c FROM base GROUP BY c HAVING SUM(CASE WHEN {instrument_col} IS NOT NULL THEN 1 ELSE 0 END) > 0),
    keep_bal AS (SELECT c FROM base WHERE c IN (SELECT c FROM keep_z) GROUP BY c HAVING COUNT(DISTINCT t) = 7),
    base_kept AS (SELECT b.* FROM base b JOIN keep_bal k USING (c))
    SELECT
        base_kept.*,
        COALESCE(MAX(CASE WHEN t = 2012 THEN y_cst_lag0 END) OVER (PARTITION BY c), 0) AS firm_size_2012
    FROM base_kept
    """
)

reg_panel = con.sql("SELECT * FROM regression_panel").df()
if (y_cst_lagm3_min is not None) or (y_cst_lagm3_max is not None):
    if "y_cst_lagm3" not in reg_panel.columns:
        print("[warn] y_cst_lagm3 filter requested but column 'y_cst_lagm3' is missing; skipping filter.")
    else:
        _n0 = len(reg_panel)
        _m = reg_panel["y_cst_lagm3"].notna()
        if y_cst_lagm3_min is not None:
            _m &= reg_panel["y_cst_lagm3"] >= float(y_cst_lagm3_min)
        if y_cst_lagm3_max is not None:
            _m &= reg_panel["y_cst_lagm3"] <= float(y_cst_lagm3_max)
        reg_panel = reg_panel.loc[_m].copy()
        print(
            f"[info] Applied y_cst_lagm3 filter"
            f" (min={y_cst_lagm3_min}, max={y_cst_lagm3_max}): "
            f"{_n0:,} -> {len(reg_panel):,} rows."
        )
_derived_treatment_cols = {"x_bin_any_nonzero", "x_bin_above_year_median", "x_bin_topbot_quartile"}
_is_derived_lag_bin = bool(
    re.fullmatch(r"x_cst_lag(m?\d+)_bin_(any_nonzero|above_year_median|topbot_quartile)", treatment_col)
)
if (treatment_col not in reg_panel.columns) and (treatment_col not in _derived_treatment_cols) and (not _is_derived_lag_bin):
    raise ValueError(
        f"Treatment column '{treatment_col}' not found in regression panel. "
        f"Available example treatment columns include: masters_opt_hires, valid_masters_opt_hires, masters_opt_hires_correction_aware."
    )
if instrument_col not in reg_panel.columns:
    raise ValueError(f"Instrument column '{instrument_col}' not found in regression panel.")

if use_state_year_fe:
    if "company_state" not in reg_panel.columns:
        print(
            "[warn] use_state_year_fe=true but column 'company_state' is missing in analysis panel; "
            "state-year FE will be skipped."
        )
        use_state_year_fe = False
    else:
        _state = reg_panel["company_state"].astype("string").str.strip().str.upper()
        _state = _state.mask(_state.isna() | (_state == ""), "UNK")
        reg_panel["company_state"] = _state
        _t_int = pd.to_numeric(reg_panel["t"], errors="coerce").astype("Int64")
        reg_panel["state_year_fe"] = _state.astype(str) + "_" + _t_int.astype(str)
        print(
            "[info] use_state_year_fe=true: added state_year_fe "
            f"(n_states={reg_panel['company_state'].nunique(dropna=True)})."
        )

if use_log_instrument:
    _base_instrument_col = instrument_col
    _log_instrument_col = f"log_{_base_instrument_col}"
    reg_panel[_log_instrument_col] = np.where(
        reg_panel[_base_instrument_col].notna() & (reg_panel[_base_instrument_col] > 0),
        np.log(reg_panel[_base_instrument_col]),
        np.nan,
    )
    instrument_col = _log_instrument_col
    print(
        f"[info] use_log_instrument=true: using instrument_col='{instrument_col}' "
        f"(from '{_base_instrument_col}', log for values > 0)."
    )
else:
    reg_panel[instrument_col] = reg_panel[instrument_col].fillna(0)

post_period_dummy_col = f"post_t_ge_{instrument_interact_post_year}"
_t_num = pd.to_numeric(reg_panel["t"], errors="coerce")
reg_panel[post_period_dummy_col] = np.where(
    _t_num.notna() & (_t_num >= instrument_interact_post_year),
    1,
    0,
).astype("int8")
instrument_post_interaction_col = f"{instrument_col}_X_post{instrument_interact_post_year}"
reg_panel[instrument_post_interaction_col] = reg_panel[instrument_col] * reg_panel[post_period_dummy_col]
instrument_coef_term = (
    instrument_post_interaction_col if instrument_interact_post2016 else instrument_col
)
instrument_coef_label = (
    f"{instrument_col} x 1[t >= {instrument_interact_post_year}]"
    if instrument_interact_post2016
    else instrument_col
)
if instrument_interact_post2016:
    print(
        "[info] instrument_interact_post2016=true: regressions include "
        f"'{instrument_col}' and '{instrument_post_interaction_col}'. "
        f"Reported/plotted focal term: '{instrument_coef_term}'."
    )

# Binarized treatment/instrument versions for quick robustness checks.
# z: above yearly median; x: any non-zero.
x_source_col = "masters_opt_hires_correction_aware"
if x_source_col not in reg_panel.columns:
    x_source_col = treatment_col

reg_panel["x_bin_any_nonzero"] = (reg_panel[x_source_col].fillna(0) != 0).astype("int8")
x_year_median = reg_panel.groupby("t")[x_source_col].transform("median")
reg_panel["x_bin_above_year_median"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
x_mask = reg_panel[x_source_col].notna() & x_year_median.notna()
if x_mask.any():
    x_valid_idx = reg_panel.index[x_mask]
    x_cmp = reg_panel.loc[x_valid_idx, x_source_col] > x_year_median.loc[x_valid_idx]
    reg_panel.loc[x_valid_idx, "x_bin_above_year_median"] = x_cmp.astype("int8")

x_year_p25 = reg_panel.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.25))
x_year_p75 = reg_panel.groupby("t")[x_source_col].transform(lambda s: s.quantile(0.75))
reg_panel["x_bin_topbot_quartile"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
x_q_mask = reg_panel[x_source_col].notna() & x_year_p25.notna() & x_year_p75.notna()
if x_q_mask.any():
    x_valid_q_idx = reg_panel.index[x_q_mask]
    x_valid = reg_panel.loc[x_valid_q_idx, x_source_col]
    x_p25_valid = x_year_p25.loc[x_valid_q_idx]
    x_p75_valid = x_year_p75.loc[x_valid_q_idx]
    reg_panel.loc[x_valid_q_idx[x_valid <= x_p25_valid], "x_bin_topbot_quartile"] = 0
    reg_panel.loc[x_valid_q_idx[x_valid >= x_p75_valid], "x_bin_topbot_quartile"] = 1

z_year_median = reg_panel.groupby("t")[instrument_col].transform("median")
reg_panel["z_bin_above_year_median"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
z_mask = reg_panel[instrument_col].notna() & z_year_median.notna()
reg_panel.loc[z_mask, "z_bin_above_year_median"] = (
    reg_panel.loc[z_mask, instrument_col] > z_year_median[z_mask]
).astype("int8")
z_year_p25 = reg_panel.groupby("t")[instrument_col].transform(lambda s: s.quantile(0.25))
z_year_p75 = reg_panel.groupby("t")[instrument_col].transform(lambda s: s.quantile(0.75))
reg_panel["z_bin_topbot_quartile"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
z_q_mask = reg_panel[instrument_col].notna() & z_year_p25.notna() & z_year_p75.notna()
reg_panel.loc[z_q_mask & (reg_panel[instrument_col] <= z_year_p25), "z_bin_topbot_quartile"] = 0
reg_panel.loc[z_q_mask & (reg_panel[instrument_col] >= z_year_p75), "z_bin_topbot_quartile"] = 1

# Build lagged binary x variants for every x_cst_lag* column.
_base_x_lag_cols = [c for c in reg_panel.columns if re.fullmatch(r"x_cst_lag(m?\d+)", c)]
for _xcol in _base_x_lag_cols:
    reg_panel[f"{_xcol}_bin_any_nonzero"] = (reg_panel[_xcol].fillna(0) != 0).astype("int8")

    _x_year_median = reg_panel.groupby("t")[_xcol].transform("median")
    reg_panel[f"{_xcol}_bin_above_year_median"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
    _x_mask = reg_panel[_xcol].notna() & _x_year_median.notna()
    if _x_mask.any():
        _valid_idx = reg_panel.index[_x_mask]
        _cmp = reg_panel.loc[_valid_idx, _xcol] > _x_year_median.loc[_valid_idx]
        reg_panel.loc[_valid_idx, f"{_xcol}_bin_above_year_median"] = _cmp.astype("int8")

    _x_year_p25 = reg_panel.groupby("t")[_xcol].transform(lambda s: s.quantile(0.25))
    _x_year_p75 = reg_panel.groupby("t")[_xcol].transform(lambda s: s.quantile(0.75))
    reg_panel[f"{_xcol}_bin_topbot_quartile"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int8")
    _x_q_mask = reg_panel[_xcol].notna() & _x_year_p25.notna() & _x_year_p75.notna()
    if _x_q_mask.any():
        _valid_q_idx = reg_panel.index[_x_q_mask]
        _x_valid = reg_panel.loc[_valid_q_idx, _xcol]
        _p25_valid = _x_year_p25.loc[_valid_q_idx]
        _p75_valid = _x_year_p75.loc[_valid_q_idx]
        reg_panel.loc[_valid_q_idx[_x_valid <= _p25_valid], f"{_xcol}_bin_topbot_quartile"] = 0
        reg_panel.loc[_valid_q_idx[_x_valid >= _p75_valid], f"{_xcol}_bin_topbot_quartile"] = 1

reduced_form_outcome_col = f"{outcome_prefix}0"


def _resolve_lagm1_outcome_col(df: pd.DataFrame, prefix: str) -> str | None:
    preferred = f"{prefix}m1"
    if preferred in df.columns:
        return preferred
    if "y_cst_lagm1" in df.columns:
        return "y_cst_lagm1"
    lagm1_candidates = sorted(col for col in df.columns if col.endswith("lagm1"))
    return lagm1_candidates[0] if lagm1_candidates else None


lagm1_outcome_col = _resolve_lagm1_outcome_col(reg_panel, outcome_prefix)
if lagm1_outcome_col is None:
    print(
        "[warn] No lag-1 outcome column found (expected "
        f"'{outcome_prefix}m1'). Lagm1 interaction/ventile steps will be skipped."
    )
elif lagm1_outcome_col != f"{outcome_prefix}m1":
    print(
        f"[info] Using fallback lag-1 outcome column '{lagm1_outcome_col}' "
        f"(expected '{outcome_prefix}m1')."
    )

_alt_event_outcome_cfg = reg_cfg.get("alt_event_outcome_col", None)
if _alt_event_outcome_cfg is None or str(_alt_event_outcome_cfg).strip().lower() in {"", "none", "null"}:
    alt_event_outcome_col = treatment_col
else:
    alt_event_outcome_col = str(_alt_event_outcome_cfg)

if use_instrument_x_firm_size_ventile:
    if "y_cst_lagm3" not in reg_panel.columns:
        raise ValueError(
            "Toggle use_instrument_x_firm_size_ventile=true requires column 'y_cst_lagm3' in reg_panel."
        )
    if firm_size_ventile_n < 2:
        raise ValueError("firm_size_ventile_n must be >= 2.")
    reg_panel["firm_size_ventile"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int64")
    _ventile_mask = reg_panel["y_cst_lagm3"].notna()
    if _ventile_mask.any():
        reg_panel.loc[_ventile_mask, "firm_size_ventile"] = (
            pd.qcut(
                reg_panel.loc[_ventile_mask, "y_cst_lagm3"].rank(method="first"),
                firm_size_ventile_n,
                labels=False,
            )
            + 1
        ).astype("Int64")
    print(
        "[info] use_instrument_x_firm_size_ventile=true: built firm_size_ventile from y_cst_lagm3 "
        f"(n={firm_size_ventile_n}); regressions stay non-interacted."
    )

    if "n_universities" in reg_panel.columns and reg_panel["n_universities"].notna().any():
        if n_universities_ventile_n < 2:
            raise ValueError("n_universities_ventile_n must be >= 2.")
        reg_panel["n_universities_ventile"] = pd.Series(pd.NA, index=reg_panel.index, dtype="Int64")
        _nu_mask = reg_panel["n_universities"].notna()
        reg_panel.loc[_nu_mask, "n_universities_ventile"] = (
            pd.qcut(
                reg_panel.loc[_nu_mask, "n_universities"].rank(method="first"),
                n_universities_ventile_n,
                labels=False,
                duplicates="drop",
            )
            + 1
        ).astype("Int64")
        print(
            "[info] use_instrument_x_firm_size_ventile=true: built n_universities_ventile from n_universities "
            f"(n={n_universities_ventile_n})."
        )
    else:
        print(
            "[warn] use_instrument_x_firm_size_ventile=true but n_universities is missing/empty; "
            "skipping n_universities_ventile^t FE augmentation."
        )


def _instrument_rhs(has_company_fe: bool) -> str:
    if instrument_interact_post2016:
        return f"{instrument_col} + {instrument_post_interaction_col}"
    return instrument_col


def _fe_terms(fe: str) -> set[str]:
    if not fe:
        return set()
    return {x.strip() for x in fe.split("+") if x.strip()}


def _augment_fe(fe: str) -> str:
    terms = _fe_terms(fe)
    fe_out = fe
    if use_state_year_fe and "t" in terms and "state_year_fe" in reg_panel.columns and "state_year_fe" not in terms:
        fe_out = f"{fe_out} + state_year_fe" if fe_out else "state_year_fe"
        terms.add("state_year_fe")
    if use_instrument_x_firm_size_ventile and {"c", "t"}.issubset(terms):
        if "firm_size_ventile" in reg_panel.columns and "firm_size_ventile^t" not in terms:
            fe_out = f"{fe_out} + firm_size_ventile^t"
            terms.add("firm_size_ventile^t")
        if "n_universities_ventile" in reg_panel.columns and "n_universities_ventile^t" not in terms:
            fe_out = f"{fe_out} + n_universities_ventile^t"
    return fe_out


def _parse_int_list(value: object) -> list[int]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(x) for x in value]


def _extract_est_se(
    tidy: pd.DataFrame,
    term: str,
    *,
    context: str,
    warn_on_missing: bool = True,
) -> tuple[float, float]:
    if term not in tidy.index:
        if warn_on_missing:
            print(f"[warn] term '{term}' missing in {context}; returning NaN.")
        return float("nan"), float("nan")
    return float(tidy.loc[term, "Estimate"]), float(tidy.loc[term, "Std. Error"])


def _first_stage_by_year_profile(
    sample: pd.DataFrame,
    rhs_col: str,
    *,
    context: str,
) -> pd.DataFrame:
    year_levels = sorted(int(y) for y in sample["t"].dropna().unique())
    if not year_levels:
        return pd.DataFrame(columns=["year", "coef", "se", "lo", "hi"])
    year_sample = sample.copy()
    year_inter_cols = []
    for yr in year_levels:
        col = f"{rhs_col}_X_year_{yr}"
        year_sample[col] = year_sample[rhs_col] * (year_sample["t"] == yr).astype("int8")
        year_inter_cols.append(col)
    try:
        fit_year = pf.feols(
            f"{treatment_col} ~ {' + '.join(year_inter_cols)} + 1",
            data=year_sample,
            vcov={"CRV1": "c"},
            demeaner_backend="rust",
        )
        tidy_year = fit_year.tidy()
    except ValueError as e:
        if "All variables are collinear" in str(e):
            print(f"[warn] Collinear year-profile model in {context}; returning NaNs.")
            tidy_year = pd.DataFrame()
        else:
            raise

    year_rows = []
    for yr, col in zip(year_levels, year_inter_cols):
        est, se = _extract_est_se(tidy_year, col, context=context, warn_on_missing=False)
        year_rows.append(
            {
                "year": int(yr),
                "coef": est,
                "se": se,
                "lo": est - 1.96 * se if pd.notna(se) else float("nan"),
                "hi": est + 1.96 * se if pd.notna(se) else float("nan"),
            }
        )
    return pd.DataFrame(year_rows).sort_values("year")


def _ensure_derived_outcome_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        return

    # Common derived outcomes used elsewhere in this script.
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

    base_x = x_source_col if x_source_col in df.columns else None
    if base_x is None and treatment_col in df.columns:
        base_x = treatment_col

    if col == "x_bin_any_nonzero" and base_x is not None:
        df[col] = (df[base_x].fillna(0) != 0).astype("int8")
        return

    if col == "x_bin_above_year_median" and base_x is not None:
        x_year_median_df = df.groupby("t")[base_x].transform("median")
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        x_mask_df = df[base_x].notna() & x_year_median_df.notna()
        if x_mask_df.any():
            valid_idx = df.index[x_mask_df]
            cmp = df.loc[valid_idx, base_x] > x_year_median_df.loc[valid_idx]
            df.loc[valid_idx, col] = cmp.astype("int8")
        return

    if col == "x_bin_topbot_quartile" and base_x is not None:
        x_year_p25_df = df.groupby("t")[base_x].transform(lambda s: s.quantile(0.25))
        x_year_p75_df = df.groupby("t")[base_x].transform(lambda s: s.quantile(0.75))
        df[col] = pd.Series(pd.NA, index=df.index, dtype="Int8")
        x_q_mask_df = df[base_x].notna() & x_year_p25_df.notna() & x_year_p75_df.notna()
        if x_q_mask_df.any():
            valid_q_idx = df.index[x_q_mask_df]
            x_valid_df = df.loc[valid_q_idx, base_x]
            p25_valid_df = x_year_p25_df.loc[valid_q_idx]
            p75_valid_df = x_year_p75_df.loc[valid_q_idx]
            df.loc[valid_q_idx[x_valid_df <= p25_valid_df], col] = 0
            df.loc[valid_q_idx[x_valid_df >= p75_valid_df], col] = 1
        return

    # Lagged binary-x outcomes: x_cst_lag* variants.
    m = re.fullmatch(r"(x_cst_lagm?\d+)_bin_any_nonzero", col)
    if m and m.group(1) in df.columns:
        df[col] = (df[m.group(1)].fillna(0) != 0).astype("int8")
        return


_base_reduced_form_outcome_col = f"{outcome_prefix}0"
if _base_reduced_form_outcome_col not in reg_panel.columns:
    raise ValueError(
        f"Reduced-form outcome column '{_base_reduced_form_outcome_col}' not found. "
        f"Set shift_share_regressions.outcome_prefix to match an existing '*0' column."
    )

if use_log_outcome_for_reduced_form:
    reduced_form_outcome_col = f"log1p_{_base_reduced_form_outcome_col}"
    reg_panel[reduced_form_outcome_col] = np.where(
        reg_panel[_base_reduced_form_outcome_col].notna() & (reg_panel[_base_reduced_form_outcome_col] >= 0),
        np.log1p(reg_panel[_base_reduced_form_outcome_col]),
        np.nan,
    )
else:
    reduced_form_outcome_col = _base_reduced_form_outcome_col

## REGRESSION STUFF
if regs:
    reg_panel['z_X_firmsize'] = reg_panel[instrument_col] * reg_panel['firm_size_2012']
    if lagm1_outcome_col is not None:
        reg_panel["z_X_y_lagm1"] = reg_panel[instrument_col] * reg_panel[lagm1_outcome_col]
    else:
        reg_panel["z_X_y_lagm1"] = np.nan

    def _fml(lhs: str, rhs: str, fe: str) -> str:
        return f"{lhs} ~ {rhs}" if not fe else f"{lhs} ~ {rhs} | {fe}"


    def _has_company_fe(fe: str) -> bool:
        return "c" in _fe_terms(fe)


    fe_specs = [("no_fes", ""), ("year_company_fes", "c + t")]
    #fe_specs = [("year_company_fes", "c + t")]
    reg_specs = [
        ("first_stage", treatment_col, "__INSTRUMENT_RHS__"),
    #    ("ols", "y_cst_lag0", treatment_col),
        ("reduced_form", reduced_form_outcome_col, "__INSTRUMENT_RHS__"),
        #("reduced_form2", "y_cst_lag3", instrument_col)
    ]
    control_specs = [("baseline", "")]#, ("with_firmsize_interact", " + firm_size_2012 + z_X_firmsize"), ("with_y_lagm1_interact", " + <lagm1_outcome_col> + z_X_y_lagm1")]

    models = []
    model_heads = []
    for reg_name, lhs, rhs in reg_specs:
        for ctrl_name, ctrl_rhs in control_specs:
            for fe_name, fe in fe_specs:
                rhs_core = _instrument_rhs(_has_company_fe(fe)) if rhs == "__INSTRUMENT_RHS__" else rhs
                rhs_use = rhs_core + ctrl_rhs
                if reg_name == "first_stage" and _has_company_fe(fe):
                    rhs_use = rhs_use.replace(" + firm_size_2012", "")
                fe_use = _augment_fe(fe)
                fit = pf.feols(
                    _fml(lhs, rhs_use, fe_use),
                    data=reg_panel,
                    vcov={"CRV1": "c"},
                    demeaner_backend="rust",
                )
                models.append(fit)
                model_heads.append(f"{reg_name} | {ctrl_name} | {fe_name}")
                tidy = fit.tidy()
                print(f"\n[{reg_name} | {ctrl_name} | {fe_name}]")
                #print(tidy)
                report_terms = [instrument_coef_term, "z_X_firmsize", "z_X_y_lagm1"]
                if not instrument_interact_post2016:
                    report_terms = ["z_ct", "z_ct_full", instrument_col, "z_X_firmsize", "z_X_y_lagm1"]
                for term in dict.fromkeys(report_terms):
                    if term in tidy.index:
                        est = float(tidy.loc[term, "Estimate"])
                        se = float(tidy.loc[term, "Std. Error"])
                        t = float("nan") if (not pd.notna(se) or abs(se) < 1e-12) else est / se
                        print(f"Coef ({term}) = {est:.5f}")
                        if pd.notna(t):
                            print(f"F-stat ({term}) = {t*t:.3f}")
                        else:
                            print(f"F-stat ({term}) = NA (Std. Error is zero or missing)")

    print("\nModel table (pyfixest etable):")
    # round coefs to 5 decimals, SEs to 3 decimals in etable output


    reg_out=     pf.etable(
            models,
            type="df",
            model_heads=model_heads,
            show_fe=True,
            show_se_type=True,
            coef_fmt="b (se)"
    )
    print(reg_out.to_string())
    if slides_out is not None:
        reg_out.to_csv(slides_out / "reg_table.csv")
        print(f"[info] Saved reg_table.csv")

###### COEF PLOTS: FIRST STAGE BY FIRM SIZE AND YEAR #######
if coef_plots_bycat:
    # Pooled (all-years) benchmark coefficients for subgroup plots.
    fit_first_stage_pooled_ct = pf.feols(
        f"{treatment_col} ~ {_instrument_rhs(True)} | {_augment_fe('c + t')}",
        data=reg_panel,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    first_stage_pooled_coef_ct, _ = _extract_est_se(
        fit_first_stage_pooled_ct.tidy(),
        instrument_coef_term,
        context="first-stage pooled c+t FE",
    )

    fit_first_stage_pooled_nofe = pf.feols(
        f"{treatment_col} ~ {_instrument_rhs(False)} + 1",
        data=reg_panel,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    first_stage_pooled_coef_nofe, _ = _extract_est_se(
        fit_first_stage_pooled_nofe.tidy(),
        instrument_coef_term,
        context="first-stage pooled no FE",
    )

    fit_reduced_form_pooled = pf.feols(
        f"{reduced_form_outcome_col} ~ {_instrument_rhs(True)} | {_augment_fe('c + t')}",
        data=reg_panel,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    reduced_form_pooled_coef, _ = _extract_est_se(
        fit_reduced_form_pooled.tidy(),
        instrument_coef_term,
        context="reduced-form pooled c+t FE",
    )

    # Build firm-size categories for interaction plots.
    if lagm1_outcome_col is None:
        print("\n[warn] Skipping size-ventile plots: no lag-1 outcome column available.")
    else:
        lagm1_plot_col = "lagm1_value"
        firm_groups = (
            reg_panel[["c", "t", lagm1_outcome_col]]
            .rename(columns={lagm1_outcome_col: lagm1_plot_col})
            .dropna(subset=[lagm1_plot_col])
            .copy()
        )
        if firm_groups.empty:
            print(f"\n[warn] Skipping size-ventile plots: '{lagm1_outcome_col}' has no non-missing values.")
        else:
            firm_groups = firm_groups.sort_values(["c", "t"])
            firm_lagm1_2012 = firm_groups.loc[firm_groups["t"] == 2012, ["c", lagm1_plot_col]].drop_duplicates("c")
            firm_lagm1_fallback = firm_groups.groupby("c", as_index=False).first()[["c", lagm1_plot_col]]
            firm_groups = firm_lagm1_fallback.merge(
                firm_lagm1_2012.rename(columns={lagm1_plot_col: f"{lagm1_plot_col}_2012"}),
                on="c",
                how="left",
            )
            firm_groups["lagm1_group"] = firm_groups[f"{lagm1_plot_col}_2012"].fillna(firm_groups[lagm1_plot_col])
            firm_groups["size_ventile"] = pd.qcut(
                firm_groups["lagm1_group"].rank(method="first"),
                firm_size_ventile_n,
                labels=False,
            ) + 1
            reg_panel = reg_panel.drop(columns=["size_ventile"], errors="ignore")
            reg_panel = reg_panel.merge(firm_groups[["c", "size_ventile"]], on="c", how="left")

            # Interacted model by firm-size category: y ~ sum_d instrument * 1[size_ventile=d]
            size_sample = reg_panel.loc[reg_panel["size_ventile"].notna()].copy()
            if size_sample.empty:
                print("\n[warn] Skipping size-ventile regressions: no rows with non-missing size_ventile.")
            else:
                size_inter_cols = []
                for d in range(1, firm_size_ventile_n + 1):
                    col = f"{instrument_col}_X_size_ventile_{d}"
                    size_sample[col] = size_sample[instrument_col] * (size_sample["size_ventile"] == d).astype("int8")
                    size_inter_cols.append(col)

                fit_size_fs = pf.feols(
                    f"{treatment_col} ~ {' + '.join(size_inter_cols)} | {_augment_fe('c + t')}",
                    data=size_sample,
                    vcov={"CRV1": "c"},
                    demeaner_backend="rust",
                )
                tidy_size_fs = fit_size_fs.tidy()
                ventile_rows = []
                for d, col in enumerate(size_inter_cols, start=1):
                    if col in tidy_size_fs.index:
                        est = float(tidy_size_fs.loc[col, "Estimate"])
                        se = float(tidy_size_fs.loc[col, "Std. Error"])
                        ventile_rows.append({"ventile": d, "coef": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se})
                    else:
                        ventile_rows.append({"ventile": d, "coef": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")})
                ventile_res = pd.DataFrame(ventile_rows).sort_values("ventile")

                fit_size_rf = pf.feols(
                    f"{reduced_form_outcome_col} ~ {' + '.join(size_inter_cols)} | {_augment_fe('c + t')}",
                    data=size_sample,
                    vcov={"CRV1": "c"},
                    demeaner_backend="rust",
                )
                tidy_size_rf = fit_size_rf.tidy()
                rf_ventile_rows = []
                for d, col in enumerate(size_inter_cols, start=1):
                    if col in tidy_size_rf.index:
                        est_rf = float(tidy_size_rf.loc[col, "Estimate"])
                        se_rf = float(tidy_size_rf.loc[col, "Std. Error"])
                        rf_ventile_rows.append(
                            {"ventile": d, "coef": est_rf, "se": se_rf, "lo": est_rf - 1.96 * se_rf, "hi": est_rf + 1.96 * se_rf}
                        )
                    else:
                        rf_ventile_rows.append({"ventile": d, "coef": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")})
                rf_ventile_res = pd.DataFrame(rf_ventile_rows).sort_values("ventile")
                ventile_bins = (
                    firm_groups.groupby("size_ventile", as_index=False)
                    .agg(
                        n_firms=("c", "nunique"),
                        lagm1_min=("lagm1_group", "min"),
                        lagm1_max=("lagm1_group", "max"),
                    )
                    .sort_values("size_ventile")
                )
                print(f"\n[{lagm1_outcome_col} ventile bins]")
                print(ventile_bins)
                print("\n[first_stage_by_size_ventile | c+t_fes]")
                print(ventile_res)
                print("\n[reduced_form_by_size_ventile | c+t_fes]")
                print(rf_ventile_res)

                plt.figure(figsize=(8, 4.5))
                plt.errorbar(
                    ventile_res["ventile"],
                    ventile_res["coef"],
                    yerr=1.96 * ventile_res["se"],
                    fmt="o",
                    capsize=4,
                )
                plt.axhline(0, color="black", linewidth=1)
                plt.axhline(
                    first_stage_pooled_coef_ct,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Pooled (all years): {first_stage_pooled_coef_ct:.4f}",
                )
                plt.xticks(range(1, firm_size_ventile_n + 1))
                plt.xlabel(f"Lagged firm size ventile ({lagm1_outcome_col})")
                plt.ylabel(f"Coef by Firm Size: effect of {instrument_coef_label} on # OPT hires")
                plt.legend()
                plt.tight_layout()
                _savefig("fs_by_firmsize.png")
                plt.show()

                plt.figure(figsize=(8, 4.5))
                plt.errorbar(
                    rf_ventile_res["ventile"],
                    rf_ventile_res["coef"],
                    yerr=1.96 * rf_ventile_res["se"],
                    fmt="o",
                    capsize=4,
                )
                plt.axhline(0, color="black", linewidth=1)
                plt.axhline(
                    reduced_form_pooled_coef,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Pooled (all years): {reduced_form_pooled_coef:.4f}",
                )
                plt.xticks(range(1, firm_size_ventile_n + 1))
                plt.xlabel(f"Firm ventile ({lagm1_outcome_col})")
                plt.ylabel(f"Coefficient on {instrument_coef_label}")
                plt.title(f"Reduced form by {lagm1_outcome_col} ventile")
                plt.legend()
                plt.tight_layout()
                _savefig("rf_by_firmsize.png")
                plt.show()

    # Interacted model by instrument ventile: y ~ sum_d instrument * 1[instrument_ventile=d]
    inst_sample = reg_panel.loc[reg_panel[instrument_col].notna()].copy()
    inst_sample["instrument_ventile"] = (
        pd.qcut(
            inst_sample[instrument_col].rank(method="first"),
            firm_size_ventile_n,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    inst_levels = sorted(int(v) for v in inst_sample["instrument_ventile"].dropna().unique())
    inst_inter_cols = []
    for d in inst_levels:
        col = f"{instrument_col}_X_instrument_ventile_{d}"
        inst_sample[col] = inst_sample[instrument_col] * (inst_sample["instrument_ventile"] == d).astype("int8")
        inst_inter_cols.append(col)

    fit_inst_fs = pf.feols(
        f"{treatment_col} ~ {' + '.join(inst_inter_cols)} | {_augment_fe('c + t')}",
        data=inst_sample,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    tidy_inst_fs = fit_inst_fs.tidy()
    inst_fs_rows = []
    for d, col in zip(inst_levels, inst_inter_cols):
        if col in tidy_inst_fs.index:
            est = float(tidy_inst_fs.loc[col, "Estimate"])
            se = float(tidy_inst_fs.loc[col, "Std. Error"])
            inst_fs_rows.append({"ventile": d, "coef": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se})
        else:
            inst_fs_rows.append({"ventile": d, "coef": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")})
    inst_fs_res = pd.DataFrame(inst_fs_rows).sort_values("ventile")

    fit_inst_rf = pf.feols(
        f"{reduced_form_outcome_col} ~ {' + '.join(inst_inter_cols)} | {_augment_fe('c + t')}",
        data=inst_sample,
        vcov={"CRV1": "c"},
        demeaner_backend="rust",
    )
    tidy_inst_rf = fit_inst_rf.tidy()
    inst_rf_rows = []
    for d, col in zip(inst_levels, inst_inter_cols):
        if col in tidy_inst_rf.index:
            est = float(tidy_inst_rf.loc[col, "Estimate"])
            se = float(tidy_inst_rf.loc[col, "Std. Error"])
            inst_rf_rows.append({"ventile": d, "coef": est, "se": se, "lo": est - 1.96 * se, "hi": est + 1.96 * se})
        else:
            inst_rf_rows.append({"ventile": d, "coef": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")})
    inst_rf_res = pd.DataFrame(inst_rf_rows).sort_values("ventile")

    print("\n[first_stage_by_instrument_ventile | c+t_fes]")
    print(inst_fs_res)
    print("\n[reduced_form_by_instrument_ventile | c+t_fes]")
    print(inst_rf_res)

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        inst_fs_res["ventile"],
        inst_fs_res["coef"],
        yerr=1.96 * inst_fs_res["se"],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        first_stage_pooled_coef_ct,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Pooled (all years): {first_stage_pooled_coef_ct:.4f}",
    )
    plt.xticks(inst_levels)
    plt.xlabel(f"Instrument ventile ({instrument_col})")
    plt.ylabel(f"First-stage coefficient on {instrument_coef_label}")
    plt.legend()
    plt.tight_layout()
    _savefig("fs_by_instrument_ventile.png")
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        inst_rf_res["ventile"],
        inst_rf_res["coef"],
        yerr=1.96 * inst_rf_res["se"],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        reduced_form_pooled_coef,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Pooled (all years): {reduced_form_pooled_coef:.4f}",
    )
    plt.xticks(inst_levels)
    plt.xlabel(f"Instrument ventile ({instrument_col})")
    plt.ylabel(f"Reduced-form coefficient on {instrument_coef_label}")
    plt.legend()
    plt.tight_layout()
    _savefig("rf_by_instrument_ventile.png")
    plt.show()

    # Interacted model by year: y ~ sum_t instrument * 1[t=year]
    year_res = _first_stage_by_year_profile(
        reg_panel,
        instrument_coef_term,
        context="first-stage by year",
    )
    print("\n[first_stage_by_year | focal term]")
    print(year_res)

    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        year_res["year"],
        year_res["coef"],
        yerr=1.96 * year_res["se"],
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(
        first_stage_pooled_coef_nofe,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label=f"Pooled (all years): {first_stage_pooled_coef_nofe:.4f}",
    )
    plt.xticks(year_res["year"])
    plt.xlabel("Year")
    plt.ylabel(f"Coef by Year: effect of {instrument_coef_label} on # OPT hires")
    #plt.title("Effect of instrument on # OPT Hires by year (no FEs)")
    plt.legend()
    plt.tight_layout()
    _savefig("fs_by_year.png")
    plt.show()

    if instrument_interact_post2016:
        split_specs = [
            (0, f"t < {instrument_interact_post_year}"),
            (1, f"t >= {instrument_interact_post_year}"),
        ]
        split_profiles: list[tuple[str, pd.DataFrame]] = []
        for post_val, split_label in split_specs:
            split_sample = reg_panel.loc[reg_panel[post_period_dummy_col] == post_val].copy()
            if split_sample.empty:
                print(f"\n[warn] No rows in split '{split_label}'; skipping non-interacted year profile.")
                continue
            split_res = _first_stage_by_year_profile(
                split_sample,
                instrument_col,
                context=f"first-stage by year (non-interacted, {split_label})",
            )
            if split_res.empty:
                print(f"\n[warn] Empty non-interacted year profile for split '{split_label}'.")
                continue
            print(f"\n[first_stage_by_year_noninteracted | {split_label}]")
            print(split_res)
            split_profiles.append((split_label, split_res))

        if split_profiles:
            fig, axes = plt.subplots(
                len(split_profiles),
                1,
                figsize=(8, 4.2 * len(split_profiles)),
                sharex=False,
            )
            if len(split_profiles) == 1:
                axes = [axes]
            for ax, (split_label, split_res) in zip(axes, split_profiles):
                ax.errorbar(
                    split_res["year"],
                    split_res["coef"],
                    yerr=1.96 * split_res["se"],
                    fmt="o",
                    capsize=4,
                )
                ax.axhline(0, color="black", linewidth=1)
                ax.set_xticks(split_res["year"])
                ax.set_ylabel(f"Coef on {instrument_col}")
                ax.set_title(f"Non-interacted by year ({split_label})")
            axes[-1].set_xlabel("Year")
            plt.tight_layout()
            _savefig("fs_by_year_split.png")
            plt.show()

##### INSTRUMENT Z_CT HISTOGRAM AND AVERAGES BY YEAR/FIRM SIZE #####
# produces: zct_hist, zct_by_year, zct_by_firmsize
if zct_hists:
    # Instrument diagnostics: histogram and averages by lagged outcome decile/year.
    instr_hist = reg_panel[instrument_col].dropna()
    plt.figure(figsize=(8, 4.5))
    sns.histplot(instr_hist, bins=50)
    plt.xlabel("'OPT Hiring pool' size (z_ct)")#instrument_col)
    plt.ylabel("Count")
    #plt.title(f"Histogram of {instrument_col}")
    plt.tight_layout()
    _savefig("zct_hist.png")
    plt.show()

    def _build_bins_by_log_size(value_col: str, out_col: str, n_bins: int = 25) -> pd.DataFrame:
        if "y_cst_lagm3" not in reg_panel.columns:
            print("\n[warn] Column 'y_cst_lagm3' not found; skipping binscatter plots.")
            return pd.DataFrame()
        if value_col not in reg_panel.columns:
            print(f"\n[warn] Column '{value_col}' not found; skipping binscatter plot.")
            return pd.DataFrame()
        cols = [value_col, "y_cst_lagm3"]
        has_nu = "n_universities" in reg_panel.columns
        if has_nu:
            cols.append("n_universities")
        tmp = reg_panel[cols].dropna().copy()
        tmp = tmp.loc[tmp["y_cst_lagm3"] >= 0].copy()
        if tmp.empty:
            return pd.DataFrame()
        tmp["log_y_cst_lagm3"] = np.log1p(tmp["y_cst_lagm3"])
        tmp["lagm3_bin"] = (
            pd.qcut(
                tmp["log_y_cst_lagm3"].rank(method="first"),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            + 1
        )
        agg_dict = {
            "lagged_firm_size_bin": ("log_y_cst_lagm3", "mean"),
            out_col: (value_col, "mean"),
        }
        if has_nu:
            agg_dict["mean_n_universities"] = ("n_universities", "mean")
        return tmp.groupby("lagm3_bin", as_index=False).agg(**agg_dict).sort_values("lagged_firm_size_bin")

    instr_bins_by_lagm3 = _build_bins_by_log_size(instrument_col, "avg_instrument", n_bins=25)

    treatment_bins_by_lagm3 = _build_bins_by_log_size(treatment_col, "avg_treatment", n_bins=25)

    log_y0_col = "log1p_y_cst_lag0"
    reg_panel[log_y0_col] = np.where(
        reg_panel["y_cst_lag0"].notna() & (reg_panel["y_cst_lag0"] >= 0),
        np.log1p(reg_panel["y_cst_lag0"]),
        np.nan,
    )
    y0_bins_by_lagm3 = _build_bins_by_log_size(log_y0_col, "avg_y0", n_bins=25)

    avg_instr_by_year = (
        reg_panel[[instrument_col, "t"]]
        .dropna()
        .groupby("t", as_index=False)
        .agg(avg_instrument=(instrument_col, "mean"))
        .sort_values("t")
    )

    if not instr_bins_by_lagm3.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in instr_bins_by_lagm3.columns:
            sc = plt.scatter(
                instr_bins_by_lagm3["lagged_firm_size_bin"],
                instr_bins_by_lagm3["avg_instrument"],
                c=instr_bins_by_lagm3["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean n_universities in bin")
        else:
            plt.scatter(
                instr_bins_by_lagm3["lagged_firm_size_bin"],
                instr_bins_by_lagm3["avg_instrument"],
                s=30,
                alpha=0.9,
            )
        plt.xlabel("Three-Year Lagged log(firm size)")
        plt.ylabel(f"Average 'OPT Hiring pool' size (z_ct)")#instrument_col)
        #plt.title(f"Binscatter (25 bins): Average 'OPT Hiring pool' size (z_ct) by lagged firm size")
        plt.tight_layout()
        _savefig("zct_by_logfirmsize.png")
        plt.show()
    else:
        print("\n[warn] No binscatter data available for y_cst_lagm3 plot.")

    if not treatment_bins_by_lagm3.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in treatment_bins_by_lagm3.columns:
            sc = plt.scatter(
                treatment_bins_by_lagm3["lagged_firm_size_bin"],
                treatment_bins_by_lagm3["avg_treatment"],
                c=treatment_bins_by_lagm3["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean n_universities in bin")
        else:
            plt.scatter(
                treatment_bins_by_lagm3["lagged_firm_size_bin"],
                treatment_bins_by_lagm3["avg_treatment"],
                s=30,
                alpha=0.9,
            )
        plt.xlabel("Three-Year Lagged log(firm size)")
        plt.ylabel(f"Average # OPT Hires at t = 0 (opthires_ct)")#{treatment_col}")
        plt.tight_layout()
        _savefig("xct_by_logfirmsize.png")
        plt.show()
    else:
        print(f"\n[warn] No binscatter data available for {treatment_col} plot.")

    if not y0_bins_by_lagm3.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in y0_bins_by_lagm3.columns:
            sc = plt.scatter(
                y0_bins_by_lagm3["lagged_firm_size_bin"],
                y0_bins_by_lagm3["avg_y0"],
                c=y0_bins_by_lagm3["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean n_universities in bin")
        else:
            plt.scatter(
                y0_bins_by_lagm3["lagged_firm_size_bin"],
                y0_bins_by_lagm3["avg_y0"],
                s=30,
                alpha=0.9,
            )
        plt.xlabel("log(firm size) at t = -3")
        plt.ylabel("Average log(firm size) at t = 0")
        plt.tight_layout()
        _savefig("yct_by_logfirmsize.png")
        plt.show()
    else:
        print("\n[warn] No binscatter data available for y_cst_lag0 plot.")

    if not avg_instr_by_year.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(avg_instr_by_year["t"], avg_instr_by_year["avg_instrument"], marker="o", linewidth=1.5)
        plt.xticks(avg_instr_by_year["t"])
        plt.xlabel("Year")
        plt.ylabel(f"Average 'OPT Hiring pool' size (z_ct)")#instrument_col)
        #plt.title(f"Average 'OPT Hiring pool' size (z_ct) by year")
        plt.tight_layout()
        _savefig("zct_by_year.png")
        plt.show()
    else:
        print("\n[warn] No year data available for year-average plot.")

# binscatters for regressions
if binscatters_for_regs:
    xy_rf_outcome_col = None
    if lagm1_outcome_col is not None:
        if use_log_outcome_for_reduced_form:
            xy_rf_outcome_col = f"log1p_{lagm1_outcome_col}"
            _ensure_derived_outcome_col(reg_panel, xy_rf_outcome_col)
        else:
            xy_rf_outcome_col = lagm1_outcome_col
    _xy_rf_label = (
        f"Mean log(1 + {xy_rf_outcome_col[len('log1p_'):]})"
        if (xy_rf_outcome_col is not None and xy_rf_outcome_col.startswith("log1p_"))
        else "Mean Firm Size (emp_ct) in t - 1" #f"Mean {xy_rf_outcome_col}"
    )

    # Binscatter: instrument (x) vs reduced-form outcome at lag m1 (y).
    xy_bins = pd.DataFrame()
    if (
        xy_rf_outcome_col is not None
        and instrument_col in reg_panel.columns
        and xy_rf_outcome_col in reg_panel.columns
    ):
        cols = [instrument_col, xy_rf_outcome_col]
        has_nu = "n_universities" in reg_panel.columns
        if has_nu:
            cols.append("n_universities")
        xy_df = reg_panel[cols].dropna().copy()
        if not xy_df.empty:
            xy_df["bin_id"] = (
                pd.qcut(
                    xy_df[instrument_col].rank(method="first"),
                    q=25,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            agg_dict = {
                "x_bin": (instrument_col, "mean"),
                "y_bin": (xy_rf_outcome_col, "mean"),
            }
            if has_nu:
                agg_dict["mean_n_universities"] = ("n_universities", "mean")
            xy_bins = xy_df.groupby("bin_id", as_index=False).agg(**agg_dict).sort_values("x_bin")
    if not xy_bins.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in xy_bins.columns:
            sc = plt.scatter(
                xy_bins["x_bin"],
                xy_bins["y_bin"],
                c=xy_bins["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean # universities contributing to z_ct")
        else:
            plt.scatter(xy_bins["x_bin"], xy_bins["y_bin"], s=30, alpha=0.9)
        plt.xlabel("OPT Hiring Pool Size (z_ct)")#instrument_col)
        plt.ylabel(_xy_rf_label)
        plt.tight_layout()
        _savefig("rf_binscatter_by_nuniv.png")
        plt.show()
    else:
        if xy_rf_outcome_col is None:
            print("\n[warn] No lag-m1 reduced-form outcome column found; skipping x-y binscatter.")
        else:
            print(
                f"\n[warn] No binscatter data available for x={instrument_col}, "
                f"y={xy_rf_outcome_col}."
            )

    # Binscatter: instrument (x) vs treatment (y), pooled across years.
    x_treat_bins = pd.DataFrame()
    if instrument_col in reg_panel.columns and treatment_col in reg_panel.columns:
        cols = [instrument_col, treatment_col]
        has_nu = "n_universities" in reg_panel.columns
        if has_nu:
            cols.append("n_universities")
        x_treat_df = reg_panel[cols].dropna().copy()
        if not x_treat_df.empty:
            x_treat_df["bin_id"] = (
                pd.qcut(
                    x_treat_df[instrument_col].rank(method="first"),
                    q=25,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )
            agg_dict = {
                "x_bin": (instrument_col, "mean"),
                "y_bin": (treatment_col, "mean"),
            }
            if has_nu:
                agg_dict["mean_n_universities"] = ("n_universities", "mean")
            x_treat_bins = x_treat_df.groupby("bin_id", as_index=False).agg(**agg_dict).sort_values("x_bin")
    if not x_treat_bins.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in x_treat_bins.columns:
            sc = plt.scatter(
                x_treat_bins["x_bin"],
                x_treat_bins["y_bin"],
                c=x_treat_bins["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean # universities contributing to z_ct")
        else:
            plt.scatter(x_treat_bins["x_bin"], x_treat_bins["y_bin"], s=30, alpha=0.9)
        plt.xlabel("OPT Hiring Pool Size (z_ct)")#instrument_col)
        plt.ylabel("Mean # OPT Hires at t = 0")#treatment_col   )
        plt.tight_layout()
        _savefig("fs_binscatter_by_nuniv.png")
        plt.show()
    else:
        print(
            f"\n[warn] No binscatter data available for x={instrument_col}, "
            f"y={treatment_col}."
        )

    # Binscatter: reduced-form outcome (y) vs within-firm YoY percent change in instrument (x), pooled.
    z_pct_y0_bins = pd.DataFrame()
    if instrument_col in reg_panel.columns and reduced_form_outcome_col in reg_panel.columns:
        cols = ["c", "t", instrument_col, reduced_form_outcome_col]
        has_nu = "n_universities" in reg_panel.columns
        if has_nu:
            cols.append("n_universities")
        z_pct_df = reg_panel[cols].dropna(subset=["c", "t", instrument_col, reduced_form_outcome_col]).copy()
        if not z_pct_df.empty:
            z_pct_df = z_pct_df.sort_values(["c", "t"])
            z_pct_df["z_lag1"] = z_pct_df.groupby("c")[instrument_col].shift(1)
            z_pct_df["z_pct_change"] = np.where(
                z_pct_df["z_lag1"].notna() & (z_pct_df["z_lag1"] != 0),
                (z_pct_df[instrument_col] - z_pct_df["z_lag1"]) / z_pct_df["z_lag1"],
                np.nan,
            )
            z_pct_df = z_pct_df.dropna(subset=["z_pct_change", reduced_form_outcome_col]).copy()
            if not z_pct_df.empty:
                z_pct_df["bin_id"] = (
                    pd.qcut(
                        z_pct_df["z_pct_change"].rank(method="first"),
                        q=25,
                        labels=False,
                        duplicates="drop",
                    )
                    + 1
                )
                agg_dict = {
                    "x_bin": ("z_pct_change", "mean"),
                    "y_bin": (reduced_form_outcome_col, "mean"),
                }
                if has_nu:
                    agg_dict["mean_n_universities"] = ("n_universities", "mean")
                z_pct_y0_bins = (
                    z_pct_df.groupby("bin_id", as_index=False)
                    .agg(**agg_dict)
                    .sort_values("x_bin")
                )
    if not z_pct_y0_bins.empty:
        plt.figure(figsize=(8, 4.5))
        if "mean_n_universities" in z_pct_y0_bins.columns:
            sc = plt.scatter(
                z_pct_y0_bins["x_bin"],
                z_pct_y0_bins["y_bin"],
                c=z_pct_y0_bins["mean_n_universities"],
                cmap="viridis",
                s=30,
                alpha=0.9,
            )
            plt.colorbar(sc, label="Mean n_universities in bin")
        else:
            plt.scatter(z_pct_y0_bins["x_bin"], z_pct_y0_bins["y_bin"], s=30, alpha=0.9)
        plt.xlabel(f"YoY % change in {instrument_col} (within firm)")
        plt.ylabel(reduced_form_outcome_col)
        plt.tight_layout()
        _savefig("rf_pctchange_binscatter.png")
        plt.show()
    else:
        print(
            f"\n[warn] No binscatter data available for x=YoY % change in {instrument_col}, "
            f"y={reduced_form_outcome_col}."
        )


## COEFFICIENTS BY LAGGED X
# Coefficients by x lag: x_cst_lag[num] ~ instrument (no c,t FEs), separately by year
if coefs_by_xlag:
    _x_lag_suffix_map = {
        "masters_opt_hires_correction_aware": "",
        "x_bin_any_nonzero": "_bin_any_nonzero",
        "x_bin_above_year_median": "_bin_above_year_median",
        "x_bin_topbot_quartile": "_bin_topbot_quartile",
    }
    _x_lag_suffix = _x_lag_suffix_map.get(treatment_col, "")
    if treatment_col not in _x_lag_suffix_map and treatment_col not in {
        "masters_opt_hires",
        "valid_masters_opt_hires",
    }:
        print(
            f"\n[warn] treatment_col='{treatment_col}' has no explicit lag mapping; "
            "defaulting to x_cst_lag* for x-lag plot."
        )

    def _parse_x_lag(colname: str) -> int | None:
        m = re.fullmatch(rf"x_cst_lag(m?\d+){re.escape(_x_lag_suffix)}", colname)
        if not m:
            return None
        token = m.group(1)
        return -int(token[1:]) if token.startswith("m") else int(token)

    x_lag_cols = [c for c in reg_panel.columns if _parse_x_lag(c) is not None]
    x_lag_cols = sorted(x_lag_cols, key=lambda c: _parse_x_lag(c))

    if not x_lag_cols:
        print(
            f"\n[warn] No lag columns found for treatment_col='{treatment_col}' "
            f"(expected suffix '{_x_lag_suffix}'); skipping lag-coefficient plot."
        )
    elif not plot_x_lag_by_year:
        print("\n[info] Skipping x_lag-by-year plot (set shift_share_regressions.plot_x_lag_by_year: true to enable).")
    else:
        available_years = sorted(int(y) for y in reg_panel["t"].dropna().unique())
        pooled_all_years = x_lag_by_year_years_cfg in (None, "", [])
        if x_lag_by_year_years_cfg in (None, "", []):
            selected_years = []
        elif isinstance(x_lag_by_year_years_cfg, str):
            selected_years = [int(y.strip()) for y in x_lag_by_year_years_cfg.split(",") if y.strip()]
        elif isinstance(x_lag_by_year_years_cfg, (int, float)):
            selected_years = [int(x_lag_by_year_years_cfg)]
        else:
            selected_years = [int(y) for y in x_lag_by_year_years_cfg]

        if not pooled_all_years:
            selected_years = [y for y in selected_years if y in available_years]
        if (not pooled_all_years) and (not selected_years):
            print("\n[warn] No selected years overlap reg_panel years; skipping x_lag-by-year plot.")
        else:
            x_lag_rows = []
            iter_specs = [("Pooled", reg_panel.copy())] if pooled_all_years else [
                (str(yr), reg_panel.loc[reg_panel["t"] == yr].copy()) for yr in selected_years
            ]
            for series_label, samp in iter_specs:
                for col in x_lag_cols:
                    lag_num = _parse_x_lag(col)
                    try:
                        pooled_fe_term = _augment_fe("c + t")
                        lag_fml = (
                            f"{col} ~ {_instrument_rhs(True)} | {pooled_fe_term}"
                            if pooled_all_years
                            else f"{col} ~ {_instrument_rhs(False)} + 1"
                        )
                        fit_lag = pf.feols(
                            lag_fml,
                            data=samp,
                            vcov={"CRV1": "c"},
                            demeaner_backend="rust",
                        )
                        tl = fit_lag.tidy()
                        est, se = _extract_est_se(
                            tl,
                            instrument_coef_term,
                            context=f"x-lag model ({series_label}, {col})",
                            warn_on_missing=False,
                        )
                        x_lag_rows.append({"series": series_label, "lag": lag_num, "coef": est, "se": se})
                    except ValueError as e:
                        if "All variables are collinear" in str(e):
                            print(f"[warn] collinear model for series={series_label}, {col}; storing NaN.")
                            x_lag_rows.append(
                                {"series": series_label, "lag": lag_num, "coef": float("nan"), "se": float("nan")}
                            )
                        else:
                            raise
                    except KeyError:
                        print(
                            f"[warn] focal term '{instrument_coef_term}' missing for "
                            f"series={series_label}, {col}; storing NaN."
                        )
                        x_lag_rows.append(
                            {"series": series_label, "lag": lag_num, "coef": float("nan"), "se": float("nan")}
                        )

            x_lag_res = pd.DataFrame(x_lag_rows).sort_values(["series", "lag"])
            x_lag_res["lo"] = x_lag_res["coef"] - 1.96 * x_lag_res["se"]
            x_lag_res["hi"] = x_lag_res["coef"] + 1.96 * x_lag_res["se"]
            print("\n[first_stage_by_x_lag]")
            print(x_lag_res)

            plt.figure(figsize=(9, 5))
            for series_label in x_lag_res["series"].drop_duplicates().tolist():
                yr_df = x_lag_res.loc[x_lag_res["series"] == series_label].sort_values("lag")
                (line,) = plt.plot(yr_df["lag"], yr_df["coef"], marker="o", label=series_label)
                if yr_df["se"].notna().any():
                    plt.fill_between(
                        yr_df["lag"],
                        yr_df["lo"],
                        yr_df["hi"],
                        color=line.get_color(),
                        alpha=0.18,
                    )
            plt.axvline(0, color="black", linestyle="--", linewidth=1)
            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(sorted(x_lag_res["lag"].dropna().unique()))
            plt.xlabel("Year Relative to Event")
            plt.ylabel(f"Coef: effect of {instrument_coef_label} on # OPT hires by event time")
            # plt.title(
            #     f"Lagged treatment ({treatment_col}) ~ instrument | c + t, pooled all years"
            #     if pooled_all_years
            #     else f"Lagged treatment ({treatment_col}) ~ instrument (no FE), by year"
            # )
            if not pooled_all_years:
                plt.legend(title="Year")
            plt.tight_layout()
            _savefig("fs_by_eventtime.png")
            plt.show()

## ALTERNATE INSTRUMENT EVENT STUDY
if alt_event_study:
    es_source = con.sql(
        f"""
        SELECT *
        FROM analysis_panel
        WHERE t BETWEEN {alt_event_data_min_t} AND {alt_event_data_max_t}
        """
    ).df()
    if alt_event_instrument_col not in es_source.columns:
        raise ValueError(
            f"alt_event_instrument_col '{alt_event_instrument_col}' not found in event-study source columns."
        )
    _ensure_derived_outcome_col(es_source, alt_event_outcome_col)
    if alt_event_outcome_col not in es_source.columns:
        raise ValueError(
            f"alt_event_outcome_col '{alt_event_outcome_col}' not found in event-study source columns."
        )
    if not (0 < alt_event_control_pctile < alt_event_treat_pctile < 100):
        raise ValueError(
            "Require 0 < alt_event_control_pctile < alt_event_treat_pctile < 100."
        )
    if alt_event_time_min >= alt_event_time_max:
        raise ValueError("alt_event_time_min must be < alt_event_time_max.")
    if alt_event_event_min_t > alt_event_event_max_t:
        raise ValueError("alt_event_event_min_t must be <= alt_event_event_max_t.")

    es_df = es_source[["c", "t", alt_event_instrument_col, alt_event_outcome_col]].copy()
    es_df = es_df.sort_values(["c", "t"])
    es_df["z_lag1"] = es_df.groupby("c")[alt_event_instrument_col].shift(1)
    es_df["z_pct_change"] = np.where(
        es_df["z_lag1"].notna() & (es_df["z_lag1"] != 0),
        (es_df[alt_event_instrument_col] - es_df["z_lag1"]) / es_df["z_lag1"],
        np.nan,
    )
    es_df["event_year_eligible"] = es_df["t"].between(alt_event_event_min_t, alt_event_event_max_t)

    pct_valid = es_df.loc[es_df["event_year_eligible"], "z_pct_change"].dropna()
    if pct_valid.empty:
        print("\n[warn] No valid z_pct_change values; skipping alt event-study section.")
    else:
        treat_cutoff = float(np.nanpercentile(pct_valid, alt_event_treat_pctile))
        control_cutoff = float(np.nanpercentile(pct_valid, alt_event_control_pctile))

        es_df["is_treat_event"] = es_df["event_year_eligible"] & (es_df["z_pct_change"] > treat_cutoff)
        es_df["is_control_event"] = es_df["event_year_eligible"] & (es_df["z_pct_change"] > control_cutoff)

        treat_events = (
            es_df.loc[es_df["is_treat_event"], ["c", "t"]]
            .groupby("c", as_index=False)
            .agg(n_events=("t", "count"), event_year=("t", "min"))
        )
        treated_firms = treat_events.loc[treat_events["n_events"] == 1, ["c", "event_year"]].copy()
        treated_firms["treated"] = 1

        control_counts = (
            es_df.groupby("c", as_index=False)
            .agg(n_control_events=("is_control_event", "sum"))
        )
        control_firms = control_counts.loc[control_counts["n_control_events"] == 0, ["c"]].copy()
        control_firms = control_firms.loc[~control_firms["c"].isin(set(treated_firms["c"]))].copy()

        if treated_firms.empty or control_firms.empty:
            print(
                "\n[warn] Treated or control firm set is empty under current thresholds; "
                "skipping alt event-study section."
            )
        else:
            rng = np.random.default_rng(alt_event_seed)
            control_firms = control_firms.copy()
            matched_controls_used = False
            if alt_event_match_controls_on_y_cst_lagm3:
                if "y_cst_lagm3" not in es_source.columns:
                    print(
                        "\n[warn] alt_event_match_controls_on_y_cst_lagm3=true but y_cst_lagm3 is missing; "
                        "falling back to random control-year assignment."
                    )
                else:
                    y_lookup = es_source[["c", "t", "y_cst_lagm3"]].rename(columns={"t": "event_year"})
                    treated_match = treated_firms.merge(
                        y_lookup, on=["c", "event_year"], how="left"
                    ).rename(columns={"y_cst_lagm3": "treated_y_cst_lagm3"})

                    control_pool = control_firms.merge(y_lookup, on="c", how="left")
                    available_controls = set(control_firms["c"].tolist())
                    matched_rows = []
                    _can_match = True
                    for _, tr in treated_match.sort_values(["event_year", "c"]).iterrows():
                        tr_event_year = int(tr["event_year"])
                        cand = control_pool.loc[
                            (control_pool["event_year"] == tr_event_year)
                            & (control_pool["c"].isin(available_controls))
                        ].copy()
                        if cand.empty:
                            _can_match = False
                            break
                        tr_y = tr["treated_y_cst_lagm3"]
                        if pd.notna(tr_y):
                            cand_nonmissing = cand.loc[cand["y_cst_lagm3"].notna()].copy()
                            if not cand_nonmissing.empty:
                                cand_nonmissing["abs_diff"] = (cand_nonmissing["y_cst_lagm3"] - tr_y).abs()
                                chosen = cand_nonmissing.sort_values(["abs_diff", "c"]).iloc[0]
                            else:
                                chosen = cand.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
                        else:
                            chosen = cand.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
                        chosen_c = chosen["c"]
                        available_controls.remove(chosen_c)
                        matched_rows.append({"c": chosen_c, "event_year": tr_event_year})

                    if _can_match and len(matched_rows) == len(treated_firms):
                        control_firms = pd.DataFrame(matched_rows)
                        matched_controls_used = True
                    else:
                        print(
                            "\n[warn] Could not construct full matched controls on y_cst_lagm3; "
                            "falling back to random control-year assignment."
                        )

            if not matched_controls_used:
                treat_year_probs = (
                    treated_firms["event_year"].value_counts(normalize=True).sort_index()
                )
                draw_years = rng.choice(
                    treat_year_probs.index.to_numpy(dtype=int),
                    size=len(control_firms),
                    replace=True,
                    p=treat_year_probs.to_numpy(),
                )
                control_firms["event_year"] = draw_years

            control_firms["treated"] = 0

            event_firms = pd.concat(
                [
                    treated_firms[["c", "event_year", "treated"]],
                    control_firms[["c", "event_year", "treated"]],
                ],
                ignore_index=True,
            )
            event_firms["group"] = np.where(event_firms["treated"] == 1, "treated", "control")

            event_counts = (
                event_firms.groupby(["event_year", "group"], as_index=False)
                .agg(n_firms=("c", "nunique"))
                .sort_values(["event_year", "group"])
            )
            event_counts_wide = (
                event_counts.pivot(index="event_year", columns="group", values="n_firms")
                .fillna(0)
                .astype(int)
                .reset_index()
                .sort_values("event_year")
            )
            print("\n[alt_event_study_counts_by_event_year]")
            print(event_counts_wide)

            if "y_cst_lagm3" in es_source.columns:
                event_firm_sizes = (
                    event_firms.merge(
                        es_source[["c", "t", "y_cst_lagm3"]].rename(columns={"t": "event_year"}),
                        on=["c", "event_year"],
                        how="left",
                    )
                )
                mean_lagm3_by_group = (
                    event_firm_sizes.groupby("group", as_index=False)
                    .agg(
                        mean_y_cst_lagm3=("y_cst_lagm3", "mean"),
                        n_firms=("c", "nunique"),
                        n_nonmissing_y_cst_lagm3=("y_cst_lagm3", "count"),
                    )
                    .sort_values("group")
                )
                mean_lagm3_by_year_group = (
                    event_firm_sizes.groupby(["event_year", "group"], as_index=False)
                    .agg(
                        mean_y_cst_lagm3=("y_cst_lagm3", "mean"),
                        n_firms=("c", "nunique"),
                        n_nonmissing_y_cst_lagm3=("y_cst_lagm3", "count"),
                    )
                    .sort_values(["event_year", "group"])
                )
                print("\n[alt_event_study_mean_y_cst_lagm3_by_group]")
                print(mean_lagm3_by_group)
                print("\n[alt_event_study_mean_y_cst_lagm3_by_event_year_group]")
                print(mean_lagm3_by_year_group)
            else:
                print("\n[warn] Column 'y_cst_lagm3' not found in event-study source; skipping lagm3 summary tables.")

            es_sample = es_source.merge(event_firms, on="c", how="inner")
            es_sample["event_time"] = es_sample["t"] - es_sample["event_year"]
            es_sample = es_sample.loc[
                (es_sample["event_time"] >= alt_event_time_min)
                & (es_sample["event_time"] <= alt_event_time_max)
            ].copy()

            if es_sample.empty:
                print("\n[warn] Event-study sample empty after event-time window filter; skipping section.")
            else:
                # (1) Raw means by event time for treated vs control.
                raw_means = (
                    es_sample.groupby(["event_time", "treated"], as_index=False)[alt_event_outcome_col]
                    .mean()
                    .rename(columns={alt_event_outcome_col: "mean_outcome"})
                )
                plt.figure(figsize=(8, 4.5))
                for tr, label in [(0, "Control"), (1, "Treated")]:
                    s = raw_means.loc[raw_means["treated"] == tr].sort_values("event_time")
                    if not s.empty:
                        plt.plot(s["event_time"], s["mean_outcome"], marker="o", label=label)
                plt.axvline(0, color="black", linestyle="--", linewidth=1)
                plt.xlabel("Event time")
                plt.ylabel(f"Mean {alt_event_outcome_col}")
                plt.legend()
                plt.tight_layout()
                _savefig("abs_opthires_by_eventtime.png")
                plt.show()

                # (2) Raw means by event time for 2012/2014/2016 event cohorts.
                plot_years = _parse_int_list(alt_event_plot_years_cfg) or [2012, 2014, 2016]
                fig, axes = plt.subplots(len(plot_years), 1, figsize=(8, 4.2 * len(plot_years)), sharex=True)
                if len(plot_years) == 1:
                    axes = [axes]
                for ax, event_year in zip(axes, plot_years):
                    sub = es_sample.loc[es_sample["event_year"] == event_year].copy()
                    sub_means = (
                        sub.groupby(["event_time", "treated"], as_index=False)[alt_event_outcome_col]
                        .mean()
                        .rename(columns={alt_event_outcome_col: "mean_outcome"})
                    )
                    for tr, label in [(0, "Control"), (1, "Treated")]:
                        s = sub_means.loc[sub_means["treated"] == tr].sort_values("event_time")
                        if not s.empty:
                            ax.plot(s["event_time"], s["mean_outcome"], marker="o", label=label)
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set_ylabel(f"Mean {alt_event_outcome_col}")
                    ax.set_title(f"Event year = {event_year}")
                    ax.legend()
                axes[-1].set_xlabel("Event time")
                plt.tight_layout()
                _savefig("abs_raw_by_cohort.png")
                plt.show()

                # (3) TWFE event study: coefficients on treated x event-time.
                event_times = sorted(int(k) for k in es_sample["event_time"].dropna().unique())
                ref_k = -1 if -1 in event_times else min(event_times)
                rhs_terms = []
                kept_ks = []
                for k in event_times:
                    if k == ref_k:
                        continue
                    col = f"evt_treat_{'m' + str(abs(k)) if k < 0 else 'p' + str(k)}"
                    es_sample[col] = ((es_sample["event_time"] == k) & (es_sample["treated"] == 1)).astype("int8")
                    rhs_terms.append(col)
                    kept_ks.append((k, col))

                if not rhs_terms:
                    print("\n[warn] No non-reference event-time indicators available for TWFE plot.")
                else:
                    twfe_fit = pf.feols(
                        f"{alt_event_outcome_col} ~ {' + '.join(rhs_terms)} | {_augment_fe('c + t')}",
                        data=es_sample,
                        vcov={"CRV1": "c"},
                        demeaner_backend="rust",
                    )
                    twfe_tidy = twfe_fit.tidy()
                    twfe_rows = []
                    for k, col in kept_ks:
                        if col in twfe_tidy.index:
                            est = float(twfe_tidy.loc[col, "Estimate"])
                            se = float(twfe_tidy.loc[col, "Std. Error"])
                            twfe_rows.append(
                                {
                                    "event_time": k,
                                    "coef": est,
                                    "se": se,
                                    "lo": est - 1.96 * se,
                                    "hi": est + 1.96 * se,
                                }
                            )
                    twfe_res = pd.DataFrame(twfe_rows).sort_values("event_time")
                    if twfe_res.empty:
                        print("\n[warn] TWFE regression returned no event-time coefficients to plot.")
                    else:
                        ref_point = pd.DataFrame(
                            [{"event_time": ref_k, "coef": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0}]
                        )
                        twfe_plot = (
                            pd.concat([twfe_res, ref_point], ignore_index=True)
                            .drop_duplicates(subset=["event_time"], keep="first")
                            .sort_values("event_time")
                        )
                        plt.figure(figsize=(8, 4.5))
                        plt.errorbar(
                            twfe_plot["event_time"],
                            twfe_plot["coef"],
                            yerr=1.96 * twfe_plot["se"],
                            fmt="o",
                            capsize=4,
                        )
                        plt.axhline(0, color="black", linewidth=1)
                        plt.axvline(0, color="black", linestyle="--", linewidth=1)
                        plt.xlabel("Event time")
                        plt.ylabel(f"TWFE coef on Treated x Event-time ({alt_event_outcome_col})")
                        plt.tight_layout()
                        _savefig("abs_fs_eventstudy.png")
                        plt.show()

                        # Additional TWFE split plot by event-year cohorts.
                        cohort_specs = [
                            ("Events 2012-2014", (2012, 2014), "tab:blue"),
                            ("Events 2015-2018", (2015, 2018), "tab:orange"),
                        ]
                        cohort_plot_rows: list[pd.DataFrame] = []
                        for cohort_label, (y0, y1), _ in cohort_specs:
                            cohort_sample = es_sample.loc[
                                (es_sample["event_year"] >= y0) & (es_sample["event_year"] <= y1)
                            ].copy()
                            if cohort_sample.empty or cohort_sample["treated"].nunique() < 2:
                                continue
                            cohort_fit = pf.feols(
                                f"{alt_event_outcome_col} ~ {' + '.join(rhs_terms)} | {_augment_fe('c + t')}",
                                data=cohort_sample,
                                vcov={"CRV1": "c"},
                                demeaner_backend="rust",
                            )
                            cohort_tidy = cohort_fit.tidy()
                            rows = []
                            for k, col in kept_ks:
                                if col in cohort_tidy.index:
                                    est = float(cohort_tidy.loc[col, "Estimate"])
                                    se = float(cohort_tidy.loc[col, "Std. Error"])
                                    rows.append(
                                        {
                                            "event_time": k,
                                            "coef": est,
                                            "se": se,
                                            "cohort": cohort_label,
                                        }
                                    )
                            if rows:
                                cdf = pd.DataFrame(rows)
                                cdf = pd.concat(
                                    [
                                        cdf,
                                        pd.DataFrame(
                                            [
                                                {
                                                    "event_time": ref_k,
                                                    "coef": 0.0,
                                                    "se": 0.0,
                                                    "cohort": cohort_label,
                                                }
                                            ]
                                        ),
                                    ],
                                    ignore_index=True,
                                )
                                cdf = cdf.drop_duplicates(subset=["event_time"], keep="first").sort_values(
                                    "event_time"
                                )
                                cdf["lo"] = cdf["coef"] - 1.96 * cdf["se"]
                                cdf["hi"] = cdf["coef"] + 1.96 * cdf["se"]
                                cohort_plot_rows.append(cdf)

                        if cohort_plot_rows:
                            cohort_plot = pd.concat(cohort_plot_rows, ignore_index=True)
                            plt.figure(figsize=(8, 4.5))
                            for cohort_label, _, color in cohort_specs:
                                cdf = cohort_plot.loc[cohort_plot["cohort"] == cohort_label].sort_values("event_time")
                                if cdf.empty:
                                    continue
                                plt.plot(
                                    cdf["event_time"],
                                    cdf["coef"],
                                    marker="o",
                                    color=color,
                                    label=cohort_label,
                                )
                                plt.fill_between(
                                    cdf["event_time"],
                                    cdf["lo"],
                                    cdf["hi"],
                                    color=color,
                                    alpha=0.18,
                                )
                            plt.axhline(0, color="black", linewidth=1)
                            plt.axvline(0, color="black", linestyle="--", linewidth=1)
                            plt.xlabel("Event time")
                            plt.ylabel(f"TWFE coef on Treated x Event-time ({alt_event_outcome_col})")
                            plt.legend()
                            plt.tight_layout()
                            _savefig("abs_fs_bycohort.png")
                            plt.show()
                        else:
                            print("\n[warn] Could not estimate cohort-split TWFE event-study plot.")

## BINSCATTERS
# # Binscatter-style plots of x against z (raw and two-way demeaned by company/year)
# scatter_df = reg_panel[[instrument_col, treatment_col, "y_cst_lagm1", "c", "t"]].dropna().copy()


# def _make_binned_points(df: pd.DataFrame, x_col: str, y_col: str, size_col: str, bins: int = 50) -> pd.DataFrame:
#     tmp = df[[x_col, y_col, size_col]].dropna().copy()
#     tmp["bin_id"] = pd.qcut(
#         tmp[x_col].rank(method="first"),
#         q=bins,
#         labels=False,
#         duplicates="drop",
#     )
#     out = (
#         tmp.groupby("bin_id", as_index=False)
#         .agg(
#             x_bin=(x_col, "mean"),
#             y_bin=(y_col, "mean"),
#             mean_firm_size=(size_col, "mean"),
#         )
#         .sort_values("x_bin")
#     )
#     return out

# x_dm = (
#     scatter_df[treatment_col]
#     - scatter_df.groupby("c")[treatment_col].transform("mean")
#     - scatter_df.groupby("t")[treatment_col].transform("mean")
#     + scatter_df[treatment_col].mean()
# )
# z_dm = (
#     scatter_df[instrument_col]
#     - scatter_df.groupby("c")[instrument_col].transform("mean")
#     - scatter_df.groupby("t")[instrument_col].transform("mean")
#     + scatter_df[instrument_col].mean()
# )

# scatter_dm = pd.DataFrame({"z_raw": scatter_df[instrument_col], "x_dm": x_dm}).dropna()

# scatter_dm_both = pd.DataFrame({"z_dm": z_dm, "x_dm": x_dm}).dropna()
# scatter_raw = pd.DataFrame(
#     {
#         "z_raw": scatter_df[instrument_col],
#         "x_raw": scatter_df[treatment_col],
#         "y_cst_lagm1": scatter_df["y_cst_lagm1"],
#     }
# ).dropna()
# scatter_dm = pd.DataFrame(
#     {"z_raw": scatter_df[instrument_col], "x_dm": x_dm, "y_cst_lagm1": scatter_df["y_cst_lagm1"]}
# ).dropna()
# scatter_dm_both = pd.DataFrame(
#     {"z_dm": z_dm, "x_dm": x_dm, "y_cst_lagm1": scatter_df["y_cst_lagm1"]}
# ).dropna()

# raw_bins = _make_binned_points(scatter_raw, "z_raw", "x_raw", "y_cst_lagm1", bins=50)
# dm_rawz_bins = _make_binned_points(scatter_dm, "z_raw", "x_dm", "y_cst_lagm1", bins=50)
# dm_both_bins = _make_binned_points(scatter_dm_both, "z_dm", "x_dm", "y_cst_lagm1", bins=50)

# all_size_means = pd.concat(
#     [
#         raw_bins["mean_firm_size"],
#         dm_rawz_bins["mean_firm_size"],
#         dm_both_bins["mean_firm_size"],
#     ],
#     ignore_index=True,
# )
# size_vmin = float(all_size_means.min())
# size_vmax = float(all_size_means.max())

# def _plot_binned_with_size_color(df: pd.DataFrame, xlabel: str, ylabel: str, title: str) -> None:
#     plt.figure(figsize=(8, 4.5))
#     sc = plt.scatter(
#         df["x_bin"],
#         df["y_bin"],
#         c=df["mean_firm_size"],
#         cmap="viridis",
#         vmin=size_vmin,
#         vmax=size_vmax,
#         s=40,
#         alpha=0.9,
#     )
#     cbar = plt.colorbar(sc)
#     cbar.set_label("Mean y_cst_lagm1 in bin")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# _plot_binned_with_size_color(
#     raw_bins,
#     instrument_col,
#     treatment_col,
#     f"Binscatter: {treatment_col} vs {instrument_col}",
# )
# _plot_binned_with_size_color(
#     dm_rawz_bins,
#     f"{instrument_col} (raw)",
#     f"{treatment_col} (demeaned by c and t)",
#     f"Binscatter: demeaned {treatment_col} vs raw {instrument_col}",
# )
# _plot_binned_with_size_color(
#     dm_both_bins,
#     f"{instrument_col} (demeaned by c and t)",
#     f"{treatment_col} (demeaned by c and t)",
#     f"Binscatter: demeaned {treatment_col} vs demeaned {instrument_col}",
# )

# scatter_x = pd.DataFrame({"x_raw": scatter_df[treatment_col], "x_dm": x_dm}).dropna()
# plt.figure(figsize=(8, 4.5))
# sns.regplot(
#     data=scatter_x,
#     x="x_raw",
#     y="x_dm",
#     x_bins=50,
#     fit_reg=False,
#     ci=None,
#     scatter_kws={"s": 20, "alpha": 0.6},
# )
# plt.xlabel(f"{treatment_col} (raw)")
# plt.ylabel(f"{treatment_col} (demeaned by c and t)")
# plt.title(f"Binscatter: demeaned vs raw {treatment_col}")
# plt.tight_layout()
# plt.show()
