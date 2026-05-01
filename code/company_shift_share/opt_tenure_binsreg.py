"""Binscatter and binsreg-style summaries for tenure and OPT hiring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_SRC = str(Path(__file__).resolve().parents[1])
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from company_shift_share.config_loader import get_cfg_section, load_config
from company_shift_share.source_exposure_data import (
    load_or_build_source_opt_counts,
    load_or_build_wrds_company_year_workforce_cache,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "company_shift_share_exposure_event_study.yaml"
)


DEFAULT_TENURE_COL = "avg_tenure_years_annual"
DEFAULT_OPT_HIRE_COL = "any_opt_hires_correction_aware"
DEFAULT_FIRM_SIZE_COL = "total_headcount_wrds_annual"
DEFAULT_N_HIRES_COL = "n_new_hires_wrds_annual"
TENURE_SCOPE_ALL = "all"
TENURE_SCOPE_NEW_HIRES = "new_hires"
TENURE_SCOPE_RECENT_GRADS = "recent_grads"
TENURE_SCOPE_CHOICES = (TENURE_SCOPE_ALL, TENURE_SCOPE_NEW_HIRES, TENURE_SCOPE_RECENT_GRADS)
SCRIPT_CFG_SECTION = "opt_tenure_binsreg"


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_interactive() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return None


def _coerce_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return Path(stripped) if stripped else None
    return None


def _apply_script_config_defaults(args: argparse.Namespace, cfg: dict[str, Any], argv: list[str]) -> argparse.Namespace:
    script_cfg = get_cfg_section(cfg, SCRIPT_CFG_SECTION)
    if not script_cfg:
        return args

    def _set_if_missing(flag: str, key: str, cast: callable | None = None) -> None:
        if _argv_has_option(argv, flag):
            return
        if key not in script_cfg:
            return
        raw = script_cfg.get(key)
        if raw is None:
            return
        if cast is None:
            setattr(args, key, raw)
            return
        value = cast(raw)
        if value is not None:
            setattr(args, key, value)

    _set_if_missing("--data-min-t", "data_min_t", int)
    _set_if_missing("--data-max-t", "data_max_t", int)
    _set_if_missing("--tenure-col", "tenure_col", str)
    _set_if_missing("--opt-count-col", "opt_count_col", str)
    _set_if_missing("--n-hires-col", "n_hires_col", str)
    _set_if_missing("--firm-size-col", "firm_size_col", str)
    _set_if_missing("--metric", "metric", str)
    _set_if_missing("--force-rebuild", "force_rebuild", _coerce_bool)
    _set_if_missing("--bins", "bins", int)
    _set_if_missing("--out-dir", "out_dir", _coerce_optional_path)
    _set_if_missing("--prefix", "prefix", str)
    _set_if_missing("--by-year-bins", "by_year_bins", _coerce_bool)
    _set_if_missing("--tenure-scope", "tenure_scope", str)
    _set_if_missing("--show-plots", "show_plots", _coerce_bool)
    _set_if_missing("--log-opt-hires", "log_opt_hires", _coerce_bool)
    return args


def _resolve_year_window(cfg: dict, args: argparse.Namespace) -> tuple[int, int]:
    return _resolve_year_window_values(cfg, args.data_min_t, args.data_max_t)


def _resolve_year_window_values(
    cfg: dict,
    data_min_t: int | None,
    data_max_t: int | None,
) -> tuple[int, int]:
    exposure_cfg = get_cfg_section(cfg, "exposure_event_study")
    default_min = int(exposure_cfg.get("data_min_t", 2010))
    default_max = int(exposure_cfg.get("data_max_t", 2022))

    y_min = int(data_min_t) if data_min_t is not None else default_min
    y_max = int(data_max_t) if data_max_t is not None else default_max
    if y_min > y_max:
        raise ValueError("data_min_t must be <= data_max_t")
    return y_min, y_max


def _escape_sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _resolve_optional_path(cfg: dict, key: str) -> Path | None:
    paths_cfg = get_cfg_section(cfg, "paths")
    raw = paths_cfg.get(key)
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _resolve_selected_positions_path(cfg: dict) -> Path | None:
    path = _resolve_optional_path(cfg, "wrds_workforce_selected_us_positions_out")
    if path is not None:
        return path
    return _resolve_optional_path(cfg, "wrds_workforce_positions_out")


def _resolve_users_path(cfg: dict) -> Path | None:
    return _resolve_optional_path(cfg, "wrds_workforce_users_out")


def _load_wrds_workforce_cache_or_build(
    cfg: dict,
    cfg_path: Path,
    y_min: int,
    y_max: int,
    *,
    force_rebuild: bool,
) -> pd.DataFrame:
    paths_cfg = get_cfg_section(cfg, "paths")
    workforce_path_text = paths_cfg.get("wrds_company_year_workforce_out")
    if not workforce_path_text:
        raise RuntimeError("Config is missing paths.wrds_company_year_workforce_out")
    workforce_path = Path(workforce_path_text)
    if workforce_path.exists() and not force_rebuild:
        workforce = pd.read_parquet(workforce_path)
        has_required = {"c", "t", "avg_tenure_years_annual"}
        if has_required.issubset(set(workforce.columns)):
            t_series = _safe_numeric(workforce["t"])
            if t_series.notna().any():
                t_min = int(t_series.min())
                t_max = int(t_series.max())
                if t_min <= y_min and t_max >= y_max:
                    return workforce
        else:
            print("[warn] Workforce cache exists but missing avg_tenure_years_annual; rebuilding from cache source.")
    elif force_rebuild:
        print("[warn] force_rebuild requested; rebuilding workforce panel.")

    # Fall back to builder. This may use WRDS credentials if no compatible cache exists.
    try:
        workforce, _ = load_or_build_wrds_company_year_workforce_cache(
            config_path=cfg_path,
            cfg=cfg,
            year_min=y_min,
            year_max=y_max,
            force_rebuild=force_rebuild,
        )
        return workforce
    except Exception as exc:
        print(f"[warn] Unable to build workforce cache via WRDS pipeline: {type(exc).__name__}: {exc}")
        if force_rebuild:
            raise

    workforce = _build_workforce_from_cached_positions(cfg, y_min, y_max)
    if workforce.empty:
        raise RuntimeError(
            "Fallback workforce reconstruction from cached positions returned no rows. "
            "Try forcing rebuild with WRDS access."
        )
    return workforce


def _build_workforce_from_cached_positions(
    cfg: dict,
    y_min: int,
    y_max: int,
) -> pd.DataFrame:
    paths_cfg = get_cfg_section(cfg, "paths")
    selected_positions = paths_cfg.get("wrds_workforce_selected_us_positions_out")
    positions_path = Path(selected_positions) if selected_positions else None
    if positions_path is None or not positions_path.exists():
        fallback = Path(paths_cfg.get("wrds_workforce_positions_out"))
        positions_path = fallback if fallback.exists() else None
    if positions_path is None or not positions_path.exists():
        raise RuntimeError(
            "No cached WRDS workforce positions file found in config paths. "
            "Run workforce cache rebuild or pass --force-rebuild with WRDS credentials."
        )

    con = ddb.connect()
    try:
        sql = f"""
        WITH us_positions AS (
            SELECT
                CAST(p.rcid AS BIGINT) AS c,
                CAST(p.user_id AS BIGINT) AS user_id,
                TRY_CAST(p.startdate AS DATE) AS startdate,
                COALESCE(TRY_CAST(p.enddate AS DATE), DATE '2025-12-31') AS enddate
            FROM read_parquet('{_escape_sql_path(positions_path)}') AS p
            WHERE p.user_id IS NOT NULL
              AND TRY_CAST(p.startdate AS DATE) IS NOT NULL
              AND EXTRACT(YEAR FROM TRY_CAST(p.startdate AS DATE))::INT <= {int(y_max)}
              AND EXTRACT(YEAR FROM COALESCE(TRY_CAST(p.enddate AS DATE), DATE '2025-12-31'))::INT >= {int(y_min)}
        ),
        company_user_first_start AS (
            SELECT
                c,
                user_id,
                MIN(startdate) AS first_start
            FROM us_positions
            GROUP BY 1, 2
        ),
        new_hires AS (
            SELECT
                c,
                EXTRACT(YEAR FROM first_start)::INTEGER AS t,
                COUNT(DISTINCT user_id)::DOUBLE PRECISION AS n_new_hires_wrds_annual
            FROM company_user_first_start
            WHERE EXTRACT(YEAR FROM first_start)::INT BETWEEN {int(y_min)} AND {int(y_max)}
            GROUP BY 1, 2
        ),
        active_user_year AS (
            SELECT
                p.c,
                gs.year::INT AS t,
                p.user_id,
                MAX(cufs.first_start) AS first_start
            FROM us_positions p
            JOIN company_user_first_start cufs
              ON cufs.user_id = p.user_id
             AND cufs.c = p.c
            JOIN LATERAL generate_series(
                GREATEST(EXTRACT(YEAR FROM p.startdate)::INT, {int(y_min)}),
                LEAST(EXTRACT(YEAR FROM p.enddate)::INT, {int(y_max)})
            ) AS gs(year) ON TRUE
            GROUP BY 1, 2, 3
        )
        SELECT
            CAST(ay.c AS BIGINT) AS c,
            CAST(ay.t AS INTEGER) AS t,
            COUNT(DISTINCT ay.user_id)::DOUBLE PRECISION AS total_headcount_wrds_annual,
            COALESCE(MAX(nh.n_new_hires_wrds_annual), 0.0) AS n_new_hires_wrds_annual,
            AVG(GREATEST(0.0, (make_date(ay.t, 12, 31) - ay.first_start)::DOUBLE PRECISION / 365.25))
                AS avg_tenure_years_annual
        FROM active_user_year ay
        LEFT JOIN new_hires nh
          ON nh.c = ay.c
         AND nh.t = ay.t
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
        workforce = con.sql(sql).df()
    finally:
        con.close()
    return workforce


def _build_new_hire_tenure_panel(
    cfg: dict,
    y_min: int,
    y_max: int,
    positions_path: Path | None = None,
) -> pd.DataFrame:
    positions_path = Path(positions_path) if positions_path is not None else _resolve_selected_positions_path(cfg)
    if positions_path is None:
        return pd.DataFrame(
            columns=[
                "c",
                "t",
                "avg_tenure_new_hires_annual",
                "n_new_hires_year",
            ],
        )

    con = ddb.connect()
    try:
        sql = f"""
        WITH pos AS (
            SELECT
                CAST(rcid AS BIGINT) AS c,
                CAST(user_id AS BIGINT) AS user_id,
                CAST(startdate AS DATE) AS startdate,
                COALESCE(TRY_CAST(enddate AS DATE), DATE '2026-12-31') AS enddate
            FROM read_parquet('{_escape_sql_path(positions_path)}')
            WHERE c IS NOT NULL
              AND user_id IS NOT NULL
              AND startdate IS NOT NULL
        ),
        first_start_with_end AS (
            SELECT
                c,
                user_id,
                MIN(startdate) AS first_start,
                MAX(enddate) AS last_end
            FROM pos
            GROUP BY 1, 2
        ),
        new_hires AS (
            SELECT
                c,
                EXTRACT(YEAR FROM first_start)::INTEGER AS t,
                COUNT(*)::BIGINT AS n_new_hires_year,
                AVG(
                    GREATEST(
                        0.0,
                        (last_end - first_start)::DOUBLE PRECISION / 365.25
                    )
                ) AS avg_tenure_new_hires_annual
            FROM first_start_with_end
            WHERE EXTRACT(YEAR FROM first_start) BETWEEN {int(y_min)} AND {int(y_max)}
            GROUP BY 1, 2
        ),
        new_hire_tenure AS (
            SELECT
                c,
                t,
                avg_tenure_new_hires_annual,
                n_new_hires_year
            FROM new_hires
        )
        SELECT
            c,
            t,
            avg_tenure_new_hires_annual,
            n_new_hires_year
        FROM new_hire_tenure
        """
        out = con.sql(sql).df()
    finally:
        con.close()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "c",
                "t",
                "avg_tenure_new_hires_annual",
                "n_new_hires_year",
            ],
        )
    out["c"] = _safe_numeric(out["c"]).astype("Int64")
    out["t"] = _safe_numeric(out["t"]).astype("Int64")
    out["avg_tenure_new_hires_annual"] = _safe_numeric(out["avg_tenure_new_hires_annual"])
    out["n_new_hires_year"] = _safe_numeric(out["n_new_hires_year"])
    return out


def _build_recent_grads_tenure_panel(
    cfg: dict,
    y_min: int,
    y_max: int,
    positions_path: Path | None = None,
    users_path: Path | None = None,
    grad_lag_days: int = 365,
) -> pd.DataFrame:
    positions_path = Path(positions_path) if positions_path is not None else _resolve_selected_positions_path(cfg)
    users_path = Path(users_path) if users_path is not None else _resolve_users_path(cfg)
    if positions_path is None or users_path is None:
        return pd.DataFrame(
            columns=[
                "c",
                "t",
                "avg_tenure_recent_grads_annual",
                "n_recent_grads_year",
            ],
        )

    con = ddb.connect()
    try:
        sql = f"""
        WITH users AS (
            SELECT
                CAST(user_id AS BIGINT) AS user_id,
                MAX(TRY_CAST(ed_enddate AS DATE)) AS grad_date
            FROM read_parquet('{_escape_sql_path(users_path)}')
            WHERE user_id IS NOT NULL
              AND ed_enddate IS NOT NULL
            GROUP BY 1
        ),
        pos AS (
            SELECT
                CAST(rcid AS BIGINT) AS c,
                CAST(user_id AS BIGINT) AS user_id,
                CAST(startdate AS DATE) AS startdate
            FROM read_parquet('{_escape_sql_path(positions_path)}')
            WHERE user_id IS NOT NULL
              AND startdate IS NOT NULL
        ),
        first_grad_hire AS (
            SELECT
                p.user_id,
                p.c,
                p.startdate,
                ROW_NUMBER() OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.startdate ASC
                ) AS hire_order
            FROM pos p
            JOIN users u
              ON u.user_id = p.user_id
            WHERE p.startdate >= u.grad_date
              AND p.startdate <= u.grad_date + INTERVAL '{int(grad_lag_days)} days'
              AND EXTRACT(YEAR FROM p.startdate)::INT BETWEEN {int(y_min)} AND {int(y_max)}
        )
        SELECT
            CAST(c AS BIGINT) AS c,
            EXTRACT(YEAR FROM startdate)::INTEGER AS t,
            AVG(
                GREATEST(
                    0.0,
                    (MAKE_DATE(EXTRACT(YEAR FROM startdate)::INTEGER, 12, 31) - startdate)::DOUBLE / 365.25
                )
            ) AS avg_tenure_recent_grads_annual,
            COUNT(*)::BIGINT AS n_recent_grads_year
        FROM first_grad_hire
        WHERE hire_order = 1
        GROUP BY 1, 2
        """
        out = con.sql(sql).df()
    finally:
        con.close()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "c",
                "t",
                "avg_tenure_recent_grads_annual",
                "n_recent_grads_year",
            ],
        )
    out["c"] = _safe_numeric(out["c"]).astype("Int64")
    out["t"] = _safe_numeric(out["t"]).astype("Int64")
    out["avg_tenure_recent_grads_annual"] = _safe_numeric(out["avg_tenure_recent_grads_annual"])
    out["n_recent_grads_year"] = _safe_numeric(out["n_recent_grads_year"])
    return out


def _resolve_tenure_scope_panel(
    cfg: dict,
    workforce: pd.DataFrame,
    y_min: int,
    y_max: int,
    *,
    tenure_scope: str,
    tenure_col: str,
) -> tuple[pd.DataFrame, str, str]:
    if tenure_col != DEFAULT_TENURE_COL:
        if tenure_col not in workforce.columns:
            raise RuntimeError(f"wrds_company_year_workforce cache is missing '{tenure_col}'")
        workforce["tenure_scope"] = "custom"
        return workforce, tenure_col, "custom"

    if tenure_scope == TENURE_SCOPE_ALL:
        if tenure_col not in workforce.columns:
            raise RuntimeError(f"wrds_company_year_workforce cache is missing '{DEFAULT_TENURE_COL}'")
        workforce["tenure_scope"] = TENURE_SCOPE_ALL
        return workforce, DEFAULT_TENURE_COL, TENURE_SCOPE_ALL

    if tenure_scope == TENURE_SCOPE_NEW_HIRES:
        hires_panel = _build_new_hire_tenure_panel(cfg, y_min, y_max)
        if not hires_panel.empty:
            workforce = workforce.merge(
                hires_panel[["c", "t", "avg_tenure_new_hires_annual"]],
                on=["c", "t"],
                how="left",
            )
            workforce["tenure_scope"] = TENURE_SCOPE_NEW_HIRES
            return workforce, "avg_tenure_new_hires_annual", TENURE_SCOPE_NEW_HIRES
        print(
            "[warn] Unable to build new-hire tenure panel from cached positions. "
            "Falling back to recent-grad tenure."
        )

    if tenure_scope in (TENURE_SCOPE_NEW_HIRES, TENURE_SCOPE_RECENT_GRADS):
        grad_panel = _build_recent_grads_tenure_panel(cfg, y_min, y_max)
        if not grad_panel.empty:
            workforce = workforce.merge(
                grad_panel[["c", "t", "avg_tenure_recent_grads_annual"]],
                on=["c", "t"],
                how="left",
            )
            workforce["tenure_scope"] = TENURE_SCOPE_RECENT_GRADS
            return workforce, "avg_tenure_recent_grads_annual", TENURE_SCOPE_RECENT_GRADS
        print(
            "[warn] Unable to build recent-grad tenure panel from cached positions/users. "
            "Falling back to all-employee tenure."
        )

    if DEFAULT_TENURE_COL not in workforce.columns:
        raise RuntimeError(
            "wrds_company_year_workforce cache is missing avg_tenure_years_annual, "
            "and selected tenure scope could not be derived from caches."
        )
    workforce["tenure_scope"] = TENURE_SCOPE_ALL
    return workforce, DEFAULT_TENURE_COL, TENURE_SCOPE_ALL


def _load_tenure_opt_panel(
    cfg_path: Path,
    cfg: dict,
    args: argparse.Namespace,
    *,
    log_opt_hires: bool = False,
) -> tuple[pd.DataFrame, str, str]:
    y_min, y_max = _resolve_year_window(cfg, args)
    workforce = _load_wrds_workforce_cache_or_build(
        cfg=cfg,
        cfg_path=cfg_path,
        y_min=y_min,
        y_max=y_max,
        force_rebuild=bool(args.force_rebuild),
    )
    if (workforce["t"].between(y_min, y_max).sum() == 0) and not args.force_rebuild:
        workforce = _build_workforce_from_cached_positions(cfg, y_min, y_max)
    workforce, tenure_col, tenure_scope = _resolve_tenure_scope_panel(
        cfg,
        workforce,
        y_min,
        y_max,
        tenure_scope=getattr(args, "tenure_scope", TENURE_SCOPE_ALL),
        tenure_col=args.tenure_col,
    )
    opt_counts, _ = load_or_build_source_opt_counts(
        config_path=cfg_path,
        cfg=cfg,
        year_min=y_min,
        year_max=y_max,
        force_rebuild=bool(args.force_rebuild),
    )
    if args.opt_count_col not in opt_counts.columns:
        raise RuntimeError(f"source_opt_counts cache is missing '{args.opt_count_col}'")
    if "c" not in opt_counts.columns:
        raise RuntimeError("source_opt_counts cache is missing firm-id column 'c'")
    for col in (args.firm_size_col, tenure_col, args.n_hires_col):
        if col not in workforce.columns:
            raise RuntimeError(f"wrds_company_year_workforce cache is missing '{col}'")

    panel = workforce[["c", "t", args.firm_size_col, tenure_col, args.n_hires_col, "tenure_scope"]].copy()
    panel = panel.merge(
        opt_counts[["c", "t", args.opt_count_col]].rename(columns={args.opt_count_col: "opt_hires_count"}),
        on=["c", "t"],
        how="left",
    )

    panel["c"] = _safe_numeric(panel["c"]).astype("Int64")
    panel["t"] = _safe_numeric(panel["t"]).astype("Int64")
    panel[tenure_col] = _safe_numeric(panel[tenure_col])
    panel[args.firm_size_col] = _safe_numeric(panel[args.firm_size_col])
    panel["opt_hires_count"] = _safe_numeric(panel["opt_hires_count"]).fillna(0.0)
    panel[args.n_hires_col] = _safe_numeric(panel[args.n_hires_col])

    panel["firm_size_log"] = np.log1p(panel[args.firm_size_col])
    panel["size_q"] = pd.qcut(
        panel.loc[panel[args.firm_size_col].fillna(0) >= 0, "firm_size_log"],
        q=4,
        labels=False,
        duplicates="drop",
    ) + 1

    denominator = panel[args.n_hires_col].copy()
    panel["opt_rate"] = np.where(
        (denominator > 0) & pd.notna(panel[tenure_col]),
        panel["opt_hires_count"] / denominator,
        np.nan,
    )
    panel["opt_hires"] = panel["opt_hires_count"]
    if args.metric == "rate":
        panel["opt_metric"] = panel["opt_rate"]
    elif args.metric == "count":
        panel["opt_metric"] = panel["opt_hires"]
        if log_opt_hires:
            panel["opt_metric"] = np.log1p(panel["opt_metric"])
    else:
        raise ValueError("--metric must be 'rate' or 'count'")

    valid = panel[
        panel["opt_metric"].notna()
        & panel[tenure_col].notna()
        & panel[args.firm_size_col].notna()
        & panel["c"].notna()
        & panel["t"].notna()
    ].copy()
    valid["tenure_scope"] = tenure_scope
    valid["tenure_col"] = tenure_col
    return valid, tenure_col, tenure_scope


def _two_way_residualize(df: pd.DataFrame, value_col: str, id_col: str, year_col: str) -> pd.Series:
    x = _safe_numeric(df[value_col])
    firm_mean = x.groupby(df[id_col]).transform("mean")
    year_mean = x.groupby(df[year_col]).transform("mean")
    return x - firm_mean - year_mean + x.mean()


def _bin_stats(df: pd.DataFrame, x_col: str, y_col: str, n_bins: int) -> pd.DataFrame:
    work = df[[x_col, y_col]].copy()
    work = work.dropna()
    if work.empty:
        return pd.DataFrame()

    x = _safe_numeric(work[x_col])
    unique_bins = int(x.nunique())
    if unique_bins < 2:
        return pd.DataFrame()
    bins = max(2, min(int(n_bins), unique_bins))
    work["bin"] = (
        pd.qcut(x, q=bins, labels=False, duplicates="drop") + 1
    )
    work["bin"] = pd.to_numeric(work["bin"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame()

    out = (
        work.groupby("bin", as_index=False)
        .agg(x=(x_col, "mean"), y=(y_col, "mean"), n=(y_col, "size"))
        .sort_values("x")
    )
    return out


def _show_or_save_figure(
    fig: plt.Figure,
    out_path: Path,
    *,
    show_plots: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)

    if show_plots:
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            plt.show(block=False)

    if not show_plots:
        plt.close(fig)


def _save_bins_plot(
    bins: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    out_path: Path,
    *,
    show_plots: bool,
) -> None:
    if bins.empty:
        print(f"[warn] No data to plot for {out_path.name}; wrote nothing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bins["x"], bins["y"], s=36, alpha=0.8)
    ax.plot(bins["x"], bins["y"], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.margins(x=0.02)
    fig.tight_layout()
    _show_or_save_figure(fig, out_path=out_path, show_plots=show_plots)


def _plot_size_quartiles(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    q_col: str,
    n_bins: int,
    out_path: Path,
    x_label: str | None = None,
    y_label: str | None = None,
    *,
    show_plots: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if q_col not in df.columns:
        raise ValueError(f"Missing quartile column: {q_col}")
    valid = df[[q_col, x_col, y_col]].dropna()
    if valid.empty:
        return pd.DataFrame(), []

    groups = []
    q_values = sorted(valid[q_col].dropna().unique().tolist())
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 5))

    all_bins = []
    for idx, q in enumerate(q_values):
        subset = valid.loc[valid[q_col] == q]
        bins = _bin_stats(subset, x_col, y_col, n_bins)
        if bins.empty:
            continue
        bins = bins.copy()
        bins["size_q"] = int(q)
        all_bins.append(bins)

        color = colors[idx % len(colors)]
        label = f"Firm-size quartile {int(q)}"
        ax.scatter(bins["x"], bins["y"], s=30, alpha=0.85, color=color, label=label)
        ax.plot(bins["x"], bins["y"], linewidth=1.2, color=color)
        groups.append(label)

    if all_bins:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_title("Binsreg by firm-size quartile (FE-adjusted)")
        ax.set_xlabel(x_label or f"FE-adjusted {x_col}")
        ax.set_ylabel(y_label or f"FE-adjusted {y_col}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _show_or_save_figure(fig, out_path=out_path, show_plots=show_plots)
    elif not show_plots:
        plt.close(fig)

    if all_bins:
        return pd.concat(all_bins, ignore_index=True), groups
    return pd.DataFrame(), groups


def _plot_bins_by_year(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    year_col: str,
    n_bins: int,
    out_path: Path,
    *,
    show_plots: bool,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
) -> pd.DataFrame:
    if year_col not in df.columns:
        raise ValueError(f"Missing year column: {year_col}")
    valid = df[[year_col, x_col, y_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    years = sorted(valid[year_col].dropna().astype(int).unique().tolist())
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(9, 5))
    all_bins: list[pd.DataFrame] = []
    labels: list[str] = []

    for idx, year in enumerate(years):
        subset = valid[valid[year_col].astype(int) == int(year)]
        bins = _bin_stats(subset, x_col, y_col, n_bins)
        if bins.empty:
            continue
        bins = bins.copy()
        bins["t"] = int(year)
        all_bins.append(bins)

        color = colors[idx % len(colors)]
        label = f"Year {int(year)}"
        ax.scatter(bins["x"], bins["y"], s=22, alpha=0.8, color=color, label=label)
        ax.plot(bins["x"], bins["y"], linewidth=1.1, color=color)
        labels.append(label)

    if all_bins:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_title(title or "Binsreg by year")
        ax.set_xlabel(x_label or f"Residualized {x_col}")
        ax.set_ylabel(y_label or f"Residualized {y_col}")
        ax.grid(alpha=0.25)
        if len(labels) <= 24:
            ax.legend(fontsize=8)
        fig.tight_layout()
        _show_or_save_figure(fig, out_path=out_path, show_plots=show_plots)
    elif not show_plots:
        plt.close(fig)

    if all_bins:
        return pd.concat(all_bins, ignore_index=True)
    return pd.DataFrame()


def run_tenure_opt_binsreg(
    *,
    cfg_path: Path,
    cfg: dict[str, Any] | None = None,
    data_min_t: int | None = None,
    data_max_t: int | None = None,
    tenure_col: str = DEFAULT_TENURE_COL,
    opt_count_col: str = DEFAULT_OPT_HIRE_COL,
    n_hires_col: str = DEFAULT_N_HIRES_COL,
    firm_size_col: str = DEFAULT_FIRM_SIZE_COL,
    metric: str = "rate",
    tenure_scope: str = TENURE_SCOPE_ALL,
    log_opt_hires: bool = False,
    bins: int = 20,
    out_dir: Path | None = None,
    prefix: str = "tenure_opt",
    force_rebuild: bool = False,
    save_outputs: bool = True,
    by_year_bins: bool = False,
    show_plots: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Run binsreg-style tenure/OPT analysis and optionally write outputs."""
    cfg_full = cfg if cfg is not None else load_config(cfg_path)
    y_min, y_max = _resolve_year_window_values(cfg_full, data_min_t, data_max_t)

    if not save_outputs:
        # out_dir still needed for internal plotting helpers.
        out_dir = Path("/tmp") / "tenure_opt_binsreg" if out_dir is None else out_dir
    else:
        out_dir = (
            out_dir
            if out_dir is not None
            else Path(get_cfg_section(cfg_full, "paths").get("company_shift_share_out_dir", str(Path.cwd())))
            / "tenure_opt_binsreg"
        )

    if show_plots is None:
        show_plots = _is_interactive()

    class _Args(argparse.Namespace):
        pass

    args = _Args(
        data_min_t=y_min,
        data_max_t=y_max,
        tenure_col=tenure_col,
        opt_count_col=opt_count_col,
        n_hires_col=n_hires_col,
        firm_size_col=firm_size_col,
        tenure_scope=tenure_scope,
        metric=metric,
        bins=bins,
        force_rebuild=force_rebuild,
        log_opt_hires=bool(log_opt_hires),
    )

    df, tenure_col_used, tenure_scope_used = _load_tenure_opt_panel(
        cfg_path,
        cfg_full,
        args,
        log_opt_hires=bool(log_opt_hires),
    )
    df = df[(df["t"] >= y_min) & (df["t"] <= y_max)].copy()
    if df.empty:
        raise RuntimeError("No valid rows found for requested year window.")

    if metric == "count" and log_opt_hires:
        opt_label = "log(1 + OPT hires)"
    elif metric == "count":
        opt_label = "OPT hires"
    else:
        opt_label = "OPT hiring rate"

    raw_bins = _bin_stats(df, "opt_metric", tenure_col_used, args.bins)
    df["tenure_fe"] = _two_way_residualize(df, tenure_col_used, "c", "t")
    df["opt_metric_fe"] = _two_way_residualize(df, "opt_metric", "c", "t")
    fe_bins = _bin_stats(df, "opt_metric_fe", "tenure_fe", args.bins)
    quartile_bins, _ = _plot_size_quartiles(
        df,
        x_col="opt_metric_fe",
        y_col="tenure_fe",
        q_col="size_q",
        n_bins=args.bins,
        out_path=out_dir / f"{prefix}_binsreg_fe_by_sizeq.png",
        x_label=f"FE-adjusted {opt_label}",
        y_label=f"FE-adjusted {tenure_col_used}",
        show_plots=bool(show_plots),
    )
    year_raw_bins = pd.DataFrame()
    year_fe_bins = pd.DataFrame()
    if by_year_bins:
        year_raw_bins = _plot_bins_by_year(
            df,
            x_col="opt_metric",
            y_col=tenure_col_used,
            year_col="t",
            n_bins=args.bins,
            out_path=out_dir / f"{prefix}_binsreg_by_year_raw.png",
            show_plots=bool(show_plots),
            x_label=opt_label,
            y_label=tenure_col_used,
            title="Binsreg by year (raw scale)",
        )
        year_fe_bins = _plot_bins_by_year(
            df,
            x_col="opt_metric_fe",
            y_col="tenure_fe",
            year_col="t",
            n_bins=args.bins,
            out_path=out_dir / f"{prefix}_binsreg_by_year_fe.png",
            show_plots=bool(show_plots),
            x_label=f"FE-adjusted {opt_label}",
            y_label=f"FE-adjusted {tenure_col_used}",
            title="Binsreg by year (FE-adjusted)",
        )

    if save_outputs:
        _save_bins_plot(
            raw_bins,
            x_label=opt_label,
            y_label=tenure_col_used,
            title=f"Binscatter: {opt_label} vs {tenure_col_used}",
            out_path=out_dir / f"{prefix}_binscatter_raw.png",
            show_plots=bool(show_plots),
        )
        _save_bins_plot(
            fe_bins,
            x_label=f"FE-adjusted {opt_label}",
            y_label=f"FE-adjusted {tenure_col_used}",
            title="Binsreg (year + firm FE residualized)",
            out_path=out_dir / f"{prefix}_binsreg_fe.png",
            show_plots=bool(show_plots),
        )
        if not quartile_bins.empty:
            # _plot_size_quartiles already wrote its own plot as a side effect.
            pass
        if by_year_bins and not year_raw_bins.empty:
            # _plot_bins_by_year already wrote the raw-by-year plot as a side effect.
            pass
        if by_year_bins and not year_fe_bins.empty:
            # _plot_bins_by_year already wrote the FE-by-year plot as a side effect.
            pass

        raw_bins_out = out_dir / f"{prefix}_raw_bins.csv"
        fe_bins_out = out_dir / f"{prefix}_fe_bins.csv"
        quartile_bins_out = out_dir / f"{prefix}_fe_sizeq_bins.csv"
        raw_by_year_bins_out = out_dir / f"{prefix}_raw_by_year_bins.csv"
        fe_by_year_bins_out = out_dir / f"{prefix}_fe_by_year_bins.csv"
        raw_bins.to_csv(raw_bins_out, index=False)
        fe_bins.to_csv(fe_bins_out, index=False)
        if not quartile_bins.empty:
            quartile_bins.to_csv(quartile_bins_out, index=False)
        if by_year_bins and not year_raw_bins.empty:
            year_raw_bins.to_csv(raw_by_year_bins_out, index=False)
        if by_year_bins and not year_fe_bins.empty:
            year_fe_bins.to_csv(fe_by_year_bins_out, index=False)
        print(f"[done] wrote plots and bin tables to {out_dir}")

    return {
        "raw_bins": raw_bins,
        "fe_bins": fe_bins,
        "quartile_bins": quartile_bins,
        "year_raw_bins": year_raw_bins,
        "year_fe_bins": year_fe_bins,
        "panel": df,
        "tenure_col": tenure_col_used,
        "tenure_scope": tenure_scope_used,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create tenure-vs-OPT binscatter/binsreg figures.",
        allow_abbrev=False,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Config path (default: configs/company_shift_share_exposure_event_study.yaml).",
    )
    p.add_argument("--data-min-t", type=int, default=None, help="Panel start year.")
    p.add_argument("--data-max-t", type=int, default=None, help="Panel end year.")
    p.add_argument("--tenure-col", type=str, default=DEFAULT_TENURE_COL, help="Tenure column from workforce cache.")
    p.add_argument(
        "--opt-count-col",
        type=str,
        default=DEFAULT_OPT_HIRE_COL,
        help="OPT hires count column to use from source_opt_counts.",
    )
    p.add_argument(
        "--n-hires-col",
        type=str,
        default=DEFAULT_N_HIRES_COL,
        help="Denominator column for OPT hiring rate.",
    )
    p.add_argument(
        "--firm-size-col",
        type=str,
        default=DEFAULT_FIRM_SIZE_COL,
        help="Firm size column for size quartiles.",
    )
    p.add_argument(
        "--metric",
        choices=("rate", "count"),
        default="rate",
        help="Whether binsreg should use OPT count (any_opt_hires) or rate.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Rebuild workforce/opt-count caches for the requested year window. "
            "Only use this if WRDS credentials are available."
        ),
    )
    p.add_argument("--bins", type=int, default=20, help="Number of bins.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    p.add_argument("--prefix", type=str, default="tenure_opt", help="Output filename prefix.")
    p.add_argument(
        "--by-year-bins",
        action="store_true",
        default=None,
        help="Create additional FE/level binsreg curves separately by year.",
    )
    p.add_argument(
        "--tenure-scope",
        choices=TENURE_SCOPE_CHOICES,
        default=TENURE_SCOPE_ALL,
        help=(
            "Tenure metric to use: all (all employees), new_hires (first start per firm), "
            "or recent_grads (first qualifying post-grad role within 365 days)."
        ),
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        default=None,
        help=(
            "Show figures in the current interactive output (in addition to saving). "
            "Defaults to true in IPython/Jupyter."
        ),
    )
    p.add_argument(
        "--log-opt-hires",
        action="store_true",
        default=False,
        help="Use log(1 + OPT hires count) for count-based metric mode.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, unknown = p.parse_known_args(args=raw_argv)
    if unknown:
        if _is_interactive():
            print(f"[warn] Ignoring extra CLI args from interactive launch: {unknown}")
        else:
            raise SystemExit(f"unrecognized arguments: {' '.join(unknown)}")
    _apply_script_config_defaults(args, load_config(args.config), raw_argv)
    if args.show_plots is None:
        args.show_plots = _is_interactive()
    if args.by_year_bins is None:
        args.by_year_bins = _is_interactive()
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv=argv)
    run_tenure_opt_binsreg(
        cfg_path=args.config,
        data_min_t=args.data_min_t,
        data_max_t=args.data_max_t,
        tenure_col=args.tenure_col,
        tenure_scope=args.tenure_scope,
        opt_count_col=args.opt_count_col,
        n_hires_col=args.n_hires_col,
        firm_size_col=args.firm_size_col,
        metric=args.metric,
        log_opt_hires=args.log_opt_hires,
        bins=args.bins,
        out_dir=args.out_dir,
        prefix=args.prefix,
        force_rebuild=args.force_rebuild,
        save_outputs=True,
        by_year_bins=args.by_year_bins,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
