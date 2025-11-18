"""Exploratory shift-share analysis for the IHMA project."""

from __future__ import annotations

import duckdb as ddb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from ihma_clean import IHMACleanTables, prepare_clean_data

Y_COLS = [f"y_it{i}" for i in range(1, 9)]
Y_POSTGRAD_COLS = [f"y_it_pg{i}" for i in range(1, len(Y_COLS) + 1)]
Y_NO_POSTGRAD_COLS = [f"y_it_no_pg{i}" for i in range(1, len(Y_COLS) + 1)]
Y_ALT_COLS = [f"y_it_alt{i}" for i in range(1, len(Y_COLS) + 1)]
OUTCOME_GROUPS = {
    "all": Y_COLS,
    "postgrad": Y_POSTGRAD_COLS,
    "nonpostgrad": Y_NO_POSTGRAD_COLS,
}
REQUIRED_BALANCED_COLS = ["x_it", "z_it", "n"] + OUTCOME_GROUPS["all"]


def balanced_complete(df: pd.DataFrame, id_col: str = "i", time_col: str = "t", req_vars: list[str] | None = None) -> pd.DataFrame:
    """Restrict to a balanced panel over the requested variables."""

    if req_vars is None:
        req_vars = []
    frame = df[[id_col, time_col] + req_vars].copy()

    frame[time_col] = frame[time_col].astype(int)
    frame = frame.sort_values([id_col, time_col]).drop_duplicates([id_col, time_col], keep="last")

    units = frame[id_col].unique()
    periods = np.sort(frame[time_col].unique())
    full_index = pd.MultiIndex.from_product([units, periods], names=[id_col, time_col])

    balanced = frame.set_index([id_col, time_col]).reindex(full_index).reset_index()
    good_units = balanced.groupby(id_col)[req_vars].apply(lambda g: g.notna().all().all())
    keep_ids = good_units[good_units].index
    balanced = balanced[balanced[id_col].isin(keep_ids)].reset_index(drop=True)
    balanced = balanced.dropna(subset=req_vars)
    return balanced


def _get_outcome_cols(group: str = "all", *, alt: bool = False) -> list[str]:
    if alt:
        return Y_ALT_COLS
    try:
        return OUTCOME_GROUPS[group]
    except KeyError as exc:  # pragma: no cover - defensive for interactive use
        raise ValueError(f"Unknown outcome group '{group}'. Expected one of {sorted(OUTCOME_GROUPS)}.") from exc


def _get_outcome_column_for_horizon(horizon: int, *, group: str = "all", alt: bool = False) -> str:
    cols = _get_outcome_cols(group, alt=alt)
    if horizon < 1 or horizon > len(cols):
        raise ValueError(f"Horizon {horizon} is out of bounds for group '{group}'.")
    return cols[horizon - 1]


def _scalar(con: ddb.DuckDBPyConnection, query: str) -> float:
    return con.sql(query).fetchone()[0]


def describe_sample(
    con: ddb.DuckDBPyConnection,
    tables: IHMACleanTables,
    plot_distributions: bool = False,
    balanced_panel: pd.DataFrame | None = None,
) -> None:
    base_sample = _scalar(
        con,
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_user} WHERE origin_gradyr IS NOT NULL",
    )
    geo_matched = _scalar(
        con,
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_user} WHERE NOT us_group_id = -1",
    )
    us_rsid = _scalar(con, f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_user} WHERE us_rsid IS NOT NULL")
    worked_in_us = _scalar(
        con,
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.pos_clean_msa} WHERE pos_geo_geoname_id IS NOT NULL AND pos_geo_country_name = 'United States'",
    )

    print(f"Share with US higher ed: {us_rsid / base_sample:.2%}")
    print(f"Share of US higher-ed users with IPEDS match: {geo_matched / us_rsid:.2%}")
    print(f"Share working in US post-grad: {worked_in_us / base_sample:.2%}")

    if plot_distributions and balanced_panel is not None:
        panel_counts = balanced_panel[["i", "t", "n"]].dropna()
        if not panel_counts.empty:
            log_vals = np.log10(panel_counts["n"].astype(float).replace(0, np.nan).dropna())
            plt.figure(figsize=(6, 4))
            plt.hist(log_vals, bins=30, edgecolor="black")
            plt.xlabel("log10 Users per (i,t)")
            plt.ylabel("Frequency")
            plt.title("Distribution of users contributing to x_it")
            plt.tight_layout()
            plt.show()
            threshold = panel_counts["n"].quantile(0.99)
            high = panel_counts[panel_counts["n"] >= threshold].sort_values("n", ascending=False).head(10)
            if not high.empty:
                print("Top (i,t) cells by user count:")
                print(high.to_string(index=False))

        share_counts = con.sql(
            f"""
            SELECT i, k, n_ikt
            FROM {tables.educ_cbsa_shares}
            WHERE n_ikt IS NOT NULL AND n_ikt > 0
            """
        ).df()
        if not share_counts.empty:
            log_vals = np.log10(share_counts["n_ikt"].astype(float))
            plt.figure(figsize=(6, 4))
            plt.hist(log_vals, bins=30, edgecolor="black")
            plt.xlabel("log10 Users per (i,k)")
            plt.ylabel("Frequency")
            plt.title("Distribution of users contributing to location shares")
            plt.tight_layout()
            plt.show()
            threshold = share_counts["n_ikt"].quantile(0.99)
            high = share_counts[share_counts["n_ikt"] >= threshold].sort_values("n_ikt", ascending=False).head(10)
            if not high.empty:
                print("Top (i,k) share cells by user count:")
                print(high.to_string(index=False))


def fetch_analysis_panel(con: ddb.DuckDBPyConnection, tables: IHMACleanTables) -> pd.DataFrame:
    y_select = ",\n            ".join(f"dep.{col}" for col in Y_COLS)
    y_postgrad_select = ",\n            ".join(f"dep.{col}" for col in Y_POSTGRAD_COLS)
    y_no_postgrad_select = ",\n            ".join(f"dep.{col}" for col in Y_NO_POSTGRAD_COLS)
    y_alt_select = ",\n            ".join(f"dep.{col}" for col in Y_ALT_COLS)
    query = f"""
        SELECT
            indep.i,
            indep.t,
            indep.origin_group_name,
            indep.x_it,
            indep.n,
            instr.z_it,
            instr.z_it_all,
            {y_select},
            {y_postgrad_select},
            {y_no_postgrad_select},
            {y_alt_select}
        FROM {tables.indep_constr} AS indep
        JOIN {tables.dep_constr} AS dep
            ON indep.i = dep.i
           AND indep.t = dep.t
        JOIN {tables.instrument_panel} AS instr
            ON indep.i = instr.i
           AND indep.t = instr.t
        ORDER BY indep.i, indep.t
    """
    return con.sql(query).df()


def instrument_contribution_table(
    con: ddb.DuckDBPyConnection,
    tables: IHMACleanTables,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank US universities by their average absolute contribution to z_it."""

    query = f"""
        WITH component_stats AS (
            SELECT
                k,
                AVG(ABS(z_it)) AS avg_abs_contribution,
                AVG(z_it) AS avg_signed_contribution,
                SUM(z_it) AS total_signed_contribution,
                COUNT(*) AS contributing_cells
            FROM {tables.instrument_components}
            GROUP BY k
        ),
        educ AS (
            SELECT
                k,
                MAX(location_name) AS location_name
            FROM {tables.educ_cbsa_shares_main}
            GROUP BY k
        )
        SELECT
            stats.k,
            COALESCE(educ.location_name, '[unknown]') AS us_institution,
            stats.avg_abs_contribution,
            stats.avg_signed_contribution,
            stats.total_signed_contribution,
            stats.contributing_cells
        FROM component_stats AS stats
        LEFT JOIN educ ON stats.k = educ.k
        ORDER BY stats.avg_abs_contribution DESC
        LIMIT {int(top_n)}
    """
    table = con.sql(query).df()
    return table.fillna({"us_institution": "[unknown]"})


def leave_one_k_out_sensitivity(
    panel: pd.DataFrame,
    base_results: pd.DataFrame,
    con: ddb.DuckDBPyConnection,
    tables: IHMACleanTables,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute IV coefficients while omitting one US university at a time."""

    ranked = instrument_contribution_table(con, tables, top_n=top_k)
    if ranked.empty:
        return pd.DataFrame(), pd.DataFrame()

    needed_ks = ranked["k"].dropna().tolist()
    if not needed_ks:
        return pd.DataFrame(), pd.DataFrame()

    filter_df = pd.DataFrame({"k": needed_ks})
    con.register("leave_one_out_k_filter", filter_df)
    try:
        components = con.sql(
            f"""
            SELECT
                ic.i,
                ic.t,
                ic.k,
                SUM(ic.z_it) AS z_it
            FROM {tables.instrument_components} AS ic
            JOIN leave_one_out_k_filter AS filt
              ON ic.k = filt.k
            GROUP BY 1, 2, 3
            """
        ).df()
    finally:
        con.unregister("leave_one_out_k_filter")

    if components.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = {k: df for k, df in components.groupby("k")}
    base_iv = base_results.set_index("y_index")["iv_b"]
    records: list[dict[str, object]] = []

    for _, entry in ranked.iterrows():
        key = entry["k"]
        if key not in grouped:
            continue
        adjustments = grouped[key].groupby(["i", "t"])["z_it"].sum().reset_index().rename(columns={"z_it": "z_remove"})

        variant = panel.merge(adjustments, on=["i", "t"], how="left")
        variant["z_remove"] = variant["z_remove"].fillna(0)
        variant["z_it"] = variant["z_it"] - variant["z_remove"]
        variant = variant.drop(columns=["z_remove"])

        variant_results = run_regressions(variant)
        for _, res in variant_results.iterrows():
            base_coef = float(base_iv.loc[res["y_index"]])
            records.append(
                {
                    "k": key,
                    "us_institution": entry["us_institution"],
                    "y_index": res["y_index"],
                    "iv_b": res["iv_b"],
                    "iv_delta": res["iv_b"] - base_coef,
                }
            )

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    loo_results = pd.DataFrame.from_records(records)
    summary_records: list[dict[str, object]] = []
    for y_index, subset in loo_results.groupby("y_index"):
        summary_records.append(
            {
                "y_index": y_index,
                "baseline_iv": float(base_iv.loc[y_index]),
                "min_iv": subset["iv_b"].min(),
                "max_iv": subset["iv_b"].max(),
                "std_iv": subset["iv_b"].std(ddof=0),
                "min_institution": subset.loc[subset["iv_b"].idxmin(), "us_institution"],
                "max_institution": subset.loc[subset["iv_b"].idxmax(), "us_institution"],
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    return loo_results, summary


def run_regressions(
    panel: pd.DataFrame,
    instr: str = "z_it",
    y_alt: bool = False,
    earnings_group: str = "all",
    include_fixed_effects: bool = True,
) -> pd.DataFrame:
    records = []
    panel = panel.copy()
    fe_term = " + C(i) + C(t)" if include_fixed_effects else ""
    n_contr = "+ n" if not y_alt else ""

    outcome_cols = _get_outcome_cols(earnings_group, alt=y_alt)
    for idx, col in enumerate(outcome_cols, start=1):
        ols = sm.OLS.from_formula(f"{col} ~ x_it {n_contr}{fe_term}", data=panel).fit()
        rf = sm.OLS.from_formula(f"{col} ~ {instr} {n_contr}{fe_term}", data=panel).fit()
        iv = IV2SLS.from_formula(
            f"{col} ~ 1 {n_contr}{fe_term} + [x_it ~ {instr}]",
            data=panel,
        ).fit(cov_type="robust")

        records.append(
            {
                "y_index": idx,
                "ols_b": ols.params["x_it"],
                "ols_se": ols.bse["x_it"],
                "rf_b": rf.params[instr],
                "rf_se": rf.bse[instr],
                "iv_b": iv.params["x_it"],
                "iv_se": iv.std_errors["x_it"],
            }
        )

    return pd.DataFrame.from_records(records)


def plot_coefficients(results: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].errorbar(
        results["y_index"],
        results["ols_b"],
        yerr=1.96 * results["ols_se"],
        fmt="o-",
        capsize=3,
        color="tab:blue",
    )
    axes[0].axhline(0, color="grey", lw=1)
    axes[0].set_ylabel("OLS effect on earnings")
    axes[0].set_title("OLS Estimates")

    axes[1].errorbar(
        results["y_index"],
        results["iv_b"],
        yerr=1.96 * results["iv_se"],
        fmt="s--",
        capsize=3,
        color="tab:orange",
    )
    axes[1].axhline(0, color="grey", lw=1)
    axes[1].set_xlabel("Years after graduation")
    axes[1].set_ylabel("IV effect on earnings")
    axes[1].set_title("IV Estimates")

    fig.suptitle("Non-US Undergrads", y=0.98)
    fig.tight_layout()
    plt.show()
    return axes[0], axes[1]


def run_single_horizon_diagnostics(
    panel: pd.DataFrame,
    horizon: int = 6,
    include_fixed_effects: bool = True,
    y_alt: bool = False,
    earnings_group: str = "all",
) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    y_col = _get_outcome_column_for_horizon(horizon, group=earnings_group, alt=y_alt)
    if y_col not in panel.columns:
        raise ValueError(f"{y_col} not found in panel.")

    work = panel.copy()
    work_fe = work.set_index(["i", "t"])
    fe_term = " + C(i) + C(t)" if include_fixed_effects else ""
    n_contr = "+ n" if not y_alt else ""

    ols = sm.OLS.from_formula(f"{y_col} ~ x_it {n_contr}{fe_term}", data=work).fit()
    first_stage = sm.OLS.from_formula(f"x_it ~ z_it{fe_term}", data=work).fit()
    reduced = sm.OLS.from_formula(f"{y_col} ~ z_it {n_contr}{fe_term}", data=work).fit()
    iv = IV2SLS.from_formula(
        f"{y_col} ~ 1 {n_contr}{fe_term} + [x_it ~ z_it]",
        data=work,
    ).fit(cov_type="robust")

    print(f"Horizon {horizon}: OLS beta={ols.params['x_it']:.4f}, IV beta={iv.params['x_it']:.4f}")
    return {"ols": ols, "first_stage": first_stage, "reduced": reduced, "iv": iv}


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = ddb.connect()
    tables = prepare_clean_data(con, force_resave=True)
    describe_sample(con, tables, plot_distributions=True)

    panel = fetch_analysis_panel(con, tables)
    filtered = panel[(panel["t"] > 2012) & (panel["t"] <= 2020)].copy()
    balanced = balanced_complete(filtered[filtered["t"] <= 2017], req_vars=REQUIRED_BALANCED_COLS)

    print(f"Balanced panel: {balanced['i'].nunique()} units × {balanced['t'].nunique()} periods")

    regression_results = run_regressions(balanced)
    plot_coefficients(regression_results)
    run_single_horizon_diagnostics(balanced, horizon=6)
    contrib_table = instrument_contribution_table(con, tables, top_n=20)
    print("Top instrument contributors (avg |contribution| to z_it):")
    print(contrib_table.to_string(index=False))
    loo_results, loo_summary = leave_one_k_out_sensitivity(
        balanced,
        regression_results,
        con,
        tables,
        top_k=10,
    )
    if not loo_summary.empty:
        print("Leave-one-k-out IV sensitivity (top contributors):")
        print(loo_summary.to_string(index=False))

    # return balanced, regression_results


# if __name__ == "__main__":
#     main()

con = ddb.connect()
tables = prepare_clean_data(con, force_resave=True, save_intermediate=True,require_us_work_after_grad=False, us_group_by="us_cluster_ipeds_ids", origin_group_by="origin_rsid")
panel = fetch_analysis_panel(con, tables)

def do_analysis(
    panel,
    con,
    startt: int = 2007,
    endt: int = 2017,
    alt_instr: bool = False,
    alt_y: bool = False,
    include_fixed_effects: bool = True,
    earnings_group: str = "all",
):
    filtered = panel[(panel["t"] > startt) & (panel["t"] <= endt)].copy()
    required_y = _get_outcome_cols(earnings_group, alt=alt_y)
    req_vars = ["x_it", "z_it" if not alt_instr else "z_it_all", "n"] + required_y
    balanced = balanced_complete(filtered[filtered["t"] <= endt], req_vars=req_vars)

    print(f"Balanced panel: {balanced['i'].nunique()} units × {balanced['t'].nunique()} periods")
    
    #describe_sample(con, tables, plot_distributions=True, balanced_panel=balanced)

    regression_results = run_regressions(
        balanced,
        instr="z_it_all" if alt_instr else "z_it",
        y_alt=alt_y,
        earnings_group=earnings_group,
        include_fixed_effects=include_fixed_effects,
    )
    plots = plot_coefficients(regression_results)
    single_horizon = run_single_horizon_diagnostics(
        balanced,
        horizon=6,
        include_fixed_effects=include_fixed_effects,
        y_alt=alt_y,
        earnings_group=earnings_group,
    )
        
    # instrument_contributions = instrument_contribution_table(con, tables, top_n=20)
    # print("Top instrument contributors (avg |contribution| to z_it):")
    # print(instrument_contributions.to_string(index=False))

    return balanced, regression_results, plots, single_horizon

balanced, regression_results, plots, single_horizon = do_analysis(panel, con, startt=2010, endt=2017, alt_instr=False, alt_y=False, include_fixed_effects=True, earnings_group="all")

# leave_one_out_results, leave_one_out_summary = leave_one_k_out_sensitivity(
#     balanced,
#     regression_results,
#     con,
#     tables,
#     top_k=10,
# )
# if not leave_one_out_summary.empty:
#     print("Leave-one-k-out IV sensitivity (top contributors):")
#     print(leave_one_out_summary.to_string(index=False))

# descriptive: total change in us going and us working rates over time (graph)
us_going = con.sql(f"""
    SELECT
        t,
        100*SUM(x_it) AS total_us_going_time_100,
        SUM(n) AS total_n,
        SUM(x_it) / SUM(n) AS us_going_rate
    FROM {tables.indep_constr}
    WHERE x_it IS NOT NULL AND n IS NOT NULL AND t > 2000 AND t < 2026
    GROUP BY t
    ORDER BY t
""").df()
plt.figure(figsize=(7,4))
plt.plot(us_going["t"], us_going["total_us_going_time_100"], marker='o')
plt.plot(us_going["t"], us_going["total_n"], marker='x')
plt.title("US Going Rate Over Time")
plt.xlabel("Graduation Year")
plt.ylabel("US Ever Go Rate times 100 and Total N")
plt.legend(["US Going times 100", "Total N"])
plt.grid()
plt.tight_layout()
plt.show()

us_working = con.sql(f"""
    SELECT dep.t,
        100*SUM(CASE WHEN y_it_pg8 IS NOT NULL THEN y_it_pg8 ELSE 0 END) AS total_us_working_time_100,
        SUM(n) AS total_n,
        SUM(CASE WHEN y_it_pg8 IS NOT NULL THEN y_it_pg8 ELSE 0 END) / SUM(n) AS us_working_rate
    FROM {tables.dep_constr} AS dep
    JOIN {tables.indep_constr} AS indep
        ON dep.i = indep.i
       AND dep.t = indep.t
    WHERE indep.x_it IS NOT NULL AND indep.n IS NOT NULL AND dep.y_it_pg8 IS NOT NULL AND indep.t > 2000 AND indep.t < 2026
    GROUP BY dep.t 
    ORDER BY dep.t
""").df()
                     

plt.figure(figsize=(7,4))
plt.plot(us_working["t"], us_working["total_us_working_time_100"], marker='o')
plt.plot(us_working["t"], us_working["total_n"], marker='x')
plt.title("US Working Rate Over Time")
plt.xlabel("Graduation Year")
plt.ylabel("US Ever Work Rate times 100 and Total N")
plt.legend(["US Working times 100", "Total N"])
plt.grid()
plt.tight_layout()
plt.show()