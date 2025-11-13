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
REQUIRED_BALANCED_COLS = ["x_it", "z_it", "n"] + Y_COLS


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
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_samp} WHERE gradyr IS NOT NULL",
    )
    us_higher_ed = _scalar(con, f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_us_educ_clean}")
    geo_matched = _scalar(
        con,
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_user} WHERE us_cluster_geo_city_id IS NOT NULL",
    )
    us_rsid = _scalar(con, f"SELECT COUNT(DISTINCT user_id) FROM {tables.ihma_user} WHERE us_rsid IS NOT NULL")
    worked_in_us = _scalar(
        con,
        f"SELECT COUNT(DISTINCT user_id) FROM {tables.pos_clean_msa} WHERE pos_geo_geoname_id IS NOT NULL AND pos_geo_country_name = 'United States'",
    )

    print(f"Share with US higher ed: {us_higher_ed / base_sample:.2%}")
    print(f"Share of US higher-ed users with geoname match: {geo_matched / us_rsid:.2%}")
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
            SELECT i, k, n_cbsa
            FROM {tables.cbsa_shares}
            WHERE n_cbsa IS NOT NULL AND n_cbsa > 0
            """
        ).df()
        if not share_counts.empty:
            log_vals = np.log10(share_counts["n_cbsa"].astype(float))
            plt.figure(figsize=(6, 4))
            plt.hist(log_vals, bins=30, edgecolor="black")
            plt.xlabel("log10 Users per (i,k)")
            plt.ylabel("Frequency")
            plt.title("Distribution of users contributing to location shares")
            plt.tight_layout()
            plt.show()
            threshold = share_counts["n_cbsa"].quantile(0.99)
            high = share_counts[share_counts["n_cbsa"] >= threshold].sort_values("n_cbsa", ascending=False).head(10)
            if not high.empty:
                print("Top (i,k) share cells by user count:")
                print(high.to_string(index=False))


def fetch_analysis_panel(con: ddb.DuckDBPyConnection, tables: IHMACleanTables) -> pd.DataFrame:
    y_select = ",\n            ".join(f"dep.{col}" for col in Y_COLS)
    query = f"""
        SELECT
            indep.i,
            indep.t,
            indep.origin_group_name,
            indep.x_it,
            indep.n,
            instr.z_it,
            {y_select}
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


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    records = []
    panel = panel.copy()
    panel["const"] = 1

    for idx, col in enumerate(Y_COLS, start=1):
        ols = sm.OLS(panel[col], panel[["const", "x_it", "n"]]).fit()
        rf = sm.OLS(panel[col], panel[["const", "z_it", "n"]]).fit()
        iv = IV2SLS.from_formula(f"{col} ~ 1 + [x_it ~ z_it] + n", data=panel).fit(cov_type="robust")

        records.append(
            {
                "y_index": idx,
                "ols_b": ols.params["x_it"],
                "ols_se": ols.bse["x_it"],
                "rf_b": rf.params["z_it"],
                "rf_se": rf.bse["z_it"],
                "iv_b": iv.params["x_it"],
                "iv_se": iv.std_errors["x_it"],
            }
        )

    return pd.DataFrame.from_records(records)


def plot_coefficients(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(results["y_index"], results["ols_b"], yerr=1.96 * results["ols_se"], fmt="o-", capsize=3, label="OLS")
    ax.errorbar(results["y_index"], results["iv_b"], yerr=1.96 * results["iv_se"], fmt="s--", capsize=3, label="IV")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Years after graduation")
    ax.set_ylabel("Effect of US postgrad on cohort average earnings")
    ax.set_title("Non-US Undergrads")
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_single_horizon_diagnostics(panel: pd.DataFrame, horizon: int = 6) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    y_col = f"y_it{horizon}"
    if y_col not in panel.columns:
        raise ValueError(f"{y_col} not found in panel.")

    work = panel.copy()
    work["const"] = 1

    ols = sm.OLS(work[y_col], work[["const", "x_it", "n"]]).fit()
    first_stage = sm.OLS(work["x_it"], work[["const", "z_it"]]).fit()
    reduced = sm.OLS(work[y_col], work[["const", "z_it", "n"]]).fit()
    iv = IV2SLS.from_formula(f"{y_col} ~ 1 + [x_it ~ z_it] + n", data=work).fit(cov_type="robust")

    print(f"Horizon {horizon}: OLS beta={ols.params['x_it']:.4f}, IV beta={iv.params['x_it']:.4f}")
    return {"ols": ols, "first_stage": first_stage, "reduced": reduced, "iv": iv}


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = ddb.connect()
    tables = prepare_clean_data(con, force_resave=True)
    describe_sample(con, tables, plot_distributions=True)

    panel = fetch_analysis_panel(con, tables)
    filtered = panel[(panel["t"] > 2007) & (panel["t"] <= 2020)].copy()
    balanced = balanced_complete(filtered[filtered["t"] <= 2017], req_vars=REQUIRED_BALANCED_COLS)

    print(f"Balanced panel: {balanced['i'].nunique()} units × {balanced['t'].nunique()} periods")

    regression_results = run_regressions(balanced)
    plot_coefficients(regression_results)
    run_single_horizon_diagnostics(balanced, horizon=6)

    # return balanced, regression_results


# if __name__ == "__main__":
#     main()

con = ddb.connect()
tables = prepare_clean_data(con, force_resave=True, us_group_by="us_cluster_ipeds_ids", origin_group_by = "origin_rsid")
describe_sample(con, tables, plot_distributions=True)
