from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import StrMethodFormatter


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
USER_ROOT = REPO_ROOT.parent

DEFAULT_INPUT_PATH = (
    USER_ROOT
    / "data"
    / "int"
    / "f1_indiv_merge"
    / "01_f1_foia_clean"
    / "foia_person_panel_apr2026v1.parquet"
)
DEFAULT_OUTPUT_DIR = CODE_ROOT / "output" / "f1_indiv_merge"
DEFAULT_COUNTS_OUTPUT_PNG = (
    DEFAULT_OUTPUT_DIR / "foia_cohort_count_comparison_apr2026v1.png"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the share of FOIA person_ids with any OPT activity by "
            "program-end (graduation) year."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to FOIA person panel parquet (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=date.today().isoformat(),
        help="Only include rows with program_end_date on or before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-persons",
        type=int,
        default=1000,
        help="Drop graduation years with fewer than this many distinct person_ids.",
    )
    parser.add_argument(
        "--censor-start-year",
        type=int,
        default=None,
        help=(
            "Optional first graduation year to visually mark as right-censored. "
            "Default is as_of_year - 2."
        ),
    )
    parser.add_argument(
        "--x-min-year",
        type=int,
        default=None,
        help="Optional lower bound for included graduation years.",
    )
    parser.add_argument(
        "--x-max-year",
        type=int,
        default=None,
        help="Optional upper bound for included graduation years.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "foia_opt_share_by_grad_year_apr2026v1.png",
        help="Path for the output plot PNG.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "foia_opt_share_by_grad_year_apr2026v1.csv",
        help="Path for the output summary CSV.",
    )
    parser.add_argument(
        "--counts-output-png",
        type=Path,
        default=DEFAULT_COUNTS_OUTPUT_PNG,
        help="Path for the cohort-count comparison PNG.",
    )
    return parser.parse_args()


def build_opt_share_table(
    *,
    input_path: Path,
    as_of_date: str,
    min_persons: int,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing FOIA input parquet: {input_path}")

    con = duckdb.connect()
    try:
        query = f"""
            WITH raw_base AS (
                SELECT
                    person_id,
                    individual_key,
                    TRY_CAST(year AS INTEGER) AS report_year,
                    CAST(program_end_date AS DATE) AS program_end_date,
                    CAST(EXTRACT(YEAR FROM program_end_date) AS INTEGER) AS grad_year,
                    CASE
                        WHEN COALESCE(
                            opt_employer_start_date,
                            opt_authorization_start_date,
                            authorization_start_date
                        ) IS NOT NULL THEN 1
                        ELSE 0
                    END AS has_opt
                FROM read_parquet('{str(input_path).replace("'", "''")}')
                WHERE program_end_date IS NOT NULL
                  AND CAST(program_end_date AS DATE) <= DATE '{as_of_date}'
            ),
            base AS (
                SELECT *
                FROM raw_base
                WHERE person_id IS NOT NULL
            ),
            person_grad AS (
                SELECT
                    grad_year,
                    person_id,
                    MAX(has_opt) AS ever_opt
                FROM base
                WHERE grad_year IS NOT NULL
                GROUP BY grad_year, person_id
            ),
            person_eq_grad AS (
                SELECT
                    grad_year,
                    COUNT(DISTINCT person_id) AS n_person_ids_with_year_eq_grad
                FROM base
                WHERE grad_year IS NOT NULL
                  AND person_id IS NOT NULL
                  AND report_year = grad_year
                GROUP BY grad_year
            ),
            key_eq_grad AS (
                SELECT
                    grad_year,
                    individual_key,
                    MAX(has_opt) AS ever_opt
                FROM raw_base
                WHERE grad_year IS NOT NULL
                  AND individual_key IS NOT NULL
                  AND report_year = grad_year
                GROUP BY grad_year, individual_key
            ),
            key_eq_grad_summary AS (
                SELECT
                    grad_year,
                    COUNT(*) AS n_individual_keys_year_eq_grad,
                    SUM(ever_opt) AS n_key_opt_users_year_eq_grad,
                    AVG(CAST(ever_opt AS DOUBLE)) AS opt_share_individual_key_year_eq_grad
                FROM key_eq_grad
                GROUP BY grad_year
            )
            SELECT
                pg.grad_year,
                COUNT(*) AS n_persons,
                SUM(ever_opt) AS n_opt_users,
                AVG(CAST(ever_opt AS DOUBLE)) AS opt_share,
                peg.n_person_ids_with_year_eq_grad,
                keg.n_individual_keys_year_eq_grad,
                keg.n_key_opt_users_year_eq_grad,
                keg.opt_share_individual_key_year_eq_grad
            FROM person_grad AS pg
            LEFT JOIN person_eq_grad AS peg
              ON pg.grad_year = peg.grad_year
            LEFT JOIN key_eq_grad_summary AS keg
              ON pg.grad_year = keg.grad_year
            GROUP BY
                pg.grad_year,
                peg.n_person_ids_with_year_eq_grad,
                keg.n_individual_keys_year_eq_grad,
                keg.n_key_opt_users_year_eq_grad,
                keg.opt_share_individual_key_year_eq_grad
            HAVING COUNT(*) >= {int(min_persons)}
            ORDER BY pg.grad_year
        """
        out = con.execute(query).df()
        out["person_to_key_ratio"] = (
            out["n_persons"] / out["n_individual_keys_year_eq_grad"]
        )
        out["share_person_ids_observed_in_grad_year"] = (
            out["n_person_ids_with_year_eq_grad"] / out["n_persons"]
        )
        return out
    finally:
        con.close()


def plot_opt_share(
    summary: pd.DataFrame,
    *,
    as_of_date: str,
    censor_start_year: int,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        summary["grad_year"],
        summary["opt_share"],
        color="#1f5aa6",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="person_id by program_end year",
    )
    ax.plot(
        summary["grad_year"],
        summary["opt_share_individual_key_year_eq_grad"],
        color="#b04a2b",
        linewidth=2.2,
        marker="s",
        markersize=4.2,
        label="individual_key where year = grad_year",
    )

    plot_x_min = int(summary["grad_year"].min())
    plot_x_max = int(summary["grad_year"].max())

    if censor_start_year <= int(summary["grad_year"].max()) and censor_start_year <= plot_x_max:
        ax.axvspan(
            censor_start_year - 0.5,
            float(plot_x_max) + 0.5,
            color="#f2c57c",
            alpha=0.25,
            zorder=0,
        )
        text_transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            max(censor_start_year + 0.1, plot_x_min + 0.1),
            0.05,
            f"Recent cohorts (>= {censor_start_year})\nmay be right-censored",
            fontsize=9,
            color="#7a4b00",
            transform=text_transform,
            va="bottom",
        )

    ax.set_title("FOIA OPT Usage by Graduation Year")
    ax.set_xlabel("Program end year")
    ax.set_ylabel("Share with any OPT activity")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    note = (
        f"Source: {input_path_name(summary)}\n"
        "OPT activity = any non-null OPT-related start date "
        "(opt_employer_start_date, opt_authorization_start_date, or authorization_start_date).\n"
        "Blue line: distinct person_id within program_end year. "
        "Red line: distinct individual_key among rows with year == grad_year.\n"
        f"Restricted to program_end_date <= {as_of_date}."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_count_comparison(
    summary: pd.DataFrame,
    *,
    as_of_date: str,
    censor_start_year: int,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        summary["grad_year"],
        summary["n_persons"],
        color="#1f5aa6",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Unique person_id by program_end year",
    )
    ax.plot(
        summary["grad_year"],
        summary["n_individual_keys_year_eq_grad"],
        color="#b04a2b",
        linewidth=2.2,
        marker="s",
        markersize=4.2,
        label="Unique individual_key where year = grad_year",
    )

    plot_x_min = int(summary["grad_year"].min())
    plot_x_max = int(summary["grad_year"].max())
    if censor_start_year <= plot_x_max:
        ax.axvspan(
            max(censor_start_year - 0.5, plot_x_min - 0.5),
            float(plot_x_max) + 0.5,
            color="#f2c57c",
            alpha=0.25,
            zorder=0,
        )
        text_transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            max(censor_start_year + 0.1, plot_x_min + 0.1),
            0.05,
            f"Recent cohorts (>= {censor_start_year})\nmay be right-censored",
            fontsize=9,
            color="#7a4b00",
            transform=text_transform,
            va="bottom",
        )

    ax.set_title("FOIA Cohort Count Comparison by Graduation Year")
    ax.set_xlabel("Program end year")
    ax.set_ylabel("Distinct cohort count")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    note = (
        f"Source: {input_path_name(summary)}\n"
        "Blue line: distinct person_id within program_end year. "
        "Red line: distinct individual_key among rows with year == grad_year.\n"
        f"Restricted to program_end_date <= {as_of_date}."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def input_path_name(summary: pd.DataFrame) -> str:
    source = summary.attrs.get("input_path")
    if isinstance(source, Path):
        return source.name
    return "foia_person_panel"


def filter_summary_years(
    summary: pd.DataFrame,
    *,
    x_min_year: int | None,
    x_max_year: int | None,
) -> pd.DataFrame:
    filtered = summary.copy()
    if x_min_year is not None:
        filtered = filtered[filtered["grad_year"] >= int(x_min_year)]
    if x_max_year is not None:
        filtered = filtered[filtered["grad_year"] <= int(x_max_year)]
    return filtered.reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    summary = build_opt_share_table(
        input_path=args.input_path,
        as_of_date=args.as_of_date,
        min_persons=int(args.min_persons),
    )
    summary = filter_summary_years(
        summary,
        x_min_year=args.x_min_year,
        x_max_year=args.x_max_year,
    )
    summary.attrs["input_path"] = args.input_path

    if summary.empty:
        raise ValueError("No graduation-year cohorts satisfy the requested filters.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    as_of_year = int(args.as_of_date[:4])
    censor_start_year = (
        int(args.censor_start_year)
        if args.censor_start_year is not None
        else as_of_year - 2
    )
    plot_opt_share(
        summary,
        as_of_date=args.as_of_date,
        censor_start_year=censor_start_year,
        output_png=args.output_png,
    )
    plot_count_comparison(
        summary,
        as_of_date=args.as_of_date,
        censor_start_year=censor_start_year,
        output_png=args.counts_output_png,
    )

    print(f"Wrote summary CSV: {args.output_csv}")
    print(f"Wrote OPT-share plot PNG: {args.output_png}")
    print(f"Wrote cohort-count comparison PNG: {args.counts_output_png}")
    display_cols = [
        "grad_year",
        "n_persons",
        "n_individual_keys_year_eq_grad",
        "n_opt_users",
        "opt_share",
        "n_key_opt_users_year_eq_grad",
        "opt_share_individual_key_year_eq_grad",
        "person_to_key_ratio",
    ]
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
