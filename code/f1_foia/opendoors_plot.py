"""
Plot OpenDoors totals: total international students, enrolled international students,
and the enrolled share of US enrollment (secondary axis) from
/home/yk0581/data/raw/opendoors_total_intl_enr.xlsx.

Columns are auto-detected by name; expected candidates:
  - year: ["year", "academic_year", "AY"]
  - total intl (all): ["total_intl", "total_international", "intl_total"]
  - enrolled intl: ["intl_enrolled", "enrolled_intl", "international_enrollment"]
  - US enrollment: ["us_enrollment", "us_total_enrollment", "total_us_enrollment"]
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *  # noqa: F401,F403

BASE_FONT_SIZE = 12 * 1.4
IHMP_COLORS = {
    "IHMP": "#2e8b57",      # medium sea green
    "Non-IHMP": "#e07a5f",  # terracotta
}

DATA_PATH = Path("/home/yk0581/data/raw/opendoors_total_intl_enr.xlsx")
DEGREE_PATH = Path("/home/yk0581/data/raw/opendoors_intl_enr_by_degree.xlsx")
MPI_PATH = Path("/home/yk0581/data/raw/mpi_imm_over_time.xlsx")
FIG_DIR = Path(f"{root}/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

YEAR_CANDS = ["year", "academic_year", "ay"]
TOTAL_INTL_CANDS = ["total_intl_students", "total_international", "intl_total"]
ENROLLED_INTL_CANDS = ["enr_intl_students", "enrolled_intl", "international_enrollment"]
US_ENROLL_CANDS = ["us_enrollment", "us_total_enrollment", "total_us_enrollment"]


def first_present(cols, cands, label):
    cols_lower = {c.lower(): c for c in cols}
    for cand in cands:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Could not find {label}. Available columns: {cols}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected Excel file at {path}")
    # Headers are split across lines 3 and 4, so read a multi-row header and flatten.
    try:
        df = pd.read_excel(path, header=[2, 3])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(part).strip() for part in col if pd.notna(part) and str(part).strip() != ""]).lower().replace(" ", "_")
                for col in df.columns
            ]
    except Exception:
        df = pd.read_excel(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = ["year", "enr_intl_students", "opt_students", "total_intl_students", "annual_pct_change", "us_total_enrollment", "intl_share_of_us_enrollment"]
    year_col = first_present(df.columns, YEAR_CANDS, "year column")
    total_intl_col = first_present(df.columns, TOTAL_INTL_CANDS, "total intl column")
    enrolled_intl_col = first_present(df.columns, ENROLLED_INTL_CANDS, "enrolled intl column")
    us_enroll_col = first_present(df.columns, US_ENROLL_CANDS, "US enrollment column")

    df = df.rename(
        columns={
            year_col: "year",
            total_intl_col: "total_intl",
            enrolled_intl_col: "intl_enrolled",
            us_enroll_col: "us_enrollment",
        }
    )
    df = df[["year", "total_intl", "intl_enrolled", "us_enrollment"]].dropna()
    df["year"] = df["year"].str.slice(0, 4).astype(int)
    df = df[~pd.isnull(df["year"])]
    df = df.sort_values("year")
    # Compute share of US enrollment, coerce intl_enrolled to numeric
    df["intl_enrolled"] = pd.to_numeric(df["intl_enrolled"], errors="coerce")
    df["intl_share_us_enrollment"] = df["intl_enrolled"] / df["us_enrollment"]
    return df


def plot(df: pd.DataFrame, out_path: Path, show: bool = True, include_counts: bool = True, include_share: bool = True) -> None:
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_counts = ax if include_counts else None
    ax_share = None
    if include_share and include_counts:
        ax_share = ax.twinx()
    elif include_share:
        ax_share = ax

    lines = []
    labels = []

    if include_counts:
        ax_counts.plot(
            df["year"],
            df["total_intl"] / 1_000_000,
            marker="o",
            label="Total international students",
            color=IHMP_COLORS["IHMP"],
        )
        ax_counts.set_ylabel("Students (millions)")
        ax_counts.set_xlabel("Year")
        l_c, lab_c = ax_counts.get_legend_handles_labels()
        lines += l_c
        labels += lab_c

    if include_share and ax_share is not None:
        ax_share.plot(
            df["year"],
            df["intl_share_us_enrollment"],
            marker="^",
            label="Intl enrolled / US enrollment",
            color=IHMP_COLORS["Non-IHMP"],
        )
        ax_share.set_ylabel("Share of US enrollment")
        ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_share.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        max_share = df["intl_share_us_enrollment"].max()
        ax_share.set_ylim(0, max_share * 1.2 if pd.notnull(max_share) else 1)
        if not include_counts:
            ax_share.set_xlabel("Year")
        l_s, lab_s = ax_share.get_legend_handles_labels()
        lines += l_s
        labels += lab_s

    if lines:
        ax.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_opt_counts(df: pd.DataFrame, out_path: Path, show: bool = True) -> None:
    """
    Plot OPT-only counts (total_intl - intl_enrolled) over time.
    """
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    plot_df = df.copy()
    plot_df["opt_only"] = plot_df["total_intl"] - plot_df["intl_enrolled"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        plot_df["year"],
        plot_df["opt_only"] / 1_000.0,
        marker="o",
        label="OPT students",
        color=IHMP_COLORS["IHMP"],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Students (thousands)")
    ax.set_title("")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def load_degree_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected Excel file at {path}")
    df = pd.read_excel(path, sheet_name="Broad Academic Levels", header=[3], nrows = 51)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = ['year', 'undergrad_count', 'undergrad_share', 'graduate_count', 'graduate_share', 'nondegree_count', 'nondegree_share', 'opt_count', 'opt_share', 'total']
    year_col = first_present(df.columns, YEAR_CANDS, "year column")
    df = df.rename(columns={year_col: "year"})
    value_cols = ['undergrad_count', 'graduate_count', 'opt_count', 'total']
    # drop rows with missing year or year that doesn't start with a number
    df = df[df["year"].str.match(r"^\d{4}")]
    df["year"] = df["year"].str.slice(0, 4).astype(int)
    df = df[["year"] + value_cols].dropna(subset=["year"])
    long = df.melt(id_vars=["year",'total'], value_vars=value_cols, var_name="degree_type", value_name="intl_students")
    long['intl_students'] = pd.to_numeric(long['intl_students'], errors='coerce')
    long['share_of_total'] = long['intl_students'] / long['total']
    return long.sort_values(["degree_type", "year"])


def plot_degree_breakdown(df_long: pd.DataFrame, out_path: Path, yvar = "intl_students", show: bool = True) -> None:
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    fig, ax = plt.subplots(figsize=(10, 6))
    for degree, grp in df_long.groupby("degree_type"):
        ax.plot(grp["year"], grp[yvar], marker="o", label=degree)
    ax.set_xlabel("Year")
    ax.set_ylabel("International students" if yvar == "intl_students" else "Share of total")
    ax.set_title("International students by degree type (OpenDoors)")
    ax.legend(title="Degree type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def load_mpi(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected MPI file at {path}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    df = df.rename(columns={"year": "year", "n_imm": "n_imm", "imm_share": "imm_share"})
    df = df[["year", "n_imm", "imm_share"]].dropna()
    df["year"] = df["year"].astype(int)
    return df


def plot_intl_vs_imm(intl_df: pd.DataFrame, mpi_df: pd.DataFrame, out_path: Path, use_share: bool = False, show: bool = True) -> None:
    plt.rcParams.update({"font.size": BASE_FONT_SIZE})
    merged = intl_df.merge(mpi_df, on="year", how="inner")
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = ax_left.twinx()
    if use_share:
        ax_left.plot(
            merged["year"],
            merged["intl_share_us_enrollment"],
            marker="o",
            label="Intl share of US enrollment",
            color="tab:blue",
        )
        ax_left.set_ylabel("Intl share of US enrollment")
        ax_left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        ax_right.plot(
            merged["year"],
            merged["imm_share"],
            marker="s",
            label="Immigrant share",
            color="tab:red",
        )
        ax_right.set_ylabel("Immigrant share")
        ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_left.set_title("International students vs. immigrants (shares)")
    else:
        ax_left.plot(
            merged["year"],
            merged["total_intl"],
            marker="o",
            label="Total international students",
            color="tab:blue",
        )
        ax_left.set_ylabel("International students")

        ax_right.plot(
            merged["year"],
            merged["n_imm"],
            marker="s",
            label="Total immigrants",
            color="tab:red",
        )
        ax_right.set_ylabel("Immigrants")
        ax_left.set_title("International students vs. immigrants (counts)")

    ax_left.set_xlabel("Year")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main():
    df = load_data(DATA_PATH)
    combined_out = FIG_DIR / "opendoors_international_enrollment.png"
    counts_only_out = FIG_DIR / "opendoors_international_total_only.png"
    share_only_out = FIG_DIR / "opendoors_international_share_only.png"
    opt_only_out = FIG_DIR / "opendoors_opt_only.png"
    plot(df, combined_out, show=True, include_counts=True, include_share=True)
    plot(df, counts_only_out, show=False, include_counts=True, include_share=False)
    plot(df, share_only_out, show=False, include_counts=False, include_share=True)
    plot_opt_counts(df, opt_only_out, show=False)
    print(f"Saved plots to {combined_out}, {counts_only_out}, {share_only_out}, and {opt_only_out}")

    deg_df = load_degree_data(DEGREE_PATH)
    deg_out = FIG_DIR / "opendoors_intl_by_degree.png"
    plot_degree_breakdown(deg_df, deg_out, yvar = "share_of_total", show=True)
    print(f"Saved degree-type plot to {deg_out}")

    mpi_df = load_mpi(MPI_PATH)
    counts_out = FIG_DIR / "opendoors_intl_vs_imm_counts.png"
    shares_out = FIG_DIR / "opendoors_intl_vs_imm_shares.png"
    plot_intl_vs_imm(df, mpi_df, counts_out, use_share=False, show=True)
    plot_intl_vs_imm(df, mpi_df, shares_out, use_share=True, show=True)
    print(f"Saved intl vs. imm plots to {counts_out} and {shares_out}")


if __name__ == "__main__":
    main()
