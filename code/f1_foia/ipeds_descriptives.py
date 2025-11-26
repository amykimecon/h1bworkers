# INITIAL DESCRIPTIVES OF IHMPs from IPEDS data 
# Imports and Paths
import duckdb as ddb
import pandas as pd
import sys
import os
import time
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import zipfile
import requests
try:
    import pgeocode
except Exception:  # pragma: no cover - optional dependency
    pgeocode = None
try:
    import geopandas as gpd
except Exception:  # pragma: no cover - optional dependency
    gpd = None

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
INT_FOLDER = f"{root}/data/int/int_files_nov2025"
IPEDS_GEO_PATH = f"{root}/data/raw/ipeds_cw_2021.csv"
# Optional shapefiles for county/ZIP outlines (set to your local paths)
ZIP_SHAPEFILE = os.environ.get("ZIP_SHAPEFILE_PATH", "")
COUNTY_SHAPEFILE = os.environ.get("COUNTY_SHAPEFILE_PATH", "")
ZCTA_CACHE_PATH = os.path.join(INT_FOLDER, "cb_2022_us_zcta520_500k.geojson")

# READING IN RAW DATA
ipeds = pd.read_parquet(f"{INT_FOLDER}/ipeds_completions_all.parquet")
ipeds['ihmp'] = (ipeds['share_intl'] >= 0.5)&(ipeds['awlevel'] == 7)&(ipeds['majornum'] == 1)&(ipeds['STEMOPT'] == 1)
ipeds['awlevel_group'] = np.where(ipeds['awlevel'].isin([5, 7, 17, 9]), ipeds['awlevel'].replace({
    5: 'Bachelor',
    7: 'Master',
    17: 'Doctor',
    9: 'Doctor'
}), 'Other')
ipeds['intl_ihmp'] = ipeds['ihmp']*ipeds['cnralt']

# DESCRIPTIVE ONE: Share of all degree completions by international students over time
ipeds_by_year = ipeds[ipeds['awlevel_group'] != 'Other'].groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_year['share_intl'] = ipeds_by_year['cnralt'] / ipeds_by_year['ctotalt']
ipeds_by_year['share_ihmp_intl'] = ipeds_by_year['intl_ihmp'] / ipeds_by_year['ctotalt']
ipeds_by_deg_year = ipeds[ipeds['awlevel_group'] != 'Other'].groupby(['year', 'awlevel_group']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
ipeds_by_deg_year['share_intl'] = ipeds_by_deg_year['cnralt'] / ipeds_by_deg_year['ctotalt']

sns.lineplot(data=ipeds_by_year, x='year', y='cnralt', color = 'blue')
sns.lineplot(data=ipeds_by_year, x='year', y='ctotalt', color = 'gray')

sns.lineplot(data=ipeds_by_year, x='year', y='share_intl', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year, x='year', y='share_intl', hue='awlevel_group')

sns.lineplot(data=ipeds_by_year, x='year', y='cnralt', color = 'gray')
sns.lineplot(data=ipeds_by_deg_year, x='year', y='cnralt', hue='awlevel_group')

# DESCRIPTIVE TWO: Share of Master's IHMPs over time
ipeds_masters = ipeds[ipeds['awlevel'] == 7]
ipeds_masters_by_year = ipeds_masters.groupby(['year']).agg({'ctotalt': 'sum', 'cnralt': 'sum', 'intl_ihmp': 'sum'}).reset_index()
# ipeds_masters_by_year['share_intl'] = ipeds_masters_by_year['cnralt'] / ipeds_masters_by_year['ctotalt

# DESCRIPTIVE THREE: Spatial distribution of IHMPs over time
def _resolve_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_zcta_shapes() -> "gpd.GeoDataFrame | None":
    """
    Load ZCTA boundaries via Census cartographic files.

    Tries, in order:
    1) ZIP_SHAPEFILE env var (if provided)
    2) Cached GeoJSON at ZCTA_CACHE_PATH
    3) Download Census cartographic ZCTA (cb_2022_us_zcta520_500k.zip)
    """

    if not gpd:
        return None

    if ZIP_SHAPEFILE and os.path.exists(ZIP_SHAPEFILE):
        try:
            return gpd.read_file(ZIP_SHAPEFILE)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read ZIP_SHAPEFILE {ZIP_SHAPEFILE} ({exc}); continuing.")

    if os.path.exists(ZCTA_CACHE_PATH):
        try:
            return gpd.read_file(ZCTA_CACHE_PATH)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read cached ZCTA geojson at {ZCTA_CACHE_PATH} ({exc}); re-downloading.")

    url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_zcta520_500k.zip"
    try:
        print("Downloading ZCTA boundaries from Census (cb_2022_us_zcta520_500k)...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(INT_FOLDER)
        shp_path = os.path.join(INT_FOLDER, "cb_2022_us_zcta520_500k.shp")
        gdf = gpd.read_file(shp_path)
        try:
            gdf.to_file(ZCTA_CACHE_PATH, driver="GeoJSON")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to cache ZCTA geojson ({exc})")
        return gdf
    except Exception as exc:  # pragma: no cover
        print(f"Failed to download Census ZCTA boundaries ({exc}); falling back to point plot.")
        return None


if os.path.exists(IPEDS_GEO_PATH):
    ipeds_geo = pd.read_csv(IPEDS_GEO_PATH)
    lat_col = _resolve_col(ipeds_geo, ["lat", "latitude", "lat_dd"])
    lon_col = _resolve_col(ipeds_geo, ["lon", "lng", "longitude", "long_dd"])
    zip_col = _resolve_col(ipeds_geo, ["zip", "zip_code", "postalcode", "postal_code"])
    unit_col = _resolve_col(ipeds_geo, ["unitid", "UNITID"])
    state_col = _resolve_col(ipeds_geo, ["stabbr", "state", "state_abbr"])

    # If lat/lon missing, try deriving them from ZIP codes via pgeocode (if available).
    if zip_col and (not lat_col or not lon_col):
        if pgeocode is None:
            print("Skipping spatial IHMP plot: no lat/lon columns and pgeocode not installed to derive them from ZIP.")
        else:
            # Normalize ZIPs to 5-digit strings
            zip_norm_col = "zip_norm_tmp"
            ipeds_geo[zip_norm_col] = ipeds_geo[zip_col].astype(str).str.slice(0, 5).str.zfill(5)
            nomi = pgeocode.Nominatim("us")
            zip_lookup = ipeds_geo[zip_norm_col].dropna().unique()
            zip_df = pd.DataFrame({"zip_norm": zip_lookup})
            zip_df["lat_dd"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).latitude)
            zip_df["long_dd"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).longitude)
            zip_df["state_code"] = zip_df["zip_norm"].apply(lambda z: nomi.query_postal_code(z).state_code)
            ipeds_geo = ipeds_geo.merge(
                zip_df,
                left_on=zip_norm_col,
                right_on="zip_norm",
                how="left",
            )
            lat_col = "lat_dd"
            lon_col = "long_dd"
            if state_col is None and "state_code" in ipeds_geo.columns:
                state_col = "state_code"
            if zip_col is None:
                zip_col = zip_norm_col

    if lat_col and lon_col and unit_col and zip_col:
        ihmp_geo = (
            ipeds_masters.loc[(ipeds_masters["ihmp"]) & (ipeds_masters["ctotalt"] >= 10)]
            .merge(
                ipeds_geo[[col for col in [unit_col, lat_col, lon_col, zip_col, state_col] if col]],
                left_on="unitid",
                right_on=unit_col,
                how="left",
            )
        )
        if state_col:
            ihmp_geo = ihmp_geo[~ihmp_geo[state_col].isin(["AK", "HI"])]

        # total unitids per zip for denominator
        zip_totals = (
            ipeds_geo.groupby(zip_col, as_index=False)[unit_col].nunique().rename(columns={unit_col: "total_unitids"})
        )

        ihmp_geo_agg = (
            ihmp_geo.groupby(["year", zip_col], as_index=False)
            .agg(
                ihmp_unitids=("unitid", "nunique"),
                lat=(lat_col, "first"),
                lon=(lon_col, "first"),
                state=(state_col, "first") if state_col else ("unitid", "size"),
            )
            .merge(zip_totals, on=zip_col, how="left")
            .rename(columns={"lon": "plot_lon", "lat": "plot_lat", zip_col: "zip_code"})
        )
        ihmp_geo_agg["ihmp_share"] = ihmp_geo_agg["ihmp_unitids"] / ihmp_geo_agg["total_unitids"]

        shapes = _load_zcta_shapes() if gpd else None
        if gpd and shapes is not None:
            geo_key = _resolve_col(shapes, ["GEOID", "ZCTA5CE10", "ZCTA5CE20", "ZIP", "ZIPCODE", "ZIP_CODE", "ZCTA"])
            if geo_key:
                shapes = shapes.rename(columns={geo_key: "zip_code"})
                shapes["zip_code"] = shapes["zip_code"].astype(str).str.extract(r"(\\d{3,5})")[0].str.zfill(5)
                merged = shapes.merge(ihmp_geo_agg, on="zip_code", how="left")
                if state_col and "state" in merged.columns:
                    merged = merged[~merged["state"].isin(["AK", "HI"])]
                years = sorted(merged["year"].dropna().unique())
                ncols = 4
                nrows = int(np.ceil(len(years) / ncols)) or 1
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = np.atleast_1d(axes).flatten()
                for ax, yr in zip(axes, years):
                    merged[merged["year"] == yr].plot(
                        column="ihmp_share",
                        cmap="viridis",
                        linewidth=0.05,
                        edgecolor="gray",
                        legend=False,
                        ax=ax,
                    )
                    ax.set_title(f"Year {yr}")
                    ax.set_axis_off()
                for ax in axes[len(years) :]:
                    ax.set_axis_off()
                fig.suptitle("Spatial distribution of IHMP programs (ZIP polygons, ctotalt>=10, excluding AK/HI)", y=0.92)
                plt.tight_layout()
            else:
                print("Could not resolve ZIP field in Census shapes; falling back to point plot.")
                shapes = None
        if not gpd or shapes is None:
            g = sns.relplot(
                data=ihmp_geo_agg,
                x="plot_lon",
                y="plot_lat",
                size="ihmp_unitids",
                hue="ihmp_share",
                col="year",
                col_wrap=4,
                height=3,
                palette="viridis",
                sizes=(20, 200),
            )
            g.set_titles("Year {col_name}")
            g.set_axis_labels("Longitude", "Latitude")
            plt.suptitle("Spatial distribution of IHMP programs (ZIP aggregated, ctotalt>=10, excluding AK/HI)", y=1.02)
            plt.tight_layout()
    else:
        print("Skipping spatial IHMP plot: could not resolve lat/lon/zip/unitid columns in crosswalk.")
else:
    print(f"Skipping spatial IHMP plot: geoname crosswalk not found at {IPEDS_GEO_PATH}.")

# DESCRIPTIVE FOUR: Econ field composition over time (relative to 2010)
def _pad_cip(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else None)


ipeds["cip_str"] = _pad_cip(ipeds.get("cipcode", pd.Series(dtype="Int64")))

def _assign_econ_group(cip: str) -> str | None:
    if cip is None:
        return None
    if cip == "450601":
        return "General Economics"
    if cip == "450603":
        return "Econometrics"
    if cip in {"450602", "450604", "450605", "450699"}:
        return "Other Economics"
    return None


def _assign_fin_math_group(cip: str) -> str | None:
    if cip is None:
        return None
    if cip.startswith("5208"):
        return "Finance (5208xx)"
    if cip.startswith("2701"):
        return "General Math (2701xx)"
    if cip.startswith("2703"):
        return "Applied Math (2703xx)"
    if cip.startswith("2705") or cip.startswith("2706"):
        return "Statistics (2705/2706xx)"
    return None


# Econ comparison
econ = ipeds[ipeds["awlevel_group"] != "Other"].copy()
econ["econ_group"] = econ["cip_str"].apply(_assign_econ_group)
econ = econ[econ["econ_group"].notna()]
econ_grouped = (
    econ.groupby(["awlevel_group", "year", "econ_group"], as_index=False)["ctotalt"]
    .sum()
    .rename(columns={"ctotalt": "completions"})
)
baseline_econ = econ_grouped[econ_grouped["year"] == 2010][["awlevel_group", "econ_group", "completions"]].rename(
    columns={"completions": "base_2010"}
)
econ_grouped = econ_grouped.merge(baseline_econ, on=["awlevel_group", "econ_group"], how="left")
econ_grouped["rel_to_2010"] = econ_grouped["completions"] / econ_grouped["base_2010"]
econ_grouped = econ_grouped[econ_grouped['awlevel_group']=="Master"]

fig, axes = plt.subplots(1, len(econ_grouped["awlevel_group"].unique()), figsize=(14, 4), sharey=True)
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
for ax, degree in zip(axes, sorted(econ_grouped["awlevel_group"].unique())):
    subset = econ_grouped[econ_grouped["awlevel_group"] == degree]
    for group_name, grp in subset.groupby("econ_group"):
        ax.plot(grp["year"], grp["completions"], marker="o", label=group_name)
    ax.set_title(degree)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Completions (relative to 2010)")
    ax.set_xlabel("Year")
    ax.legend()
fig.suptitle("Economics field completions relative to 2010 (by degree type)")
fig.tight_layout()

# Finance / Math / Statistics comparison
tabfield = 'cnralt'  # can switch to 'ctotalt' for total completions instead of intl
fin_math = ipeds[ipeds["awlevel_group"] != "Other"].copy()
fin_math["field_group"] = fin_math["cip_str"].apply(_assign_fin_math_group)
fin_math = fin_math[fin_math["field_group"].notna()]
fin_grouped = (
    fin_math.groupby(["awlevel_group", "year", "field_group"], as_index=False)[tabfield]
    .sum()
    .rename(columns={tabfield: "completions"})
)
baseline_fin = fin_grouped[fin_grouped["year"] == 2010][["awlevel_group", "field_group", "completions"]].rename(
    columns={"completions": "base_2010"}
)
fin_grouped = fin_grouped.merge(baseline_fin, on=["awlevel_group", "field_group"], how="left")
fin_grouped["rel_to_2010"] = fin_grouped["completions"] / fin_grouped["base_2010"]

fin_grouped = fin_grouped[fin_grouped['awlevel_group']=="Master"]

fig2, axes2 = plt.subplots(1, len(fin_grouped["awlevel_group"].unique()), figsize=(8, 6), sharey=True)
if not isinstance(axes2, np.ndarray):
    axes2 = np.array([axes2])
for ax, degree in zip(axes2, sorted(fin_grouped["awlevel_group"].unique())):
    subset = fin_grouped[fin_grouped["awlevel_group"] == degree]
    for group_name, grp in subset.groupby("field_group"):
        ax.plot(grp["year"], grp["completions"], marker="o", label=group_name)
    ax.set_title(degree)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Completions (relative to 2010)")
    ax.set_xlabel("Year")
    ax.legend()
fig2.suptitle("Finance/Math/Statistics completions relative to 2010 (by degree type)")
fig2.tight_layout()


### random
con = ddb.connect()
FOIA_PATH = f"{root}/data/int/foia_sevp_combined_raw.parquet"
foia_raw = con.read_parquet(FOIA_PATH)
foia_clean = 