"""
Unit tests for ProgramClusterer in program_partition package.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import root  # type: ignore
from program_partition import ProgramClusterConfig, ProgramClusterer  # type: ignore


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": np.arange(1, 7),
            "university_name": [
                "New York University",
                "New York Univ.",
                "Columbia University",
                "Columbia University",
                "New York University",
                "New York Univ.",
            ],
            "university_raw": [
                "New York University",
                "New York Univ.",
                "Columbia University",
                "Columbia Univ.",
                "NYU",
                "New York Univ.",
            ],
            "field": [
                "Computer Science",
                "Comp Sci",
                "Applied Mathematics",
                "Applied Math",
                "Economics",
                "Econ",
            ],
            "field_raw": [
                "Computer Science",
                "Comp Sci",
                "Applied Mathematics",
                "Applied Math",
                "Economics",
                "Econ",
            ],
            "degree": [
                "Bachelor of Science",
                "B.S.",
                "Master of Science",
                "M.S.",
                "Bachelor of Arts",
                "B.A.",
            ],
            "degree_raw": [
                "Bachelor of Science",
                "BS",
                "Master of Science",
                "MS",
                "Bachelor of Arts",
                "BA",
            ],
        }
    )


def test_clusterer_groups_universities_and_programs() -> None:
    df = build_sample_frame()
    config = ProgramClusterConfig(
        distance_threshold=0.45,
        min_group_size_for_clustering=1,
        cluster_universities=True,
        university_distance_threshold=0.35,
        verbose=False,
    )

    clusterer = ProgramClusterer(config)
    assignments, summary = clusterer.fit_transform(df)

    canonical_universities = set(assignments["canonical_university"])
    assert canonical_universities == {"new york university", "columbia university"}

    ny_assignments = assignments.loc[assignments["canonical_university"] == "new york university"]
    assert set(ny_assignments["canonical_field"]) == {"computer", "economics"}

    ny_summary = summary.loc[summary["canonical_university"] == "new york university"]
    assert ny_summary["university_cluster_size"].iat[0] == 4
    assert ny_summary["cluster_size"].sum() == 4  # two NYU programs, two records each


def test_clusterer_respects_university_toggle() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "university_name": [
                "Massachusetts Institute of Technology",
                "MIT",
            ],
            "field": ["Computer Science", "Computer Science"],
            "degree": ["Bachelor of Science", "Bachelor of Science"],
        }
    )

    config = ProgramClusterConfig(
        min_group_size_for_clustering=1,
        cluster_universities=False,
        verbose=False,
    )
    clusterer = ProgramClusterer(config)
    assignments, _ = clusterer.fit_transform(df)

    assert set(assignments["canonical_university"]) == {
        "massachusetts institute of technology",
        "mit",
    }


def test_clusterer_requires_field_information() -> None:
    df = pd.DataFrame({"university_name": ["Test University"]})
    clusterer = ProgramClusterer(ProgramClusterConfig(verbose=False))

    with pytest.raises(ValueError):
        clusterer.fit_transform(df)


def test_high_school_separation() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "university_name": [
                "Springfield High School",
                "Springfield Secondary School",
                "Springfield University",
                "Springfield College",
            ],
            "field": [
                "Science",
                "General Studies",
                "Computer Science",
                "Economics",
            ],
            "degree": [
                "High School Diploma",
                "High School Diploma",
                "Bachelor of Science",
                "Bachelor of Arts",
            ],
        }
    )

    config = ProgramClusterConfig(
        min_group_size_for_clustering=1,
        cluster_universities=True,
        university_distance_threshold=0.35,
        verbose=False,
    )

    clusterer = ProgramClusterer(config)
    assignments, summary = clusterer.fit_transform(df)

    hs_clusters = set(assignments.loc[assignments["is_high_school"], "university_cluster_id"].unique())
    uni_clusters = set(assignments.loc[~assignments["is_high_school"], "university_cluster_id"].unique())

    assert hs_clusters, "High school clusters should be identified."
    assert uni_clusters, "Higher-ed clusters should be identified."
    assert hs_clusters.isdisjoint(uni_clusters), "High school clusters must not mix with university clusters."
    assert summary.loc[summary["is_high_school"], "canonical_university"].str.contains("high school|secondary school").any()


def test_city_refined_clustering() -> None:
    geonames_path = Path(root) / "data/crosswalks/geonames/cities500.txt"
    if not geonames_path.exists():
        pytest.skip("Geonames reference data unavailable; skipping city clustering test.")

    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "university_name": [
                "Hanyang University Law School",
                "Anyang University",
                "Toronto University",
                "Oronto Univ",
            ],
            "university_location": [
                "Seoul, South Korea",
                "Anyang, South Korea",
                "Toronto, Canada",
                "Oronto, Canada",
            ],
            "field": [
                "Law",
                "Law",
                "Economics",
                "Economics",
            ],
            "degree": [
                "J.D.",
                "J.D.",
                "B.A.",
                "B.S.",
            ],
        }
    )

    config = ProgramClusterConfig(
        min_group_size_for_clustering=1,
        cluster_universities=True,
        verbose=False,
    )
    config.geonames_cities_path = geonames_path

    clusterer = ProgramClusterer(config)
    assignments, summary = clusterer.fit_transform(df)

    hanyang_cluster = assignments.loc[assignments["clean_university"].str.contains("hanyang"), "university_cluster_id"].unique()
    anyang_cluster = assignments.loc[assignments["clean_university"].str.contains("anyang"), "university_cluster_id"].unique()
    toronto_cluster = assignments.loc[assignments["clean_university"].str.contains("toronto"), "university_cluster_id"].unique()
    oronto_cluster = assignments.loc[assignments["clean_university"].str.contains("oronto"), "university_cluster_id"].unique()

    assert hanyang_cluster.size == 1
    assert anyang_cluster.size == 1
    assert hanyang_cluster[0] != anyang_cluster[0], "Different cities should yield distinct clusters."

    assert toronto_cluster.size == 1
    assert oronto_cluster.size == 1
    assert toronto_cluster[0] == oronto_cluster[0], "Similar city names should merge into one cluster."

    toronto_summary = summary.loc[summary["canonical_city"] == "toronto"]
    assert not toronto_summary.empty


REAL_DATA_PATHS = [
    Path(root) / "data/int/ihma_educ_all_oct20.parquet",
    Path(root) / "data/int/linkedin_education_history.parquet",
]


def get_real_data_path() -> Path | None:
    for candidate in REAL_DATA_PATHS:
        if candidate.exists():
            return candidate
    return None


@pytest.mark.skipif(get_real_data_path() is None, reason="Real LinkedIn education parquet not available.")
def test_clusterer_real_subset() -> None:
    path = get_real_data_path()
    assert path is not None  # for mypy/typing

    df = pd.read_parquet(path).head(500).copy()
    df = df.rename(columns=str.lower)

    required = {"university_name", "field"}
    missing = required.difference(df.columns)
    if missing:
        pytest.skip(f"Dataset {path} missing required columns: {missing}")

    subset = df.loc[df["university_name"].notna() & df["field"].notna()].head(200).copy()
    if subset.empty or subset["university_name"].nunique() == 0:
        pytest.skip("Subset lacks usable education records for clustering.")

    config = ProgramClusterConfig(
        verbose=False,
        min_group_size_for_clustering=2,
        distance_threshold=0.4,
        university_distance_threshold=0.3,
    )

    clusterer = ProgramClusterer(config)
    assignments, summary = clusterer.fit_transform(subset)

    assert not assignments.empty
    assert not summary.empty
    assert assignments["program_id"].nunique() == summary.shape[0]
    assert assignments["canonical_university"].isin(summary["canonical_university"]).all()

import duckdb as ddb
import openai 

def llm_resolver(context, catalog):
    prompt = (
        "Given the program description below, pick the best 6-digit CIP code.\n\n"
        f"Program text: {context['program_feature']}\n"
        f"Canonical field: {context['canonical_field']}\n"
        f"Canonical degree: {context['canonical_degree']}\n"
        f"Canonical university: {context['canonical_university']}\n"
        "Respond with JSON: {\"cip_code\": \"xxxxxx\", \"reason\": \"...\"}"
    )
    client = openai.OpenAI()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content
    result = json.loads(text)
    code = result.get("cip_code", "")
    if not code or code not in catalog["cip_code"].values:
        return None
    row = catalog.loc[catalog["cip_code"] == code].iloc[0]
    return {
        "cip_code": code,
        "cip_title": row.get("cip_title", ""),
        "cip_level": row.get("cip_level", ""),
        "similarity": 0.0,  # optional confidence score
    }

con = ddb.connect()

con.read_csv("/home/yk0581/data/clean/wrds_clusters_oct2025.csv").df().shape

x = con.read_parquet(f"{root}/data/int/wrds_users_sep2.parquet")
df = con.sql("SELECT * FROM x").to_df().sample(100000)

config = ProgramClusterConfig(
    distance_threshold=0.15,
    min_group_size_for_clustering=1,
    verbose=True,
    enable_translation=False,
    enable_cip_matching = False,
    infer_city_from_university=False,
    cip_reference_path=f"{root}/data/crosswalks/CIPCode2010.csv",
    cip_code_column="CIPCode",
    cip_title_column="CIPTitle",
    cip_description_column="CIPDefinition",
    enable_university_hints=False,
    analyzer = "word",
    min_df = 2
)

clusterer = ProgramClusterer(config)

# df_prep = clusterer._prepare(df)
# df_prep.to_parquet(f"{root}/data/int/test_linkedin_prep_kr.parquet", index=False)

# df_prep = pd.read_parquet(f"{root}/data/int/test_linkedin_prep_kr.parquet")
# df_prep[['university_raw','clean_university','canonical_university','university_cluster_size', 'clean_city', 'canonical_city_x', 'canonical_city_y','university_cluster_id']]

assignments, summary = clusterer.fit_transform(df)
assignments.to_parquet(f"{root}/data/int/test_linkedin_program_partition_nov4.parquet", index=False)
# number of clusters
# df_prep['university_cluster_id'].nunique()

## examining assignments

# assignments[['program_id','program_name', 'cluster_size', 'program_cluster', 'university_source', 'canonical_university', 'canonical_city', 'university_cluster_size', 'field_source', 'canonical_field', 'degree_source', 'canonical_degree', 'cip_code','cip_title','cip_similarity']]

