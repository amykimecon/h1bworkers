from pathlib import Path
from typing import Dict, Sequence, Set, List, Tuple

import sys 
import json
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

import numpy as np
import pandas as pd
import pytest
import sys 
import os
import duckdb as ddb 
import adaptive_fuzzy as af_module
import pyarrow.parquet as pq

EXPECTED_CLUSTER_COLUMNS = {
    "cluster_root",
    "university_name",
    "member_cities",
    "cluster_cities",
    "matched_city",
    "matched_geo_city_id",
    "matched_institution_id",
    "ipeds_ids",
}

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 

con = ddb.connect()
from adaptive_fuzzy import (
    LabelledExample,
    build_clusters,
    clusters_to_frame,
    compute_features,
    fit_classifier,
    generate_pair_candidates,
    prepare_normalized_name_parquet,
    prepare_name_groups,
    set_feature_cache,
    set_token_statistics_from_names,
    build_faiss_index,
    AnnRetriever,
)


def test_compute_features_returns_expected_length():
    set_feature_cache(None)
    set_token_statistics_from_names([
        "University of California",
        "University of California Los Angeles",
        "Massachusetts Institute of Technology",
    ])
    features = compute_features("University of California", "University of California")
    assert features.shape == (20,)
    assert features[0] == 100.0  # identical strings => perfect match
    assert 0.0 <= features[-1] <= 1.0
    city_features = features[10:12]
    country_features = features[12:14]
    idf_features = features[14:18]
    school_features = features[18:]
    # No explicit location tokens => both flags zero
    assert city_features.tolist() == [0.0, 0.0]
    assert country_features.tolist() == [0.0, 0.0]
    assert idf_features.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert school_features.tolist() == [1.0, 0.0]


def test_build_clusters_merges_high_probability_pairs():
    names = [
        "University of California Los Angeles",
        "UCLA",
        "Massachusetts Institute of Technology",
    ]
    set_feature_cache(None)
    set_token_statistics_from_names(names)
    candidates = generate_pair_candidates(names)
    labelled = [
        LabelledExample(
            compute_features("University of California Los Angeles", "UCLA"), 1
        ),
        LabelledExample(
            compute_features("Massachusetts Institute of Technology", "UCLA"), 0
        ),
        LabelledExample(
            compute_features("UCLA", "Massachusetts Institute of Technology"), 0
        ),
    ]
    model = fit_classifier(labelled)
    clusters = build_clusters(names, model, candidates, threshold=0.6)

    # Expect UCLA and the long form to share a cluster, MIT separate
    cluster_sizes = sorted(len(members) for members in clusters.values())
    assert cluster_sizes == [1, 2]

    df = clusters_to_frame(clusters)
    assert set(df.columns) == EXPECTED_CLUSTER_COLUMNS
    assert len(df) == len(names)
    assert df["member_cities"].apply(lambda value: isinstance(value, tuple)).all()
    assert df["cluster_cities"].apply(lambda value: isinstance(value, tuple)).all()
    assert df["ipeds_ids"].apply(lambda value: isinstance(value, tuple)).all()


def test_generate_candidates_subset(tmp_path):
    set_feature_cache(None)
    csv_path = tmp_path / "demo_names.csv"
    df = pd.DataFrame(
        {
            "university_name": [
                "Seoul National University",
                "Seoul Natl Univ",
                "Korea University",
                "Yonsei University",
            ]
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = pd.read_csv(csv_path)["university_name"].tolist()
    candidates = generate_pair_candidates(loaded)

    assert any(
        {cand.name_a, cand.name_b} == {"Seoul National University", "Seoul Natl Univ"}
        for cand in candidates
    )


def test_location_features_detect_city_and_country():
    set_feature_cache(None)
    set_token_statistics_from_names([
        "University of California, Berkeley, USA",
        "California State University, Berkeley, United States",
    ])
    name_a = "University of California, Berkeley, USA"
    name_b = "California State University, Berkeley, United States"
    features = compute_features(name_a, name_b)

    city_features = features[10:12]
    country_features = features[12:14]
    idf_features = features[14:18]
    school_features = features[18:]

    assert city_features[0] == 1.0  # same city
    assert city_features[1] == 0.0  # not different cities

    assert country_features[0] == 1.0  # same country
    assert country_features[1] == 0.0
    assert len(idf_features) == 4
    assert school_features.tolist() == [1.0, 0.0]


def test_clusters_to_frame_city_metadata_union():
    clusters = {
        "University of California, Berkeley": [
            "University of California, Berkeley",
            "University of California at Berkeley",
        ],
        "Massachusetts Institute of Technology": [
            "Massachusetts Institute of Technology",
        ],
    }

    df = clusters_to_frame(clusters)
    assert set(df.columns) == EXPECTED_CLUSTER_COLUMNS

    berkeley_rows = df[df["cluster_root"] == "University of California, Berkeley"]
    assert not berkeley_rows.empty
    assert berkeley_rows["cluster_cities"].nunique() == 1
    assert berkeley_rows["cluster_cities"].iloc[0] == ("berkeley",)
    assert berkeley_rows["member_cities"].apply(lambda cities: "berkeley" in cities).all()

    mit_rows = df[df["cluster_root"] == "Massachusetts Institute of Technology"]
    assert mit_rows["member_cities"].tolist() == [tuple()]
    assert mit_rows["cluster_cities"].tolist() == [tuple()]
    assert mit_rows["matched_city"].isna().all()


def test_prepare_name_groups_deduplicates_variants(monkeypatch):
    raw_names = [
        "Massachusetts Institute of Technology (MIT)",
        "MIT",
        "MIT.",
        "University of California, Main Campus",
        " University of California ",
        "University of California, Berkeley",
    ]

    def canonical(name: str) -> str:
        return af_module._canonicalize_institution_alias(name)

    ground_truth = {
        canonical("Massachusetts Institute of Technology"): {"mit-id"},
        canonical("MIT"): {"mit-id"},
        canonical("University of California"): {"uc-id"},
        canonical("University of California, Main Campus"): {"uc-id"},
        canonical("University of California, Berkeley"): {"ucb-id"},
    }

    af_module._institution_ground_truth.cache_clear()
    set_feature_cache(None)

    def fake_ground_truth() -> Dict[str, Set[str]]:
        return ground_truth

    monkeypatch.setattr(af_module, "_institution_ground_truth", fake_ground_truth)
    set_token_statistics_from_names(raw_names)
    representatives, mapping, ids = prepare_name_groups(raw_names)

    assert len(representatives) == 3
    assert "Massachusetts Institute of Technology (MIT)" in representatives
    assert "University of California, Main Campus" in representatives
    assert "University of California, Berkeley" in representatives

    mit_members = mapping["Massachusetts Institute of Technology (MIT)"]
    assert {"Massachusetts Institute of Technology (MIT)", "MIT", "MIT."} <= set(mit_members)

    uc_members = mapping["University of California, Main Campus"]
    assert "University of California, Main Campus" in uc_members
    assert "University of California" in uc_members

    assert mapping["University of California, Berkeley"] == ["University of California, Berkeley"]

    mit_rep = next(rep for rep in representatives if "Massachusetts Institute" in rep)
    uc_rep = "University of California, Main Campus"
    ucb_rep = "University of California, Berkeley"

    assert ids[mit_rep] == "mit-id"
    assert ids[uc_rep] == "uc-id"
    assert ids[ucb_rep] == "ucb-id"


def test_program_suffixes_merge_with_parent(monkeypatch):
    raw_names = [
        "Drexel University",
        "Drexel University College of Business",
        "California State University Fullerton",
        "California State University, Long Beach",
    ]

    def canonical(name: str) -> str:
        return af_module._canonicalize_institution_alias(name)

    ground_truth = {
        canonical("Drexel University"): {"drexel"},
        canonical("Drexel University College of Business"): {"drexel"},
        canonical("California State University Fullerton"): {"csuf"},
        canonical("California State University, Long Beach"): {"csulb"},
    }

    af_module._institution_ground_truth.cache_clear()
    set_feature_cache(None)

    def fake_ground_truth() -> Dict[str, Set[str]]:
        return ground_truth

    monkeypatch.setattr(af_module, "_institution_ground_truth", fake_ground_truth)
    set_token_statistics_from_names(raw_names)
    representatives, mapping, ids = prepare_name_groups(raw_names)

    drexel_rep = next(rep for rep in representatives if "Drexel University" in rep)
    csu_fullerton_rep = next(rep for rep in representatives if "Fullerton" in rep)
    csu_long_beach_rep = next(rep for rep in representatives if "Long Beach" in rep)

    assert ids[drexel_rep] == "drexel"
    assert "Drexel University" in mapping[drexel_rep]
    assert "Drexel University College of Business" in mapping[drexel_rep]

    assert ids[csu_fullerton_rep] == "csuf"
    assert ids[csu_long_beach_rep] == "csulb"
    assert csu_fullerton_rep != csu_long_beach_rep


def test_prepare_name_groups_splits_composite_entries():
    composite = (
        "Concordia Junior College, Bronxville, NY  A.A.; "
        "Concordia Senior College, Ft. Wayne, IN, B.S.in Psychology; "
        "Concordia Seminary, St. Louis, MO, M.Div."
    )
    composite_log: List[Tuple[str, List[str]]] = []
    representatives, mapping, _ids = prepare_name_groups(
        [composite],
        composite_log=composite_log,
    )

    assert composite_log, "Expected composite entry to be detected and split."
    original, segments = composite_log[0]
    assert original == composite
    assert len(segments) == 3
    assert set(representatives) == set(segments)
    flattened = {name for members in mapping.values() for name in members}
    assert composite not in flattened
    for segment in segments:
        assert segment in flattened


def test_prepare_normalized_name_parquet(tmp_path):
    input_path = tmp_path / "names.csv"
    df = pd.DataFrame(
        {
            "name": [
                "Concordia Junior College, Bronxville, NY  A.A.; Concordia Senior College, Ft. Wayne, IN, B.S.",
                "Massachusetts Institute of Technology, Inc.",
                "École Polytechnique Fédérale de Lausanne",
                "  ",
            ]
        }
    )
    df.to_csv(input_path, index=False)
    output_path = tmp_path / "names.parquet"

    written = prepare_normalized_name_parquet(input_path, "name", output_path, chunk_size=2)
    assert written == 3
    table = pq.read_table(output_path)
    assert {"id", "name", "norm", "tokens", "shingles", "alnum", "length", "first_token", "last_token"} <= set(
        table.column_names
    )
    data = table.to_pydict()
    assert data["norm"][1] == "massachusetts institute of technology"
    assert "mit" not in data["tokens"][1]  # tokens unique sorted


def _names_hash_for_test(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def test_ann_retriever_produces_candidates(tmp_path):
    pytest.importorskip("faiss")

    names = [
        "Alpha University",
        "Alpha Univ",
        "Beta College",
    ]
    dim = 4
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    emb_path = tmp_path / "ann_embeddings.f32"
    mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=embeddings.shape)
    mmap[:] = embeddings
    mmap.flush()

    meta = {
        "rows": len(names),
        "dim": dim,
        "model": "unit-test",
        "normalize": True,
        "batch_size": 1,
        "names_hash": _names_hash_for_test(names),
        "created_at": "1970-01-01T00:00:00Z",
    }
    meta_path = emb_path.with_suffix(emb_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))

    index_path = tmp_path / "ann.index"
    build_faiss_index(
        emb_path,
        index_path,
        nlist=1,
        m=1,
        nbits=8,
        training_samples=len(names),
        random_seed=123,
    )

    retriever = AnnRetriever(names, emb_path, index_path, top_k=2, nprobe=1)
    pairs = retriever.generate_candidates(names, limit=None)

    assert pairs, "Expected ANN retriever to produce candidate pairs."
    pair_sets = [{a, b} for a, b, _ in pairs]
    assert {"Alpha University", "Alpha Univ"} in pair_sets

def test_generate_candidates_real_dataset():
    duckdb = pytest.importorskip("duckdb")

    dataset_path = Path(root) / "data/int/wrds_users_sep2.parquet"
    if not dataset_path.exists():
        pytest.skip(f"Dataset not available at {dataset_path}")

    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT university_raw
        FROM read_parquet('{dataset_path}')
        WHERE university_country = 'South Korea'
        LIMIT 1000
    """
    df = con.sql(query).to_df()
    names = df["university_raw"].dropna().tolist()
    assert names, "No names retrieved from dataset"

    set_feature_cache(None)
    candidates = generate_pair_candidates(names)
    assert len(candidates) > 0


class QueuePrompter:
    """Mock prompt returning predefined responses."""

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        if not self._responses:
            return "q"
        return self._responses.pop(0)


def test_interactive_training_with_mocked_prompts(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")

    dataset_path = Path(root) / "data/int/wrds_users_sep2.parquet"
    if not dataset_path.exists():
        pytest.skip(f"Dataset not available at {dataset_path}")

    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT university_raw
        FROM read_parquet('{dataset_path}')
        WHERE university_country = 'South Korea'
        LIMIT 200
    """
    df = con.sql(query).to_df()
    names = df["university_raw"].dropna().tolist()
    if len(names) < 10:
        pytest.skip("Not enough names after filtering")

    set_feature_cache(None)
    candidates = generate_pair_candidates(names)
    responses = ["y", "y", "n", "", "y", "n", "q"]
    monkeypatch.setattr("builtins.input", QueuePrompter(responses))
    set_token_statistics_from_names(names)

    from adaptive_fuzzy.cli import collect_initial_labels, interactive_training

    archive_path = tmp_path / "labels.csv"
    initial_labelled = collect_initial_labels(candidates, to_label=3, archive_path=archive_path)
    assert len(initial_labelled) > 0
    assert archive_path.exists()

    model, labelled, _ = interactive_training(
        candidates,
        initial_labels=2,
        batch_size=3,
        archive_path=archive_path,
        max_iterations=2,
        convergence_threshold=0.05,
        initial_labelled=initial_labelled,
    )

    assert len(labelled) > 0

    clusters = build_clusters(names, model, candidates, threshold=0.6)
    assert clusters

    df = clusters_to_frame(clusters)
    assert set(df.columns) == EXPECTED_CLUSTER_COLUMNS
    assert len(df) == len(names)


@pytest.mark.skipif(
    os.environ.get("AF_INTERACTIVE") != "1",
    reason="Set AF_INTERACTIVE=1 to run the manual interactive test.",
)
def test_interactive_training_manual(tmp_path):
    names = [
        "University of California Los Angeles",
        "UCLA",
        "Massachusetts Institute of Technology",
        "MIT",
        "California Institute of Technology",
    ]
    af_module.set_feature_cache(None)
    candidates = generate_pair_candidates(names)

    from adaptive_fuzzy.cli import collect_initial_labels, interactive_training

    archive_path = tmp_path / "manual_labels.csv"
    print(
        "\nManual interactive test:\n"
        "  - Respond to the prompts with y/n/u/q.\n"
        "  - The loop will stop after one iteration or when you enter 'q'.\n"
    )
    initial_labelled = collect_initial_labels(
        candidates,
        to_label=2,
        archive_path=archive_path,
    )
    assert len(initial_labelled) > 0

    set_token_statistics_from_names(names)
    model, labelled, _ = interactive_training(
        candidates,
        initial_labels=1,
        batch_size=2,
        archive_path=archive_path,
        max_iterations=1,
        convergence_threshold=0.2,
        initial_labelled=initial_labelled,
    )

    clusters = build_clusters(names, model, candidates, threshold=0.6)
    df = clusters_to_frame(clusters)
    assert set(df.columns) == EXPECTED_CLUSTER_COLUMNS
    assert len(df) == len(names)
