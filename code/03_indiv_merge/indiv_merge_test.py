# File Description: Consistency checks for indiv_merge pipeline outputs
# Author: Amy Kim
# Date Updated: Feb 23 2026

from __future__ import annotations

import ast
import os
import sys
import traceback
from pathlib import Path

import pyarrow.parquet as pq

if "__file__" in globals():
    _THIS_DIR = Path(__file__).resolve().parent
else:
    _THIS_DIR = Path.cwd() / "03_indiv_merge"

_CODE_DIR = _THIS_DIR.parent
sys.path.append(str(_THIS_DIR))
sys.path.append(str(_CODE_DIR))

import indiv_merge_config as icfg


def _path_pairs() -> dict[str, tuple[str, str]]:
    return {
        "baseline": (icfg.MERGE_FILT_BASELINE_PARQUET, icfg.MERGE_FILT_BASELINE_PARQUET_LEGACY),
        "prefilt": (icfg.MERGE_FILT_PREFILT_PARQUET, icfg.MERGE_FILT_PREFILT_PARQUET_LEGACY),
        "mult2": (icfg.MERGE_FILT_MULT2_PARQUET, icfg.MERGE_FILT_MULT2_PARQUET_LEGACY),
        "mult4": (icfg.MERGE_FILT_MULT4_PARQUET, icfg.MERGE_FILT_MULT4_PARQUET_LEGACY),
        "mult6": (icfg.MERGE_FILT_MULT6_PARQUET, icfg.MERGE_FILT_MULT6_PARQUET_LEGACY),
    }


def _strict_path() -> str:
    return icfg.MERGE_FILT_STRICT_PARQUET


def _resolved_paths() -> dict[str, str]:
    return {
        name: icfg.choose_path(primary, fallback)
        for name, (primary, fallback) in _path_pairs().items()
    }


def _schema_cols(path: str) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _get_default_int_arg_from_source(func_name: str, arg_name: str) -> int:
    source_path = _THIS_DIR / "indiv_merge.py"
    module = ast.parse(source_path.read_text())

    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != func_name:
            continue

        args = node.args.args
        defaults = node.args.defaults
        default_start = len(args) - len(defaults)

        for i, arg in enumerate(args):
            if arg.arg != arg_name:
                continue
            default_idx = i - default_start
            if default_idx < 0:
                raise AssertionError(f"{func_name}.{arg_name} has no default value")
            default_node = defaults[default_idx]
            if not isinstance(default_node, ast.Constant) or not isinstance(default_node.value, int):
                raise AssertionError(
                    f"{func_name}.{arg_name} default is non-integer: {ast.unparse(default_node)}"
                )
            return default_node.value

    raise AssertionError(f"Could not find function or arg: {func_name}.{arg_name}")


def test_choose_path_resolution_is_consistent():
    for name, (primary, fallback) in _path_pairs().items():
        resolved = icfg.choose_path(primary, fallback)
        expected = primary if os.path.exists(primary) else fallback
        assert resolved == expected, (
            f"{name}: choose_path returned {resolved}, expected {expected}"
        )


def test_config_testing_controls_are_loaded():
    assert isinstance(icfg.BUILD_OVERWRITE, bool)
    assert isinstance(icfg.TESTING_ENABLED, bool)
    assert isinstance(icfg.TESTING_SAMPLE_MATCHES, int)
    assert icfg.TESTING_SAMPLE_MATCHES >= 1
    assert icfg.TESTING_RANDOM_SEED is None or isinstance(icfg.TESTING_RANDOM_SEED, int)
    assert icfg.TESTING_FIRM_UID is None or isinstance(icfg.TESTING_FIRM_UID, str)
    assert icfg.TESTING_LOTTERY_YEAR is None or isinstance(icfg.TESTING_LOTTERY_YEAR, str)
    assert isinstance(icfg.TESTING_MATERIALIZE_INTERMEDIATE_TABLES, bool)
    assert isinstance(icfg.TESTING_TABLE_PREFIX, str)
    assert len(icfg.TESTING_TABLE_PREFIX.strip()) > 0


def test_merge_output_parquets_exist_and_nonempty():
    for name, path in _resolved_paths().items():
        assert os.path.exists(path), f"{name}: missing parquet at {path}"
        assert os.path.getsize(path) > 0, f"{name}: empty parquet at {path}"


def test_merge_output_core_columns_present():
    required_cols = {
        "foia_indiv_id",
        "foia_firm_uid",
        "FEIN",
        "lottery_year",
        "status_type",
        "user_id",
        "weight_norm",
        "ade_ind",
        "ade_year",
        "last_grad_year",
        "enddatediff",
        "still_at_firm1",
        "still_at_firm2",
        "diff_firm1",
        "diff_firm2",
        "new_diff_firm1",
        "new_diff_firm2",
        "other1",
        "other2",
        "in_us1",
        "in_us2",
        "new_educ1",
        "new_educ2",
        "continuing_educ1",
        "continuing_educ2",
        "in_home_country1",
        "in_home_country2",
        "agg_compensation1",
        "agg_compensation2",
    }

    for name, path in _resolved_paths().items():
        cols = _schema_cols(path)
        missing = sorted(required_cols - cols)
        assert not missing, f"{name}: missing required columns: {missing}"


def test_tvar_horizon_defaults_match_current_indiv_merge():
    assert _get_default_int_arg_from_source("get_rel_year_inds_wide", "t1") == 2
    assert _get_default_int_arg_from_source("get_rel_year_inds_wide_by_t", "t1") == 2
    assert _get_default_int_arg_from_source("with_t_vars_query", "t1") == 2


def test_strict_config_constants_loaded():
    assert isinstance(icfg.STRICT_MIN_WEIGHT_NORM, float), "STRICT_MIN_WEIGHT_NORM should be float"
    assert isinstance(icfg.STRICT_MIN_TOTAL_SCORE, float), "STRICT_MIN_TOTAL_SCORE should be float"
    assert isinstance(icfg.STRICT_MIN_FIRM_QUALITY, float), "STRICT_MIN_FIRM_QUALITY should be float"
    assert isinstance(icfg.STRICT_MIN_COUNTRY_SCORE, float), "STRICT_MIN_COUNTRY_SCORE should be float"
    assert isinstance(icfg.STRICT_REQUIRE_EST_YOB, bool), "STRICT_REQUIRE_EST_YOB should be bool"
    assert icfg.STRICT_MAX_N_MATCH_FILT is None or isinstance(icfg.STRICT_MAX_N_MATCH_FILT, int), (
        "STRICT_MAX_N_MATCH_FILT should be None or int"
    )
    assert isinstance(icfg.MERGE_FILT_STRICT_PARQUET, str), "MERGE_FILT_STRICT_PARQUET should be str"


def test_strict_parquet_exists_and_nonempty():
    path = _strict_path()
    assert os.path.exists(path), f"strict: missing parquet at {path}"
    assert os.path.getsize(path) > 0, f"strict: empty parquet at {path}"


def test_strict_parquet_satisfies_thresholds():
    import pyarrow.compute as pc

    path = _strict_path()
    if not os.path.exists(path):
        return  # covered by test_strict_parquet_exists_and_nonempty

    tbl = pq.read_table(path)
    cols = set(tbl.schema.names)
    n = len(tbl)
    assert n > 0, "strict: parquet is empty"

    # All rows must be rank=1
    assert "match_rank" in cols, "strict: missing match_rank column"
    ranks = tbl.column("match_rank").to_pylist()
    assert all(r == 1 for r in ranks), "strict: found match_rank != 1"

    # weight_norm threshold
    assert "weight_norm" in cols, "strict: missing weight_norm column"
    wn = tbl.column("weight_norm").to_pylist()
    violations = sum(1 for w in wn if w is not None and w < icfg.STRICT_MIN_WEIGHT_NORM)
    assert violations == 0, f"strict: {violations} rows have weight_norm < {icfg.STRICT_MIN_WEIGHT_NORM}"

    # total_score threshold
    assert "total_score" in cols, "strict: missing total_score column"
    ts = tbl.column("total_score").to_pylist()
    violations = sum(1 for s in ts if s is not None and s < icfg.STRICT_MIN_TOTAL_SCORE)
    assert violations == 0, f"strict: {violations} rows have total_score < {icfg.STRICT_MIN_TOTAL_SCORE}"

    # firm_match_quality_mult threshold
    assert "firm_match_quality_mult" in cols, "strict: missing firm_match_quality_mult column"
    fq = tbl.column("firm_match_quality_mult").to_pylist()
    violations = sum(1 for v in fq if v is not None and v < icfg.STRICT_MIN_FIRM_QUALITY)
    assert violations == 0, f"strict: {violations} rows have firm_match_quality_mult < {icfg.STRICT_MIN_FIRM_QUALITY}"

    # country_score threshold
    assert "country_score" in cols, "strict: missing country_score column"
    cs = tbl.column("country_score").to_pylist()
    violations = sum(1 for v in cs if v is not None and v < icfg.STRICT_MIN_COUNTRY_SCORE)
    assert violations == 0, f"strict: {violations} rows have country_score < {icfg.STRICT_MIN_COUNTRY_SCORE}"

    # est_yob non-null if required
    if icfg.STRICT_REQUIRE_EST_YOB and "est_yob" in cols:
        nulls = sum(1 for v in tbl.column("est_yob").to_pylist() if v is None)
        assert nulls == 0, f"strict: {nulls} rows have null est_yob despite strict_require_est_yob=True"

    # n_match_filt cap if set
    if icfg.STRICT_MAX_N_MATCH_FILT is not None and "n_match_filt" in cols:
        nmf = tbl.column("n_match_filt").to_pylist()
        violations = sum(1 for v in nmf if v is not None and v > icfg.STRICT_MAX_N_MATCH_FILT)
        assert violations == 0, f"strict: {violations} rows exceed strict_max_n_match_filt={icfg.STRICT_MAX_N_MATCH_FILT}"

    # Each foia_indiv_id appears at most once (rank=1 only)
    assert "foia_indiv_id" in cols, "strict: missing foia_indiv_id column"
    ids = tbl.column("foia_indiv_id").to_pylist()
    assert len(ids) == len(set(ids)), "strict: duplicate foia_indiv_id (expected one row per applicant)"

    # Print retention stats
    baseline_path = icfg.choose_path(icfg.MERGE_FILT_BASELINE_PARQUET, icfg.MERGE_FILT_BASELINE_PARQUET_LEGACY)
    if os.path.exists(baseline_path):
        baseline_ids = set(pq.read_table(baseline_path, columns=["foia_indiv_id"]).column("foia_indiv_id").to_pylist())
        strict_ids = set(ids)
        pct = 100.0 * len(strict_ids) / len(baseline_ids) if baseline_ids else 0.0
        print(f"  strict sample retains {len(strict_ids):,} / {len(baseline_ids):,} baseline apps ({pct:.1f}%)")
        extra = strict_ids - baseline_ids
        assert not extra, f"strict: {len(extra)} foia_indiv_ids not in baseline (strict should be a subset)"


def test_tvar_horizon_columns_match_t1_equals_2_outputs():
    must_exist = {
        "still_at_firm1",
        "still_at_firm2",
        "diff_firm1",
        "diff_firm2",
        "other1",
        "other2",
        "in_us1",
        "in_us2",
        "new_educ1",
        "new_educ2",
        "continuing_educ1",
        "continuing_educ2",
    }
    must_not_exist = {
        "still_at_firm3",
        "diff_firm3",
        "other3",
        "in_us3",
        "new_educ3",
        "in_home_country3",
        "agg_compensation3",
    }

    for name, path in _resolved_paths().items():
        cols = _schema_cols(path)
        missing_expected = sorted(must_exist - cols)
        unexpected_extra = sorted(c for c in must_not_exist if c in cols)
        assert not missing_expected, f"{name}: missing expected t<=2 columns: {missing_expected}"
        assert not unexpected_extra, (
            f"{name}: found unexpected t=3 columns despite t1=2 default: {unexpected_extra}"
        )


def _run_as_script() -> int:
    tests = sorted(
        name for name, obj in globals().items() if name.startswith("test_") and callable(obj)
    )
    failures: list[tuple[str, str]] = []

    print(f"Running {len(tests)} indiv_merge consistency checks...")
    for name in tests:
        try:
            globals()[name]()
            print(f"[PASS] {name}")
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"[FAIL] {name}")

    if failures:
        print("\nFailure details:")
        for name, tb in failures:
            print(f"\n{name}\n{tb}")
        return 1

    print("All indiv_merge consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
