from __future__ import annotations

import io
import importlib.util
import math
import subprocess
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PIPELINE_ROOT / "05_indiv_merge"

spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage05_test_module",
    STAGE_DIR / "stage_main.py",
)
assert spec is not None and spec.loader is not None
stage_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_main)

merge_spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage05_merge_logic_test_module",
    STAGE_DIR / "merge_logic.py",
)
assert merge_spec is not None and merge_spec.loader is not None
merge_logic = importlib.util.module_from_spec(merge_spec)
merge_spec.loader.exec_module(merge_logic)


class Stage05Tests(unittest.TestCase):
    def _write_fixture_inputs(self, tempdir: Path, *, school_cw_variant: str = "standard") -> dict[str, Path]:
        f1 = pd.DataFrame(
            [
                {
                    "original_row_num": 1,
                    "person_id": 1,
                    "individual_key": "INDIV-1",
                    "school_name": "Stanford University",
                    "campus_city": "Palo Alto",
                    "campus_state": "CA",
                    "campus_zip_code": "94305",
                    "country_of_birth": "INDIA",
                    "student_edu_level_desc": "MASTER'S",
                    "program_start_date": "2019-09-01",
                    "program_end_date": "2021-06-01",
                    "major_1_cip_code": "11.0701",
                    "employment_opt_type": "STEM",
                    "year": 2021,
                    "year_int": 2021,
                    "employer_name": "Google LLC",
                    "employer_city": "Mountain View",
                    "employer_state": "CA",
                    "employer_zip_code": "94043",
                }
            ]
        )
        f1_path = tempdir / "f1_clean.parquet"
        f1.to_parquet(f1_path, index=False)

        school_cw_row = {
            "f1_school_name": "Stanford University",
            "f1_instname_clean": "stanford university",
            "rev_university_raw": "Stanford University",
            "rev_instname_clean": "stanford university",
            "match_score": 0.99,
            "lev_sim": 0.99,
            "jw_sim": 0.99,
            "match_ambiguous_ind": 0,
            "school_match_rank": 1,
            "match_score_gap_from_top": 0.0,
            "n_rev_university_raw_variants": 1,
        }
        if school_cw_variant == "standard":
            school_cw_row["n_rev_users"] = 1
        elif school_cw_variant == "legacy_records":
            school_cw_row["n_revelio_institution_records"] = 1
        school_cw = pd.DataFrame([school_cw_row])
        school_cw_path = tempdir / "school_family.parquet"
        school_cw.to_parquet(school_cw_path, index=False)

        f1_inst_unitid = pd.DataFrame(
            [
                {
                    "f1_row_num": 7,
                    "school_name": "Stanford University",
                    "f1_city_clean": "palo alto",
                    "f1_state_clean": "CA",
                    "f1_zip_clean": "94305",
                    "unitid": 1001,
                }
            ]
        )
        f1_inst_unitid_path = tempdir / "f1_inst_unitid.parquet"
        f1_inst_unitid.to_parquet(f1_inst_unitid_path, index=False)

        employer_lookup = pd.DataFrame(
            [
                {
                    "employer_name": "Google LLC",
                    "employer_city_clean": "mountain view",
                    "employer_state_clean": "CA",
                    "employer_zip_clean": "94043",
                    "foia_row_uid": "row1",
                    "foia_firm_uid": "firm1",
                    "rcid": 900,
                    "lookup_rcid_count": 1,
                    "lookup_rcid_ambiguous_ind": 0,
                    "lookup_has_direct_ind": 1,
                }
            ]
        )
        employer_lookup_path = tempdir / "employer_lookup.parquet"
        employer_lookup.to_parquet(employer_lookup_path, index=False)

        rev_users_core = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "fullname": "Ravi Patel",
                    "est_yob": 1997,
                    "stem_ind_any": 1,
                    "f_prob": 0.1,
                    "fields_json": '["computer science"]',
                    "highest_ed_level": "Master",
                }
            ]
        )
        rev_users_core_path = tempdir / "rev_users_core.parquet"
        rev_users_core.to_parquet(rev_users_core_path, index=False)

        rev_match_ready = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "education_number": 1,
                    "country_candidate": "India",
                    "country_score": 0.95,
                    "nanat_score": 0.95,
                    "institution_score": 0.80,
                    "nametrace_score": 0.70,
                    "nanat_subregion_score": 0.95,
                    "nt_subregion_score": 0.85,
                    "subregion_candidate": "South Asia",
                    "country_uncertain_ind": 0,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": 1107,
                    "employer_key": "google",
                    "university_raw": "Stanford University",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-06-01",
                    "school_match_score": 0.99,
                },
                {
                    "user_id": 10,
                    "education_number": 2,
                    "country_candidate": "India",
                    "country_score": 0.95,
                    "nanat_score": 0.95,
                    "institution_score": 0.80,
                    "nametrace_score": 0.70,
                    "nanat_subregion_score": 0.95,
                    "nt_subregion_score": 0.85,
                    "subregion_candidate": "South Asia",
                    "country_uncertain_ind": 0,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": 1107,
                    "employer_key": "google",
                    "university_raw": "Stanford University",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-06-01",
                    "school_match_score": 0.99,
                }
            ]
        )
        rev_match_ready_path = tempdir / "rev_match_ready.parquet"
        rev_match_ready.to_parquet(rev_match_ready_path, index=False)

        rev_educ = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "education_number": 1,
                    "university_raw": "Stanford University",
                    "degree_clean": "Master",
                    "field_clean": "computer science",
                    "cip": 1107,
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-06-01",
                },
                {
                    "user_id": 10,
                    "education_number": 2,
                    "university_raw": "Stanford University",
                    "degree_clean": "Master",
                    "field_clean": "computer science",
                    "cip": 1107,
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-06-01",
                }
            ]
        )
        rev_educ_path = tempdir / "rev_educ_clean_long.parquet"
        rev_educ.to_parquet(rev_educ_path, index=False)

        rev_pos = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "position_id": "p1",
                    "rcid": 900,
                    "country": "India",
                    "company_raw": "Google LLC",
                    "startdate": "2021-07-01",
                    "enddate": "2023-01-01",
                }
            ]
        )
        rev_pos_path = tempdir / "rev_pos_clean_long.parquet"
        rev_pos.to_parquet(rev_pos_path, index=False)

        legacy_indiv = pd.DataFrame(
            [
                {
                    "user_id": 10,
                    "fullname": "Ravi Patel",
                    "country": "India",
                    "subregion": "South Asia",
                    "nanat_score": 0.95,
                    "nanat_subregion_score": 0.95,
                    "nt_subregion_score": 0.85,
                    "country_uncertain_ind": 0,
                    "est_yob": 1997,
                    "stem_ind": 1,
                    "f_prob": 0.1,
                    "fields": '["computer science"]',
                    "highest_ed_level": "Master",
                }
            ]
        )
        legacy_indiv_path = tempdir / "legacy_rev_indiv.parquet"
        legacy_indiv.to_parquet(legacy_indiv_path, index=False)

        return {
            "f1": f1_path,
            "school_cw": school_cw_path,
            "f1_inst_unitid": f1_inst_unitid_path,
            "employer_lookup": employer_lookup_path,
            "rev_users_core": rev_users_core_path,
            "rev_match_ready": rev_match_ready_path,
            "rev_educ": rev_educ_path,
            "rev_pos": rev_pos_path,
            "legacy_indiv": legacy_indiv_path,
        }

    def _write_config(self, tempdir: Path, inputs: dict[str, Path], *, use_stage4_indiv: bool) -> Path:
        cfg = {
            "run_tag": "test",
            "build": {
                "overwrite": True,
                "default_stages": ["05_indiv_merge"],
                "stop_on_deferred_stage": True,
                "allow_legacy_fallbacks": True,
                "w_emp_max": 0.95,
                "emp_n_scale": 1,
            },
            "testing": {
                "enabled": False,
                "sample_n_persons": 10,
                "random_seed": 7,
                "materialize_intermediate_tables": False,
                "table_prefix": "f1mt",
            },
            "stages": {
                "05_indiv_merge": {
                    "f1_foia_input_parquet": str(inputs["f1"]),
                    "f1_foia_fallback_parquet": str(inputs["f1"]),
                    "school_family_crosswalk_input_parquet": str(inputs["school_cw"]),
                    "school_resolution_input_parquet": str(tempdir / "missing_school_resolution.parquet"),
                    "f1_inst_unitid_crosswalk_input_parquet": str(inputs["f1_inst_unitid"]),
                    "employer_lookup_input_parquet": str(inputs["employer_lookup"]),
                    "rev_users_core_input_parquet": str(inputs["rev_users_core"]) if use_stage4_indiv else str(tempdir / "missing_users_core.parquet"),
                    "rev_match_ready_input_parquet": str(inputs["rev_match_ready"]) if use_stage4_indiv else str(tempdir / "missing_match_ready.parquet"),
                    "rev_educ_clean_long_input_parquet": str(inputs["rev_educ"]),
                    "rev_pos_clean_long_input_parquet": str(inputs["rev_pos"]),
                    "econ_relabels_input_parquet": str(tempdir / "missing_econ_relabels.parquet"),
                    "restrict_to_relabel_programs": False,
                    "employment_history_filter_enabled": True,
                    "legacy_rev_indiv_parquet": str(inputs["legacy_indiv"]),
                    "legacy_rev_educ_long_parquet": str(inputs["rev_educ"]),
                    "legacy_rev_pos_parquet": str(inputs["rev_pos"]),
                    "baseline_parquet": str(tempdir / "out" / "baseline.parquet"),
                    "mult2_parquet": str(tempdir / "out" / "mult2.parquet"),
                    "mult4_parquet": str(tempdir / "out" / "mult4.parquet"),
                    "mult6_parquet": str(tempdir / "out" / "mult6.parquet"),
                    "strict_parquet": str(tempdir / "out" / "strict.parquet"),
                    "person_agg_parquet": str(tempdir / "out" / "person_agg.parquet"),
                    "person_baseline_parquet": str(tempdir / "out" / "person_baseline.parquet"),
                    "person_strict_parquet": str(tempdir / "out" / "person_strict.parquet"),
                    "person_shard_count": None,
                    "person_shard_id": None,
                    "compare_to_reference_outputs": False,
                }
            },
        }
        cfg_path = tempdir / ("pipeline_stage4.yaml" if use_stage4_indiv else "pipeline_legacy.yaml")
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path

    def _read_stage_cfg(self, cfg_path: Path) -> dict:
        return yaml.safe_load(cfg_path.read_text())["stages"]["05_indiv_merge"]

    def _shard_artifact_paths(self, cfg_path: Path, *, shard_count: int, shard_id: int) -> dict[str, Path]:
        stage_cfg = self._read_stage_cfg(cfg_path)
        return {
            "baseline": Path(
                stage_main._shard_output_path(
                    stage_cfg["baseline_parquet"],
                    shard_count=shard_count,
                    shard_id=shard_id,
                )
            ),
            "person_agg": Path(
                stage_main._shard_output_path(
                    stage_cfg["person_agg_parquet"],
                    shard_count=shard_count,
                    shard_id=shard_id,
                )
            ),
        }

    def _assert_outputs(self, tempdir: Path) -> None:
        baseline = pd.read_parquet(tempdir / "out" / "baseline.parquet")
        strict = pd.read_parquet(tempdir / "out" / "strict.parquet")
        person_baseline = pd.read_parquet(tempdir / "out" / "person_baseline.parquet")
        person_strict = pd.read_parquet(tempdir / "out" / "person_strict.parquet")

        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(strict), 1)
        self.assertEqual(len(person_baseline), 1)
        self.assertEqual(len(person_strict), 1)
        self.assertEqual(int(baseline.iloc[0]["match_rank"]), 1)
        self.assertEqual(int(person_baseline.iloc[0]["person_match_rank"]), 1)
        self.assertGreaterEqual(float(strict.iloc[0]["weight_norm"]), 0.85)
        self.assertGreaterEqual(float(person_strict.iloc[0]["person_weight_norm"]), 0.85)

    def test_stage05_builds_outputs_from_stage04_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            stage_main.run(config_path=cfg_path, testing=False)
            self._assert_outputs(tempdir)

    def test_stage05_falls_back_to_legacy_indiv_when_stage04_country_artifacts_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=False)

            stage_main.run(config_path=cfg_path, testing=False)
            self._assert_outputs(tempdir)

    def test_stage05_accepts_legacy_school_crosswalk_without_n_rev_users(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir, school_cw_variant="legacy_records")
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            stage_main.run(config_path=cfg_path, testing=False)
            self._assert_outputs(tempdir)

    def test_stage05_shard_run_writes_only_shard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            stage_main.run(
                config_path=cfg_path,
                testing=False,
                shard_count=2,
                shard_id=1,
            )

            shard_paths = self._shard_artifact_paths(cfg_path, shard_count=2, shard_id=1)
            self.assertTrue(shard_paths["baseline"].exists())
            self.assertTrue(shard_paths["person_agg"].exists())
            self.assertFalse((tempdir / "out" / "baseline.parquet").exists())
            self.assertFalse((tempdir / "out" / "mult2.parquet").exists())
            self.assertFalse((tempdir / "out" / "mult4.parquet").exists())
            self.assertFalse((tempdir / "out" / "mult6.parquet").exists())
            self.assertFalse((tempdir / "out" / "strict.parquet").exists())
            self.assertFalse((tempdir / "out" / "person_baseline.parquet").exists())
            self.assertFalse((tempdir / "out" / "person_strict.parquet").exists())

            baseline_shard = pd.read_parquet(shard_paths["baseline"])
            person_agg_shard = pd.read_parquet(shard_paths["person_agg"])
            self.assertEqual(len(baseline_shard), 1)
            self.assertEqual(len(person_agg_shard), 1)
            self.assertEqual(int(baseline_shard.iloc[0]["person_id"]), 1)
            self.assertEqual(int(person_agg_shard.iloc[0]["person_id"]), 1)
            self.assertEqual(int(baseline_shard.iloc[0]["user_id"]), 10)

    def test_stage05_merge_shards_reproduces_unsharded_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            for shard_id in range(2):
                stage_main.run(
                    config_path=cfg_path,
                    testing=False,
                    shard_count=2,
                    shard_id=shard_id,
                )

            merged = stage_main._merge_stage05_sharded_outputs(
                config_path=cfg_path,
                shard_count=2,
            )

            self.assertEqual(merged["person_shard_count"], 2)
            self.assertIn("baseline_parquet", merged["merged_outputs"])
            self.assertIn("person_strict_parquet", merged["merged_outputs"])
            self._assert_outputs(tempdir)

    def test_stage05_merge_shards_runs_global_assignment_across_person_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)

            pd.DataFrame(
                [
                    {
                        "original_row_num": 1,
                        "person_id": 1,
                        "individual_key": "INDIV-1",
                        "school_name": "Stanford University",
                        "campus_city": "Palo Alto",
                        "campus_state": "CA",
                        "campus_zip_code": "94305",
                        "country_of_birth": "INDIA",
                        "student_edu_level_desc": "MASTER'S",
                        "program_start_date": "2019-09-01",
                        "program_end_date": "2021-06-01",
                        "major_1_cip_code": "11.0701",
                        "employment_opt_type": "STEM",
                        "year": 2021,
                        "year_int": 2021,
                        "employer_name": "Google LLC",
                        "employer_city": "Mountain View",
                        "employer_state": "CA",
                        "employer_zip_code": "94043",
                    },
                    {
                        "original_row_num": 2,
                        "person_id": 2,
                        "individual_key": "INDIV-2",
                        "school_name": "Stanford University",
                        "campus_city": "Palo Alto",
                        "campus_state": "CA",
                        "campus_zip_code": "94305",
                        "country_of_birth": "INDIA",
                        "student_edu_level_desc": "MASTER'S",
                        "program_start_date": "2019-09-01",
                        "program_end_date": "2021-06-01",
                        "major_1_cip_code": "11.0701",
                        "employment_opt_type": "STEM",
                        "year": 2021,
                        "year_int": 2021,
                        "employer_name": "Meta Platforms",
                        "employer_city": "Menlo Park",
                        "employer_state": "CA",
                        "employer_zip_code": "94025",
                    },
                ]
            ).to_parquet(inputs["f1"], index=False)
            pd.DataFrame(
                [
                    {
                        "employer_name": "Google LLC",
                        "employer_city_clean": "mountain view",
                        "employer_state_clean": "CA",
                        "employer_zip_clean": "94043",
                        "foia_row_uid": "row1",
                        "foia_firm_uid": "firm1",
                        "rcid": 900,
                        "lookup_rcid_count": 1,
                        "lookup_rcid_ambiguous_ind": 0,
                        "lookup_has_direct_ind": 1,
                    },
                    {
                        "employer_name": "Meta Platforms",
                        "employer_city_clean": "menlo park",
                        "employer_state_clean": "CA",
                        "employer_zip_clean": "94025",
                        "foia_row_uid": "row2",
                        "foia_firm_uid": "firm2",
                        "rcid": 901,
                        "lookup_rcid_count": 1,
                        "lookup_rcid_ambiguous_ind": 0,
                        "lookup_has_direct_ind": 1,
                    },
                ]
            ).to_parquet(inputs["employer_lookup"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "fullname": "Ravi Patel",
                        "est_yob": 1997,
                        "stem_ind_any": 1,
                        "f_prob": 0.1,
                        "fields_json": '["computer science"]',
                        "highest_ed_level": "Master",
                    },
                    {
                        "user_id": 20,
                        "fullname": "Asha Singh",
                        "est_yob": 1996,
                        "stem_ind_any": 1,
                        "f_prob": 0.2,
                        "fields_json": '["computer science"]',
                        "highest_ed_level": "Master",
                    },
                ]
            ).to_parquet(inputs["rev_users_core"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "education_number": 1,
                        "country_candidate": "India",
                        "country_score": 0.95,
                        "nanat_score": 0.95,
                        "institution_score": 0.80,
                        "nametrace_score": 0.70,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "subregion_candidate": "South Asia",
                        "country_uncertain_ind": 0,
                        "unitid": 1001,
                        "degree_clean": "masters",
                        "cip": 1107,
                        "employer_key": "google",
                        "university_raw": "Stanford University",
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                        "school_match_score": 0.99,
                    },
                    {
                        "user_id": 20,
                        "education_number": 1,
                        "country_candidate": "India",
                        "country_score": 0.95,
                        "nanat_score": 0.95,
                        "institution_score": 0.80,
                        "nametrace_score": 0.70,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "subregion_candidate": "South Asia",
                        "country_uncertain_ind": 0,
                        "unitid": 1001,
                        "degree_clean": "masters",
                        "cip": 1107,
                        "employer_key": "meta",
                        "university_raw": "Stanford University",
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                        "school_match_score": 0.99,
                    },
                ]
            ).to_parquet(inputs["rev_match_ready"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "education_number": 1,
                        "university_raw": "Stanford University",
                        "degree_clean": "Master",
                        "field_clean": "computer science",
                        "cip": 1107,
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                    },
                    {
                        "user_id": 20,
                        "education_number": 1,
                        "university_raw": "Stanford University",
                        "degree_clean": "Master",
                        "field_clean": "computer science",
                        "cip": 1107,
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                    },
                ]
            ).to_parquet(inputs["rev_educ"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "position_id": "p1",
                        "rcid": 900,
                        "country": "India",
                        "company_raw": "Google LLC",
                        "startdate": "2021-07-01",
                        "enddate": "2023-01-01",
                    },
                    {
                        "user_id": 20,
                        "position_id": "p2",
                        "rcid": 901,
                        "country": "India",
                        "company_raw": "Meta Platforms",
                        "startdate": "2021-07-01",
                        "enddate": "2023-01-01",
                    },
                ]
            ).to_parquet(inputs["rev_pos"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "fullname": "Ravi Patel",
                        "country": "India",
                        "subregion": "South Asia",
                        "nanat_score": 0.95,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "country_uncertain_ind": 0,
                        "est_yob": 1997,
                        "stem_ind": 1,
                        "f_prob": 0.1,
                        "fields": '["computer science"]',
                        "highest_ed_level": "Master",
                    },
                    {
                        "user_id": 20,
                        "fullname": "Asha Singh",
                        "country": "India",
                        "subregion": "South Asia",
                        "nanat_score": 0.95,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "country_uncertain_ind": 0,
                        "est_yob": 1996,
                        "stem_ind": 1,
                        "f_prob": 0.2,
                        "fields": '["computer science"]',
                        "highest_ed_level": "Master",
                    },
                ]
            ).to_parquet(inputs["legacy_indiv"], index=False)

            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)
            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["stages"]["05_indiv_merge"]["employment_history_filter_enabled"] = False
            cfg_path.write_text(yaml.safe_dump(cfg))

            for shard_id in range(2):
                stage_main.run(
                    config_path=cfg_path,
                    testing=False,
                    shard_count=2,
                    shard_id=shard_id,
                )

            stage_main._merge_stage05_sharded_outputs(
                config_path=cfg_path,
                shard_count=2,
            )

            person_baseline = pd.read_parquet(tempdir / "out" / "person_baseline.parquet")
            baseline = pd.read_parquet(tempdir / "out" / "baseline.parquet")
            mult2 = pd.read_parquet(tempdir / "out" / "mult2.parquet")
            assignment = {
                int(row["person_id"]): int(row["user_id"])
                for _, row in person_baseline.iterrows()
            }
            self.assertEqual(assignment, {1: 10, 2: 20})
            self.assertEqual(baseline["spell_id"].nunique(), 2)
            self.assertEqual(mult2["spell_id"].nunique(), 2)

    def test_stage05_shard_person_agg_preserves_multi_spell_person_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)

            pd.DataFrame(
                [
                    {
                        "original_row_num": 1,
                        "person_id": 1,
                        "individual_key": "INDIV-1",
                        "school_name": "Stanford University",
                        "campus_city": "Palo Alto",
                        "campus_state": "CA",
                        "campus_zip_code": "94305",
                        "country_of_birth": "INDIA",
                        "student_edu_level_desc": "MASTER'S",
                        "program_start_date": "2017-09-01",
                        "program_end_date": "2021-06-01",
                        "major_1_cip_code": "11.0701",
                        "employment_opt_type": "STEM",
                        "year": 2021,
                        "year_int": 2021,
                        "employer_name": "Google LLC",
                        "employer_city": "Mountain View",
                        "employer_state": "CA",
                        "employer_zip_code": "94043",
                    },
                    {
                        "original_row_num": 2,
                        "person_id": 1,
                        "individual_key": "INDIV-1",
                        "school_name": "Stanford University",
                        "campus_city": "Palo Alto",
                        "campus_state": "CA",
                        "campus_zip_code": "94305",
                        "country_of_birth": "INDIA",
                        "student_edu_level_desc": "MASTER'S",
                        "program_start_date": "2019-09-01",
                        "program_end_date": "2021-06-01",
                        "major_1_cip_code": "11.0701",
                        "employment_opt_type": "STEM",
                        "year": 2021,
                        "year_int": 2021,
                        "employer_name": "Google LLC",
                        "employer_city": "Mountain View",
                        "employer_state": "CA",
                        "employer_zip_code": "94043",
                    },
                ]
            ).to_parquet(inputs["f1"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "education_number": 1,
                        "country_candidate": "India",
                        "country_score": 0.95,
                        "nanat_score": 0.95,
                        "institution_score": 0.80,
                        "nametrace_score": 0.70,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "subregion_candidate": "South Asia",
                        "country_uncertain_ind": 0,
                        "unitid": 1001,
                        "degree_clean": "masters",
                        "cip": 1107,
                        "employer_key": "google",
                        "university_raw": "Stanford University",
                        "ed_startdate": "2017-09-01",
                        "ed_enddate": "2021-06-01",
                        "school_match_score": 0.99,
                    },
                    {
                        "user_id": 10,
                        "education_number": 2,
                        "country_candidate": "India",
                        "country_score": 0.95,
                        "nanat_score": 0.95,
                        "institution_score": 0.80,
                        "nametrace_score": 0.70,
                        "nanat_subregion_score": 0.95,
                        "nt_subregion_score": 0.85,
                        "subregion_candidate": "South Asia",
                        "country_uncertain_ind": 0,
                        "unitid": 1001,
                        "degree_clean": "masters",
                        "cip": 1107,
                        "employer_key": "google",
                        "university_raw": "Stanford University",
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                        "school_match_score": 0.99,
                    },
                ]
            ).to_parquet(inputs["rev_match_ready"], index=False)
            pd.DataFrame(
                [
                    {
                        "user_id": 10,
                        "education_number": 1,
                        "university_raw": "Stanford University",
                        "degree_clean": "Master",
                        "field_clean": "computer science",
                        "cip": 1107,
                        "ed_startdate": "2017-09-01",
                        "ed_enddate": "2021-06-01",
                    },
                    {
                        "user_id": 10,
                        "education_number": 2,
                        "university_raw": "Stanford University",
                        "degree_clean": "Master",
                        "field_clean": "computer science",
                        "cip": 1107,
                        "ed_startdate": "2019-09-01",
                        "ed_enddate": "2021-06-01",
                    },
                ]
            ).to_parquet(inputs["rev_educ"], index=False)

            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)
            stage_main.run(
                config_path=cfg_path,
                testing=False,
                shard_count=2,
                shard_id=1,
            )

            shard_paths = self._shard_artifact_paths(cfg_path, shard_count=2, shard_id=1)
            baseline_shard = pd.read_parquet(shard_paths["baseline"])
            person_agg_shard = pd.read_parquet(shard_paths["person_agg"])

            self.assertEqual(len(baseline_shard), 1)
            self.assertEqual(len(person_agg_shard), 1)
            self.assertEqual(int(person_agg_shard.iloc[0]["person_id"]), 1)
            self.assertEqual(int(person_agg_shard.iloc[0]["n_spell_matches"]), 2)
            self.assertGreater(
                float(person_agg_shard.iloc[0]["person_score_sum"]),
                float(baseline_shard.iloc[0]["total_score"]),
            )

    def test_stage05_merge_requires_shard_count_when_config_has_no_shard_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            with self.assertRaisesRegex(ValueError, "Shard merge requires `shard_count`"):
                stage_main._merge_stage05_sharded_outputs(config_path=cfg_path)

    def test_stage05_run_rejects_invalid_person_shard_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            with self.assertRaisesRegex(ValueError, "requires both `shard_count` and `shard_id`"):
                stage_main.run(config_path=cfg_path, testing=False, shard_count=2)
            with self.assertRaisesRegex(ValueError, "0 <= person_shard_id < person_shard_count"):
                stage_main.run(config_path=cfg_path, testing=False, shard_count=2, shard_id=2)

    def test_stage05_wrapper_rejects_forwarded_shard_management_flags(self) -> None:
        script_path = PIPELINE_ROOT / "05_indiv_merge" / "run_stage05_shards.sh"
        result = subprocess.run(
            [
                "bash",
                str(script_path),
                "--merge-only",
                "--",
                "--merge-shards",
            ],
            cwd=PIPELINE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("wrapper-managed flag forwarded via stage_main args: --merge-shards", result.stderr)

    def test_stage05_audit_person_candidates_prints_raw_candidates_before_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            _, merge_module = stage_main._load_local_merge_modules(cfg_path)
            merge_module.build_f1_merge_inputs(testing=True, con=merge_module.con_f1)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                merge_module.audit_person_candidates(1, testing=True, con=merge_module.con_f1)
            text = buffer.getvalue()

            self.assertIn("RAW CANDIDATE AUDIT: person_id=1", text)
            self.assertIn("Candidate source:", text)
            self.assertIn("SPELL 1 | raw candidates=2 rows", text)
            self.assertIn("rev_gradyr=2021", text)
            self.assertIn("rev_gradyr=2020", text)

    def test_stage05_check_person_prints_user_name_and_country_score(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            _, merge_module = stage_main._load_local_merge_modules(cfg_path)
            merge_module.build_f1_merge_inputs(testing=True, con=merge_module.con_f1)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                merge_module.check_person(1, testing=True, con=merge_module.con_f1)
            text = buffer.getvalue()

            self.assertIn("name=Ravi Patel", text)
            self.assertIn("country_score=1.000", text)

    def test_match_unit_rows_split_same_individual_key_across_years(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE MACRO initcap(x) AS
                    CASE
                        WHEN x IS NULL OR length(x) = 0 THEN x
                        ELSE upper(left(x, 1)) || lower(substr(x, 2))
                    END
                """
            )
            con.execute(
                """
                CREATE TABLE f1_foia AS
                SELECT *
                FROM (
                    VALUES
                        (1::BIGINT, 'INDIV-1', 'Stanford University', 'INDIA', 'MASTER''S',
                         DATE '2018-09-01', DATE '2020-06-01', '11.0701', 'STEM', 2020::BIGINT, 2020::BIGINT),
                        (1::BIGINT, 'INDIV-1', 'Stanford University', 'INDIA', 'MASTER''S',
                         DATE '2020-09-01', DATE '2021-06-01', '11.0701', 'STEM', 2021::BIGINT, 2021::BIGINT)
                ) AS t(
                    person_id,
                    individual_key,
                    school_name,
                    country_of_birth,
                    student_edu_level_desc,
                    program_start_date,
                    program_end_date,
                    major_1_cip_code,
                    employment_opt_type,
                    year,
                    year_int
                )
                """
            )
            con.execute(
                """
                CREATE TABLE _country_cw AS
                SELECT * FROM (
                    VALUES ('INDIA', 'India')
                ) AS t(key_upper, std_country)
                """
            )

            match_unit_q = merge_logic._build_f1_match_unit_rows_query("f1_foia")
            con.execute(f"CREATE TABLE f1_match_unit_rows AS {match_unit_q}")
            con.execute(
                """
                CREATE TABLE f1_school_rows AS
                SELECT
                    *,
                    1::BIGINT AS school_match_count_pre,
                    1::BIGINT AS school_match_count_post_row,
                    NULL::BIGINT AS resolved_f1_row_num_row,
                    NULL::BIGINT AS resolved_unitid_row
                FROM f1_match_unit_rows
                """
            )

            spells_q = merge_logic._build_f1_educ_spells_query("f1_school_rows")
            con.execute(f"CREATE TABLE f1_educ_spells AS {spells_q}")

            match_unit_stats = con.sql(
                """
                SELECT
                    COUNT(DISTINCT individual_key) AS n_individual_keys,
                    COUNT(DISTINCT person_id) AS n_match_person_ids
                FROM f1_match_unit_rows
                """
            ).fetchone()
            self.assertEqual(int(match_unit_stats[0]), 1)
            self.assertEqual(int(match_unit_stats[1]), 2)

            spell_counts = con.sql(
                """
                SELECT person_id, COUNT(*) AS n_spells
                FROM f1_educ_spells
                GROUP BY person_id
                ORDER BY person_id
                """
            ).fetchall()
            self.assertEqual(spell_counts, [(1, 1), (2, 1)])
        finally:
            con.close()

    def test_duckdb_connections_use_distinct_temp_directories(self) -> None:
        con_a = merge_logic.get_duckdb_connection()
        con_b = merge_logic.get_duckdb_connection()
        try:
            temp_dir_a = Path(con_a.execute("SELECT current_setting('temp_directory')").fetchone()[0])
            temp_dir_b = Path(con_b.execute("SELECT current_setting('temp_directory')").fetchone()[0])
            self.assertNotEqual(temp_dir_a, temp_dir_b)
            self.assertEqual(temp_dir_a.parent, temp_dir_b.parent)
            self.assertEqual(temp_dir_a.parent.name, "f1_indiv_merge_duckdb")
            self.assertTrue(temp_dir_a.is_dir())
            self.assertTrue(temp_dir_b.is_dir())
        finally:
            con_a.close()
            con_b.close()

    def test_stage05_launch_ipython_session_publishes_namespace_inside_ipykernel(self) -> None:
        fake_merge_module = types.SimpleNamespace(
            con_f1=object(),
            build_f1_merge_inputs=Mock(),
            get_runtime_table_names=Mock(return_value={"candidates": "_f1_candidates"}),
            audit_person_candidates=Mock(),
            check_person=Mock(),
            cfg=types.SimpleNamespace(
                TESTING_SAMPLE_N_PERSONS=100,
                TESTING_MATERIALIZE_INTERMEDIATE_TABLES=True,
                TESTING_ENABLED=False,
                TESTING_INDIVIDUAL_KEYS=None,
                TESTING_PERSON_IDS=None,
            ),
        )
        cfg = {"testing": {"enabled": False}}
        with patch.object(stage_main, "_running_in_ipykernel", return_value=True), patch.object(
            stage_main,
            "_load_local_merge_modules",
            return_value=(types.SimpleNamespace(), fake_merge_module),
        ), patch.object(stage_main, "_publish_interactive_namespace") as publish_namespace, patch(
            "IPython.start_ipython"
        ) as start_ipython:
            namespace = stage_main.launch_ipython_session(
                pipeline_cfg=cfg,
                testing=True,
            )

        fake_merge_module.build_f1_merge_inputs.assert_called_once_with(testing=True, con=fake_merge_module.con_f1)
        publish_namespace.assert_called_once()
        start_ipython.assert_not_called()
        self.assertEqual(fake_merge_module.cfg.TESTING_SAMPLE_N_PERSONS, 10)
        self.assertIsNone(fake_merge_module.cfg.TESTING_INDIVIDUAL_KEYS)
        self.assertIsNone(fake_merge_module.cfg.TESTING_PERSON_IDS)
        self.assertIs(namespace["merge"], fake_merge_module)
        self.assertIs(namespace["con"], fake_merge_module.con_f1)
        self.assertIn("audit_person", namespace)
        self.assertIn("check_person", namespace)

        namespace["audit_person"](123)
        namespace["check_person"](456)
        fake_merge_module.audit_person_candidates.assert_called_once_with(123, testing=True, con=fake_merge_module.con_f1)
        fake_merge_module.check_person.assert_called_once_with(456, testing=True, con=fake_merge_module.con_f1)

    def test_stage05_main_auto_launches_interactive_session_in_ipykernel(self) -> None:
        cfg = {"testing": {"enabled": False}}
        with patch.object(stage_main, "sanitize_ipykernel_argv", return_value=[]), patch.object(
            stage_main,
            "_running_in_ipykernel",
            return_value=True,
        ), patch.object(stage_main.cfg_loader, "load_config", return_value=cfg), patch.object(
            stage_main,
            "launch_ipython_session",
        ) as launch_ipython_session, patch.object(stage_main, "run") as run_stage:
            stage_main.main()

        launch_ipython_session.assert_called_once()
        run_stage.assert_not_called()

    def test_stage05_launch_ipython_session_accepts_explicit_individual_keys(self) -> None:
        fake_merge_module = types.SimpleNamespace(
            con_f1=object(),
            build_f1_merge_inputs=Mock(),
            get_runtime_table_names=Mock(return_value={"candidates": "_f1_candidates"}),
            audit_person_candidates=Mock(),
            check_person=Mock(),
            cfg=types.SimpleNamespace(
                TESTING_SAMPLE_N_PERSONS=100,
                TESTING_MATERIALIZE_INTERMEDIATE_TABLES=True,
                TESTING_ENABLED=False,
                TESTING_INDIVIDUAL_KEYS=None,
                TESTING_PERSON_IDS=None,
            ),
        )
        cfg = {"testing": {"enabled": False}}
        with patch.object(stage_main, "_running_in_ipykernel", return_value=True), patch.object(
            stage_main,
            "_load_local_merge_modules",
            return_value=(types.SimpleNamespace(), fake_merge_module),
        ), patch.object(stage_main, "_publish_interactive_namespace"), patch("IPython.start_ipython"):
            stage_main.launch_ipython_session(
                pipeline_cfg=cfg,
                testing=True,
                individual_keys=["abc", "xyz"],
            )

        self.assertEqual(fake_merge_module.cfg.TESTING_INDIVIDUAL_KEYS, ["abc", "xyz"])
        self.assertIsNone(fake_merge_module.cfg.TESTING_PERSON_IDS)
        self.assertEqual(fake_merge_module.cfg.TESTING_SAMPLE_N_PERSONS, 2)

    def test_relabel_sample_query_uses_school_campus_join_not_original_row_num(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE foia_input AS
                SELECT *
                FROM (
                    VALUES
                        (1001::BIGINT, 'Stanford University', 'Palo Alto', 'CA', '94305', DATE '2018-06-01', 2018, 'MASTER''S', '45.0601')
                ) AS t(
                    original_row_num,
                    school_name,
                    campus_city,
                    campus_state,
                    campus_zip_code,
                    program_end_date,
                    year_int,
                    student_edu_level_desc,
                    major_1_cip_code
                )
                """
            )
            con.execute(
                """
                CREATE TABLE f1_inst_unitid AS
                SELECT *
                FROM (
                    VALUES
                        (7::BIGINT, 'Stanford University', 'palo alto', 'CA', '94305', 12345::BIGINT)
                ) AS t(
                    f1_row_num,
                    school_name,
                    f1_city_clean,
                    f1_state_clean,
                    f1_zip_clean,
                    unitid
                )
                """
            )
            con.execute(
                """
                CREATE TABLE treated_events AS
                SELECT *
                FROM (
                    VALUES
                        (12345::BIGINT, 2018::BIGINT, 'econ_to_econometrics')
                ) AS t(unitid, relabel_year, relabel_type)
                """
            )
            con.execute(
                """
                CREATE TABLE matched_pairs AS
                SELECT *
                FROM (
                    VALUES
                        (NULL::BIGINT, NULL::BIGINT, NULL::VARCHAR)
                ) AS t(control_unitid, relabel_year, relabel_type)
                WHERE 1 = 0
                """
            )

            sample_query = merge_logic._build_f1_relabel_sample_query(
                "foia_input",
                "f1_inst_unitid",
                "treated_events",
                "matched_pairs",
                gradyear_window=5,
            )
            sample_df = con.sql(sample_query).df()

            self.assertEqual(sample_df.shape[0], 1)
            self.assertEqual(int(sample_df.iloc[0]["sample_unitid"]), 12345)
            self.assertEqual(sample_df.iloc[0]["sample_role"], "treated")
            self.assertEqual(int(sample_df.iloc[0]["original_row_num"]), 1001)
        finally:
            con.close()

    def test_testing_individual_key_pin_bypasses_random_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            inputs = self._write_fixture_inputs(tempdir)
            cfg_path = self._write_config(tempdir, inputs, use_stage4_indiv=True)

            _, merge_module = stage_main._load_local_merge_modules(cfg_path)
            merge_module.cfg.TESTING_INDIVIDUAL_KEYS = ["INDIV-1"]
            merge_module.cfg.TESTING_PERSON_IDS = None
            merge_module.cfg.TESTING_SAMPLE_N_PERSONS = 1

            try:
                merge_module._load_data(con=merge_module.con_f1)
                query = merge_module.build_f1_merge_inputs.__globals__["_build_f1_school_rows_query"]  # keep module loaded
                _ = query
                seed = merge_module.cfg.TESTING_RANDOM_SEED or 42
                n_sample = merge_module.cfg.TESTING_SAMPLE_N_PERSONS
                individual_keys_pin = list(getattr(merge_module.cfg, "TESTING_INDIVIDUAL_KEYS", []) or [])
                person_ids_pin = list(getattr(merge_module.cfg, "TESTING_PERSON_IDS", []) or [])
                f1_foia_src = merge_module._build_testing_f1_foia_source_query(
                    "f1_foia",
                    n_sample=n_sample,
                    seed=seed,
                    individual_keys_pin=individual_keys_pin,
                    person_ids_pin=person_ids_pin,
                    school_pin=merge_module.cfg.TESTING_SCHOOL,
                    country_pin=merge_module.cfg.TESTING_COUNTRY,
                )
                result = merge_module.con_f1.sql(
                    f"SELECT COUNT(*), MIN(person_id), MAX(person_id), MIN(individual_key), MAX(individual_key) FROM {f1_foia_src}"
                ).fetchone()
                self.assertEqual(int(result[0]), 1)
                self.assertEqual(int(result[1]), 1)
                self.assertEqual(int(result[2]), 1)
                self.assertEqual(result[3], "INDIV-1")
                self.assertEqual(result[4], "INDIV-1")
            finally:
                merge_module.con_f1.close()

    def test_testing_random_sampling_samples_distinct_individual_keys(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            rows = [
                {
                    "person_id": idx + 1,
                    "individual_key": "INDIV-1",
                    "program_end_date": "2021-06-01",
                    "year_int": 2021,
                    "school_name": "School A",
                    "country_of_birth": "INDIA",
                }
                for idx in range(1000)
            ]
            rows.extend(
                {
                    "person_id": 1000 + idx,
                    "individual_key": f"INDIV-{idx}",
                    "program_end_date": "2021-06-01",
                    "year_int": 2021,
                    "school_name": "School A",
                    "country_of_birth": "INDIA",
                }
                for idx in range(2, 22)
            )
            con.register("f1_input_py", pd.DataFrame(rows))
            con.execute("CREATE TABLE f1_foia AS SELECT * FROM f1_input_py")

            f1_foia_src = merge_logic._build_testing_f1_foia_source_query(
                "f1_foia",
                n_sample=10,
                seed=20,
            )
            result = con.sql(
                f"SELECT COUNT(DISTINCT individual_key) AS n_individual_keys FROM {f1_foia_src}"
            ).fetchone()

            self.assertEqual(int(result[0]), 10)
        finally:
            con.close()

    def test_relabel_sample_cache_is_reused_when_force_rebuild_false(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cache_path = tempdir / "relabel_sample.parquet"
            pd.DataFrame(
                [
                    {
                        "original_row_num": 1,
                        "person_id": 11,
                        "sample_unitid": 12345,
                        "sample_grad_year": 2018,
                        "sample_relabel_year": 2018,
                        "sample_relabel_type": "econ_to_econometrics",
                        "sample_role": "treated",
                    }
                ]
            ).to_parquet(cache_path, index=False)

            con = duckdb.connect(database=":memory:")
            try:
                with patch.object(merge_logic, "materialize_table", wraps=merge_logic.materialize_table) as materialize_table, patch.object(
                    merge_logic,
                    "write_query_to_parquet",
                ) as write_query_to_parquet:
                    table_name = merge_logic._materialize_relabel_sample(
                        con,
                        "SELECT 999 AS impossible_query_should_not_run",
                        cache_path=str(cache_path),
                        force_rebuild=False,
                    )

                self.assertEqual(table_name, "_f1_foia_relabel_sample")
                self.assertEqual(materialize_table.call_count, 1)
                self.assertIn("read_parquet", materialize_table.call_args.args[1])
                self.assertEqual(
                    int(con.sql("SELECT COUNT(*) FROM _f1_foia_relabel_sample").fetchone()[0]),
                    1,
                )
                write_query_to_parquet.assert_not_called()
            finally:
                con.close()

    def test_employer_match_pairs_use_buffered_year_overlap(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE candidates AS
                SELECT 1::BIGINT AS person_id, 10::BIGINT AS user_id
                """
            )
            con.execute(
                """
                CREATE TABLE f1_opt_employers AS
                SELECT
                    1::BIGINT AS person_id,
                    'google llc' AS employer_name_clean,
                    'google' AS employer_name_normed,
                    900::BIGINT AS rcid,
                    2018::BIGINT AS min_f1_year,
                    2020::BIGINT AS max_f1_year
                """
            )
            con.execute(
                """
                CREATE TABLE rev_pos_full AS
                SELECT
                    10::BIGINT AS user_id,
                    900::BIGINT AS rcid,
                    'google llc' AS company_raw_clean,
                    'google' AS company_raw_normed,
                    2021::BIGINT AS pos_start_year,
                    2021::BIGINT AS pos_end_year
                """
            )
            con.execute(
                """
                CREATE TABLE emp_idf AS
                SELECT 900::BIGINT AS rcid, 1.0::DOUBLE AS idf_weight
                """
            )
            con.execute(
                """
                CREATE TABLE token_idf AS
                SELECT * FROM (VALUES ('google', 1::BIGINT, 1.0::DOUBLE)) AS t(token, n_companies, token_idf)
                """
            )

            q0 = merge_logic._build_match_pairs_query(
                "candidates",
                "f1_opt_employers",
                "rev_pos_full",
                "emp_idf",
                "token_idf",
                EMPLOYER_MATCH_YEAR_BUFFER=0,
            )
            q2 = merge_logic._build_match_pairs_query(
                "candidates",
                "f1_opt_employers",
                "rev_pos_full",
                "emp_idf",
                "token_idf",
                EMPLOYER_MATCH_YEAR_BUFFER=2,
            )

            self.assertEqual(int(con.sql(f"SELECT COUNT(*) FROM ({q0})").fetchone()[0]), 0)
            self.assertEqual(int(con.sql(f"SELECT COUNT(*) FROM ({q2})").fetchone()[0]), 1)
        finally:
            con.close()

    def test_multiplicative_score_only_multiplies_emp_against_nonemp_average(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE candidates AS
                SELECT *
                FROM (
                    VALUES
                        (1::BIGINT, 101::BIGINT, 0.25::DOUBLE, 0.50::DOUBLE, 0.75::DOUBLE, 1.00::DOUBLE),
                        (2::BIGINT, 202::BIGINT, 0.20::DOUBLE, 0.40::DOUBLE, 0.60::DOUBLE, 0.80::DOUBLE)
                ) AS t(
                    person_id,
                    user_id,
                    gradyr_score,
                    country_score,
                    inst_score,
                    field_score
                )
                """
            )
            con.execute(
                """
                CREATE TABLE emp_scores AS
                SELECT *
                FROM (
                    VALUES
                        (1::BIGINT, 101::BIGINT, 2::BIGINT, 1::BIGINT, 1::INTEGER, 0.80::DOUBLE)
                ) AS t(
                    person_id,
                    user_id,
                    n_f1_employers,
                    n_emp_matched,
                    has_any_emp_match,
                    employer_score
                )
                """
            )

            query = merge_logic._build_merge_scored_query(
                "candidates",
                "emp_scores",
                MULTIPLICATIVE_SCORE=True,
                W_EMP_MAX=0.60,
                EMP_N_SCALE=1.0,
                W_GRADYR=0.20,
                W_COUNTRY=0.30,
                W_INST=0.10,
                W_FIELD=0.40,
            )
            result = con.sql(
                f"""
                SELECT person_id, user_id, total_score, w_emp_eff, emp_score_available_ind
                FROM ({query})
                ORDER BY person_id
                """
            ).df()

            w_emp_eff = 0.60 * (1.0 - math.exp(-2.0 / 1.0))
            nonemp_with_emp = 0.20 * 0.25 + 0.30 * 0.50 + 0.10 * 0.75 + 0.40 * 1.00
            expected_with_emp = (0.80 ** w_emp_eff) * (nonemp_with_emp ** (1.0 - w_emp_eff))
            nonemp_no_emp = 0.20 * 0.20 + 0.30 * 0.40 + 0.10 * 0.60 + 0.40 * 0.80

            self.assertEqual(int(result.iloc[0]["emp_score_available_ind"]), 1)
            self.assertAlmostEqual(float(result.iloc[0]["w_emp_eff"]), w_emp_eff, places=8)
            self.assertAlmostEqual(float(result.iloc[0]["total_score"]), expected_with_emp, places=8)
            self.assertEqual(int(result.iloc[1]["emp_score_available_ind"]), 0)
            self.assertAlmostEqual(float(result.iloc[1]["w_emp_eff"]), 0.0, places=8)
            self.assertAlmostEqual(float(result.iloc[1]["total_score"]), nonemp_no_emp, places=8)
        finally:
            con.close()

    def test_relative_score_filter_applies_field_before_country(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE merge_scored AS
                SELECT *
                FROM (
                    VALUES
                        (1::BIGINT, 101::BIGINT, 1::INTEGER, 1::INTEGER, 0::INTEGER, 0.62::DOUBLE, 0.0::DOUBLE, NULL::DOUBLE),
                        (1::BIGINT, 202::BIGINT, 1::INTEGER, 1::INTEGER, 0::INTEGER, 0.07::DOUBLE, 1.0::DOUBLE, NULL::DOUBLE),
                        (1::BIGINT, 303::BIGINT, 1::INTEGER, 1::INTEGER, 0::INTEGER, 0.05::DOUBLE, 0.0::DOUBLE, NULL::DOUBLE)
                ) AS t(
                    spell_id,
                    user_id,
                    employment_history_pass_ind,
                    field_candidate_pass_ind,
                    emp_score_available_ind,
                    country_score,
                    field_score,
                    employer_score
                )
                """
            )

            q = merge_logic._build_candidate_filtered_query(
                "merge_scored",
                relative_score_filter_enabled=True,
                employer_score_relative_buffer=0.15,
                employer_score_relative_apply_min=0.35,
                field_score_relative_buffer=0.20,
                field_score_relative_apply_min=0.75,
                country_score_relative_buffer=0.20,
                country_score_relative_apply_min=0.35,
            )
            kept = con.sql(f"SELECT user_id FROM ({q}) ORDER BY user_id").df()["user_id"].tolist()
            self.assertEqual(kept, [202])
        finally:
            con.close()

    def test_relative_score_filter_applies_country_when_no_field_signal(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE merge_scored AS
                SELECT *
                FROM (
                    VALUES
                        (2::BIGINT, 111::BIGINT, 1::INTEGER, 1::INTEGER, 0::INTEGER, 0.45::DOUBLE, 0.0::DOUBLE, NULL::DOUBLE),
                        (2::BIGINT, 222::BIGINT, 1::INTEGER, 1::INTEGER, 0::INTEGER, 0.10::DOUBLE, 0.0::DOUBLE, NULL::DOUBLE)
                ) AS t(
                    spell_id,
                    user_id,
                    employment_history_pass_ind,
                    field_candidate_pass_ind,
                    emp_score_available_ind,
                    country_score,
                    field_score,
                    employer_score
                )
                """
            )

            q = merge_logic._build_candidate_filtered_query(
                "merge_scored",
                relative_score_filter_enabled=True,
                employer_score_relative_buffer=0.15,
                employer_score_relative_apply_min=0.35,
                field_score_relative_buffer=0.20,
                field_score_relative_apply_min=0.75,
                country_score_relative_buffer=0.20,
                country_score_relative_apply_min=0.35,
            )
            kept = con.sql(f"SELECT user_id FROM ({q}) ORDER BY user_id").df()["user_id"].tolist()
            self.assertEqual(kept, [111])
        finally:
            con.close()

    def test_relative_score_filter_applies_employer_first(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(
                """
                CREATE TABLE merge_scored AS
                SELECT *
                FROM (
                    VALUES
                        (3::BIGINT, 111::BIGINT, 1::INTEGER, 1::INTEGER, 1::INTEGER, 0.50::DOUBLE, 0.0::DOUBLE, 0.55::DOUBLE),
                        (3::BIGINT, 222::BIGINT, 1::INTEGER, 1::INTEGER, 1::INTEGER, 0.49::DOUBLE, 0.0::DOUBLE, 0.30::DOUBLE),
                        (3::BIGINT, 333::BIGINT, 1::INTEGER, 1::INTEGER, 1::INTEGER, 0.48::DOUBLE, 0.0::DOUBLE, 0.10::DOUBLE)
                ) AS t(
                    spell_id,
                    user_id,
                    employment_history_pass_ind,
                    field_candidate_pass_ind,
                    emp_score_available_ind,
                    country_score,
                    field_score,
                    employer_score
                )
                """
            )

            q = merge_logic._build_candidate_filtered_query(
                "merge_scored",
                relative_score_filter_enabled=True,
                employer_score_relative_buffer=0.15,
                employer_score_relative_apply_min=0.35,
                field_score_relative_buffer=0.20,
                field_score_relative_apply_min=0.75,
                country_score_relative_buffer=0.20,
                country_score_relative_apply_min=0.35,
            )
            kept = con.sql(f"SELECT user_id FROM ({q}) ORDER BY user_id").df()["user_id"].tolist()
            self.assertEqual(kept, [111])
        finally:
            con.close()

    def test_unitid_candidates_use_general_cip_agreement_scoring(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.create_function("initcap", lambda value: value.title() if value else value, ["VARCHAR"], "VARCHAR")
            con.execute(
                """
                CREATE TABLE _country_cw AS
                SELECT *
                FROM (
                    VALUES ('INDIA', 'India')
                ) AS t(key_upper, std_country)
                """
            )
            con.execute(
                """
                CREATE TABLE rev_match_ready AS
                SELECT *
                FROM (
                    VALUES
                        (101::BIGINT, 1::BIGINT, 1001::BIGINT, 'masters', 'India', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.70::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 'South Asia', 0::INTEGER, 1107::BIGINT, 0.80::DOUBLE, 1::INTEGER, 'Stanford University', 'computer science', '2019-09-01', '2021-06-01', 0.99::DOUBLE),
                        (202::BIGINT, 1::BIGINT, 1001::BIGINT, 'masters', 'India', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.70::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 'South Asia', 0::INTEGER, 1101::BIGINT, 0.80::DOUBLE, 1::INTEGER, 'Stanford University', 'computer science', '2019-09-01', '2021-06-01', 0.99::DOUBLE),
                        (303::BIGINT, 1::BIGINT, 1001::BIGINT, 'masters', 'India', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.70::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 'South Asia', 0::INTEGER, 1409::BIGINT, 0.80::DOUBLE, 1::INTEGER, 'Stanford University', 'computer science', '2019-09-01', '2021-06-01', 0.99::DOUBLE)
                ) AS t(
                    user_id,
                    education_number,
                    unitid,
                    degree_clean,
                    country_candidate,
                    country_score,
                    nanat_score,
                    institution_score,
                    nametrace_score,
                    nanat_subregion_score,
                    nt_subregion_score,
                    subregion_candidate,
                    country_uncertain_ind,
                    cip,
                    cip_score,
                    field_mapped_ind,
                    university_raw,
                    field_clean,
                    ed_startdate,
                    ed_enddate,
                    school_match_score
                )
                """
            )
            rev_match_ready_educ_q = merge_logic._build_rev_match_ready_educ_query(
                "rev_match_ready",
                merge_logic._describe_relation_columns(con, "rev_match_ready"),
            )
            con.execute(f"CREATE TABLE rev_match_ready_educ AS {rev_match_ready_educ_q}")
            con.execute(
                """
                CREATE TABLE f1_spells AS
                SELECT *
                FROM (
                    VALUES
                        (1::BIGINT, 1::BIGINT, 'Stanford University', 'India', 'Master', 2019::BIGINT, 2021::BIGINT, '11.0701', 1107::BIGINT, 'STEM', 1::BIGINT, 2021::BIGINT, 2021::BIGINT, 'location_unique', 7::BIGINT, 1001::BIGINT, 1::INTEGER, 1::BIGINT, 1::BIGINT)
                ) AS t(
                    spell_id,
                    person_id,
                    school_name,
                    f1_country_std,
                    f1_degree_level,
                    f1_prog_start_year,
                    f1_prog_end_year,
                    f1_cip6,
                    f1_cip4,
                    f1_opt_type,
                    n_f1_years,
                    f1_year_min,
                    f1_year_max,
                    school_resolution_status,
                    resolved_f1_row_num,
                    resolved_unitid,
                    school_block_applied_ind,
                    school_match_count_pre,
                    school_match_count_post
                )
                """
            )
            con.execute(
                """
                CREATE TABLE rev_indiv AS
                SELECT *
                FROM (
                    VALUES
                        (101::BIGINT, 'Ravi Patel', 'India', 'South Asia', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 0.95::DOUBLE, 0::INTEGER, 1997::BIGINT, 1::INTEGER, 0.10::DOUBLE, '["computer science"]', 'Master', 'India'),
                        (202::BIGINT, 'Asha Rao', 'India', 'South Asia', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 0.95::DOUBLE, 0::INTEGER, 1997::BIGINT, 1::INTEGER, 0.10::DOUBLE, '["computer science"]', 'Master', 'India'),
                        (303::BIGINT, 'Neha Das', 'India', 'South Asia', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 0.95::DOUBLE, 0::INTEGER, 1997::BIGINT, 1::INTEGER, 0.10::DOUBLE, '["computer science"]', 'Master', 'India')
                ) AS t(
                    user_id,
                    fullname,
                    country,
                    subregion,
                    country_score,
                    country_score_stage4,
                    institution_score,
                    nanat_score,
                    nt_subregion_score,
                    nanat_subregion_score,
                    country_uncertain_ind,
                    est_yob,
                    stem_ind,
                    f_prob,
                    fields,
                    highest_ed_level,
                    country_std
                )
                """
            )

            q = merge_logic._build_candidates_unitid_query(
                "f1_spells",
                "rev_match_ready_educ",
                "rev_indiv",
                FIELD_FILTER_MIN_SCORE=0.25,
                FIELD_CIP2_MATCH_MULTIPLIER=0.70,
            )
            result = con.sql(
                f"""
                SELECT user_id, ROUND(field_score, 3) AS field_score, field_candidate_pass_ind
                FROM ({q})
                ORDER BY user_id
                """
            ).df()

            self.assertEqual(result["user_id"].tolist(), [101, 202, 303])
            self.assertEqual(result["field_score"].tolist(), [0.8, 0.56, 0.0])
            self.assertEqual(result["field_candidate_pass_ind"].tolist(), [1, 1, 0])
        finally:
            con.close()

    def test_rev_match_ready_active_path_does_not_apply_field_clean_heuristic(self) -> None:
        con = duckdb.connect(database=":memory:")
        try:
            con.create_function("initcap", lambda value: value.title() if value else value, ["VARCHAR"], "VARCHAR")
            con.execute(
                """
                CREATE TABLE _country_cw AS
                SELECT *
                FROM (
                    VALUES ('INDIA', 'India')
                ) AS t(key_upper, std_country)
                """
            )
            con.execute(
                """
                CREATE TABLE rev_match_ready AS
                SELECT *
                FROM (
                    VALUES
                        (101::BIGINT, 1::BIGINT, 1001::BIGINT, 'masters', 'India', 0.95::DOUBLE, 0.95::DOUBLE, 0.80::DOUBLE, 0.70::DOUBLE, 0.95::DOUBLE, 0.85::DOUBLE, 'South Asia', 0::INTEGER, NULL::BIGINT, NULL::DOUBLE, 0::INTEGER, 'Stanford University', 'computer science', '2019-09-01', '2021-06-01', 0.99::DOUBLE)
                ) AS t(
                    user_id,
                    education_number,
                    unitid,
                    degree_clean,
                    country_candidate,
                    country_score,
                    nanat_score,
                    institution_score,
                    nametrace_score,
                    nanat_subregion_score,
                    nt_subregion_score,
                    subregion_candidate,
                    country_uncertain_ind,
                    cip,
                    cip_score,
                    field_mapped_ind,
                    university_raw,
                    field_clean,
                    ed_startdate,
                    ed_enddate,
                    school_match_score
                )
                """
            )
            rev_match_ready_educ_q = merge_logic._build_rev_match_ready_educ_query(
                "rev_match_ready",
                merge_logic._describe_relation_columns(con, "rev_match_ready"),
            )
            row = con.execute(rev_match_ready_educ_q).fetchdf().iloc[0]
            self.assertTrue(pd.isna(row["rev_cip4"]))
            self.assertTrue(pd.isna(row["rev_cip_score"]))
            self.assertEqual(int(row["rev_field_mapped_ind"]), 0)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
