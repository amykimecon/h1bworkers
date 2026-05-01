from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "relabels_revelio" / "relabel_indiv_analysis.py"
spec = importlib.util.spec_from_file_location(
    "relabel_indiv_analysis_test_module",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class RelabelIndivAnalysisTests(unittest.TestCase):
    def _write_fixture_inputs(self, tempdir: Path) -> dict[str, object]:
        stage04 = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "education_number": 10,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450601",
                    "university_raw": "Treated University",
                    "field_clean": "economics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-05-15",
                    "school_match_score": 0.97,
                },
                {
                    "user_id": 1,
                    "education_number": 11,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2017-09-01",
                    "ed_enddate": "2019-05-15",
                    "school_match_score": 0.96,
                },
                {
                    "user_id": 2,
                    "education_number": 20,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "520201",
                    "university_raw": "Treated University",
                    "field_clean": "business",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.95,
                },
                {
                    "user_id": 10,
                    "education_number": 100,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "520201",
                    "university_raw": "Treated University",
                    "field_clean": "business",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.95,
                },
                {
                    "user_id": 10,
                    "education_number": 101,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "040201",
                    "university_raw": "Treated University",
                    "field_clean": "architecture",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.95,
                },
                {
                    "user_id": 11,
                    "education_number": 110,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "52.02",
                    "university_raw": "Treated University",
                    "field_clean": "business",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.95,
                },
                {
                    "user_id": 3,
                    "education_number": 30,
                    "unitid": None,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Unknown University",
                    "field_clean": "economics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-05-15",
                    "school_match_score": 0.80,
                },
                {
                    "user_id": 4,
                    "education_number": 40,
                    "unitid": 2001,
                    "degree_clean": "MBA",
                    "cip": "45.0603",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": "2017-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.91,
                },
                {
                    "user_id": 5,
                    "education_number": 50,
                    "unitid": 2001,
                    "degree_clean": "Doctor",
                    "cip": "450699",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": "2016-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.89,
                },
                {
                    "user_id": 6,
                    "education_number": 60,
                    "unitid": 1001,
                    "degree_clean": "bachelors",
                    "cip": "450699",
                    "university_raw": "Treated University",
                    "field_clean": "economics",
                    "ed_startdate": "2013-09-01",
                    "ed_enddate": None,
                    "school_match_score": 0.88,
                },
                {
                    "user_id": 7,
                    "education_number": 70,
                    "unitid": 2001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": None,
                    "ed_enddate": None,
                    "school_match_score": 0.87,
                },
                {
                    "user_id": 8,
                    "education_number": 80,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-05-15",
                    "school_match_score": 0.94,
                },
                {
                    "user_id": 8,
                    "education_number": 81,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-12-31",
                    "school_match_score": 0.94,
                },
                {
                    "user_id": 9,
                    "education_number": 90,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-06-01",
                    "school_match_score": 0.93,
                },
                {
                    "user_id": 9,
                    "education_number": 91,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450603",
                    "university_raw": "Treated University",
                    "field_clean": "econometrics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2019-06-01",
                    "school_match_score": 0.93,
                },
                {
                    "user_id": 12,
                    "education_number": 120,
                    "unitid": 1001,
                    "degree_clean": "masters",
                    "cip": "450601",
                    "university_raw": "Treated University",
                    "field_clean": "economics",
                    "ed_startdate": "2019-09-01",
                    "ed_enddate": "2021-05-15",
                    "school_match_score": 0.92,
                },
                {
                    "user_id": 13,
                    "education_number": 130,
                    "unitid": 2001,
                    "degree_clean": "masters",
                    "cip": "450601",
                    "university_raw": "Control University",
                    "field_clean": "economics",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-05-15",
                    "school_match_score": 0.92,
                },
            ]
        )
        stage04_path = tempdir / "rev_match_ready.parquet"
        stage04.to_parquet(stage04_path, index=False)

        stage05 = pd.DataFrame(
            [
                {"person_id": 100, "user_id": 1, "person_match_rank": 1},
                {"person_id": 101, "user_id": 4, "person_match_rank": 1},
                {"person_id": 102, "user_id": 5, "person_match_rank": 2},
                {"person_id": 103, "user_id": 999, "person_match_rank": 1},
            ]
        )
        stage05_path = tempdir / "person_baseline.parquet"
        stage05.to_parquet(stage05_path, index=False)

        user_core = pd.DataFrame(
            [
                {"user_id": 1, "f_prob": 0.7, "top_country_candidate": "India", "est_yob": 1995},
                {"user_id": 4, "f_prob": 0.4, "top_country_candidate": "China", "est_yob": 1992},
                {"user_id": 5, "f_prob": 0.2, "top_country_candidate": "India", "est_yob": 1991},
                {"user_id": 6, "f_prob": 0.3, "top_country_candidate": "India", "est_yob": 1993},
                {"user_id": 8, "f_prob": 0.6, "top_country_candidate": "Brazil", "est_yob": None},
                {"user_id": 9, "f_prob": 0.8, "top_country_candidate": None, "est_yob": 1994},
                {"user_id": 10, "f_prob": 0.5, "top_country_candidate": "India", "est_yob": 1996},
                {"user_id": 11, "f_prob": 0.5, "top_country_candidate": "India", "est_yob": 1996},
                {"user_id": 12, "f_prob": 0.6, "top_country_candidate": "India", "est_yob": 1996},
                {"user_id": 13, "f_prob": 0.4, "top_country_candidate": "China", "est_yob": 1995},
            ]
        )
        user_core_path = tempdir / "rev_users_core.parquet"
        user_core.to_parquet(user_core_path, index=False)

        educ_long = pd.DataFrame(
            [
                {"user_id": 1, "education_number": 10, "ed_startdate": "2018-09-01", "ed_enddate": "2023-12-31", "unitid": 1001, "rsid": 7001},
                {"user_id": 1, "education_number": 11, "ed_startdate": "2017-09-01", "ed_enddate": "2019-05-31", "unitid": 1001, "rsid": 7001},
                {"user_id": 4, "education_number": 40, "ed_startdate": "2017-09-01", "ed_enddate": "2019-05-31", "unitid": 2001, "rsid": 8001},
                {"user_id": 5, "education_number": 50, "ed_startdate": "2016-09-01", "ed_enddate": "2020-05-31", "unitid": 2001, "rsid": 8002},
                {"user_id": 6, "education_number": 60, "ed_startdate": "2013-09-01", "ed_enddate": "2017-05-31", "unitid": 1001, "rsid": 7003},
                {"user_id": 8, "education_number": 80, "ed_startdate": "2018-09-01", "ed_enddate": "2019-05-15", "unitid": 1001, "rsid": 7001},
                {"user_id": 8, "education_number": 81, "ed_startdate": "2018-09-01", "ed_enddate": "2019-12-31", "unitid": 1001, "rsid": 7001},
                {"user_id": 9, "education_number": 90, "ed_startdate": "2018-09-01", "ed_enddate": "2019-06-01", "unitid": 1001, "rsid": 7002},
                {"user_id": 9, "education_number": 91, "ed_startdate": "2018-09-01", "ed_enddate": "2019-06-01", "unitid": 1001, "rsid": 7002},
                {"user_id": 10, "education_number": 100, "ed_startdate": "2019-09-01", "ed_enddate": "2021-05-31", "unitid": 1001, "rsid": 7004},
                {"user_id": 10, "education_number": 101, "ed_startdate": "2019-09-01", "ed_enddate": "2021-05-31", "unitid": 1001, "rsid": 7005},
                {"user_id": 11, "education_number": 110, "ed_startdate": "2019-09-01", "ed_enddate": "2021-05-31", "unitid": 1001, "rsid": 7006},
                {"user_id": 12, "education_number": 120, "ed_startdate": "2019-09-01", "ed_enddate": "2021-05-31", "unitid": 1001, "rsid": 7001},
                {"user_id": 13, "education_number": 130, "ed_startdate": "2018-09-01", "ed_enddate": "2020-05-31", "unitid": 2001, "rsid": 8001},
            ]
        )
        educ_long_path = tempdir / "rev_educ_clean_long.parquet"
        educ_long.to_parquet(educ_long_path, index=False)

        positions = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "rcid": 9001,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-06-30",
                    "total_compensation": 120000,
                },
                {
                    "user_id": 4,
                    "rcid": 9004,
                    "country": "United States",
                    "startdate": "2019-07-01",
                    "enddate": "2021-06-30",
                    "total_compensation": 95000,
                },
                {
                    "user_id": 5,
                    "rcid": 9005,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-06-30",
                    "total_compensation": 105000,
                },
                {
                    "user_id": 12,
                    "rcid": 9012,
                    "country": "United States",
                    "startdate": "2021-07-01",
                    "enddate": "2022-06-30",
                    "total_compensation": 110000,
                },
                {
                    "user_id": 13,
                    "rcid": 9013,
                    "country": "Canada",
                    "startdate": "2020-07-01",
                    "enddate": "2021-06-30",
                    "total_compensation": 90000,
                },
            ]
        )
        positions_path = tempdir / "positions.parquet"
        positions.to_parquet(positions_path, index=False)

        ipeds_completions = pd.DataFrame(
            [
                {"unitid": 1001, "c21basic_lab": "R1"},
                {"unitid": 2001, "c21basic_lab": "Master's Colleges"},
            ]
        )
        ipeds_path = tempdir / "ipeds_completions.parquet"
        ipeds_completions.to_parquet(ipeds_path, index=False)

        hd_dir = tempdir / "directory_info_hd"
        hd_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"UNITID": 1001, "INSTSIZE": "3"},
                {"UNITID": 2001, "INSTSIZE": "2"},
            ]
        ).to_csv(hd_dir / "hd2019.csv", index=False)
        pd.DataFrame(
            [
                {"UNITID": 1001, "INSTSIZE": "4"},
                {"UNITID": 2001, "INSTSIZE": "2"},
            ]
        ).to_csv(hd_dir / "hd2020.csv", index=False)

        relabel_df = pd.DataFrame(
            [
                {
                    "unitid": 1001,
                    "relabel_year": 2020,
                    "relabel_type": "econ_to_econometrics",
                    "event_flag": 1,
                }
            ]
        )
        matched_pairs = pd.DataFrame(
            [
                {
                    "relabel_type": "econ_to_econometrics",
                    "relabel_year": 2020,
                    "treated_unitid": 1001,
                    "control_unitid": 2001,
                }
            ]
        )
        generalized_panel = pd.DataFrame(
            [
                {
                    "unitid": 1001,
                    "relabel_year": 2020,
                    "relabel_type": "business_52_to_52",
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                    "broad_bin_eligible": 1,
                    "degree_type": "Master",
                    "awlevel": 7,
                    "event_source_cip6": "520201",
                    "target_cip6": "521399",
                    "relabel_score": 4.0,
                },
                {
                    "unitid": 1001,
                    "relabel_year": 2020,
                    "relabel_type": "architecture_design_to_built_env_stem",
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                    "broad_bin_eligible": 1,
                    "degree_type": "Master",
                    "awlevel": 7,
                    "event_source_cip6": "040201",
                    "target_cip6": "040902",
                    "relabel_score": 3.0,
                },
                {
                    "unitid": 9999,
                    "relabel_year": 2020,
                    "relabel_type": "should_be_filtered",
                    "event_flag": 0,
                    "event_origin_category": "external_only",
                    "broad_bin_eligible": 1,
                    "degree_type": "Master",
                    "awlevel": 7,
                    "event_source_cip6": "520201",
                    "target_cip6": "521399",
                    "relabel_score": 1.0,
                },
            ]
        )
        generalized_panel_path = tempdir / "generalized_relabels_panel.parquet"
        generalized_panel.to_parquet(generalized_panel_path, index=False)
        generalized_matched_pairs = pd.DataFrame(
            [
                {
                    "pair_id": 101,
                    "relabel_year": 2020,
                    "relabel_type": "business_52_to_52",
                    "degree_type": "Master",
                    "awlevel": 7,
                    "broad_pair_bin": "business_52_to_52",
                    "source_cip6": "520201",
                    "target_cip6": "521399",
                    "treated_unitid": 1001,
                    "control_unitid": 2001,
                },
                {
                    "pair_id": 102,
                    "relabel_year": 2020,
                    "relabel_type": "architecture_design_to_built_env_stem",
                    "degree_type": "Master",
                    "awlevel": 7,
                    "broad_pair_bin": "architecture_design_to_built_env_stem",
                    "source_cip6": "040201",
                    "target_cip6": "040902",
                    "treated_unitid": 1001,
                    "control_unitid": 2001,
                },
            ]
        )
        output_panel_path = tempdir / "relabel_panel.parquet"
        output_did_path = tempdir / "relabel_did.parquet"
        output_dir = tempdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "stage04_path": stage04_path,
            "stage05_path": stage05_path,
            "user_core_path": user_core_path,
            "educ_long_path": educ_long_path,
            "positions_path": positions_path,
            "ipeds_path": ipeds_path,
            "hd_dir": hd_dir,
            "relabel_df": relabel_df,
            "matched_pairs": matched_pairs,
            "generalized_panel_path": generalized_panel_path,
            "generalized_matched_pairs": generalized_matched_pairs,
            "output_panel_path": output_panel_path,
            "output_did_path": output_did_path,
            "output_dir": output_dir,
        }

    def _patch_runtime(
        self,
        fixtures: dict[str, object],
        *,
        event_window: int = 1,
        outcome_horizons: tuple[int, ...] = (1,),
        event_source_mode: str = "econ_v2",
    ) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            patch.object(analysis.cfg, "STAGE04_MERGE_READY_PARQUET", str(fixtures["stage04_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "STAGE05_PERSON_BASELINE_PARQUET", str(fixtures["stage05_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REV_POS_PARQUET", str(fixtures["positions_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REV_USERS_CORE_PARQUET", str(fixtures["user_core_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REV_EDUC_CLEAN_LONG_PARQUET", str(fixtures["educ_long_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REV_POS_CLEAN_LONG_PARQUET", str(fixtures["positions_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "IPEDS_HD_DIR", str(fixtures["hd_dir"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "OUTPUT_PANEL_PARQUET", str(fixtures["output_panel_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "OUTPUT_DID_RESULTS_PARQUET", str(fixtures["output_did_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "GENERALIZED_RELABELS_PANEL_PARQUET", str(fixtures["generalized_panel_path"]))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "REVELIO_IPEDS_INST_CW_PARQUET", "")
        )
        stack.enter_context(
            patch.object(analysis.cfg, "IPEDS_CROSSWALK_PARQUET", "")
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_SAMPLE_CIP_PREFIXES", ["4506"])
        )
        stack.enter_context(
            patch.object(
                analysis.cfg,
                "BUILD_SAMPLE_VARIANTS",
                ["stage04_all", "foia_linked_person_baseline"],
            )
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_SAMPLE_GRADYEAR_WINDOW", 5)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_EVENT_WINDOW", event_window)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_EVENT_SOURCE_MODE", event_source_mode)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_OUTCOME_HORIZONS", list(outcome_horizons))
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_CAP_TO_LATEST_AVAILABLE_YEAR", True)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_RUN_DID", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_DID_PLOT_MODE", "event_study_by_cohort")
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_INSTITUTION_MATCH_QUALITY_GATE", True)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_INSTITUTION_MATCH_SCORE_MIN", 0.85)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_INSTITUTION_ALIAS_JW_MIN", 0.92)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_RSID_SUPPORT_GATE", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_RSID_SUPPORT_MIN_SHARE", 0.05)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_RSID_SUPPORT_MIN_COUNT", 10)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_FOREIGN_HETEROGENEITY", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", False)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "BUILD_DID_COUNTRY_TOP_N", 2)
        )
        stack.enter_context(
            patch.object(analysis.cfg, "TESTING_ENABLED", False)
        )
        stack.enter_context(
            patch.object(analysis.v2.base, "IPEDS_PATH", str(fixtures["ipeds_path"]))
        )
        stack.enter_context(
            patch.object(analysis.generalized.base, "IPEDS_PATH", str(fixtures["ipeds_path"]))
        )
        stack.enter_context(
            patch.object(analysis, "OUTPUT_DIR", fixtures["output_dir"])
        )
        stack.enter_context(
            patch.object(analysis.plt, "show", lambda: None)
        )
        return stack

    def test_step2_prepare_stage04_samples_filters_and_derives_grad_year(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                variants = analysis.step2_prepare_stage04_samples(con)
                self.assertEqual(
                    variants,
                    ["stage04_all", "foia_linked_person_baseline"],
                )
                sample_all = con.sql(
                    """
                    SELECT user_id, unitid, education_number, grad_year, rsid
                    FROM stage04_sample_all
                    ORDER BY user_id, education_number
                    """
                ).df()
                self.assertNotIn(2, sample_all["user_id"].tolist())
                self.assertNotIn(3, sample_all["user_id"].tolist())
                self.assertNotIn(7, sample_all["user_id"].tolist())
                grad_years = dict(
                    sample_all.loc[
                        sample_all["education_number"].isin([40, 50, 60]),
                        ["education_number", "grad_year"],
                    ].itertuples(index=False, name=None)
                )
                self.assertEqual(grad_years[40], 2019)
                self.assertEqual(grad_years[50], 2020)
                self.assertEqual(grad_years[60], 2017)
                rsids = dict(
                    sample_all.loc[
                        sample_all["education_number"].isin([10, 81, 130]),
                        ["education_number", "rsid"],
                    ].itertuples(index=False, name=None)
                )
                self.assertEqual(rsids[10], 7001)
                self.assertEqual(rsids[81], 7001)
                self.assertEqual(rsids[130], 8001)

    def test_step3_match_treated_uses_unitid_window_dedupe_and_master_only_econ_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                self.assertEqual(set(treated["user_id"].tolist()), {1, 8, 9, 12})
                self.assertTrue(treated[["user_id", "relabel_year"]].duplicated().sum() == 0)

                picked = treated.set_index("user_id")
                self.assertEqual(int(picked.loc[1, "education_number"]), 10)
                self.assertEqual(int(picked.loc[1, "grad_year"]), 2020)
                self.assertEqual(int(picked.loc[8, "education_number"]), 81)
                self.assertEqual(int(picked.loc[9, "education_number"]), 90)

    def test_step3_institution_quality_gate_uses_score_or_unitid_alias_jw(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                con.sql(
                    """
                    CREATE OR REPLACE TEMP VIEW institution_match_aliases AS
                    SELECT *
                    FROM (
                        VALUES
                            (1001::BIGINT, 'trusted school alias'::VARCHAR)
                    ) AS t(unitid, alias_clean)
                    """
                )
                con.sql(
                    """
                    CREATE OR REPLACE TEMP VIEW quality_gate_sample AS
                    SELECT *
                    FROM (
                        VALUES
                            (100::BIGINT, 1001::BIGINT, 1::BIGINT, DATE '2020-05-15',
                             'Very Wrong Raw Name'::VARCHAR, 0.96::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR),
                            (101::BIGINT, 1001::BIGINT, 1::BIGINT, DATE '2020-05-15',
                             'Trusted School Alias'::VARCHAR, 0.50::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR),
                            (102::BIGINT, 1001::BIGINT, 1::BIGINT, DATE '2020-05-15',
                             'Wrong Remote Institute'::VARCHAR, 0.50::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR),
                            (103::BIGINT, 1001::BIGINT, 1::BIGINT, DATE '2020-05-15',
                             'Trusted School Alias'::VARCHAR, 0.50::DOUBLE, 2020::INTEGER,
                             'Bachelor'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR)
                    ) AS t(
                        user_id, unitid, education_number, ed_enddate,
                        university_raw, school_match_score, grad_year,
                        degree_type, cip_match_level, cip_match_code
                    )
                    """
                )
                events = pd.DataFrame(
                    [
                        {
                            "unitid": 1001,
                            "relabel_year": 2020,
                            "relabel_type": "econ_to_econometrics",
                        }
                    ]
                )
                matched = analysis._match_individuals_to_events(
                    con,
                    "quality_gate_sample",
                    events,
                    treated_ind=1,
                    group_label="quality_gate",
                )

            self.assertEqual(set(matched["user_id"].tolist()), {100, 101})

    def test_step3_rsid_support_gate_keeps_common_school_rsids(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures), patch.object(
                analysis.cfg, "BUILD_RSID_SUPPORT_GATE", True
            ), patch.object(
                analysis.cfg, "BUILD_RSID_SUPPORT_MIN_SHARE", 0.50
            ), patch.object(
                analysis.cfg, "BUILD_RSID_SUPPORT_MIN_COUNT", 2
            ):
                con = duckdb.connect()
                con.sql(
                    """
                    CREATE OR REPLACE TEMP VIEW rsid_gate_sample AS
                    SELECT *
                    FROM (
                        VALUES
                            (200::BIGINT, 1001::BIGINT, 1::BIGINT, DATE '2020-05-15',
                             'Treated University'::VARCHAR, 0.96::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR, 7001::BIGINT),
                            (201::BIGINT, 1001::BIGINT, 2::BIGINT, DATE '2020-05-15',
                             'Treated University'::VARCHAR, 0.96::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR, 7001::BIGINT),
                            (202::BIGINT, 1001::BIGINT, 3::BIGINT, DATE '2020-05-15',
                             'Treated University'::VARCHAR, 0.96::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR, 7002::BIGINT),
                            (203::BIGINT, 1001::BIGINT, 4::BIGINT, DATE '2020-05-15',
                             'Treated University'::VARCHAR, 0.96::DOUBLE, 2020::INTEGER,
                             'Master'::VARCHAR, 'cip6'::VARCHAR, '450601'::VARCHAR, NULL::BIGINT)
                    ) AS t(
                        user_id, unitid, education_number, ed_enddate,
                        university_raw, school_match_score, grad_year,
                        degree_type, cip_match_level, cip_match_code, rsid
                    )
                    """
                )
                events = pd.DataFrame(
                    [
                        {
                            "unitid": 1001,
                            "relabel_year": 2020,
                            "relabel_type": "econ_to_econometrics",
                        }
                    ]
                )
                matched = analysis._match_individuals_to_events(
                    con,
                    "rsid_gate_sample",
                    events,
                    treated_ind=1,
                    group_label="rsid_gate",
                )

            self.assertEqual(set(matched["user_id"].tolist()), {200, 201})
            self.assertTrue((matched["rsid_unitid_match_count"] == 2).all())
            self.assertTrue((matched["rsid_unitid_total_with_rsid"] == 3).all())
            self.assertTrue((matched["rsid_unitid_required_count"] == 2).all())

    def test_step1_generalized_loads_finalized_treated_event_universe(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_source_mode="generalized_final_sample"):
                con = duckdb.connect()
                relabel_df = analysis.step1_relabels(con)
                treated_events = analysis._generalized_treated_events(relabel_df)

            self.assertEqual(len(treated_events), 2)
            self.assertEqual(
                set(treated_events["broad_pair_bin"].tolist()),
                {"business_52_to_52", "architecture_design_to_built_env_stem"},
            )
            self.assertEqual(set(treated_events["degree_type"].tolist()), {"Master"})
            self.assertTrue((treated_events["event_origin_category"] != "external_only").all())

    def test_step2_generalized_stage04_sample_keeps_non_econ_cips(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_source_mode="generalized_final_sample"):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                sample_all = con.sql(
                    """
                    SELECT user_id, education_number, cip6
                    FROM stage04_sample_all
                    ORDER BY user_id, education_number
                    """
                ).df()

            user_ids = set(sample_all["user_id"].tolist())
            self.assertIn(2, user_ids)
            self.assertIn(10, user_ids)
            self.assertIn(11, user_ids)
            self.assertNotIn(3, user_ids)
            self.assertNotIn(7, user_ids)
            self.assertIn("520201", set(sample_all["cip6"].tolist()))
            self.assertIn("040201", set(sample_all["cip6"].tolist()))

    def test_step3_generalized_matches_event_aligned_cips_and_keeps_distinct_events(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            relabel_df = pd.read_parquet(fixtures["generalized_panel_path"])
            with self._patch_runtime(fixtures, event_source_mode="generalized_final_sample"):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    relabel_df,
                    "stage04_sample_all",
                    "stage04_all",
                )

            self.assertNotIn(1, set(treated["user_id"].tolist()))
            business_users = treated[treated["broad_pair_bin"] == "business_52_to_52"]["user_id"].tolist()
            self.assertIn(2, business_users)
            self.assertIn(11, business_users)
            user11 = treated[treated["user_id"] == 11].reset_index(drop=True)
            self.assertEqual(len(user11), 1)
            self.assertEqual(user11.loc[0, "broad_pair_bin"], "business_52_to_52")
            self.assertEqual(user11.loc[0, "degree_type"], "Master")

            user10 = treated[treated["user_id"] == 10].sort_values("broad_pair_bin").reset_index(drop=True)
            self.assertEqual(len(user10), 2)
            self.assertEqual(user10["event_id"].nunique(), 2)
            self.assertEqual(
                set(user10["broad_pair_bin"].tolist()),
                {"business_52_to_52", "architecture_design_to_built_env_stem"},
            )

    def test_build_control_events_generalized_preserves_pair_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            relabel_df = pd.read_parquet(fixtures["generalized_panel_path"])
            with self._patch_runtime(fixtures, event_source_mode="generalized_final_sample"), patch.object(
                analysis.generalized,
                "match_treated_to_never_treated",
                return_value=fixtures["generalized_matched_pairs"],
            ):
                con = duckdb.connect()
                control_events = analysis.build_control_events(con, relabel_df)

            self.assertEqual(set(control_events["pair_id"].tolist()), {101, 102})
            self.assertEqual(set(control_events["unitid"].tolist()), {2001})
            self.assertEqual(set(control_events["treated_unitid"].tolist()), {1001})
            self.assertEqual(control_events["event_id"].nunique(), 2)
            self.assertEqual(
                set(control_events["broad_pair_bin"].tolist()),
                {"business_52_to_52", "architecture_design_to_built_env_stem"},
            )

    def test_step4_generalized_panel_keeps_same_user_distinct_event_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            relabel_df = pd.read_parquet(fixtures["generalized_panel_path"])
            with self._patch_runtime(
                fixtures,
                outcome_horizons=(1,),
                event_source_mode="generalized_final_sample",
            ):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    relabel_df,
                    "stage04_sample_all",
                    "stage04_all",
                )
                panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )

            user10 = panel[panel["user_id"] == 10].sort_values("event_id").reset_index(drop=True)
            self.assertEqual(len(user10), 2)
            self.assertEqual(user10["event_id"].nunique(), 2)
            self.assertEqual(
                set(user10["broad_pair_bin"].tolist()),
                {"business_52_to_52", "architecture_design_to_built_env_stem"},
            )

    def test_foia_linked_variant_only_uses_person_baseline_users(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                foia_users = con.sql(
                    """
                    SELECT DISTINCT user_id
                    FROM stage04_sample_foia_linked_person_baseline
                    ORDER BY user_id
                    """
                ).df()["user_id"].tolist()
                self.assertEqual(foia_users, [1, 4])

                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_foia_linked_person_baseline",
                    "foia_linked_person_baseline",
                )
                self.assertEqual(treated["user_id"].tolist(), [1])

    def test_control_matching_uses_same_window_on_stage04_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                control_matches = analysis._match_individuals_to_events(
                    con,
                    "stage04_sample_all",
                    fixtures["matched_pairs"].rename(columns={"control_unitid": "unitid"})[
                        ["unitid", "relabel_year", "relabel_type"]
                    ],
                    treated_ind=0,
                    group_label="control_test",
                )
                self.assertEqual(set(control_matches["user_id"].tolist()), {4, 13})
                self.assertTrue((control_matches["treated_ind"] == 0).all())
                self.assertEqual(
                    dict(control_matches.set_index("user_id")["grad_year"].to_dict()),
                    {4: 2019, 13: 2020},
                )

    def test_step4_uses_post_graduation_horizons(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, outcome_horizons=(1, 3)):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )

            user1 = panel[panel["user_id"] == 1].sort_values("horizon_years").reset_index(drop=True)
            self.assertEqual(user1["horizon_years"].tolist(), [1, 3])

            horizon1 = user1[user1["horizon_years"] == 1].iloc[0]
            self.assertEqual(int(horizon1["target_year"]), 2021)
            self.assertEqual(int(horizon1["eval_year"]), 2021)
            self.assertEqual(int(horizon1["target_year_observed"]), 1)
            self.assertEqual(int(horizon1["in_us"]), 1)
            self.assertEqual(int(horizon1["n_pos"]), 1)
            self.assertAlmostEqual(float(horizon1["salary_imputed"]), 120000.0)

            horizon3 = user1[user1["horizon_years"] == 3].iloc[0]
            self.assertEqual(int(horizon3["target_year"]), 2023)
            self.assertEqual(int(horizon3["eval_year"]), 2022)
            self.assertEqual(int(horizon3["target_year_observed"]), 0)
            self.assertEqual(int(horizon3["used_latest_avail"]), 1)

    def test_step4_enriches_panel_with_revelio_controls_and_linkedin_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, outcome_horizons=(1, 3)):
                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                analysis._ensure_enrichment_views(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )
                panel = analysis._finalize_variant_panel(panel)

            user1 = panel[panel["user_id"] == 1].sort_values("horizon_years").reset_index(drop=True)
            self.assertEqual(float(user1.loc[0, "female_prob_raw"]), 0.7)
            self.assertEqual(user1.loc[0, "origin_country_raw"], "India")
            self.assertEqual(user1.loc[0, "origin_country_bucket"], "India")
            self.assertEqual(int(user1.loc[0, "imputed_foreign_ind"]), 1)
            self.assertEqual(user1.loc[0, "imputed_foreign_label"], "Foreign")
            self.assertEqual(int(user1.loc[0, "est_yob"]), 1995)
            self.assertEqual(float(user1.loc[0, "age_at_grad"]), 25.0)
            self.assertEqual(int(user1.loc[0, "age_missing_ind"]), 0)
            self.assertEqual(pd.Timestamp(user1.loc[0, "linkedin_last_position_date"]).date().isoformat(), "2022-06-30")
            self.assertEqual(pd.Timestamp(user1.loc[0, "linkedin_last_education_date"]).date().isoformat(), "2023-12-31")
            self.assertEqual(pd.Timestamp(user1.loc[0, "linkedin_last_activity_date"]).date().isoformat(), "2023-12-31")
            self.assertEqual(int(user1.loc[0, "linkedin_last_activity_year"]), 2023)
            self.assertEqual(user1["linkedin_active_through_target_year"].tolist(), [1, 1])
            self.assertEqual(set(user1["instsize_hd"].tolist()), {"4"})
            self.assertEqual(set(user1["c21basic_lab"].tolist()), {"R1"})

            user8 = panel[panel["user_id"] == 8].iloc[0]
            self.assertEqual(int(user8["age_missing_ind"]), 1)
            self.assertFalse(pd.isna(user8["age_at_grad"]))

            user9 = panel[panel["user_id"] == 9].iloc[0]
            self.assertEqual(user9["origin_country_raw"], "Unknown")
            self.assertEqual(int(user9["imputed_foreign_ind"]), 0)
            self.assertEqual(user9["imputed_foreign_label"], "Non-foreign")

    def test_did_formula_adds_optional_controls_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures):
                base_formula = analysis._did_formula(-1)
                self.assertIn("C(cluster_unitid)", base_formula)
                self.assertNotIn("female_prob", base_formula)
                self.assertNotIn("C(origin_country_bucket)", base_formula)
                self.assertNotIn("C(grad_year):C(instsize_hd)", base_formula)

            with self._patch_runtime(fixtures) as stack:
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_INCLUDE_SCHOOL_CHAR_GRADYEAR_CONTROLS", True))
                enriched_formula = analysis._did_formula(-1)
                self.assertIn("female_prob", enriched_formula)
                self.assertIn("age_at_grad", enriched_formula)
                self.assertIn("age_missing_ind", enriched_formula)
                self.assertIn("C(origin_country_bucket)", enriched_formula)
                self.assertIn("C(grad_year):C(instsize_hd)", enriched_formula)
                self.assertIn("C(grad_year):C(c21basic_lab)", enriched_formula)

    def test_step8_did_returns_results_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=1) as stack, patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_MODEL", "panel"))

                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                treated_panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )
                control_events = analysis.build_control_events(con, fixtures["relabel_df"])
                control_panel = analysis.step6_control_group(
                    con,
                    "stage04_sample_all",
                    control_events,
                    "stage04_all",
                )
                did_results = analysis.step8_did(
                    treated_panel,
                    control_panel,
                    analysis_variant="stage04_all",
                )

            self.assertFalse(did_results.empty)
            self.assertEqual(set(did_results["did_model"].tolist()), {"simple"})
            self.assertEqual(set(did_results["analysis_variant"].tolist()), {"stage04_all"})
            self.assertTrue(
                {"horizon_years", "cohort_t", "reference_cohort_t", "coef", "se", "pval", "n_entities"}.issubset(
                    did_results.columns
                )
            )
            self.assertEqual(set(did_results["horizon_years"].unique().tolist()), {1})
            self.assertEqual(set(did_results["cohort_t"].unique().tolist()), {0})
            self.assertEqual(set(did_results["reference_cohort_t"].unique().tolist()), {-1})
            written_files = {path.name for path in Path(fixtures["output_dir"]).glob("*.png")}
            self.assertTrue(
                {
                    "did_att_active_us_stage04_all.png",
                    "did_att_active_positions_stage04_all.png",
                }.issubset(written_files)
            )

    def test_step8_did_with_individual_controls_keeps_missing_age_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=1) as stack, patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_INCLUDE_INDIVIDUAL_CONTROLS", True))

                con = duckdb.connect()
                analysis.step2_prepare_stage04_samples(con)
                analysis._ensure_enrichment_views(con)
                treated = analysis.step3_match_treated(
                    con,
                    fixtures["relabel_df"],
                    "stage04_sample_all",
                    "stage04_all",
                )
                treated_panel = analysis.step4_build_outcome_panel(
                    con,
                    treated,
                    group_label="treated_stage04_all",
                    analysis_variant="stage04_all",
                )
                control_events = analysis.build_control_events(con, fixtures["relabel_df"])
                control_panel = analysis.step6_control_group(
                    con,
                    "stage04_sample_all",
                    control_events,
                    "stage04_all",
                )
                variant_panel = analysis._finalize_variant_panel(
                    pd.concat([treated_panel, control_panel], ignore_index=True)
                )
                did_results = analysis.step8_did(
                    variant_panel[variant_panel["treated_ind"] == 1].copy(),
                    variant_panel[variant_panel["treated_ind"] == 0].copy(),
                    analysis_variant="stage04_all",
                )

            self.assertFalse(did_results.empty)
            self.assertTrue((did_results["did_include_individual_controls"] == 1).all())
            self.assertIn("female_prob", did_results["formula"].iloc[0])

    def test_step8_pooled_post_by_horizon_returns_results_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=5) as stack:
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_PLOT_MODE", "pooled_post_by_horizon"))
                base_rows = [
                    {
                        "analysis_variant": "stage04_all",
                        "relabel_year": 2020,
                        "relabel_type": "econ_to_econometrics",
                        "horizon_years": 1,
                        "target_year": 2021,
                        "eval_year": 2021,
                        "latest_available_year": 2021,
                        "target_year_observed": 1,
                        "used_latest_avail": 0,
                        "unitid": 1001,
                        "education_number": 10,
                        "female_prob_raw": 0.5,
                        "origin_country_raw": "India",
                        "origin_country_bucket": "India",
                        "est_yob": 1995,
                        "age_at_grad_raw": 25.0,
                        "age_at_grad": 25.0,
                        "age_missing_ind": 0,
                        "linkedin_last_education_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_position_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_activity_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_activity_year": 2022,
                        "instsize_hd": "4",
                        "c21basic_lab": "R1",
                        "n_pos": 1.0,
                        "salary_imputed": 100000.0,
                        "linkedin_active_through_target_year": 1,
                        "pair_id": 501,
                        "event_id": "evt",
                        "broad_pair_bin": "business_52_to_52",
                        "degree_type": "Master",
                    }
                ]
                treated_panel = pd.DataFrame(
                    base_rows
                    + [
                        {
                            **base_rows[0],
                            "user_id": 1,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 1,
                            "in_us": 0.20,
                        },
                        {
                            **base_rows[0],
                            "user_id": 2,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 1,
                            "education_number": 11,
                            "pair_id": 502,
                            "event_id": "evt2",
                            "in_us": 0.20,
                        },
                        {
                            **base_rows[0],
                            "user_id": 3,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 1,
                            "education_number": 12,
                            "pair_id": 503,
                            "event_id": "evt3",
                            "in_us": 0.50,
                        },
                        {
                            **base_rows[0],
                            "user_id": 4,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 1,
                            "education_number": 13,
                            "pair_id": 504,
                            "event_id": "evt4",
                            "in_us": 0.50,
                        },
                        {
                            **base_rows[0],
                            "user_id": 5,
                            "grad_year": 2024,
                            "cohort_t": 4,
                            "treated_ind": 1,
                            "education_number": 14,
                            "pair_id": 505,
                            "event_id": "evt5",
                            "in_us": 99.0,
                        },
                    ]
                )
                control_panel = pd.DataFrame(
                    [
                        {
                            **base_rows[0],
                            "user_id": 10,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 0,
                            "unitid": 2001,
                            "education_number": 20,
                            "pair_id": 601,
                            "event_id": "ctrl1",
                            "in_us": 0.10,
                        },
                        {
                            **base_rows[0],
                            "user_id": 11,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 0,
                            "unitid": 2001,
                            "education_number": 21,
                            "pair_id": 602,
                            "event_id": "ctrl2",
                            "in_us": 0.10,
                        },
                        {
                            **base_rows[0],
                            "user_id": 12,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 0,
                            "unitid": 2001,
                            "education_number": 22,
                            "pair_id": 603,
                            "event_id": "ctrl3",
                            "in_us": 0.10,
                        },
                        {
                            **base_rows[0],
                            "user_id": 13,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 0,
                            "unitid": 2001,
                            "education_number": 23,
                            "pair_id": 604,
                            "event_id": "ctrl4",
                            "in_us": 0.10,
                        },
                        {
                            **base_rows[0],
                            "user_id": 14,
                            "grad_year": 2024,
                            "cohort_t": 4,
                            "treated_ind": 0,
                            "unitid": 2001,
                            "education_number": 24,
                            "pair_id": 605,
                            "event_id": "ctrl5",
                            "in_us": -99.0,
                        },
                    ]
                )
                pooled_results = analysis.step8_pooled_post_by_horizon(
                    treated_panel,
                    control_panel,
                    analysis_variant="stage04_all",
                )

            self.assertFalse(pooled_results.empty)
            self.assertTrue(
                {"horizon_years", "coef", "se", "baseline_mean", "effect_size", "treated_pre_mean", "control_post_mean"}.issubset(
                    pooled_results.columns
                )
            )
            self.assertEqual(set(pooled_results["analysis_variant"].tolist()), {"stage04_all"})
            in_us_results = pooled_results[pooled_results["outcome"] == "in_us"].copy()
            self.assertEqual(len(in_us_results), 1)
            self.assertTrue(np.allclose(in_us_results["baseline_mean"], 0.20))
            self.assertTrue(np.allclose(in_us_results["coef"], 0.30))
            self.assertEqual(int(in_us_results["n_obs"].iloc[0]), 8)
            self.assertTrue(np.isfinite(in_us_results["se"]).all())
            written_files = {path.name for path in Path(fixtures["output_dir"]).glob("*.png")}
            self.assertIn("did_att_active_us_stage04_all.png", written_files)

    def test_pooled_post_formula_omits_treated_main_effect(self) -> None:
        formula = analysis._pooled_post_formula()
        self.assertIn("post_ind", formula)
        self.assertIn("post_ind:treated_ind", formula)
        self.assertNotIn("post_ind*treated_ind", formula)

    def test_step8_did_distinguishes_same_user_horizon_by_event_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=2) as stack:
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_MODEL", "simple"))

                base_rows = [
                    {
                        "relabel_year": 2020,
                        "relabel_type": "generalized_event",
                        "horizon_years": 1,
                        "target_year": 2021,
                        "eval_year": 2021,
                        "latest_available_year": 2021,
                        "target_year_observed": 1,
                        "used_latest_avail": 0,
                        "analysis_variant": "stage04_all",
                        "unitid": 1001,
                        "education_number": 10,
                        "female_prob_raw": 0.5,
                        "origin_country_raw": "India",
                        "origin_country_bucket": "India",
                        "est_yob": 1995,
                        "age_at_grad_raw": 25.0,
                        "age_at_grad": 25.0,
                        "age_missing_ind": 0,
                        "linkedin_last_education_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_position_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_activity_date": pd.Timestamp("2022-12-31"),
                        "linkedin_last_activity_year": 2022,
                        "instsize_hd": "4",
                        "c21basic_lab": "R1",
                        "n_pos": 1,
                        "salary_imputed": 100000.0,
                        "linkedin_active_through_target_year": 1,
                    }
                ]
                treated_panel = pd.DataFrame(
                    base_rows
                    + [
                        {
                            **base_rows[0],
                            "user_id": 1,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 1,
                            "pair_id": 501,
                            "event_id": "evt_ref",
                            "broad_pair_bin": "business_52_to_52",
                            "degree_type": "Master",
                            "in_us": 1,
                        },
                        {
                            **base_rows[0],
                            "user_id": 1,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 1,
                            "pair_id": 502,
                            "event_id": "evt_business",
                            "broad_pair_bin": "business_52_to_52",
                            "degree_type": "Master",
                            "in_us": 1,
                        },
                        {
                            **base_rows[0],
                            "user_id": 1,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 1,
                            "pair_id": 503,
                            "event_id": "evt_architecture",
                            "broad_pair_bin": "architecture_design_to_built_env_stem",
                            "degree_type": "Master",
                            "in_us": 0,
                        },
                    ]
                )
                control_panel = pd.DataFrame(
                    [
                        {
                            **base_rows[0],
                            "user_id": 2,
                            "grad_year": 2018,
                            "cohort_t": -2,
                            "treated_ind": 0,
                            "pair_id": 601,
                            "event_id": "ctrl_ref",
                            "broad_pair_bin": "business_52_to_52",
                            "degree_type": "Master",
                            "unitid": 2001,
                            "education_number": 20,
                            "in_us": 0,
                        },
                        {
                            **base_rows[0],
                            "user_id": 3,
                            "grad_year": 2020,
                            "cohort_t": 0,
                            "treated_ind": 0,
                            "pair_id": 602,
                            "event_id": "ctrl_event",
                            "broad_pair_bin": "business_52_to_52",
                            "degree_type": "Master",
                            "unitid": 2001,
                            "education_number": 30,
                            "in_us": 0,
                        },
                    ]
                )
                did_results = analysis.step8_did(
                    treated_panel,
                    control_panel,
                    analysis_variant="stage04_all",
                )

            self.assertFalse(did_results.empty)
            self.assertTrue((did_results["n_entities"] == 5).all())

    def test_plot_did_variant_comparison_offsets_series_and_uses_per_outcome_names(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 0,
                    "reference_cohort_t": -1,
                    "coef": 0.01,
                    "se": 0.005,
                },
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 1,
                    "reference_cohort_t": -1,
                    "coef": 0.02,
                    "se": 0.006,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 0,
                    "reference_cohort_t": -1,
                    "coef": 0.03,
                    "se": 0.010,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "cohort_t": 1,
                    "reference_cohort_t": -1,
                    "coef": 0.04,
                    "se": 0.011,
                },
            ]
        )
        saved_names: list[str] = []
        errorbar_x: list[list[float]] = []

        def _capture_save(fig, name, analysis_variant=None):  # type: ignore[no-untyped-def]
            saved_names.append(name)
            analysis.plt.close(fig)
            return Path("/tmp/ignored.png")

        original_errorbar = matplotlib.axes._axes.Axes.errorbar

        def _capture_errorbar(self, x, *args, **kwargs):  # type: ignore[no-untyped-def]
            errorbar_x.append([float(v) for v in x])
            return original_errorbar(self, x, *args, **kwargs)

        with patch.object(analysis, "_save_and_show", side_effect=_capture_save), patch.object(
            analysis.plt, "show", lambda: None
        ), patch.object(matplotlib.axes._axes.Axes, "errorbar", new=_capture_errorbar):
            analysis._plot_did_variant_comparison(results_df)

        self.assertEqual(saved_names, ["did_att_by_variant_active_us"])
        self.assertTrue(any(any(abs(v - round(v)) > 1e-9 for v in xs) for xs in errorbar_x))

    def test_plot_pooled_variant_comparison_uses_new_output_name(self) -> None:
        results_df = pd.DataFrame(
            [
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 1,
                    "coef": 0.01,
                    "se": 0.005,
                    "baseline_mean": 0.50,
                    "effect_size": 0.02,
                },
                {
                    "analysis_variant": "stage04_all",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "coef": 0.02,
                    "se": 0.006,
                    "baseline_mean": 0.50,
                    "effect_size": 0.04,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 1,
                    "coef": 0.03,
                    "se": 0.010,
                    "baseline_mean": 0.60,
                    "effect_size": 0.05,
                },
                {
                    "analysis_variant": "foia_linked_person_baseline",
                    "did_model": "simple",
                    "outcome": "in_us",
                    "horizon_years": 3,
                    "coef": 0.04,
                    "se": 0.011,
                    "baseline_mean": 0.60,
                    "effect_size": 0.07,
                },
            ]
        )
        saved_names: list[str] = []

        def _capture_save(fig, name, analysis_variant=None):  # type: ignore[no-untyped-def]
            saved_names.append(name)
            analysis.plt.close(fig)
            return Path("/tmp/ignored.png")

        with patch.object(analysis, "_save_and_show", side_effect=_capture_save), patch.object(
            analysis.plt, "show", lambda: None
        ):
            analysis._plot_pooled_variant_comparison(
                results_df,
                file_tag="variant",
                title_label="match sample",
            )

        self.assertEqual(saved_names, ["did_att_by_variant_active_us"])

    def test_append_reference_plot_row_inserts_zero_reference_point(self) -> None:
        plot_df = pd.DataFrame(
            [
                {"cohort_t": 0.0, "coef": 0.10, "se": 0.02},
                {"cohort_t": 1.0, "coef": 0.20, "se": 0.03},
            ]
        )

        out = analysis._append_reference_plot_row(
            plot_df,
            x_col="cohort_t",
            y_col="coef",
            reference_x=-1,
            extra_values={"se": 0.0},
        )

        self.assertEqual(out["cohort_t"].tolist(), [-1.0, 0.0, 1.0])
        ref_row = out[out["cohort_t"] == -1.0].iloc[0]
        self.assertEqual(float(ref_row["coef"]), 0.0)
        self.assertEqual(float(ref_row["se"]), 0.0)

    def test_main_writes_did_results_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures, event_window=1) as stack, patch.object(
                analysis,
                "step1_relabels",
                return_value=fixtures["relabel_df"],
            ), patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                stack.enter_context(patch.object(analysis.cfg, "BUILD_RUN_DID", True))
                stack.enter_context(patch.object(analysis.cfg, "BUILD_DID_MODEL", "simple"))
                analysis.main()

            did = pd.read_parquet(fixtures["output_did_path"])
            self.assertFalse(did.empty)
            self.assertEqual(
                set(did["analysis_variant"].unique().tolist()),
                {"stage04_all"},
            )
            self.assertEqual(set(did["did_model"].unique().tolist()), {"simple"})
            self.assertEqual(set(did["reference_cohort_t"].unique().tolist()), {-1})
            self.assertTrue(set(did["cohort_t"].unique().tolist()).issubset({0}))

    def test_main_runs_both_variants_and_writes_variant_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            fixtures = self._write_fixture_inputs(tempdir)
            with self._patch_runtime(fixtures), patch.object(
                analysis,
                "step1_relabels",
                return_value=fixtures["relabel_df"],
            ), patch.object(
                analysis.v2,
                "_match_treated_to_untreated_cohorts",
                return_value=fixtures["matched_pairs"],
            ):
                analysis.main()

            panel = pd.read_parquet(fixtures["output_panel_path"])
            self.assertIn("analysis_variant", panel.columns)
            self.assertEqual(
                set(panel["analysis_variant"].unique().tolist()),
                {"stage04_all", "foia_linked_person_baseline"},
            )
            expected_files = {
                "in_us_event_study_treated_stage04_all.png",
                "in_us_event_study_treated_foia_linked_person_baseline.png",
                "in_us_event_study_treated_vs_control_stage04_all.png",
                "in_us_event_study_treated_vs_control_foia_linked_person_baseline.png",
            }
            written_files = {path.name for path in Path(fixtures["output_dir"]).glob("*.png")}
            self.assertTrue(expected_files.issubset(written_files))


if __name__ == "__main__":
    unittest.main()
