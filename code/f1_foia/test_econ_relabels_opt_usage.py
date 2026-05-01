from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "f1_foia" / "econ_relabels_opt_usage.py"
spec = importlib.util.spec_from_file_location(
    "econ_relabels_opt_usage_test_module",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class EconRelabelOptUsageTests(unittest.TestCase):
    def test_cohort_outputs_include_linkedin_match_share_and_ipeds_tuition(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)

            foia_raw = pd.DataFrame(
                [
                    {
                        "school_name": "School A",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2020-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_a_1",
                        "employer_name": "Employer A",
                        "employment_opt_type": "POST-COMPLETION",
                        "opt_authorization_end_date": "2021-05-14",
                        "requested_status": "OPT",
                        "tuition__fees": 12000,
                        "year": 2020,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_b_1",
                        "employer_name": "Employer B1",
                        "employment_opt_type": "POST-COMPLETION",
                        "opt_authorization_end_date": "2022-05-14",
                        "requested_status": "OPT",
                        "tuition__fees": 20000,
                        "year": 2021,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_b_2",
                        "employer_name": "Employer B2",
                        "employment_opt_type": "STEM",
                        "opt_authorization_end_date": "2022-05-14",
                        "requested_status": "OPT",
                        "tuition__fees": 22000,
                        "year": 2021,
                    },
                    {
                        "school_name": "School A",
                        "major_1_cip_code": "400101",
                        "program_end_date": "2020-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_ctrl_a",
                        "employer_name": "Employer C1",
                        "employment_opt_type": "POST-COMPLETION",
                        "opt_authorization_end_date": "2021-05-14",
                        "requested_status": "OPT",
                        "tuition__fees": 15000,
                        "year": 2020,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "400101",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_ctrl_b",
                        "employer_name": "Employer C2",
                        "employment_opt_type": "POST-COMPLETION",
                        "opt_authorization_end_date": "2022-05-14",
                        "requested_status": "OPT",
                        "tuition__fees": 25000,
                        "year": 2021,
                    },
                ]
            )
            foia_raw_path = tempdir / "foia_raw.parquet"
            foia_raw.to_parquet(foia_raw_path, index=False)

            foia_person = pd.DataFrame(
                [
                    {
                        "school_name": "School A",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2020-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_a_1",
                        "person_id": 1,
                        "year": 2020,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_b_1",
                        "person_id": 2,
                        "year": 2021,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_b_2",
                        "person_id": 3,
                        "year": 2021,
                    },
                    {
                        "school_name": "School A",
                        "major_1_cip_code": "400101",
                        "program_end_date": "2020-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_ctrl_a",
                        "person_id": 4,
                        "year": 2020,
                    },
                    {
                        "school_name": "School B",
                        "major_1_cip_code": "400101",
                        "program_end_date": "2021-05-15",
                        "student_edu_level_desc": "MASTER'S",
                        "student_key": "s_ctrl_b",
                        "person_id": 5,
                        "year": 2021,
                    },
                ]
            )
            foia_person_path = tempdir / "foia_person.parquet"
            foia_person.to_parquet(foia_person_path, index=False)

            stage05 = pd.DataFrame(
                [
                    {"person_id": 1, "user_id": 101, "person_match_rank": 1},
                    {"person_id": 2, "user_id": 102, "person_match_rank": 1},
                    {"person_id": 4, "user_id": 104, "person_match_rank": 1},
                ]
            )
            stage05_path = tempdir / "stage05.parquet"
            stage05.to_parquet(stage05_path, index=False)

            crosswalk = pd.DataFrame(
                [
                    {"school_name": "School A", "unitid": 1001},
                    {"school_name": "School B", "unitid": 1002},
                ]
            )
            crosswalk_path = tempdir / "crosswalk.parquet"
            crosswalk.to_parquet(crosswalk_path, index=False)

            ipeds_cost = pd.DataFrame(
                [
                    {"unitid": 1001, "year": 2020, "tuition7": 10000},
                    {"unitid": 1002, "year": 2021, "tuition7": 40000},
                ]
            )
            ipeds_cost_path = tempdir / "ipeds_cost.parquet"
            ipeds_cost.to_parquet(ipeds_cost_path, index=False)

            relabel_df = pd.DataFrame(
                [
                    {
                        "unitid": 1001,
                        "year": 2020,
                        "relabel_year": 2020,
                        "relabel_type": "econ_to_econometrics",
                        "event_flag": 1,
                        "ctotalt": 50,
                        "cnralt": 30,
                    },
                    {
                        "unitid": 1002,
                        "year": 2021,
                        "relabel_year": 2021,
                        "relabel_type": "econ_to_econometrics",
                        "event_flag": 1,
                        "ctotalt": 60,
                        "cnralt": 35,
                    },
                ]
            )

            with patch.object(base, "FOIA_PATH", str(foia_raw_path)), patch.object(
                base, "F1_INST_CW_PATH", str(crosswalk_path)
            ):
                con = duckdb.connect()
                opt_usage = base.compute_opt_usage(
                    con,
                    relabel_df,
                    foia_person_panel_path=str(foia_person_path),
                    stage05_person_baseline_path=str(stage05_path),
                    ipeds_cost_panel_path=str(ipeds_cost_path),
                    ipeds_tuition_col="tuition7",
                )
                event_time = base.compute_opt_usage_event_time(opt_usage)
                control_event = base.compute_control_opt_usage_event_time(
                    con,
                    relabel_df,
                    foia_person_panel_path=str(foia_person_path),
                    stage05_person_baseline_path=str(stage05_path),
                    ipeds_cost_panel_path=str(ipeds_cost_path),
                    ipeds_tuition_col="tuition7",
                )

            self.assertTrue(
                {
                    "linkedin_matched_students",
                    "linkedin_match_total_students",
                    "linkedin_match_share",
                    "tuition_ipeds_total",
                    "avg_tuition_ipeds",
                }.issubset(opt_usage.columns)
            )
            event0 = event_time.loc[event_time["event_t"] == 0].iloc[0]
            self.assertEqual(int(event0["total_grads"]), 3)
            self.assertEqual(int(event0["linkedin_matched_students"]), 2)
            self.assertEqual(int(event0["linkedin_match_total_students"]), 3)
            self.assertAlmostEqual(float(event0["linkedin_match_share"]), 2 / 3, places=6)
            self.assertAlmostEqual(float(event0["avg_tuition_ipeds"]), 30000.0, places=6)
            self.assertAlmostEqual(float(event0["tuition_ipeds_total"]), 90000.0, places=6)

            self.assertTrue(
                {
                    "linkedin_matched_students",
                    "linkedin_match_total_students",
                    "linkedin_match_share",
                    "tuition_ipeds_total",
                    "avg_tuition_ipeds",
                }.issubset(control_event.columns)
            )
            ctrl_event0 = control_event.loc[control_event["event_t"] == 0].iloc[0]
            self.assertEqual(int(ctrl_event0["total_grads"]), 2)
            self.assertEqual(int(ctrl_event0["linkedin_matched_students"]), 1)
            self.assertEqual(int(ctrl_event0["linkedin_match_total_students"]), 2)
            self.assertAlmostEqual(float(ctrl_event0["linkedin_match_share"]), 0.5, places=6)
            self.assertAlmostEqual(float(ctrl_event0["avg_tuition_ipeds"]), 25000.0, places=6)


if __name__ == "__main__":
    unittest.main()
