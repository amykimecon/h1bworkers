from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from src.config_loader import load_config


class PipelineConfigTests(unittest.TestCase):
    def test_stage03_paths_load_and_interpolate(self) -> None:
        cfg = load_config(PIPELINE_ROOT / "pipeline.yaml")
        stage_cfg = cfg["stages"]["03_rev_crosswalks"]

        self.assertEqual(stage_cfg["status"], "implemented_with_external_cleaned_education_guard")
        self.assertIn("/03_rev_crosswalks/", stage_cfg["school_family_crosswalk_parquet"])
        self.assertIn("/03_rev_crosswalks/", stage_cfg["school_resolution_parquet"])
        self.assertTrue(stage_cfg["f1_inst_unitid_crosswalk_parquet"].endswith(".parquet"))
        self.assertIn("/03_rev_crosswalks/", stage_cfg["f1_inst_unitid_crosswalk_parquet"])
        self.assertIn("/03_rev_crosswalks/", stage_cfg["employer_key_map_parquet"])
        self.assertIn("rev_educ_long", stage_cfg["external_cleaned_education_artifact"])
        self.assertIn(
            "use_raw_wrds_users_for_field_crosswalk",
            stage_cfg,
        )
        self.assertIn("with_det_candidates_f1merge.parquet", stage_cfg["raw_wrds_users_parquet"])
        self.assertTrue(stage_cfg["cip_reference_input_path"].endswith("CIPCode2020.csv"))
        self.assertIn("ipeds_name_to_zip_crosswalk.parquet", stage_cfg["ipeds_name_crosswalk_input_parquet"])

    def test_stage04_paths_load_and_interpolate(self) -> None:
        cfg = load_config(PIPELINE_ROOT / "pipeline.yaml")
        stage_cfg = cfg["stages"]["04_rev_user_clean"]

        self.assertEqual(stage_cfg["status"], "implemented_local_stage")
        self.assertTrue(stage_cfg["enabled"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["name2nat_parquet"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["nametrace_long_parquet"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["rev_match_ready_parquet"])
        self.assertIn("with_det_candidates_f1merge.parquet", stage_cfg["wrds_users_input_parquet"])
        self.assertIn("country_score_weights", stage_cfg)
        self.assertIn("institution_country_degree_weights", stage_cfg)
        self.assertIn("write_candidate_long_artifacts", stage_cfg)
        self.assertIn("candidate_long_top_k", stage_cfg)
        self.assertIn("/04_rev_user_clean/", stage_cfg["rev_educ_inst_candidates_long_parquet"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["rev_educ_cip_candidates_long_parquet"])
        self.assertTrue(stage_cfg["cip_reference_input_path"].endswith("CIPCode2020.csv"))
        self.assertIn("ipeds_name_to_zip_crosswalk.parquet", stage_cfg["ipeds_name_crosswalk_input_parquet"])
        self.assertIn("user_degree_filter_enabled", stage_cfg)
        self.assertEqual(stage_cfg["user_degree_filter_allowed_degree_types"], ["masters"])
        self.assertEqual(stage_cfg["user_degree_filter_required_deterministic_cip_prefixes"], ["45.06xx"])
        self.assertTrue(stage_cfg["user_degree_filter_include_no_degree"])
        self.assertTrue(stage_cfg["name2nat_use_names_dataset_lookup"])
        self.assertGreater(stage_cfg["name2nat_lookup_batch_size"], stage_cfg["name2nat_fallback_batch_size"])
        self.assertTrue(stage_cfg["match_ready_include_null_employer"])
        self.assertFalse(stage_cfg["write_candidate_long_artifacts"])

    def test_stage05_paths_load_and_interpolate(self) -> None:
        cfg = load_config(PIPELINE_ROOT / "pipeline.yaml")
        stage_cfg = cfg["stages"]["05_indiv_merge"]

        self.assertEqual(stage_cfg["status"], "implemented_local_stage")
        self.assertIn("/03_rev_crosswalks/", stage_cfg["school_family_crosswalk_input_parquet"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["rev_users_core_input_parquet"])
        self.assertIn("/04_rev_user_clean/", stage_cfg["rev_match_ready_input_parquet"])
        self.assertIn("/05_indiv_merge/", stage_cfg["baseline_parquet"])
        self.assertIn("/05_indiv_merge/", stage_cfg["person_strict_parquet"])


if __name__ == "__main__":
    unittest.main()
