from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PIPELINE_ROOT / "03_rev_crosswalks"
for path in (PIPELINE_ROOT, STAGE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import stage_main


class Stage03Tests(unittest.TestCase):
    def _write_config(
        self,
        tempdir: Path,
        *,
        school_family_path: str,
        school_resolution_path: str,
        employer_lookup_path: str,
    ) -> Path:
        cfg = {
            "build": {"overwrite": True},
            "testing": {"enabled": True},
            "stages": {
                "03_rev_crosswalks": {
                    "school_family_crosswalk_parquet": str(tempdir / "school_family.parquet"),
                    "school_resolution_parquet": str(tempdir / "school_resolution.parquet"),
                    "f1_inst_unitid_crosswalk_parquet": str(tempdir / "f1_inst_unitid.parquet"),
                    "employer_lookup_parquet": str(tempdir / "employer_lookup.parquet"),
                    "employer_key_map_parquet": str(tempdir / "employer_key_map.parquet"),
                    "legacy_school_crosswalk_source_parquet": school_family_path,
                    "legacy_school_resolution_source_parquet": school_resolution_path,
                    "legacy_f1_inst_unitid_source_parquet": school_resolution_path,
                    "legacy_employer_lookup_parquet": employer_lookup_path,
                }
            },
        }
        cfg_path = tempdir / "pipeline.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path

    def test_stage03_builds_school_and_employer_artifacts_without_revelio_crosswalks(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            school_family_path = tempdir / "school_family_source.parquet"
            pd.DataFrame(
                [
                    {
                        "f1_school_name": "Stanford University",
                        "rev_university_raw": "Stanford University",
                        "rev_instname_clean": "stanford university",
                        "match_score": 0.99,
                    }
                ]
            ).to_parquet(school_family_path, index=False)

            school_resolution_path = tempdir / "school_resolution_source.parquet"
            pd.DataFrame(
                [
                    {
                        "f1_school_name": "Stanford University",
                        "school_name": "Stanford University",
                        "f1_row_num": 10,
                        "f1_instname_clean": "stanford university",
                        "UNITID": 1001,
                        "unitid": 1001,
                        "rev_university_raw": "Stanford University",
                        "university_raw": "Stanford University",
                        "rev_instname_clean": "stanford university",
                        "school_match_score": 0.99,
                    }
                ]
            ).to_parquet(school_resolution_path, index=False)

            employer_lookup_path = tempdir / "employer_lookup_source.parquet"
            pd.DataFrame(
                [
                    {
                        "employer_name": "Google LLC",
                        "employer_name_clean": "google",
                        "employer_city_clean": "mountain view",
                        "employer_state_clean": "ca",
                        "employer_zip_clean": "94043",
                        "foia_row_uid": "r1",
                        "foia_firm_uid": "f1",
                        "rcid": 900,
                        "matched_company_name": "google",
                        "match_type": "fixture",
                    }
                ]
            ).to_parquet(employer_lookup_path, index=False)

            cfg_path = self._write_config(
                tempdir,
                school_family_path=str(school_family_path),
                school_resolution_path=str(school_resolution_path),
                employer_lookup_path=str(employer_lookup_path),
            )

            outputs = stage_main.build_crosswalks(
                config_path=cfg_path,
                build_school=True,
                build_employer=True,
            )

            self.assertIn("school_family_crosswalk_parquet", outputs)
            self.assertIn("school_resolution_parquet", outputs)
            self.assertIn("f1_inst_unitid_crosswalk_parquet", outputs)
            self.assertIn("employer_lookup_parquet", outputs)
            self.assertIn("employer_key_map_parquet", outputs)
            self.assertNotIn("rev_school_unitid_crosswalk_parquet", outputs)
            self.assertNotIn("field_cip_crosswalk_parquet", outputs)
            self.assertNotIn("openalex_ipeds_crosswalk_parquet", outputs)

            f1_inst_df = pd.read_parquet(outputs["f1_inst_unitid_crosswalk_parquet"])
            employer_key_map_df = pd.read_parquet(outputs["employer_key_map_parquet"])

            self.assertEqual(int(f1_inst_df["unitid"].iloc[0]), 1001)
            self.assertEqual(int(employer_key_map_df["rcid"].iloc[0]), 900)
            self.assertEqual(employer_key_map_df["normalized_employer_name"].iloc[0], "google")


if __name__ == "__main__":
    unittest.main()
