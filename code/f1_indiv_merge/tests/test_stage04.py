from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import duckdb
import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = PIPELINE_ROOT / "04_rev_user_clean"
for path in (PIPELINE_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage04_test_module",
    STAGE_DIR / "stage_main.py",
)
assert spec is not None and spec.loader is not None
stage_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_main)

common_spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage04_common_test_module",
    STAGE_DIR / "common.py",
)
assert common_spec is not None and common_spec.loader is not None
common = importlib.util.module_from_spec(common_spec)
common_spec.loader.exec_module(common)

name2nat_spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage04_name2nat_test_module",
    STAGE_DIR / "local_name2nat.py",
)
assert name2nat_spec is not None and name2nat_spec.loader is not None
local_name2nat = importlib.util.module_from_spec(name2nat_spec)
name2nat_spec.loader.exec_module(local_name2nat)

nametrace_spec = importlib.util.spec_from_file_location(
    "f1_indiv_stage04_nametrace_test_module",
    STAGE_DIR / "local_nametrace.py",
)
assert nametrace_spec is not None and nametrace_spec.loader is not None
local_nametrace = importlib.util.module_from_spec(nametrace_spec)
nametrace_spec.loader.exec_module(local_nametrace)


class Stage04Tests(unittest.TestCase):
    def _write_raw_fixture_config(self, tempdir: Path, *, write_candidate_long: bool = False) -> tuple[Path, dict[str, Path]]:
        users = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "fullname": "Ravi Patel",
                    "profile_linkedin_url": "u1",
                    "user_location": "SF",
                    "user_country": "United States",
                    "f_prob": 0.1,
                    "updated_dt": "2026-01-01",
                    "university_name": "Stanford University",
                    "rsid": "1",
                    "education_number": 1,
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-06-01",
                    "degree": "Master",
                    "field": "Computer Science",
                    "university_country": "United States",
                    "university_location": "CA",
                    "university_raw": "Stanford University",
                    "degree_raw": "M.S.",
                    "field_raw": "Computer Science",
                    "description": "",
                },
                {
                    "user_id": 1,
                    "fullname": "Ravi Patel",
                    "profile_linkedin_url": "u1",
                    "user_location": "SF",
                    "user_country": "United States",
                    "f_prob": 0.1,
                    "updated_dt": "2026-01-01",
                    "university_name": "Indian Institute of Technology Delhi",
                    "rsid": "2",
                    "education_number": 2,
                    "ed_startdate": "2014-09-01",
                    "ed_enddate": "2018-06-01",
                    "degree": "Bachelor",
                    "field": "Electrical Engineering",
                    "university_country": "India",
                    "university_location": "Delhi",
                    "university_raw": "Indian Institute of Technology Delhi",
                    "degree_raw": "B.Tech",
                    "field_raw": "Electrical Engineering",
                    "description": "",
                },
                {
                    "user_id": 2,
                    "fullname": "Maria Garcia",
                    "profile_linkedin_url": "u2",
                    "user_location": "DC",
                    "user_country": "United States",
                    "f_prob": 0.8,
                    "updated_dt": "2026-01-02",
                    "university_name": "Georgetown University",
                    "rsid": "3",
                    "education_number": 1,
                    "ed_startdate": "2015-09-01",
                    "ed_enddate": "2019-06-01",
                    "degree": "Bachelor",
                    "field": "Marketing",
                    "university_country": "United States",
                    "university_location": "DC",
                    "university_raw": "Georgetown University",
                    "degree_raw": "B.A.",
                    "field_raw": "Marketing",
                    "description": "",
                },
            ]
        )
        users_path = tempdir / "wrds_users.parquet"
        users.to_parquet(users_path, index=False)

        positions = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "position_id": "p1",
                    "position_number": 1,
                    "rcid": 900,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-01-01",
                    "role_k17000_v3": "15-1252",
                    "salary": 100000,
                    "total_compensation": 120000,
                    "company_raw": "Google LLC",
                    "title_raw": "Software Engineer",
                },
                {
                    "user_id": 2,
                    "position_id": "p2",
                    "position_number": 1,
                    "rcid": 901,
                    "country": "United States",
                    "startdate": "2019-07-01",
                    "enddate": "2021-01-01",
                    "role_k17000_v3": "11-2021",
                    "salary": 80000,
                    "total_compensation": 90000,
                    "company_raw": "Deloitte Consulting",
                    "title_raw": "Analyst",
                },
            ]
        )
        positions_path = tempdir / "wrds_positions.parquet"
        positions.to_parquet(positions_path, index=False)

        triple_map = pd.DataFrame(
            [
                {
                    "degree_key": "m s",
                    "field_key": "computer science",
                    "inst_key": "stanford university",
                    "candidate_degree_types": [{"degree_type": "masters", "score": 0.95}],
                    "candidate_cip_codes": [
                        {"cip_code": "14.0901", "score": 0.82},
                        {"cip_code": "11.0701", "score": 0.76},
                        {"cip_code": "11.0708", "score": 0.74},
                        {"cip_code": "26.1101", "score": 0.72},
                        {"cip_code": "27.0101", "score": 0.68},
                        {"cip_code": "52.1201", "score": 0.64},
                    ],
                    "candidate_ref_inst_ids": [
                        {"ref_inst_id": "stanford_bad", "score": 0.79},
                        {"ref_inst_id": "stanford_good", "score": 0.81},
                    ],
                },
                {
                    "degree_key": "b tech",
                    "field_key": "electrical engineering",
                    "inst_key": "indian institute of technology delhi",
                    "candidate_degree_types": [{"degree_type": "bachelors", "score": 0.92}],
                    "candidate_cip_codes": [{"cip_code": "14.0901", "score": 0.83}],
                    "candidate_ref_inst_ids": [{"ref_inst_id": "iit_delhi", "score": 0.83}],
                },
                {
                    "degree_key": "b a",
                    "field_key": "marketing",
                    "inst_key": "georgetown university",
                    "candidate_degree_types": [{"degree_type": "bachelors", "score": 0.91}],
                    "candidate_cip_codes": [{"cip_code": "52.1401", "score": 0.88}],
                    "candidate_ref_inst_ids": [{"ref_inst_id": "georgetown", "score": 0.86}],
                },
            ]
        )
        triple_map_path = tempdir / "deterministic_triple_map.parquet"
        triple_map.to_parquet(triple_map_path, index=False)

        ref_inst_link = pd.DataFrame(
            [
                {"ref_inst_id": "stanford_bad", "openalex_id": "oa_stanford", "main_unitid": 2001},
                {"ref_inst_id": "stanford_good", "openalex_id": "oa_stanford", "main_unitid": 2002},
                {"ref_inst_id": "iit_delhi", "openalex_id": "oa_iitd", "main_unitid": None},
                {"ref_inst_id": "georgetown", "openalex_id": "oa_georgetown", "main_unitid": 1002},
            ]
        )
        ref_inst_link_path = tempdir / "ref_inst_link.parquet"
        ref_inst_link.to_parquet(ref_inst_link_path, index=False)

        openalex = pd.DataFrame(
            [
                {
                    "openalex_id": "oa_stanford",
                    "display_name": "Stanford University",
                    "country_code": "US",
                    "alternative_names": ["Stanford"],
                    "acronyms": [],
                },
                {
                    "openalex_id": "oa_iitd",
                    "display_name": "Indian Institute of Technology Delhi",
                    "country_code": "IN",
                    "alternative_names": ["IIT Delhi"],
                    "acronyms": ["IITD"],
                },
                {
                    "openalex_id": "oa_georgetown",
                    "display_name": "Georgetown University",
                    "country_code": "US",
                    "alternative_names": [],
                    "acronyms": [],
                },
            ]
        )
        openalex_path = tempdir / "openalex_institutions.parquet"
        openalex.to_parquet(openalex_path, index=False)

        cip_reference = pd.DataFrame(
            [
                {"CIPCode": "11.07", "CIPTitle": "Computer Science."},
                {"CIPCode": "14.09", "CIPTitle": "Electrical, Electronics and Communications Engineering."},
                {"CIPCode": "45.06", "CIPTitle": "Economics."},
                {"CIPCode": "26.11", "CIPTitle": "Biochemistry, Biophysics and Molecular Biology."},
                {"CIPCode": "27.01", "CIPTitle": "Mathematics."},
                {"CIPCode": "52.12", "CIPTitle": "Management Information Systems and Services."},
                {"CIPCode": "52.14", "CIPTitle": "Marketing."},
            ]
        )
        cip_reference_path = tempdir / "cip_reference.csv"
        cip_reference.to_csv(cip_reference_path, index=False)

        ipeds_names = pd.DataFrame(
            [
                {"UNITID": 2001, "instname": "California Technology Center", "ALIAS": False},
                {"UNITID": 2002, "instname": "Stanford University", "ALIAS": False},
                {"UNITID": 2002, "instname": "Stanford", "ALIAS": True},
                {"UNITID": 1002, "instname": "Georgetown University", "ALIAS": False},
            ]
        )
        ipeds_names_path = tempdir / "ipeds_names.parquet"
        ipeds_names.to_parquet(ipeds_names_path, index=False)

        employer = pd.DataFrame(
            [
                {"rcid": 900, "normalized_employer_name": "google", "representative_match_type": "fixture"},
                {"rcid": 901, "normalized_employer_name": "deloitte", "representative_match_type": "fixture"},
            ]
        )
        employer_path = tempdir / "employer_key_map.parquet"
        employer.to_parquet(employer_path, index=False)

        name2nat = pd.DataFrame(
            [
                {
                    "fullname_clean": "Ravi Patel",
                    "first_name_clean": "Ravi",
                    "last_name_clean": "Patel",
                    "pred_nats_full_json": '{"India": 0.9, "United States": 0.1}',
                    "pred_nats_first_json": '{"India": 0.7, "United States": 0.3}',
                    "pred_nats_last_json": '{"India": 0.95, "United States": 0.05}',
                    "pred_nats_name_json": '{"India": 0.9, "United States": 0.1}',
                },
                {
                    "fullname_clean": "Maria Garcia",
                    "first_name_clean": "Maria",
                    "last_name_clean": "Garcia",
                    "pred_nats_full_json": '{"Mexico": 0.7, "United States": 0.2, "Spain": 0.1}',
                    "pred_nats_first_json": '{"Mexico": 0.6, "United States": 0.4}',
                    "pred_nats_last_json": '{"Mexico": 0.7, "Spain": 0.3}',
                    "pred_nats_name_json": '{"Mexico": 0.7, "United States": 0.2, "Spain": 0.1}',
                },
            ]
        )
        name2nat_path = tempdir / "name2nat.parquet"
        name2nat.to_parquet(name2nat_path, index=False)

        nametrace_wide = pd.DataFrame(
            [
                {
                    "fullname_clean": "Ravi Patel",
                    "f_prob_nt": 0.1,
                    "region_probs_json": '[["South Asia", 0.9], ["North America", 0.1]]',
                },
                {
                    "fullname_clean": "Maria Garcia",
                    "f_prob_nt": 0.9,
                    "region_probs_json": '[["Central America", 0.8], ["North America", 0.2]]',
                },
            ]
        )
        nametrace_wide_path = tempdir / "nametrace_wide.parquet"
        nametrace_wide.to_parquet(nametrace_wide_path, index=False)

        nametrace_long = pd.DataFrame(
            [
                {"fullname_clean": "Ravi Patel", "f_prob_nt": 0.1, "region": "South Asia", "prob": 0.9},
                {"fullname_clean": "Ravi Patel", "f_prob_nt": 0.1, "region": "North America", "prob": 0.1},
                {"fullname_clean": "Maria Garcia", "f_prob_nt": 0.9, "region": "Central America", "prob": 0.8},
                {"fullname_clean": "Maria Garcia", "f_prob_nt": 0.9, "region": "North America", "prob": 0.2},
            ]
        )
        nametrace_long_path = tempdir / "nametrace_long.parquet"
        nametrace_long.to_parquet(nametrace_long_path, index=False)

        outputs = {
            "rev_users_core_parquet": tempdir / "rev_users_core.parquet",
            "rev_educ_clean_long_parquet": tempdir / "rev_educ_clean_long.parquet",
            "rev_pos_clean_long_parquet": tempdir / "rev_pos_clean_long.parquet",
            "rev_match_ready_parquet": tempdir / "rev_match_ready.parquet",
            "rev_educ_inst_candidates_long_parquet": tempdir / "rev_educ_inst_candidates_long.parquet",
            "rev_educ_cip_candidates_long_parquet": tempdir / "rev_educ_cip_candidates_long.parquet",
        }

        cfg = {
            "build": {"overwrite": True, "allow_legacy_fallbacks": True},
            "testing": {"enabled": True},
            "stages": {
                "04_rev_user_clean": {
                    "wrds_users_input_parquet": str(users_path),
                    "wrds_positions_input_parquet": str(positions_path),
                    "deterministic_triple_map_input_parquet": str(triple_map_path),
                    "ref_inst_link_input_parquet": str(ref_inst_link_path),
                    "openalex_institutions_input_jsonl": str(openalex_path),
                    "cip_reference_input_path": str(cip_reference_path),
                    "ipeds_name_crosswalk_input_parquet": str(ipeds_names_path),
                    "employer_key_map_input_parquet": str(employer_path),
                    "name2nat_parquet": str(name2nat_path),
                    "nametrace_wide_parquet": str(nametrace_wide_path),
                    "nametrace_long_parquet": str(nametrace_long_path),
                    "rev_users_core_parquet": str(outputs["rev_users_core_parquet"]),
                    "rev_educ_clean_long_parquet": str(outputs["rev_educ_clean_long_parquet"]),
                    "rev_pos_clean_long_parquet": str(outputs["rev_pos_clean_long_parquet"]),
                    "rev_match_ready_parquet": str(outputs["rev_match_ready_parquet"]),
                    "write_candidate_long_artifacts": write_candidate_long,
                    "candidate_long_top_k": 5,
                    "rev_educ_inst_candidates_long_parquet": str(outputs["rev_educ_inst_candidates_long_parquet"]),
                    "rev_educ_cip_candidates_long_parquet": str(outputs["rev_educ_cip_candidates_long_parquet"]),
                    "country_score_weights": {
                        "name2nat": 0.50,
                        "institution": 0.35,
                        "nametrace": 0.15,
                        "user_country_fallback": 0.25,
                    },
                    "institution_country_degree_weights": {
                        "High School": 1.0,
                        "Bachelor": 0.85,
                        "Associate": 0.60,
                        "Master": 0.60,
                        "Doctor": 0.55,
                        "MBA": 0.60,
                        "Non-Degree": 0.20,
                        "Missing": 0.35,
                    },
                    "institution_country_default_weight": 0.40,
                    "institution_country_exclude_us": True,
                    "top_country_candidates_per_user": 5,
                    "min_country_candidate_score": 0.01,
                    "country_uncertainty_gap": 0.10,
                    "match_ready_include_null_employer": True,
                    "match_ready_include_null_unitid": True,
                }
            },
        }
        cfg_path = tempdir / "pipeline.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path, outputs

    def _write_legacy_fixture_config(self, tempdir: Path) -> tuple[Path, dict[str, Path]]:
        legacy_indiv = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "fullname": "Ravi Patel",
                    "fullname_clean": "Ravi Patel",
                    "user_location": "SF",
                    "user_country": "United States",
                    "f_prob": 0.1,
                    "country": "India",
                    "subregion": "South Asia",
                    "total_score": 0.9,
                    "nanat_score": 0.9,
                    "nanat_subregion_score": 0.9,
                    "nt_subregion_score": 0.9,
                    "country_uncertain_ind": 0,
                },
                {
                    "user_id": 2,
                    "fullname": "Maria Garcia",
                    "fullname_clean": "Maria Garcia",
                    "user_location": "DC",
                    "user_country": "United States",
                    "f_prob": 0.8,
                    "country": "Mexico",
                    "subregion": "Central America",
                    "total_score": 0.8,
                    "nanat_score": 0.8,
                    "nanat_subregion_score": 0.8,
                    "nt_subregion_score": 0.8,
                    "country_uncertain_ind": 0,
                },
            ]
        )
        legacy_indiv_path = tempdir / "legacy_indiv.parquet"
        legacy_indiv.to_parquet(legacy_indiv_path, index=False)

        legacy_educ = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "education_number": 1,
                    "degree_clean": "Master",
                    "deterministic_degree_types": ["masters"],
                    "deterministic_cip_candidates": [],
                    "field_clean": "computer science",
                    "university_raw": "Stanford University",
                    "ed_startdate": "2018-09-01",
                    "ed_enddate": "2020-06-01",
                    "match_country": "United States",
                },
                {
                    "user_id": 2,
                    "education_number": 1,
                    "degree_clean": "Bachelor",
                    "deterministic_degree_types": ["bachelors"],
                    "deterministic_cip_candidates": [],
                    "field_clean": "marketing",
                    "university_raw": "Georgetown University",
                    "ed_startdate": "2015-09-01",
                    "ed_enddate": "2019-06-01",
                    "match_country": "United States",
                },
            ]
        )
        legacy_educ_path = tempdir / "legacy_educ.parquet"
        legacy_educ.to_parquet(legacy_educ_path, index=False)

        legacy_pos = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "position_number": 1,
                    "rcid": 900,
                    "country": "United States",
                    "startdate": "2020-07-01",
                    "enddate": "2022-01-01",
                    "company_raw": "Google LLC",
                    "title_raw": "Software Engineer",
                },
                {
                    "user_id": 2,
                    "position_number": 1,
                    "rcid": 901,
                    "country": "United States",
                    "startdate": "2019-07-01",
                    "enddate": "2021-01-01",
                    "company_raw": "Deloitte Consulting",
                    "title_raw": "Analyst",
                },
            ]
        )
        legacy_pos_path = tempdir / "legacy_pos.parquet"
        legacy_pos.to_parquet(legacy_pos_path, index=False)

        school = pd.DataFrame(
            [
                {
                    "university_raw": "Stanford University",
                    "unitid": 1001,
                    "school_match_score": 0.99,
                    "rev_instname_clean": "stanford university",
                },
                {
                    "university_raw": "Georgetown University",
                    "unitid": 1002,
                    "school_match_score": 0.98,
                    "rev_instname_clean": "georgetown university",
                },
            ]
        )
        school_path = tempdir / "rev_school_unitid.parquet"
        school.to_parquet(school_path, index=False)

        field = pd.DataFrame(
            [
                {"source_field_norm": "computer science", "cip_code": 1107, "mapping_source": "fixture"},
                {"source_field_norm": "marketing", "cip_code": 5214, "mapping_source": "fixture"},
            ]
        )
        field_path = tempdir / "field_cip.parquet"
        field.to_parquet(field_path, index=False)

        cip_reference = pd.DataFrame(
            [
                {"CIPCode": "11.07", "CIPTitle": "Computer Science."},
                {"CIPCode": "45.06", "CIPTitle": "Economics."},
                {"CIPCode": "52.14", "CIPTitle": "Marketing."},
            ]
        )
        cip_reference_path = tempdir / "cip_reference.csv"
        cip_reference.to_csv(cip_reference_path, index=False)

        ipeds_names = pd.DataFrame(
            [
                {"UNITID": 1001, "instname": "Stanford University", "ALIAS": False},
                {"UNITID": 1002, "instname": "Georgetown University", "ALIAS": False},
            ]
        )
        ipeds_names_path = tempdir / "ipeds_names.parquet"
        ipeds_names.to_parquet(ipeds_names_path, index=False)

        employer = pd.DataFrame(
            [
                {"rcid": 900, "normalized_employer_name": "google", "representative_match_type": "fixture"},
                {"rcid": 901, "normalized_employer_name": "deloitte", "representative_match_type": "fixture"},
            ]
        )
        employer_path = tempdir / "employer_key_map.parquet"
        employer.to_parquet(employer_path, index=False)

        outputs = {
            "rev_users_core_parquet": tempdir / "rev_users_core.parquet",
            "rev_educ_clean_long_parquet": tempdir / "rev_educ_clean_long.parquet",
            "rev_pos_clean_long_parquet": tempdir / "rev_pos_clean_long.parquet",
            "rev_match_ready_parquet": tempdir / "rev_match_ready.parquet",
        }

        cfg = {
            "build": {"overwrite": True, "allow_legacy_fallbacks": True},
            "testing": {"enabled": True},
            "paths": {
                "legacy_rev_indiv_parquet": str(legacy_indiv_path),
                "legacy_rev_educ_long_parquet": str(legacy_educ_path),
                "legacy_rev_pos_parquet": str(legacy_pos_path),
            },
            "stages": {
                "04_rev_user_clean": {
                    "legacy_rev_indiv_parquet": str(legacy_indiv_path),
                    "legacy_rev_educ_long_parquet": str(legacy_educ_path),
                    "legacy_rev_pos_parquet": str(legacy_pos_path),
                    "rev_school_unitid_crosswalk_input_parquet": str(school_path),
                    "field_cip_crosswalk_input_parquet": str(field_path),
                    "cip_reference_input_path": str(cip_reference_path),
                    "ipeds_name_crosswalk_input_parquet": str(ipeds_names_path),
                    "employer_key_map_input_parquet": str(employer_path),
                    "rev_users_core_parquet": str(outputs["rev_users_core_parquet"]),
                    "rev_educ_clean_long_parquet": str(outputs["rev_educ_clean_long_parquet"]),
                    "rev_pos_clean_long_parquet": str(outputs["rev_pos_clean_long_parquet"]),
                    "rev_match_ready_parquet": str(outputs["rev_match_ready_parquet"]),
                    "match_ready_include_null_employer": True,
                    "match_ready_include_null_unitid": True,
                }
            },
        }
        cfg_path = tempdir / "pipeline.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        return cfg_path, outputs

    def test_stage04_builds_outputs_from_raw_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            self.assertEqual(result["source_mode"], "raw_stage02")
            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertFalse(users_core.empty)
            self.assertFalse(educ.empty)
            self.assertFalse(pos.empty)
            self.assertFalse(match_ready.empty)
            self.assertEqual(
                set(["user_id", "top_country_candidate", "top_country_score"]).issubset(users_core.columns),
                True,
            )
            self.assertEqual(
                set(["user_id", "unitid", "degree_clean", "cip", "match_eligible_ind"]).issubset(educ.columns),
                True,
            )
            self.assertEqual(
                set(["user_id", "employer_key", "rcid", "company_raw"]).issubset(pos.columns),
                True,
            )
            self.assertEqual(
                set(["user_id", "unitid", "degree_clean", "country_candidate", "cip", "employer_key"]).issubset(match_ready.columns),
                True,
            )
            self.assertEqual(
                set(["cip_score", "field_mapped_ind"]).issubset(match_ready.columns),
                True,
            )
            self.assertEqual(set(match_ready["degree_clean"].dropna().tolist()), {"masters", "bachelors"})
            top_country = dict(zip(users_core["user_id"], users_core["top_country_candidate"]))
            self.assertEqual(top_country[1], "India")
            self.assertEqual(top_country[2], "Mexico")
            self.assertEqual(int(educ.loc[educ["university_raw"] == "Stanford University", "unitid"].iloc[0]), 2002)
            self.assertEqual(int(educ.loc[educ["university_raw"] == "Georgetown University", "unitid"].iloc[0]), 1002)
            self.assertTrue(educ.loc[educ["university_raw"] == "Indian Institute of Technology Delhi"].empty)
            self.assertEqual(int(educ.loc[educ["field_clean"] == "computer science", "cip"].iloc[0]), 1107)
            self.assertEqual(int(educ.loc[educ["field_clean"] == "marketing", "cip"].iloc[0]), 5214)
            self.assertEqual(
                educ.loc[educ["university_raw"] == "Stanford University", "unitid_mapping_source"].iloc[0],
                "deterministic_top1",
            )
            self.assertEqual(
                educ.loc[educ["university_raw"] == "Georgetown University", "unitid_mapping_source"].iloc[0],
                "deterministic_top1",
            )
            self.assertEqual(
                educ.loc[educ["field_clean"] == "computer science", "cip_mapping_source"].iloc[0],
                "deterministic_top1",
            )
            self.assertEqual(
                educ.loc[educ["field_clean"] == "marketing", "cip_mapping_source"].iloc[0],
                "deterministic_top1",
            )
            self.assertEqual(
                int(educ.loc[educ["university_raw"] == "Stanford University", "unitid_candidate_count"].iloc[0]),
                2,
            )
            self.assertEqual(
                int(educ.loc[educ["university_raw"] == "Georgetown University", "unitid_candidate_count"].iloc[0]),
                1,
            )
            self.assertEqual(
                int(educ.loc[educ["field_clean"] == "computer science", "cip_candidate_count"].iloc[0]),
                5,
            )
            self.assertAlmostEqual(
                float(educ.loc[educ["field_clean"] == "computer science", "cip_score"].iloc[0]),
                1.0,
                places=8,
            )
            self.assertAlmostEqual(
                float(match_ready.loc[match_ready["cip"] == 1107, "cip_score"].iloc[0]),
                1.0,
                places=8,
            )
            self.assertEqual(
                int(match_ready.loc[match_ready["cip"] == 1107, "field_mapped_ind"].iloc[0]),
                1,
            )
            self.assertNotIn("deterministic_ref_inst_candidates", educ.columns)
            self.assertNotIn("deterministic_cip_candidates", educ.columns)
            self.assertIn("google", set(pos["employer_key"].dropna().tolist()))

    def test_stage04_uses_field_key_null_heuristic_branch_for_cip_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, _outputs = self._write_raw_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]

            users_path = Path(stage_cfg["wrds_users_input_parquet"])
            users = pd.read_parquet(users_path)
            users = pd.concat(
                [
                    users,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "fullname": "Ava Chen",
                                "profile_linkedin_url": "u3",
                                "user_location": "Boston",
                                "user_country": "United States",
                                "f_prob": 0.6,
                                "updated_dt": "2026-01-03",
                                "university_name": "Stanford University",
                                "rsid": "4",
                                "education_number": 1,
                                "ed_startdate": "2017-09-01",
                                "ed_enddate": "2019-06-01",
                                "degree": "Master",
                                "field": "Business",
                                "university_country": "United States",
                                "university_location": "CA",
                                "university_raw": "Stanford University",
                                "degree_raw": "MBA",
                                "field_raw": None,
                                "description": "",
                            },
                            {
                                "user_id": 4,
                                "fullname": "Mina Kim",
                                "profile_linkedin_url": "u4",
                                "user_location": "Boston",
                                "user_country": "United States",
                                "f_prob": 0.6,
                                "updated_dt": "2026-01-04",
                                "university_name": "Stanford University",
                                "rsid": "5",
                                "education_number": 1,
                                "ed_startdate": "2017-09-01",
                                "ed_enddate": "2019-06-01",
                                "degree": "Master",
                                "field": "Business Analytics",
                                "university_country": "United States",
                                "university_location": "CA",
                                "university_raw": "Stanford University",
                                "degree_raw": "MBA",
                                "field_raw": None,
                                "description": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            users.to_parquet(users_path, index=False)
            cfg_path.write_text(yaml.safe_dump(cfg))

            session = stage_main.build_clean_users_session(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )
            try:
                self.assertEqual(session.outputs["source_mode"], "raw_stage02")
                educ_row = session.connection.execute(
                    """
                    SELECT
                        field_key,
                        cip,
                        cip_mapping_source,
                        field_mapping_source,
                        cip_score,
                        field_mapped_ind,
                        cip_candidate_count
                    FROM educ_clean_all
                    WHERE user_id = 3
                    """
                ).fetchdf().iloc[0]
                self.assertTrue(pd.isna(educ_row["field_key"]))
                self.assertEqual(int(educ_row["cip"]), 5202)
                self.assertEqual(educ_row["cip_mapping_source"], "field_key_null_heuristic")
                self.assertEqual(educ_row["field_mapping_source"], "field_key_null_heuristic")
                self.assertAlmostEqual(float(educ_row["cip_score"]), 1.0, places=8)
                self.assertEqual(int(educ_row["field_mapped_ind"]), 1)
                self.assertEqual(int(educ_row["cip_candidate_count"]), 1)

                non_category_row = session.connection.execute(
                    """
                    SELECT
                        cip,
                        cip_mapping_source,
                        cip_score,
                        field_mapped_ind,
                        cip_candidate_count
                    FROM educ_clean_all
                    WHERE user_id = 4
                    """
                ).fetchdf().iloc[0]
                self.assertTrue(pd.isna(non_category_row["cip"]))
                self.assertTrue(pd.isna(non_category_row["cip_mapping_source"]))
                self.assertTrue(pd.isna(non_category_row["cip_score"]))
                self.assertEqual(int(non_category_row["field_mapped_ind"]), 0)
                self.assertEqual(int(non_category_row["cip_candidate_count"]), 0)
            finally:
                session.close()

    def test_stage04_rejects_unfiltered_full_raw_run_outside_testing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, _outputs = self._write_raw_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]
            stage_cfg["user_degree_filter_enabled"] = False
            cfg_path.write_text(yaml.safe_dump(cfg))

            with self.assertRaisesRegex(RuntimeError, "user_degree_filter_enabled=false"):
                stage_main.build_clean_users(
                    config_path=cfg_path,
                    testing=False,
                    run_name2nat_models=False,
                    run_nametrace_model=False,
                )

    def test_stage04_raw_shard_writes_shard_scoped_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
                shard_count=2,
                shard_id=1,
            )

            users_path = Path(common.shard_output_path(outputs["rev_users_core_parquet"], shard_count=2, shard_id=1))
            educ_path = Path(common.shard_output_path(outputs["rev_educ_clean_long_parquet"], shard_count=2, shard_id=1))
            pos_path = Path(common.shard_output_path(outputs["rev_pos_clean_long_parquet"], shard_count=2, shard_id=1))
            match_ready_path = Path(common.shard_output_path(outputs["rev_match_ready_parquet"], shard_count=2, shard_id=1))

            self.assertEqual(result["user_shard_label"], "shard0001of0002")
            self.assertFalse(outputs["rev_users_core_parquet"].exists())
            self.assertTrue(users_path.exists())
            self.assertTrue(educ_path.exists())
            self.assertTrue(pos_path.exists())
            self.assertTrue(match_ready_path.exists())

            users_core = pd.read_parquet(users_path)
            educ = pd.read_parquet(educ_path)
            pos = pd.read_parquet(pos_path)
            match_ready = pd.read_parquet(match_ready_path)

            self.assertEqual(sorted(users_core["user_id"].tolist()), [1])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [1])

    def test_stage04_can_merge_sharded_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            for shard_id in (0, 1):
                stage_main.build_clean_users(
                    config_path=cfg_path,
                    testing=True,
                    run_name2nat_models=False,
                    run_nametrace_model=False,
                    shard_count=2,
                    shard_id=shard_id,
                )

            merged = stage_main._merge_stage04_sharded_outputs(
                config_path=cfg_path,
                shard_count=2,
            )

            self.assertIn("rev_users_core_parquet", merged)
            self.assertIn("rev_match_ready_parquet", merged)
            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertEqual(sorted(users_core["user_id"].tolist()), [1, 2])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [1, 2])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [1, 2])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [1, 2])

    def test_stage04_run_avoids_pandas_materialization_and_pandas_parquet_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            with patch.object(
                duckdb.DuckDBPyRelation,
                "df",
                side_effect=AssertionError("DuckDBPyRelation.df() should not be called during stage04 assembly"),
            ), patch.object(
                pd.DataFrame,
                "to_parquet",
                side_effect=AssertionError("pandas DataFrame.to_parquet() should not be called during stage04 assembly"),
            ):
                result = stage_main.build_clean_users(
                    config_path=cfg_path,
                    testing=True,
                    run_name2nat_models=False,
                    run_nametrace_model=False,
                )

            self.assertEqual(result["source_mode"], "raw_stage02")
            self.assertTrue(outputs["rev_users_core_parquet"].exists())
            self.assertTrue(outputs["rev_educ_clean_long_parquet"].exists())
            self.assertTrue(outputs["rev_pos_clean_long_parquet"].exists())
            self.assertTrue(outputs["rev_match_ready_parquet"].exists())

    def test_stage04_in_memory_only_mode_skips_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
                in_memory_only=True,
            )

            self.assertTrue(result["in_memory_only"])
            self.assertIn("duckdb_tables", result)
            self.assertIn("users_core", result["duckdb_tables"])
            self.assertIn("match_ready", result["duckdb_tables"])
            self.assertEqual(result["source_mode"], "raw_stage02")
            self.assertFalse(outputs["rev_users_core_parquet"].exists())
            self.assertFalse(outputs["rev_educ_clean_long_parquet"].exists())
            self.assertFalse(outputs["rev_pos_clean_long_parquet"].exists())
            self.assertFalse(outputs["rev_match_ready_parquet"].exists())

    def test_stage04_build_clean_users_session_returns_queryable_duckdb_session(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)

            session = stage_main.build_clean_users_session(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )
            try:
                self.assertTrue(session.outputs["in_memory_only"])
                self.assertIn("users_core", session.tables)
                self.assertIn("match_ready", session.tables)
                users_core_rows = int(session.connection.execute("SELECT COUNT(*) FROM users_core").fetchone()[0])
                match_ready_rows = int(session.connection.execute("SELECT COUNT(*) FROM match_ready").fetchone()[0])
                self.assertGreater(users_core_rows, 0)
                self.assertGreater(match_ready_rows, 0)
                self.assertFalse(outputs["rev_users_core_parquet"].exists())
                self.assertFalse(outputs["rev_educ_clean_long_parquet"].exists())
                self.assertFalse(outputs["rev_pos_clean_long_parquet"].exists())
                self.assertFalse(outputs["rev_match_ready_parquet"].exists())
            finally:
                session.close()

    def test_stage04_match_ready_detail_helpers_print_joined_match_context(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, _outputs = self._write_raw_fixture_config(tempdir)

            session = stage_main.build_clean_users_session(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    sample_keys = stage_main.print_match_ready_sample_details(
                        session,
                        sample_size=1,
                        seed=7,
                    )
                text = buffer.getvalue()
                self.assertEqual(len(sample_keys), 1)
                self.assertIn("Match Detail:", text)
                self.assertIn("fullname:", text)
                self.assertIn("country_candidates:", text)
                self.assertIn("cip_candidates:", text)
                self.assertIn("unitid_candidates:", text)
                self.assertIn("selected_cip:", text)
                self.assertIn("selected_unitid:", text)
            finally:
                session.close()

    def test_stage04_launch_ipython_session_publishes_namespace_inside_ipykernel(self) -> None:
        fake_session = types.SimpleNamespace(
            connection=object(),
            outputs={"in_memory_only": True},
            tables={"users_core": "users_core"},
        )
        with patch.object(stage_main, "_running_in_ipykernel", return_value=True), patch.object(
            stage_main,
            "build_clean_users_session",
            return_value=fake_session,
        ), patch.object(stage_main, "_publish_interactive_namespace") as publish_namespace, patch(
            "IPython.start_ipython"
        ) as start_ipython:
            result = stage_main.launch_ipython_session(testing=True, run_name2nat_models=False, run_nametrace_model=False)

        self.assertIs(result, fake_session)
        publish_namespace.assert_called_once()
        start_ipython.assert_not_called()

    def test_stage04_build_clean_users_session_defaults_skip_name_models(self) -> None:
        fake_session = stage_main.Stage04Session(
            connection=duckdb.connect(database=":memory:"),
            outputs={},
            tables={},
        )
        try:
            with patch.object(stage_main, "build_clean_users", return_value=fake_session) as build_clean_users:
                result = stage_main.build_clean_users_session(testing=True)

            self.assertIs(result, fake_session)
            build_clean_users.assert_called_once()
            _, kwargs = build_clean_users.call_args
            self.assertFalse(kwargs["run_name2nat_models"])
            self.assertFalse(kwargs["run_nametrace_model"])
            self.assertTrue(kwargs["in_memory_only"])
            self.assertTrue(kwargs["return_session"])
        finally:
            fake_session.close()

    def test_stage04_main_auto_launches_interactive_session_in_ipykernel(self) -> None:
        cfg = {"testing": {"enabled": False}}
        with patch.object(stage_main, "sanitize_ipykernel_argv", return_value=[]), patch.object(
            stage_main,
            "_running_in_ipykernel",
            return_value=True,
        ), patch.object(stage_main, "load_config", return_value=cfg), patch.object(
            stage_main,
            "launch_ipython_session",
        ) as launch_ipython_session, patch.object(stage_main, "build_clean_users") as build_clean_users:
            stage_main.main()

        launch_ipython_session.assert_called_once()
        build_clean_users.assert_not_called()

    def test_stage04_writes_opt_in_candidate_long_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir, write_candidate_long=True)

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            self.assertIn("rev_educ_inst_candidates_long_parquet", result)
            self.assertIn("rev_educ_cip_candidates_long_parquet", result)
            inst_long = pd.read_parquet(outputs["rev_educ_inst_candidates_long_parquet"])
            cip_long = pd.read_parquet(outputs["rev_educ_cip_candidates_long_parquet"])

            self.assertFalse(inst_long.empty)
            self.assertFalse(cip_long.empty)
            self.assertLessEqual(int(inst_long["candidate_rank"].max()), 5)
            self.assertLessEqual(int(cip_long["candidate_rank"].max()), 5)

            stanford_inst = inst_long.loc[
                (inst_long["user_id"] == 1) & (inst_long["education_number"] == 1)
            ].sort_values("candidate_rank")
            self.assertEqual(stanford_inst.shape[0], 2)
            self.assertEqual(int(stanford_inst["unitid"].iloc[0]), 2002)
            self.assertEqual(int(stanford_inst["selected_top1_ind"].iloc[0]), 1)

            iit_inst = inst_long.loc[
                (inst_long["user_id"] == 1) & (inst_long["education_number"] == 2)
            ].sort_values("candidate_rank")
            self.assertTrue(iit_inst.empty)

            georgetown_inst = inst_long.loc[
                (inst_long["user_id"] == 2) & (inst_long["education_number"] == 1)
            ]
            self.assertEqual(georgetown_inst.shape[0], 1)
            self.assertEqual(int(georgetown_inst["selected_top1_ind"].iloc[0]), 1)

            stanford_cip = cip_long.loc[
                (cip_long["user_id"] == 1) & (cip_long["education_number"] == 1)
            ].sort_values("candidate_rank")
            self.assertEqual(stanford_cip.shape[0], 5)
            self.assertEqual(int(stanford_cip["cip"].iloc[0]), 1107)
            self.assertEqual(int(stanford_cip["selected_top1_ind"].iloc[0]), 1)

            marketing_cip = cip_long.loc[
                (cip_long["user_id"] == 2) & (cip_long["education_number"] == 1)
            ].iloc[0]
            self.assertEqual(int(marketing_cip["cip"]), 5214)
            self.assertEqual(marketing_cip["mapping_source"], "deterministic_candidate")
            self.assertEqual(int(marketing_cip["selected_top1_ind"]), 1)

    def test_stage04_degree_filter_keeps_master_and_no_degree_users_in_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]
            stage_cfg["user_degree_filter_enabled"] = True
            stage_cfg["user_degree_filter_allowed_degree_types"] = ["masters"]
            stage_cfg["user_degree_filter_include_no_degree"] = True

            users_path = Path(stage_cfg["wrds_users_input_parquet"])
            users = pd.read_parquet(users_path)
            users = pd.concat(
                [
                    users,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "fullname": "Sam Lee",
                                "profile_linkedin_url": "u3",
                                "user_location": "New York",
                                "user_country": "United States",
                                "f_prob": 0.4,
                                "updated_dt": "2026-01-03",
                                "university_name": None,
                                "rsid": None,
                                "education_number": pd.NA,
                                "ed_startdate": None,
                                "ed_enddate": None,
                                "degree": None,
                                "field": None,
                                "university_country": None,
                                "university_location": None,
                                "university_raw": None,
                                "degree_raw": None,
                                "field_raw": None,
                                "field_key": None,
                                "degree_key": "",
                                "inst_key": "",
                                "deterministic_degree_types": ["unknown"],
                                "deterministic_cip_candidates": [],
                                "deterministic_inst_candidates": [],
                                "description": "",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            users.to_parquet(users_path, index=False)
            cfg_path.write_text(yaml.safe_dump(cfg))

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertTrue(result["user_degree_filter_enabled"])
            self.assertEqual(result["user_degree_filter_total_users"], 3)
            self.assertEqual(result["user_degree_filter_kept_users"], 2)
            self.assertEqual(result["user_degree_filter_dropped_users"], 1)
            self.assertEqual(sorted(users_core["user_id"].tolist()), [1, 3])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [1])

    def test_stage04_degree_filter_requires_configured_cip_prefix_in_raw_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_raw_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]
            stage_cfg["user_degree_filter_enabled"] = True
            stage_cfg["user_degree_filter_allowed_degree_types"] = ["masters"]
            stage_cfg["user_degree_filter_required_deterministic_cip_prefixes"] = ["45.06xx"]
            stage_cfg["user_degree_filter_include_no_degree"] = True

            users_path = Path(stage_cfg["wrds_users_input_parquet"])
            users = pd.read_parquet(users_path)
            users = pd.concat(
                [
                    users,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "fullname": "Ava Chen",
                                "profile_linkedin_url": "u3",
                                "user_location": "Boston",
                                "user_country": "United States",
                                "f_prob": 0.6,
                                "updated_dt": "2026-01-03",
                                "university_name": "Stanford University",
                                "rsid": "4",
                                "education_number": 1,
                                "ed_startdate": "2017-09-01",
                                "ed_enddate": "2019-06-01",
                                "degree": "Master",
                                "field": "Economics",
                                "university_country": "United States",
                                "university_location": "CA",
                                "university_raw": "Stanford University",
                                "degree_raw": "M.A.",
                                "field_raw": "Economics",
                                "field_key": "economics",
                                "deterministic_cip_candidates": [
                                    {"cip_code": "45.0603", "hybrid_score": 0.91},
                                    {"cip_code": "45.0601", "hybrid_score": 0.84},
                                ],
                                "deterministic_inst_candidates": [
                                    {"unitid": "2002", "hybrid_score": 0.88},
                                ],
                                "description": "",
                                "degree_key": "master of arts ma",
                                "inst_key": "stanford university",
                                "deterministic_degree_types": ["masters"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            users.to_parquet(users_path, index=False)

            positions_path = Path(stage_cfg["wrds_positions_input_parquet"])
            positions = pd.read_parquet(positions_path)
            positions = pd.concat(
                [
                    positions,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "position_id": "p3",
                                "position_number": 1,
                                "rcid": 900,
                                "country": "United States",
                                "startdate": "2019-07-01",
                                "enddate": "2021-06-01",
                                "role_k17000_v3": "19-3011",
                                "salary": 95000,
                                "total_compensation": 110000,
                                "company_raw": "Google LLC",
                                "title_raw": "Economist",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            positions.to_parquet(positions_path, index=False)
            triple_map_path = Path(stage_cfg["deterministic_triple_map_input_parquet"])
            triple_map = pd.read_parquet(triple_map_path)
            triple_map = pd.concat(
                [
                    triple_map,
                    pd.DataFrame(
                        [
                            {
                                "degree_key": "m a",
                                "field_key": "economics",
                                "inst_key": "stanford university",
                                "candidate_degree_types": [{"degree_type": "masters", "score": 0.93}],
                                "candidate_cip_codes": [{"cip_code": "45.0601", "score": 0.91}],
                                "candidate_ref_inst_ids": [{"ref_inst_id": "stanford_good", "score": 0.88}],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            triple_map.to_parquet(triple_map_path, index=False)
            cfg_path.write_text(yaml.safe_dump(cfg))

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertEqual(result["user_degree_filter_total_users"], 3)
            self.assertEqual(result["user_degree_filter_kept_users"], 1)
            self.assertEqual(result["user_degree_filter_dropped_users"], 2)
            self.assertEqual(result["user_degree_filter_required_deterministic_cip_prefixes"], ["4506"])
            self.assertEqual(sorted(users_core["user_id"].tolist()), [3])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [3])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [3])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [3])

    def test_stage04_degree_filter_keeps_master_and_no_degree_users_in_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_legacy_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]
            stage_cfg["user_degree_filter_enabled"] = True
            stage_cfg["user_degree_filter_allowed_degree_types"] = ["masters"]
            stage_cfg["user_degree_filter_include_no_degree"] = True

            legacy_indiv_path = Path(stage_cfg["legacy_rev_indiv_parquet"])
            legacy_indiv = pd.read_parquet(legacy_indiv_path)
            legacy_indiv = pd.concat(
                [
                    legacy_indiv,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "fullname": "Sam Lee",
                                "fullname_clean": "Sam Lee",
                                "user_location": "New York",
                                "user_country": "United States",
                                "f_prob": 0.4,
                                "country": "United States",
                                "subregion": "North America",
                                "total_score": 0.25,
                                "nanat_score": 0.0,
                                "nanat_subregion_score": 0.0,
                                "nt_subregion_score": 0.0,
                                "country_uncertain_ind": 1,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            legacy_indiv.to_parquet(legacy_indiv_path, index=False)
            cfg_path.write_text(yaml.safe_dump(cfg))

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertTrue(result["user_degree_filter_enabled"])
            self.assertEqual(result["user_degree_filter_total_users"], 3)
            self.assertEqual(result["user_degree_filter_kept_users"], 2)
            self.assertEqual(result["user_degree_filter_dropped_users"], 1)
            self.assertEqual(sorted(users_core["user_id"].tolist()), [1, 3])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [1])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [1])

    def test_stage04_degree_filter_requires_configured_cip_prefix_in_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_legacy_fixture_config(tempdir)
            cfg = yaml.safe_load(cfg_path.read_text())
            stage_cfg = cfg["stages"]["04_rev_user_clean"]
            stage_cfg["user_degree_filter_enabled"] = True
            stage_cfg["user_degree_filter_allowed_degree_types"] = ["masters"]
            stage_cfg["user_degree_filter_required_deterministic_cip_prefixes"] = ["45.06xx"]
            stage_cfg["user_degree_filter_include_no_degree"] = True

            legacy_indiv_path = Path(stage_cfg["legacy_rev_indiv_parquet"])
            legacy_indiv = pd.read_parquet(legacy_indiv_path)
            legacy_indiv = pd.concat(
                [
                    legacy_indiv,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "fullname": "Ava Chen",
                                "fullname_clean": "Ava Chen",
                                "user_location": "Boston",
                                "user_country": "United States",
                                "f_prob": 0.6,
                                "country": "United States",
                                "subregion": "North America",
                                "total_score": 0.7,
                                "nanat_score": 0.2,
                                "nanat_subregion_score": 0.8,
                                "nt_subregion_score": 0.8,
                                "country_uncertain_ind": 0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            legacy_indiv.to_parquet(legacy_indiv_path, index=False)

            legacy_educ_path = Path(stage_cfg["legacy_rev_educ_long_parquet"])
            legacy_educ = pd.read_parquet(legacy_educ_path)
            legacy_educ = pd.concat(
                [
                    legacy_educ,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "education_number": 1,
                                "degree_clean": "Master",
                                "deterministic_degree_types": ["masters"],
                                "deterministic_cip_candidates": [
                                    {"cip_code": "45.0603", "hybrid_score": 0.9},
                                ],
                                "field_clean": "economics",
                                "university_raw": "Stanford University",
                                "ed_startdate": "2017-09-01",
                                "ed_enddate": "2019-06-01",
                                "match_country": "United States",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            legacy_educ.to_parquet(legacy_educ_path, index=False)

            legacy_pos_path = Path(stage_cfg["legacy_rev_pos_parquet"])
            legacy_pos = pd.read_parquet(legacy_pos_path)
            legacy_pos = pd.concat(
                [
                    legacy_pos,
                    pd.DataFrame(
                        [
                            {
                                "user_id": 3,
                                "position_number": 1,
                                "rcid": 900,
                                "country": "United States",
                                "startdate": "2019-07-01",
                                "enddate": "2021-06-01",
                                "company_raw": "Google LLC",
                                "title_raw": "Economist",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            legacy_pos.to_parquet(legacy_pos_path, index=False)
            cfg_path.write_text(yaml.safe_dump(cfg))

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            users_core = pd.read_parquet(outputs["rev_users_core_parquet"])
            educ = pd.read_parquet(outputs["rev_educ_clean_long_parquet"])
            pos = pd.read_parquet(outputs["rev_pos_clean_long_parquet"])
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])

            self.assertEqual(result["user_degree_filter_total_users"], 3)
            self.assertEqual(result["user_degree_filter_kept_users"], 1)
            self.assertEqual(result["user_degree_filter_dropped_users"], 2)
            self.assertEqual(result["user_degree_filter_required_deterministic_cip_prefixes"], ["4506"])
            self.assertEqual(sorted(users_core["user_id"].tolist()), [3])
            self.assertEqual(sorted(educ["user_id"].dropna().astype(int).unique().tolist()), [3])
            self.assertEqual(sorted(pos["user_id"].dropna().astype(int).unique().tolist()), [3])
            self.assertEqual(sorted(match_ready["user_id"].dropna().astype(int).unique().tolist()), [3])

    def test_stage04_uses_legacy_fallback_when_raw_inputs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            cfg_path, outputs = self._write_legacy_fixture_config(tempdir)

            result = stage_main.build_clean_users(
                config_path=cfg_path,
                testing=True,
                run_name2nat_models=False,
                run_nametrace_model=False,
            )

            self.assertEqual(result["source_mode"], "legacy_fallback")
            match_ready = pd.read_parquet(outputs["rev_match_ready_parquet"])
            self.assertFalse(match_ready.empty)
            self.assertEqual(
                sorted(match_ready["country_candidate"].tolist()),
                ["India", "Mexico"],
            )
            self.assertEqual(
                sorted(match_ready["degree_clean"].tolist()),
                ["bachelors", "masters"],
            )
            self.assertEqual(
                sorted(match_ready["employer_key"].tolist()),
                ["deloitte", "google"],
            )

    def test_stage_clean_name_artifacts_drop_short_names_and_skip_one_word_token_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            users = pd.DataFrame(
                [
                    {"user_id": 1, "fullname": "Li"},
                    {"user_id": 2, "fullname": "J Li"},
                    {"user_id": 3, "fullname": "A"},
                    {"user_id": 4, "fullname": "A B"},
                    {"user_id": 5, "fullname": "???"},
                    {"user_id": 6, "fullname": "Li"},
                ]
            )
            users_path = tempdir / "users.parquet"
            users.to_parquet(users_path, index=False)

            artifacts = common.stage_clean_name_artifacts(
                users_path,
                artifact_dir=tempdir / "staged_names",
                overwrite=True,
            )

            base_names = pd.read_parquet(artifacts["base_names_parquet"])
            full_unique = pd.read_parquet(artifacts["full_unique_parquet"])
            first_unique = pd.read_parquet(artifacts["first_unique_parquet"])
            last_unique = pd.read_parquet(artifacts["last_unique_parquet"])

            self.assertEqual(set(base_names["fullname_clean"].tolist()), {"Li", "J Li"})
            self.assertEqual(
                base_names.set_index("fullname_clean")["name_token_count"].to_dict(),
                {"J Li": 2, "Li": 1},
            )
            self.assertEqual(set(full_unique["fullname_clean"].tolist()), {"Li", "J Li"})
            self.assertTrue(first_unique.empty)
            self.assertEqual(last_unique["last_name_clean"].tolist(), ["Li"])
            self.assertEqual(artifacts["n_single_token_names"], 1)

    def test_name2nat_only_scores_one_word_names_once(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            users = pd.DataFrame(
                [
                    {"user_id": 1, "fullname": "Li"},
                    {"user_id": 2, "fullname": "J Li"},
                    {"user_id": 3, "fullname": "A B"},
                ]
            )
            users_path = tempdir / "users.parquet"
            users.to_parquet(users_path, index=False)
            name2nat_path = tempdir / "name2nat.parquet"

            cfg = {
                "build": {"overwrite": True},
                "testing": {"enabled": False},
                "stages": {
                    "04_rev_user_clean": {
                        "wrds_users_input_parquet": str(users_path),
                        "name2nat_parquet": str(name2nat_path),
                        "name2nat_use_mock": True,
                        "name2nat_batch_size": 1,
                        "name2nat_scoring_chunk_size": 1,
                    }
                },
            }

            stats = local_name2nat.run_name2nat(
                pipeline_cfg=cfg,
                use_mock=True,
                overwrite=True,
            )

            scored = pd.read_parquet(name2nat_path).set_index("fullname_clean")
            self.assertEqual(set(scored.index.tolist()), {"Li", "J Li"})
            self.assertEqual(stats["full_prediction_rows"], 2)
            self.assertEqual(stats["first_prediction_rows"], 0)
            self.assertEqual(stats["last_prediction_rows"], 1)
            self.assertNotEqual(scored.loc["Li", "pred_nats_full_json"], "{}")
            self.assertEqual(scored.loc["Li", "pred_nats_first_json"], "{}")
            self.assertEqual(scored.loc["Li", "pred_nats_last_json"], "{}")
            self.assertEqual(scored.loc["J Li", "pred_nats_first_json"], "{}")
            self.assertNotEqual(scored.loc["J Li", "pred_nats_last_json"], "{}")

    def test_name2nat_uses_names_dataset_before_falling_back_to_name2nat(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            users = pd.DataFrame(
                [
                    {"user_id": 1, "fullname": "Ravi Patel"},
                    {"user_id": 2, "fullname": "Mystery Qxz"},
                    {"user_id": 3, "fullname": "Li"},
                ]
            )
            users_path = tempdir / "users.parquet"
            users.to_parquet(users_path, index=False)
            name2nat_path = tempdir / "name2nat.parquet"

            calls: list[list[str]] = []

            fake_names_dataset = types.ModuleType("names_dataset")

            class FakeNameDataset:
                def __init__(self, load_first_names: bool = True, load_last_names: bool = True) -> None:
                    self.load_first_names = load_first_names
                    self.load_last_names = load_last_names

                def search(self, name: str) -> dict[str, object]:
                    table = {
                        "Ravi": {"first_name": {"country": {"India": 0.8, "United States": 0.2}}},
                        "Patel": {"last_name": {"country": {"India": 0.95, "United States": 0.05}}},
                        "Li": {
                            "first_name": {"country": {"China": 0.7, "United States": 0.3}},
                            "last_name": {"country": {"China": 0.9, "United States": 0.1}},
                        },
                    }
                    return table.get(name, {})

            fake_names_dataset.NameDataset = FakeNameDataset

            fake_name2nat = types.ModuleType("name2nat")

            class FakeName2nat:
                def __call__(self, names: list[str], top_n: int = 1, mini_batch_size: int = 128):
                    calls.append(list(names))
                    mapping = {
                        "Mystery Qxz": [("India", 0.6), ("United States", 0.4)],
                        "Mystery": [("Mexico", 0.7), ("United States", 0.3)],
                        "Qxz": [("Canada", 0.9), ("United States", 0.1)],
                    }
                    return [(name, mapping[name]) for name in names]

            fake_name2nat.Name2nat = FakeName2nat

            cfg = {
                "build": {"overwrite": True},
                "testing": {"enabled": False},
                "stages": {
                    "04_rev_user_clean": {
                        "wrds_users_input_parquet": str(users_path),
                        "name2nat_parquet": str(name2nat_path),
                        "name2nat_batch_size": 1,
                        "name2nat_scoring_chunk_size": 1,
                        "name2nat_use_names_dataset_lookup": True,
                        "name2nat_names_dataset_full_first_weight": 0.40,
                        "name2nat_names_dataset_full_last_weight": 0.60,
                    }
                },
            }

            with patch.dict(sys.modules, {"names_dataset": fake_names_dataset, "name2nat": fake_name2nat}):
                stats = local_name2nat.run_name2nat(
                    pipeline_cfg=cfg,
                    overwrite=True,
                )

            scored = pd.read_parquet(name2nat_path).set_index("fullname_clean")
            work_root = tempdir / "name2nat_work"
            full_progress = json.loads((work_root / "full_prediction_chunks" / "progress.json").read_text())
            first_progress = json.loads((work_root / "first_prediction_chunks" / "progress.json").read_text())
            last_progress = json.loads((work_root / "last_prediction_chunks" / "progress.json").read_text())
            self.assertEqual(calls, [["Mystery Qxz"], ["Mystery"], ["Qxz"]])
            self.assertEqual(stats["full_names_dataset_hits"], 2)
            self.assertEqual(stats["full_name2nat_hits"], 1)
            self.assertEqual(stats["first_names_dataset_hits"], 1)
            self.assertEqual(stats["first_name2nat_hits"], 1)
            self.assertEqual(stats["last_names_dataset_hits"], 1)
            self.assertEqual(stats["last_name2nat_hits"], 1)
            self.assertEqual(full_progress["status"], "complete")
            self.assertEqual(first_progress["status"], "complete")
            self.assertEqual(last_progress["status"], "complete")
            self.assertEqual(scored.loc["Li", "pred_nats_first_json"], "{}")
            self.assertEqual(scored.loc["Li", "pred_nats_last_json"], "{}")
            li_full = json.loads(scored.loc["Li", "pred_nats_full_json"])
            self.assertAlmostEqual(li_full["China"], 0.82, places=6)
            self.assertAlmostEqual(li_full["United States"], 0.18, places=6)
            mystery_last = json.loads(scored.loc["Mystery Qxz", "pred_nats_last_json"])
            self.assertAlmostEqual(mystery_last["Canada"], 0.9, places=6)
            self.assertAlmostEqual(mystery_last["United States"], 0.1, places=6)

    def test_nametrace_writes_progress_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            users = pd.DataFrame(
                [
                    {"user_id": 1, "fullname": "Ravi Patel"},
                    {"user_id": 2, "fullname": "Maria Garcia"},
                ]
            )
            users_path = tempdir / "users.parquet"
            users.to_parquet(users_path, index=False)
            wide_path = tempdir / "nametrace_wide.parquet"
            long_path = tempdir / "nametrace_long.parquet"

            cfg = {
                "build": {"overwrite": True},
                "testing": {"enabled": False},
                "stages": {
                    "04_rev_user_clean": {
                        "wrds_users_input_parquet": str(users_path),
                        "nametrace_wide_parquet": str(wide_path),
                        "nametrace_long_parquet": str(long_path),
                        "nametrace_use_mock": True,
                        "nametrace_batch_size": 1,
                        "nametrace_scoring_chunk_size": 1,
                    }
                },
            }

            stats = local_nametrace.run_nametrace(
                pipeline_cfg=cfg,
                use_mock=True,
                overwrite=True,
            )

            progress = json.loads((tempdir / "nametrace_wide_work" / "progress.json").read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["completed_rows"], 2)
            self.assertEqual(stats["nametrace_wide_rows"], 2)
            self.assertTrue(wide_path.exists())
            self.assertTrue(long_path.exists())

    def test_duckdb_connections_use_distinct_temp_directories(self) -> None:
        con_a = common.get_duckdb_connection()
        con_b = common.get_duckdb_connection()
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


if __name__ == "__main__":
    unittest.main()
