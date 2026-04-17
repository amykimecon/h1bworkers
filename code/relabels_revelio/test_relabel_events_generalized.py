from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "relabels_revelio" / "relabel_events_generalized.py"
spec = importlib.util.spec_from_file_location(
    "relabel_events_generalized_test_module",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
generalized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generalized)


class GeneralizedRelabelEventTests(unittest.TestCase):
    def _write_ipeds_fixture(self, rows: list[dict[str, object]], tempdir: Path) -> Path:
        path = tempdir / "ipeds.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def _write_crosswalk_fixture(self, rows: list[dict[str, object]], tempdir: Path) -> Path:
        path = tempdir / "ipeds_crosswalk.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def test_load_external_candidates_infers_alias_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            candidate_path = tmpdir / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "Institution": "Example University",
                        "Likely Year": 2020,
                        "Program Name": "Applied Math",
                        "Level": "Master of Science",
                        "Comments": "messy input",
                    }
                ]
            ).to_csv(candidate_path, index=False)

            loaded = generalized.load_external_candidates(candidate_path)

            self.assertEqual(list(loaded["candidate_school_name"]), ["Example University"])
            self.assertEqual(int(loaded.loc[0, "candidate_approx_year"]), 2020)
            self.assertEqual(loaded.loc[0, "candidate_program_desc"], "Applied Math")
            self.assertEqual(loaded.loc[0, "candidate_degree_type"], "Master")
            self.assertEqual(loaded.loc[0, "candidate_notes"], "messy input")

    def test_load_external_candidate_directory_parses_llm_style_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "institution name": "UNC",
                        "degree level": "MA",
                        "program name": "Full-Time MBA",
                        "initial CIP code": "not publicly located in this pass",
                        "new CIP code": "52.1301",
                        "date of relabel event": "2021-11-15",
                        "details (+ linked source) on how you discovered the relabel event": "[high confidence] official release",
                    }
                ]
            ).to_csv(tmpdir / "stem_opt_cip_relabel_events_batch2.csv", index=False)
            (tmpdir / "stem_opt_cip_relabel_notes_batch2.txt").write_text("notes only", encoding="utf-8")

            loaded = generalized.load_external_candidates(tmpdir)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.loc[0, "candidate_school_name"], "University of North Carolina at Chapel Hill")
            self.assertEqual(int(loaded.loc[0, "candidate_approx_year"]), 2021)
            self.assertEqual(loaded.loc[0, "candidate_date_raw"], "2021-11-15")
            self.assertEqual(loaded.loc[0, "candidate_target_cip6_hint"], "521301")
            self.assertEqual(loaded.loc[0, "candidate_target_cip_status"], "exact")
            self.assertTrue(pd.isna(loaded.loc[0, "candidate_source_cip6_hint"]))
            self.assertEqual(loaded.loc[0, "candidate_source_cip_status"], "missing")
            self.assertEqual(loaded.loc[0, "candidate_confidence"], "high")
            self.assertEqual(loaded.loc[0, "candidate_source_file"], "stem_opt_cip_relabel_events_batch2.csv")

    def test_load_external_candidate_directory_dedupes_known_berkeley_journalism_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "institution name": "UC Berkeley",
                        "degree level": "MA",
                        "program name": "Master of Journalism (MJ)",
                        "initial CIP code": "09.0401",
                        "new CIP code": "9.0702",
                        "date of relabel event": "2019-01-12",
                        "details (+ linked source) on how you discovered the relabel event": "journalism-specific source",
                    }
                ]
            ).to_csv(tmpdir / "journalism_cip_relabel_events_all_schools.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "institution name": "UC Berkeley",
                        "degree level": "MA",
                        "program name": "Master of Journalism (M.J.)",
                        "initial CIP code": "09.0401",
                        "new CIP code": "9.0702",
                        "date of relabel event": "2018",
                        "details (+ linked source) on how you discovered the relabel event": "broader stem-opt source",
                    }
                ]
            ).to_csv(tmpdir / "stem_opt_cip_relabel_events.csv", index=False)

            loaded = generalized.load_external_candidates(tmpdir)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.loc[0, "candidate_program_signature"], "master of journalism")
            self.assertEqual(loaded.loc[0, "candidate_source_file"], "journalism_cip_relabel_events_all_schools.csv")
            self.assertEqual(int(loaded.loc[0, "candidate_approx_year"]), 2019)
            self.assertEqual(loaded.loc[0, "candidate_target_cip6_hint"], "090702")

    def test_normalize_degree_type(self) -> None:
        self.assertEqual(generalized.normalize_degree_type("MBA"), "Master")
        self.assertEqual(generalized.normalize_degree_type("Bachelor of Arts"), "Bachelor")
        self.assertEqual(generalized.normalize_degree_type("undergrad"), "Bachelor")
        self.assertEqual(generalized.normalize_degree_type("PhD"), "Doctor")
        self.assertEqual(generalized.normalize_degree_type("Graduate Certificate"), "Other")

    def test_parse_candidate_cip_constraints_for_hand_coded_families(self) -> None:
        unc_constraints = generalized._parse_candidate_cip_constraints(
            {
                "candidate_program_desc": "Full-Time MBA",
                "candidate_source_cip6_hint": pd.NA,
                "candidate_target_cip6_hint": pd.NA,
            }
        )
        self.assertEqual(unc_constraints["candidate_source_cip_bin"], "business_core")
        self.assertEqual(unc_constraints["candidate_target_cip_bin"], "management_science_family")
        self.assertEqual(unc_constraints["candidate_pair_bin"], "business_core_to_management_science_family")

        journalism_constraints = generalized._parse_candidate_cip_constraints(
            {
                "candidate_program_desc": "Master of Journalism (MJ)",
                "candidate_source_cip6_hint": "090401",
                "candidate_target_cip6_hint": "090702",
            }
        )
        self.assertEqual(journalism_constraints["candidate_pair_bin"], "journalism_to_digital_media")
        self.assertEqual(journalism_constraints["candidate_source_cip_bin"], "exact_090401")
        self.assertEqual(journalism_constraints["candidate_target_cip_bin"], "exact_090702")

    def test_parse_args_ignores_ipykernel_f_argument(self) -> None:
        original_argv = sys.argv[:]
        original_ipykernel = sys.modules.get("ipykernel")
        try:
            sys.modules["ipykernel"] = object()  # simulate notebook context
            sys.argv = [
                "relabel_events_generalized.py",
                "-f",
                "/run/user/123/jupyter/runtime/kernel-test.json",
            ]
            args = generalized._parse_args()
            self.assertEqual(args.foia_path, generalized.base.FOIA_PATH)
            self.assertEqual(args.candidate_path, str(generalized.DEFAULT_CANDIDATE_PATH))
        finally:
            sys.argv = original_argv
            if original_ipykernel is None:
                sys.modules.pop("ipykernel", None)
            else:
                sys.modules["ipykernel"] = original_ipykernel

    def test_resolve_school_name_exact_and_fuzzy(self) -> None:
        lookup = pd.DataFrame(
            [
                {
                    "unitid": 1001,
                    "school_name": "University of Illinois Urbana-Champaign",
                    "school_name_clean": generalized._normalize_school_name("University of Illinois Urbana-Champaign"),
                    "alias_ind": False,
                },
                {
                    "unitid": 1001,
                    "school_name": "UIUC",
                    "school_name_clean": generalized._normalize_school_name("UIUC"),
                    "alias_ind": True,
                },
            ]
        )

        exact = generalized.resolve_school_name("UIUC", lookup)
        fuzzy = generalized.resolve_school_name("Univ Illinois Urbana Champaign", lookup, min_jw=0.85)

        self.assertEqual(int(exact["matched_unitid"]), 1001)
        self.assertEqual(exact["school_match_method"], "exact_clean")
        self.assertEqual(int(fuzzy["matched_unitid"]), 1001)
        self.assertEqual(fuzzy["school_match_method"], "jaro_winkler")

    def test_event_origin_precedence(self) -> None:
        row = pd.Series({"found_in_ipeds_scan": 1, "found_in_external_candidates": 1, "external_verified": 1})
        self.assertEqual(generalized._apply_event_origin_category(row), "external_ipeds_verified")
        row = pd.Series({"found_in_ipeds_scan": 0, "found_in_external_candidates": 1, "external_verified": 0})
        self.assertEqual(generalized._apply_event_origin_category(row), "external_only")
        row = pd.Series({"found_in_ipeds_scan": 1, "found_in_external_candidates": 0, "external_verified": 0})
        self.assertEqual(generalized._apply_event_origin_category(row), "ipeds_only")

    def test_detects_same_degree_relabel_outside_econ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 100, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 100, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 100, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 100, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 100, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 100, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520801, "cipcode_lab": "52.0801-Finance, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 100, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 100, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 100, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                ],
                tmpdir,
            )
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertEqual(len(events), 1)
            self.assertEqual(int(events.loc[0, "unitid"]), 100)
            self.assertEqual(int(events.loc[0, "relabel_year"]), 2021)
            self.assertEqual(events.loc[0, "event_source_cip6"], "520801")
            self.assertEqual(events.loc[0, "target_cip6"], "270301")
            self.assertEqual(events.loc[0, "degree_type"], "Master")

    def test_relaxed_candidate_verification_can_succeed_when_strict_scan_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 200, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 200, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 200, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 200, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 200, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 200, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 200, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 200, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                    {"unitid": 200, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                ],
                tmpdir,
            )
            crosswalk_path = self._write_crosswalk_fixture(
                [{"UNITID": 200, "instname": "Example State University", "ALIAS": False}],
                tmpdir,
            )
            candidate_path = tmpdir / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "school_name": "Example State University",
                        "approx_year": 2021,
                        "description": "full-time mba",
                        "degree": "masters",
                        "initial_cip": "52.0201",
                        "new_cip": "52.1301",
                    }
                ]
            ).to_csv(candidate_path, index=False)

            con = duckdb.connect()
            strict_events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)
            self.assertTrue(strict_events.empty)

            candidates = generalized.load_external_candidates(candidate_path)
            lookup = generalized.load_school_lookup(crosswalk_path)
            resolved = generalized.resolve_candidate_schools(candidates, lookup)
            cip_map = generalized._load_ipeds_cip_map(ipeds_path)
            verified_external, audit = generalized.verify_external_candidates(
                con,
                resolved,
                ipeds_path=ipeds_path,
                cip_map=cip_map,
            )

            self.assertEqual(len(verified_external), 1)
            self.assertEqual(int(audit.loc[0, "external_verified"]), 1)
            self.assertEqual(int(verified_external.loc[0, "relabel_year"]), 2021)
            self.assertEqual(verified_external.loc[0, "event_source_cip6"], "520201")
            self.assertEqual(verified_external.loc[0, "target_cip6"], "521301")

    def test_verification_rejects_nearby_pair_when_candidate_cip_bins_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 210, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 210, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 210, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.3},
                    {"unitid": 210, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 210, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 210, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.3},
                    {"unitid": 210, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 210, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                    {"unitid": 210, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                ],
                tmpdir,
            )
            crosswalk_path = self._write_crosswalk_fixture(
                [{"UNITID": 210, "instname": "Example State University", "ALIAS": False}],
                tmpdir,
            )
            candidate_path = tmpdir / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "school_name": "Example State University",
                        "approx_year": 2021,
                        "description": "media science",
                        "degree": "masters",
                        "new_cip": "09.0702",
                    }
                ]
            ).to_csv(candidate_path, index=False)

            con = duckdb.connect()
            candidates = generalized.load_external_candidates(candidate_path)
            lookup = generalized.load_school_lookup(crosswalk_path)
            resolved = generalized.resolve_candidate_schools(candidates, lookup)
            cip_map = generalized._load_ipeds_cip_map(ipeds_path)
            verified_external, audit = generalized.verify_external_candidates(
                con,
                resolved,
                ipeds_path=ipeds_path,
                cip_map=cip_map,
            )

            self.assertTrue(verified_external.empty)
            self.assertEqual(int(audit.loc[0, "external_verified"]), 0)
            self.assertEqual(audit.loc[0, "verification_notes"], "no_relaxed_ipeds_match_for_candidate_cip_bins")

    def test_external_only_rows_keep_null_standardized_fields(self) -> None:
        strict = pd.DataFrame()
        verified_external = pd.DataFrame()
        candidate_audit = pd.DataFrame(
            [
                {
                    "candidate_id": "cand_1",
                    "candidate_school_name": "Unknown University",
                    "candidate_approx_year": 2020,
                    "candidate_program_desc": "something",
                    "candidate_degree_label": "masters",
                    "candidate_degree_type": "Master",
                    "candidate_notes": "",
                    "candidate_major": "something",
                    "matched_unitid": pd.NA,
                    "school_match_method": "unmatched",
                    "school_match_score": float("nan"),
                    "school_match_name": pd.NA,
                    "external_verified": 0,
                    "verification_notes": "school_unmatched",
                    "best_candidate_rank_score": float("nan"),
                    "best_year_distance": pd.NA,
                    "best_text_similarity": float("nan"),
                    "best_nearby_year": pd.NA,
                    "best_nearby_source_cip6": pd.NA,
                    "best_nearby_target_cip6": pd.NA,
                    "best_nearby_relabel_score": float("nan"),
                    "diagnostic_best_year": pd.NA,
                    "diagnostic_best_source_cip6": pd.NA,
                    "diagnostic_best_target_cip6": pd.NA,
                    "diagnostic_best_score": float("nan"),
                }
            ]
        )

        merged = generalized.merge_event_sources(strict, verified_external, candidate_audit)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "event_origin_category"], "external_only")
        self.assertTrue(pd.isna(merged.loc[0, "relabel_year"]))
        self.assertTrue(pd.isna(merged.loc[0, "event_source_cip6"]))
        self.assertTrue(pd.isna(merged.loc[0, "target_cip6"]))

    def test_output_dtype_coercion_allows_parquet_write_with_mixed_candidate_years(self) -> None:
        merged = pd.DataFrame(
            [
                {
                    "unitid": 1,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "year": 2020,
                    "relabel_type": "111111_to_222222",
                    "event_source_cip6": "111111",
                    "target_cip6": "222222",
                    "source_total": 4.0,
                    "source_total_prev": 8.0,
                    "source_total_intl": 4.0,
                    "source_total_intl_prev": 8.0,
                    "target_total": 6.0,
                    "target_total_prev": 0.0,
                    "target_total_intl": 6.0,
                    "target_total_intl_prev": 0.0,
                    "ctotalt": 10.0,
                    "cnralt": 8.0,
                    "source_drop": 4.0,
                    "source_drop_pct": 0.5,
                    "target_increase": 6.0,
                    "target_increase_pct": 1.0,
                    "avg5_source_drop": 4.0,
                    "avg5_source_drop_pct": 0.5,
                    "avg5_target_increase": 6.0,
                    "avg5_target_increase_pct": 1.0,
                    "source_baseline": 8.0,
                    "target_baseline": 0.0,
                    "relabel_score": 1.5,
                    "found_in_ipeds_scan": 1,
                    "found_in_external_candidates": 1,
                    "external_verified": 1,
                    "event_origin_category": "external_ipeds_verified",
                    "source_cip_label": "Source",
                    "target_cip_label": "Target",
                    "source_major": "Source",
                    "target_major": "Target",
                    "event_flag": 1,
                    "relabel_flag": 1,
                    "candidate_id": "cand_1",
                    "candidate_school_name": "Example U",
                    "candidate_approx_year": "2019 | 2020",
                    "candidate_program_desc": "Example Program",
                    "candidate_degree_label": "masters",
                    "candidate_notes": "",
                    "candidate_major": "example",
                    "n_linked_candidates": 2,
                    "school_match_method": "exact_clean",
                    "school_match_score": 1.0,
                    "school_match_name": "Example U",
                    "verification_notes": "verified_relaxed_ipeds",
                    "best_candidate_rank_score": 1.6,
                    "best_year_distance": 0,
                    "best_text_similarity": 0.8,
                    "best_nearby_year": 2020,
                    "best_nearby_source_cip6": "111111",
                    "best_nearby_target_cip6": "222222",
                    "best_nearby_relabel_score": 1.5,
                    "diagnostic_best_year": 2020,
                    "diagnostic_best_source_cip6": "111111",
                    "diagnostic_best_target_cip6": "222222",
                    "diagnostic_best_score": 1.6,
                },
                {
                    "unitid": 2,
                    "awlevel": pd.NA,
                    "degree_type": "Master",
                    "relabel_year": pd.NA,
                    "year": pd.NA,
                    "relabel_type": pd.NA,
                    "event_source_cip6": pd.NA,
                    "target_cip6": pd.NA,
                    "source_total": pd.NA,
                    "source_total_prev": pd.NA,
                    "source_total_intl": pd.NA,
                    "source_total_intl_prev": pd.NA,
                    "target_total": pd.NA,
                    "target_total_prev": pd.NA,
                    "target_total_intl": pd.NA,
                    "target_total_intl_prev": pd.NA,
                    "ctotalt": pd.NA,
                    "cnralt": pd.NA,
                    "source_drop": pd.NA,
                    "source_drop_pct": pd.NA,
                    "target_increase": pd.NA,
                    "target_increase_pct": pd.NA,
                    "avg5_source_drop": pd.NA,
                    "avg5_source_drop_pct": pd.NA,
                    "avg5_target_increase": pd.NA,
                    "avg5_target_increase_pct": pd.NA,
                    "source_baseline": pd.NA,
                    "target_baseline": pd.NA,
                    "relabel_score": pd.NA,
                    "found_in_ipeds_scan": 0,
                    "found_in_external_candidates": 1,
                    "external_verified": 0,
                    "event_origin_category": "external_only",
                    "source_cip_label": pd.NA,
                    "target_cip_label": pd.NA,
                    "source_major": pd.NA,
                    "target_major": pd.NA,
                    "event_flag": 0,
                    "relabel_flag": 0,
                    "candidate_id": "cand_2",
                    "candidate_school_name": "Other U",
                    "candidate_approx_year": 2021,
                    "candidate_program_desc": "Other Program",
                    "candidate_degree_label": "masters",
                    "candidate_notes": "",
                    "candidate_major": "other",
                    "n_linked_candidates": 1,
                    "school_match_method": "unmatched",
                    "school_match_score": pd.NA,
                    "school_match_name": pd.NA,
                    "verification_notes": "school_unmatched",
                    "best_candidate_rank_score": pd.NA,
                    "best_year_distance": pd.NA,
                    "best_text_similarity": pd.NA,
                    "best_nearby_year": pd.NA,
                    "best_nearby_source_cip6": pd.NA,
                    "best_nearby_target_cip6": pd.NA,
                    "best_nearby_relabel_score": pd.NA,
                    "diagnostic_best_year": pd.NA,
                    "diagnostic_best_source_cip6": pd.NA,
                    "diagnostic_best_target_cip6": pd.NA,
                    "diagnostic_best_score": pd.NA,
                },
            ]
        )
        merged = merged.reindex(columns=generalized.VERIFIED_EVENT_COLUMNS)
        coerced = generalized._coerce_verified_event_output_dtypes(merged)

        self.assertEqual(str(coerced["candidate_approx_year"].dtype), "string")
        self.assertEqual(str(coerced.loc[0, "candidate_approx_year"]), "2019 | 2020")
        self.assertEqual(str(coerced.loc[1, "candidate_approx_year"]), "2021")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.parquet"
            coerced.to_parquet(out, index=False)
            self.assertTrue(out.exists())

    def test_merge_dedupes_when_same_event_appears_in_ipeds_and_external(self) -> None:
        strict = pd.DataFrame(
            [
                {
                    "unitid": 300,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "year": 2020,
                    "relabel_type": "520801_to_270301",
                    "event_source_cip6": "520801",
                    "target_cip6": "270301",
                    "source_total": 4.0,
                    "source_total_prev": 20.0,
                    "source_total_intl": 4.0,
                    "source_total_intl_prev": 20.0,
                    "target_total": 18.0,
                    "target_total_prev": 0.0,
                    "target_total_intl": 18.0,
                    "target_total_intl_prev": 0.0,
                    "ctotalt": 22.0,
                    "cnralt": 12.0,
                    "source_drop": 16.0,
                    "source_drop_pct": 0.8,
                    "target_increase": 18.0,
                    "target_increase_pct": pd.NA,
                    "avg5_source_drop": -16.0,
                    "avg5_source_drop_pct": -0.8,
                    "avg5_target_increase": 18.0,
                    "avg5_target_increase_pct": pd.NA,
                    "source_baseline": 20.0,
                    "target_baseline": 0.0,
                    "relabel_score": 2.1,
                    "found_in_ipeds_scan": 1,
                    "found_in_external_candidates": 0,
                    "external_verified": 0,
                    "event_origin_category": "ipeds_only",
                    "source_cip_label": "Finance, General",
                    "target_cip_label": "Applied Mathematics, General",
                    "source_major": "Finance, General",
                    "target_major": "Applied Mathematics, General",
                    "event_flag": 1,
                    "relabel_flag": 1,
                }
            ]
        )
        verified_external = strict.copy()
        verified_external["found_in_ipeds_scan"] = 0
        verified_external["found_in_external_candidates"] = 1
        verified_external["external_verified"] = 1
        verified_external["candidate_id"] = "cand_1"
        verified_external["candidate_school_name"] = "Example U"
        verified_external["candidate_approx_year"] = 2020
        verified_external["candidate_program_desc"] = "finance to math"
        verified_external["candidate_degree_label"] = "masters"
        verified_external["candidate_notes"] = ""
        verified_external["candidate_major"] = "finance to math"
        verified_external["school_match_method"] = "exact_clean"
        verified_external["school_match_score"] = 1.0
        verified_external["school_match_name"] = "Example U"
        verified_external["verification_notes"] = "verified_relaxed_ipeds"
        verified_external["best_candidate_rank_score"] = 2.3
        verified_external["best_year_distance"] = 0
        verified_external["best_text_similarity"] = 0.8
        verified_external["best_nearby_year"] = 2020
        verified_external["best_nearby_source_cip6"] = "520801"
        verified_external["best_nearby_target_cip6"] = "270301"
        verified_external["best_nearby_relabel_score"] = 2.1
        verified_external["diagnostic_best_year"] = 2020
        verified_external["diagnostic_best_source_cip6"] = "520801"
        verified_external["diagnostic_best_target_cip6"] = "270301"
        verified_external["diagnostic_best_score"] = 2.3
        candidate_audit = pd.DataFrame(
            [
                {
                    "candidate_id": "cand_1",
                    "candidate_school_name": "Example U",
                    "candidate_approx_year": 2020,
                    "candidate_program_desc": "finance to math",
                    "candidate_degree_label": "masters",
                    "candidate_degree_type": "Master",
                    "candidate_notes": "",
                    "candidate_major": "finance to math",
                    "matched_unitid": 300,
                    "school_match_method": "exact_clean",
                    "school_match_score": 1.0,
                    "school_match_name": "Example U",
                    "external_verified": 1,
                    "verification_notes": "verified_relaxed_ipeds",
                    "best_candidate_rank_score": 2.3,
                    "best_year_distance": 0,
                    "best_text_similarity": 0.8,
                    "best_nearby_year": 2020,
                    "best_nearby_source_cip6": "520801",
                    "best_nearby_target_cip6": "270301",
                    "best_nearby_relabel_score": 2.1,
                    "diagnostic_best_year": 2020,
                    "diagnostic_best_source_cip6": "520801",
                    "diagnostic_best_target_cip6": "270301",
                    "diagnostic_best_score": 2.3,
                }
            ]
        )

        merged = generalized.merge_event_sources(strict, verified_external, candidate_audit)

        self.assertEqual(len(merged), 1)
        self.assertEqual(int(merged.loc[0, "found_in_ipeds_scan"]), 1)
        self.assertEqual(int(merged.loc[0, "found_in_external_candidates"]), 1)
        self.assertEqual(int(merged.loc[0, "external_verified"]), 1)
        self.assertEqual(merged.loc[0, "event_origin_category"], "external_ipeds_verified")

    @unittest.skipUnless(
        os.environ.get("RUN_GENERALIZED_RELABEL_SMOKE") == "1",
        "set RUN_GENERALIZED_RELABEL_SMOKE=1 to run the real-data smoke test",
    )
    def test_real_data_smoke_panel_report_and_empty_degree_plot_skip(self) -> None:
        real_events_path = Path(generalized.base.INT_FOLDER) / "econ_relabels_v2.parquet"
        if not real_events_path.exists() or not Path(generalized.base.IPEDS_PATH).exists():
            self.skipTest("real cached data not available")

        raw = pd.read_parquet(real_events_path)
        source_rows = raw[raw["event_flag"] == 1].head(2).copy()
        if source_rows.empty:
            self.skipTest("no real cached event rows available")

        cip_map = generalized._load_ipeds_cip_map(generalized.base.IPEDS_PATH)
        events = pd.DataFrame(
            [
                {
                    "unitid": int(row.unitid),
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": int(row.relabel_year),
                    "year": int(row.relabel_year),
                    "relabel_type": row.relabel_type,
                    "event_source_cip6": str(row.event_source_cip6),
                    "target_cip6": "450603",
                    "source_total": float(row.source_total),
                    "source_total_prev": float(row.source_total_prev),
                    "source_total_intl": float(row.source_total_intl),
                    "source_total_intl_prev": float(row.source_total_intl_prev),
                    "target_total": float(row.target_total),
                    "target_total_prev": float(row.target_total_prev),
                    "target_total_intl": float(row.target_total_intl),
                    "target_total_intl_prev": float(row.target_total_intl_prev),
                    "ctotalt": float(row.ctotalt),
                    "cnralt": float(row.cnralt),
                    "source_drop": float(row.source_drop),
                    "source_drop_pct": float(row.source_drop_pct),
                    "target_increase": float(row.target_increase),
                    "target_increase_pct": row.target_increase_pct,
                    "avg5_source_drop": float(row.avg5_source_drop),
                    "avg5_source_drop_pct": float(row.avg5_source_drop_pct),
                    "avg5_target_increase": float(row.avg5_target_increase),
                    "avg5_target_increase_pct": row.avg5_target_increase_pct,
                    "source_baseline": float(row.source_baseline),
                    "target_baseline": float(row.target_baseline),
                    "relabel_score": float(row.relabel_score),
                    "found_in_ipeds_scan": 1,
                    "found_in_external_candidates": 0,
                    "external_verified": 0,
                    "event_origin_category": "ipeds_only",
                    "source_cip_label": cip_map.get(str(row.event_source_cip6), ""),
                    "target_cip_label": cip_map.get("450603", ""),
                    "source_major": cip_map.get(str(row.event_source_cip6), ""),
                    "target_major": cip_map.get("450603", ""),
                    "event_flag": 1,
                    "relabel_flag": 1,
                }
                for row in source_rows.itertuples(index=False)
            ]
        )
        for column in generalized.VERIFIED_EVENT_COLUMNS:
            if column not in events.columns:
                events[column] = pd.NA
        events = events.loc[:, generalized.VERIFIED_EVENT_COLUMNS]

        con = duckdb.connect()
        panel = generalized.build_verified_event_panel(con, events, ipeds_path=generalized.base.IPEDS_PATH)
        self.assertFalse(panel.empty)
        self.assertTrue(panel["event_flag"].sum() >= 1)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            report_path = tmpdir / "report.txt"
            generalized.write_generalized_report(events, pd.DataFrame(), report_path)
            report_text = report_path.read_text()
            self.assertIn("Event counts by origin", report_text)
            self.assertIn("ipeds_only", report_text)

            plots = generalized.run_degree_plots(
                con,
                events,
                panel,
                plots_dir=tmpdir / "plots",
                yvars=["opt_share"],
                foia_path=generalized.base.FOIA_PATH,
                inst_cw_path=generalized.base.F1_INST_CW_PATH,
                ipeds_path=generalized.base.IPEDS_PATH,
            )
            self.assertTrue(any("master" in path.name for path in plots))


if __name__ == "__main__":
    unittest.main()
