from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import matplotlib
import matplotlib.pyplot as plt
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
    def test_ipeds_degree_cost_expression_uses_undergrad_for_bachelors(self) -> None:
        con = duckdb.connect()
        try:
            con.sql(
                """
                CREATE TEMP VIEW ipeds_cost_raw AS
                SELECT 1 AS unitid, 2020 AS year,
                       30000.0 AS tuition3, 50000.0 AS tuition7,
                       3000.0 AS fee3, 5000.0 AS fee7
                """
            )
            tuition_cols = generalized._resolve_ipeds_cost_columns_by_degree(
                con,
                generalized.DEFAULT_IPEDS_TUITION_COL_BY_DEGREE,
                fallback_prefix="tuition",
            )
            fee_cols = generalized._resolve_ipeds_cost_columns_by_degree(
                con,
                generalized.DEFAULT_IPEDS_FEE_COL_BY_DEGREE,
                fallback_prefix="fee",
            )
            self.assertEqual(tuition_cols["Bachelor"], "tuition3")
            self.assertEqual(tuition_cols["Master"], "tuition7")
            self.assertEqual(fee_cols["Bachelor"], "fee3")
            self.assertEqual(fee_cols["Doctor"], "fee7")
        finally:
            con.close()

    def _write_ipeds_fixture(self, rows: list[dict[str, object]], tempdir: Path) -> Path:
        path = tempdir / "ipeds.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def _write_crosswalk_fixture(self, rows: list[dict[str, object]], tempdir: Path) -> Path:
        path = tempdir / "ipeds_crosswalk.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path

    def _econ_master_phd_guard_rows(
        self,
        *,
        unitid: int = 300,
        master_year: int = 2021,
        doctor_year: int = 2024,
        master_source_cip: int = 450601,
        doctor_source_cip: int = 450601,
        doctor_source_pre: int = 18,
        doctor_source_curr: int = 6,
        doctor_target_curr: int = 10,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        cip_labels = {
            450601: "45.0601-Economics, General",
            450603: "45.0603-Econometrics and Quantitative Economics",
            450699: "45.0699-Economics, Other",
        }

        def _cip_label(cipcode: int) -> str:
            return cip_labels.get(cipcode, f"{str(cipcode).zfill(6)}-Program")

        for year in range(master_year - 3, master_year):
            rows.append(
                {
                    "unitid": unitid,
                    "year": year,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": master_source_cip,
                    "cipcode_lab": _cip_label(master_source_cip),
                    "ctotalt": 20,
                    "cnralt": 10,
                    "share_intl": 0.6,
                }
            )
        rows.extend(
            [
                {
                    "unitid": unitid,
                    "year": master_year,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": master_source_cip,
                    "cipcode_lab": _cip_label(master_source_cip),
                    "ctotalt": 4,
                    "cnralt": 2,
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": master_year,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": 12,
                    "cnralt": 7,
                    "share_intl": 0.7,
                },
                {
                    "unitid": unitid,
                    "year": master_year + 1,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": master_source_cip,
                    "cipcode_lab": _cip_label(master_source_cip),
                    "ctotalt": 4,
                    "cnralt": 2,
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": master_year + 1,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": 12,
                    "cnralt": 7,
                    "share_intl": 0.7,
                },
                {
                    "unitid": unitid,
                    "year": master_year + 2,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": master_source_cip,
                    "cipcode_lab": _cip_label(master_source_cip),
                    "ctotalt": 4,
                    "cnralt": 2,
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": master_year + 2,
                    "awlevel": 7,
                    "awlevel_group": "Master",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": 12,
                    "cnralt": 7,
                    "share_intl": 0.7,
                },
            ]
        )

        for year in range(master_year - 3, doctor_year):
            rows.append(
                {
                    "unitid": unitid,
                    "year": year,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": doctor_source_cip,
                    "cipcode_lab": _cip_label(doctor_source_cip),
                    "ctotalt": doctor_source_pre,
                    "cnralt": max(1, doctor_source_pre // 2),
                    "share_intl": 0.6,
                }
            )
        rows.extend(
            [
                {
                    "unitid": unitid,
                    "year": doctor_year,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": doctor_source_cip,
                    "cipcode_lab": _cip_label(doctor_source_cip),
                    "ctotalt": doctor_source_curr,
                    "cnralt": max(1, doctor_source_curr // 2),
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": doctor_year,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": doctor_target_curr,
                    "cnralt": max(1, doctor_target_curr // 2),
                    "share_intl": 0.7,
                },
                {
                    "unitid": unitid,
                    "year": doctor_year + 1,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": doctor_source_cip,
                    "cipcode_lab": _cip_label(doctor_source_cip),
                    "ctotalt": doctor_source_curr,
                    "cnralt": max(1, doctor_source_curr // 2),
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": doctor_year + 1,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": doctor_target_curr,
                    "cnralt": max(1, doctor_target_curr // 2),
                    "share_intl": 0.7,
                },
                {
                    "unitid": unitid,
                    "year": doctor_year + 2,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": doctor_source_cip,
                    "cipcode_lab": _cip_label(doctor_source_cip),
                    "ctotalt": doctor_source_curr,
                    "cnralt": max(1, doctor_source_curr // 2),
                    "share_intl": 0.6,
                },
                {
                    "unitid": unitid,
                    "year": doctor_year + 2,
                    "awlevel": 17,
                    "awlevel_group": "Doctor",
                    "cipcode": 450603,
                    "cipcode_lab": "45.0603-Econometrics and Quantitative Economics",
                    "ctotalt": doctor_target_curr,
                    "cnralt": max(1, doctor_target_curr // 2),
                    "share_intl": 0.7,
                },
            ]
        )
        return rows

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

        finance_constraints = generalized._parse_candidate_cip_constraints(
            {
                "candidate_program_desc": "Master of Science in Finance",
                "candidate_source_cip6_hint": pd.NA,
                "candidate_target_cip6_hint": "270305",
            }
        )
        self.assertEqual(finance_constraints["candidate_pair_bin"], "finance_to_quantitative_finance_family")
        self.assertEqual(finance_constraints["candidate_source_cip_bin"], "finance_core")
        self.assertEqual(finance_constraints["candidate_target_cip_bin"], "exact_270305")

    def test_exclude_disallowed_candidate_rows_drops_communication_to_digital_media(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "keep",
                    "candidate_pair_bin": "economics_to_econometrics",
                },
                {
                    "candidate_id": "drop",
                    "candidate_pair_bin": "journalism_to_digital_media",
                },
            ]
        )

        filtered = generalized._exclude_disallowed_candidate_rows(candidates)

        self.assertEqual(filtered["candidate_id"].tolist(), ["keep"])
        self.assertNotIn("broad_pair_bin", filtered.columns)

    def test_annotate_event_broad_bins_marks_business_non_52_target_ineligible(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_source_cip6": "520201",
                    "target_cip6": "450603",
                }
            ]
        )

        annotated = generalized.annotate_event_broad_bins(events)

        self.assertTrue(pd.isna(annotated.loc[0, "broad_pair_bin"]))
        self.assertEqual(int(annotated.loc[0, "broad_bin_eligible"]), 0)

    def test_annotate_event_broad_bins_routes_finance_to_finance_bin(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_source_cip6": "520801",
                    "target_cip6": "270501",
                }
            ]
        )

        annotated = generalized.annotate_event_broad_bins(events)

        self.assertEqual(annotated.loc[0, "broad_pair_bin"], "finance_to_quantitative_finance")
        self.assertEqual(int(annotated.loc[0, "broad_bin_eligible"]), 1)

    def test_annotate_event_broad_bins_rejects_finance_to_5213_target(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_source_cip6": "520801",
                    "target_cip6": "521301",
                }
            ]
        )

        annotated = generalized.annotate_event_broad_bins(events)

        self.assertTrue(pd.isna(annotated.loc[0, "broad_pair_bin"]))
        self.assertEqual(int(annotated.loc[0, "broad_bin_eligible"]), 0)

    def test_exclude_disallowed_event_rows_drops_communication_to_digital_media(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "unitid": 1,
                    "event_source_cip6": "090401",
                    "target_cip6": "090702",
                },
                {
                    "unitid": 2,
                    "event_source_cip6": "450601",
                    "target_cip6": "450603",
                },
            ]
        )

        filtered = generalized._exclude_disallowed_event_rows(events)

        self.assertEqual(filtered["unitid"].tolist(), [2])
        self.assertEqual(filtered["broad_pair_bin"].tolist(), ["econ_to_quant_econ"])
        self.assertEqual(filtered["broad_bin_eligible"].astype(int).tolist(), [1])

    def test_annotate_event_broad_bins_marks_public_policy_pair_ineligible(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_source_cip6": "440401",
                    "target_cip6": "450102",
                }
            ]
        )

        annotated = generalized.annotate_event_broad_bins(events)

        self.assertTrue(pd.isna(annotated.loc[0, "broad_pair_bin"]))
        self.assertEqual(int(annotated.loc[0, "broad_bin_eligible"]), 0)

    def test_build_broad_bin_membership_tightens_business_to_52_only(self) -> None:
        membership = generalized.build_broad_bin_membership(
            ["520201", "520301", "520801", "521301", "521399", "270501", "450603", "111005"]
        )

        business = membership["business_52_to_52"]
        self.assertEqual(set(business["source_cips"]), {"520201", "520301"})
        self.assertEqual(set(business["target_cips"]), {"521301", "521399"})

        finance = membership["finance_to_quantitative_finance"]
        self.assertEqual(set(finance["source_cips"]), {"520801"})
        self.assertEqual(set(finance["target_cips"]), {"270501"})

    def test_pair_control_cip_rows_expands_always_stem_control_cip2(self) -> None:
        matched_pairs = pd.DataFrame(
            [
                {
                    "pair_id": 1,
                    "control_group": generalized.CONTROL_GROUP_ALWAYS_STEM,
                    "control_cip6": "110101",
                    "control_cip2": "11",
                    "broad_pair_bin": "econ_to_quant_econ",
                }
            ]
        )
        always_stem_cips = pd.DataFrame(
            {
                "control_cip6": ["110101", "110701", "140101"],
                "control_cip2": ["11", "11", "14"],
            }
        )

        rows = generalized._pair_control_cip_rows(
            matched_pairs,
            generalized.build_broad_bin_membership(["450601", "450603"]),
            always_stem_cips=always_stem_cips,
        )

        self.assertEqual(set(rows["cip6"].tolist()), {"110101", "110701"})

    def test_derive_allowable_pair_configs_dedupes_candidate_families(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_program_desc": "Full-Time MBA",
                    "candidate_source_cip6_hint": pd.NA,
                    "candidate_target_cip6_hint": pd.NA,
                },
                {
                    "candidate_program_desc": "Evening MBA",
                    "candidate_source_cip6_hint": pd.NA,
                    "candidate_target_cip6_hint": pd.NA,
                },
                {
                    "candidate_program_desc": "Economics major",
                    "candidate_source_cip6_hint": "450601",
                    "candidate_target_cip6_hint": "450603",
                },
            ]
        )

        configs = generalized.derive_allowable_pair_configs(candidates)

        self.assertEqual(len(configs), 2)
        pair_bins = {config["candidate_pair_bin"] for config in configs}
        self.assertEqual(
            pair_bins,
            {"business_core_to_management_science_family", "economics_to_econometrics"},
        )

    def test_build_broad_treated_events_excludes_ineligible_and_keeps_earliest_year(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "unitid": 10,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2022,
                    "year": 2022,
                    "relabel_type": "450601_to_450603",
                    "event_source_cip6": "450601",
                    "target_cip6": "450603",
                    "event_origin_category": "ipeds_only",
                    "relabel_score": 1.0,
                },
                {
                    "unitid": 10,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "year": 2020,
                    "relabel_type": "450602_to_450603",
                    "event_source_cip6": "450602",
                    "target_cip6": "450603",
                    "event_origin_category": "ipeds_only",
                    "relabel_score": 2.0,
                },
                {
                    "unitid": 10,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2021,
                    "year": 2021,
                    "relabel_type": "520201_to_450603",
                    "event_source_cip6": "520201",
                    "target_cip6": "450603",
                    "event_origin_category": "ipeds_only",
                    "relabel_score": 3.0,
                },
            ]
        )

        broad_events = generalized.build_broad_treated_events(events)

        self.assertEqual(len(broad_events), 1)
        self.assertEqual(int(broad_events.loc[0, "relabel_year"]), 2020)
        self.assertEqual(broad_events.loc[0, "broad_pair_bin"], "econ_to_quant_econ")
        self.assertEqual(int(broad_events.loc[0, "broad_bin_eligible"]), 1)

    def test_build_broad_treated_events_drops_pre_2014_events(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "unitid": 10,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2012,
                    "year": 2012,
                    "relabel_type": "450601_to_450603",
                    "event_source_cip6": "450601",
                    "target_cip6": "450603",
                    "event_origin_category": "ipeds_only",
                    "relabel_score": 2.0,
                },
                {
                    "unitid": 10,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2014,
                    "year": 2014,
                    "relabel_type": "450602_to_450603",
                    "event_source_cip6": "450602",
                    "target_cip6": "450603",
                    "event_origin_category": "ipeds_only",
                    "relabel_score": 1.0,
                },
            ]
        )

        broad_events = generalized.build_broad_treated_events(events)

        self.assertEqual(len(broad_events), 1)
        self.assertEqual(int(broad_events.loc[0, "relabel_year"]), 2014)
        self.assertTrue((broad_events["relabel_year"] >= 2014).all())

    def test_build_broad_bin_year_counts_filters_to_event_rows_and_degree(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "unitid": 1,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
                {
                    "unitid": 1,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "broad_bin_eligible": 1,
                    "event_flag": 0,
                    "event_origin_category": "ipeds_only",
                },
                {
                    "unitid": 2,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "external_ipeds_verified",
                },
                {
                    "unitid": 3,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "broad_pair_bin": "business_52_to_52",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
                {
                    "unitid": 4,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2021,
                    "broad_pair_bin": "business_52_to_52",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "external_ipeds_verified",
                },
                {
                    "unitid": 5,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2021,
                    "broad_pair_bin": pd.NA,
                    "broad_bin_eligible": 0,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
                {
                    "unitid": 6,
                    "awlevel": 5,
                    "degree_type": "Bachelor",
                    "relabel_year": 2021,
                    "broad_pair_bin": "communication_to_digital_media",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
            ]
        )

        counts = generalized.build_broad_bin_year_counts(events, degree_type="Master")

        econ_2020 = counts[
            (counts["relabel_year"] == 2020)
            & counts["broad_pair_bin"].eq("econ_to_quant_econ")
        ]
        business_2020 = counts[
            (counts["relabel_year"] == 2020)
            & counts["broad_pair_bin"].eq("business_52_to_52")
        ]
        business_2021 = counts[
            (counts["relabel_year"] == 2021)
            & counts["broad_pair_bin"].eq("business_52_to_52")
        ]
        communication_2021 = counts[
            (counts["relabel_year"] == 2021)
            & counts["broad_pair_bin"].eq("communication_to_digital_media")
        ]

        self.assertEqual(int(econ_2020.iloc[0]["event_count"]), 2)
        self.assertEqual(int(business_2020.iloc[0]["event_count"]), 1)
        self.assertEqual(int(business_2021.iloc[0]["event_count"]), 1)
        self.assertEqual(int(communication_2021.iloc[0]["event_count"]), 0)

    def test_plot_broad_bin_event_counts_by_year_writes_png(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "unitid": 1,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2020,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
                {
                    "unitid": 2,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2021,
                    "broad_pair_bin": "business_52_to_52",
                    "broad_bin_eligible": 1,
                    "event_flag": 1,
                    "event_origin_category": "external_ipeds_verified",
                },
                {
                    "unitid": 3,
                    "awlevel": 7,
                    "degree_type": "Master",
                    "relabel_year": 2021,
                    "broad_pair_bin": pd.NA,
                    "broad_bin_eligible": 0,
                    "event_flag": 1,
                    "event_origin_category": "ipeds_only",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            captured_labels: list[str] = []

            def _capture(fig, out_path):
                nonlocal captured_labels
                legend = fig.axes[0].get_legend()
                if legend is not None:
                    captured_labels = [text.get_text() for text in legend.get_texts()]
                fig.savefig(out_path)
                plt.close(fig)

            with mock.patch.object(generalized, "_save_figure", side_effect=_capture):
                out_path = generalized.plot_broad_bin_event_counts_by_year(events, out_dir=tmp)

            self.assertIsNotNone(out_path)
            assert out_path is not None
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.name, "broad_bin_event_counts_by_year_all_degrees.png")
            self.assertEqual(captured_labels, ["Econ -> Quant Econ", "Business -> Management Science"])

    def test_write_broad_bin_treated_control_school_samples_uses_canonical_names(self) -> None:
        matched_pairs = pd.DataFrame(
            [
                {
                    "treated_unitid": 1001,
                    "control_unitid": 2002,
                    "broad_pair_bin": "econ_to_quant_econ",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            crosswalk_path = self._write_crosswalk_fixture(
                [
                    {"UNITID": 1001, "instname": "Example University", "ALIAS": False},
                    {"UNITID": 1001, "instname": "Example U.", "ALIAS": True},
                    {"UNITID": 2002, "instname": "Control College", "ALIAS": False},
                    {"UNITID": 2002, "instname": "Control C.", "ALIAS": True},
                ],
                tmp_path,
            )
            with mock.patch.object(
                generalized,
                "match_treated_to_never_treated",
                return_value=matched_pairs,
            ):
                out_path = generalized.write_broad_bin_treated_control_school_samples(
                    duckdb.connect(),
                    pd.DataFrame(),
                    out_dir=tmp_path,
                    crosswalk_path=crosswalk_path,
                    sample_size=10,
                    seed=42,
                )

            self.assertIsNotNone(out_path)
            assert out_path is not None
            rendered = out_path.read_text()
            self.assertIn("Example University", rendered)
            self.assertIn("Control College", rendered)
            self.assertNotIn("Example U.", rendered)
            self.assertNotIn("Control C.", rendered)
            self.assertNotIn("1001", rendered)
            self.assertNotIn("2002", rendered)

    def test_summarize_did_panel_event_time_simple_means_uses_exact_panel_rows(self) -> None:
        did_panel = pd.DataFrame(
            [
                {"event_t": -1, "treated": 1, "unitid": 10, "opt_share": 0.20, "total_grads": 10, "relabel_type": "econ_to_quant_econ"},
                {"event_t": 0, "treated": 1, "unitid": 10, "opt_share": 0.30, "total_grads": 12, "relabel_type": "econ_to_quant_econ"},
                {"event_t": 0, "treated": 1, "unitid": 11, "opt_share": 0.70, "total_grads": 18, "relabel_type": "business_52_to_52"},
                {"event_t": -1, "treated": 0, "unitid": 20, "opt_share": 0.10, "total_grads": 8, "relabel_type": "econ_to_quant_econ"},
                {"event_t": 0, "treated": 0, "unitid": 20, "opt_share": 0.25, "total_grads": 9, "relabel_type": "econ_to_quant_econ"},
                {"event_t": 0, "treated": 0, "unitid": 21, "opt_share": 0.45, "total_grads": 7, "relabel_type": "business_52_to_52"},
            ]
        )

        summary = generalized.summarize_did_panel_event_time_simple_means(did_panel, yvar="opt_share")

        treated_t0 = summary[(summary["event_t"] == 0) & (summary["treated"] == 1)].iloc[0]
        control_t0 = summary[(summary["event_t"] == 0) & (summary["treated"] == 0)].iloc[0]

        self.assertAlmostEqual(float(treated_t0["mean_outcome"]), 0.50)
        self.assertEqual(int(treated_t0["n_rows"]), 2)
        self.assertEqual(int(treated_t0["n_units"]), 2)
        self.assertAlmostEqual(float(control_t0["mean_outcome"]), 0.35)
        self.assertEqual(int(control_t0["n_rows"]), 2)
        self.assertEqual(int(control_t0["n_units"]), 2)

    def test_laborlunch_event_time_axis_relabels_without_shifting_points(self) -> None:
        rows = pd.DataFrame(
            {
                "event_t": [-5, -2, 0, 4],
                "coef": [-0.1, 0.0, 0.2, 0.3],
                "se": [0.02, 0.0, 0.04, 0.05],
                "reference_event_t": [-2, -2, -2, -2],
            }
        )
        captured: dict[str, object] = {}

        def fake_save(fig: plt.Figure, path: Path) -> None:
            captured["ax"] = fig.axes[0]
            captured["path"] = Path(path)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(generalized, "_save_figure", side_effect=fake_save):
            out_path = generalized.plot_did_event_study_generalized(
                rows,
                yvar="opt_share",
                degree_type="Pooled",
                out_dir=tmp,
            )

            self.assertIsNotNone(out_path)
            ax = captured["ax"]
            self.assertEqual(
                [label.get_text() for label in ax.get_xticklabels()],
                ["-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5"],
            )
            vertical_line_x = [
                float(xdata[0])
                for line in ax.lines
                for xdata in [list(line.get_xdata())]
                if len(xdata) == 2 and float(xdata[0]) == float(xdata[1])
            ]
            self.assertIn(generalized.LABORLUNCH_DI_D_EVENT_LINE_X, vertical_line_x)
            self.assertEqual([float(x) for x in ax.lines[0].get_xdata()], [-5.0, -2.0, 0.0, 4.0])
            self.assertEqual(pd.read_csv(Path(out_path).with_suffix(".csv"))["event_t"].tolist(), [-5, -2, 0, 4])

    def test_compute_did_event_study_supports_unweighted_estimation(self) -> None:
        did_panel = pd.DataFrame(
            [
                {"calendar_year": 2019, "event_t": -1, "treated": 1, "unitid": 101, "opt_share": 0.10, "total_grads": 1000},
                {"calendar_year": 2020, "event_t": 0, "treated": 1, "unitid": 101, "opt_share": 0.90, "total_grads": 1000},
                {"calendar_year": 2019, "event_t": -1, "treated": 1, "unitid": 102, "opt_share": 0.20, "total_grads": 10},
                {"calendar_year": 2020, "event_t": 0, "treated": 1, "unitid": 102, "opt_share": 0.20, "total_grads": 10},
                {"calendar_year": 2019, "event_t": -1, "treated": 0, "unitid": 201, "opt_share": 0.10, "total_grads": 10},
                {"calendar_year": 2020, "event_t": 0, "treated": 0, "unitid": 201, "opt_share": 0.10, "total_grads": 10},
                {"calendar_year": 2019, "event_t": -1, "treated": 0, "unitid": 202, "opt_share": 0.20, "total_grads": 10},
                {"calendar_year": 2020, "event_t": 0, "treated": 0, "unitid": 202, "opt_share": 0.20, "total_grads": 10},
            ]
        )

        weighted = generalized.v2.compute_did_event_study(did_panel=did_panel, yvar="opt_share", use_weights=True)
        unweighted = generalized.v2.compute_did_event_study(did_panel=did_panel, yvar="opt_share", use_weights=False)

        self.assertFalse(weighted.empty)
        self.assertFalse(unweighted.empty)
        weighted_t0 = float(weighted.loc[weighted["event_t"] == 0, "coef"].iloc[0])
        unweighted_t0 = float(unweighted.loc[unweighted["event_t"] == 0, "coef"].iloc[0])
        self.assertNotAlmostEqual(weighted_t0, unweighted_t0, places=6)

    def test_did_formula_omits_reference_event_time_only_from_interactions(self) -> None:
        import patsy

        formula = generalized.v2._did_event_study_formula(yvar="opt_share", reference_event_time=-1)
        design_df = pd.DataFrame(
            [
                {"opt_share": 0.1, "event_t": -2, "treated": 1, "unitid": 1, "grad_year": 2018},
                {"opt_share": 0.2, "event_t": -1, "treated": 1, "unitid": 1, "grad_year": 2019},
                {"opt_share": 0.3, "event_t": 0, "treated": 1, "unitid": 1, "grad_year": 2020},
                {"opt_share": 0.4, "event_t": -2, "treated": 0, "unitid": 2, "grad_year": 2018},
                {"opt_share": 0.5, "event_t": -1, "treated": 0, "unitid": 2, "grad_year": 2019},
                {"opt_share": 0.6, "event_t": 0, "treated": 0, "unitid": 2, "grad_year": 2020},
            ]
        )

        _, design = patsy.dmatrices(formula, design_df, return_type="dataframe")

        self.assertNotIn("C(event_t, Treatment(reference=-1))[-1]:treated", design.columns)
        self.assertNotIn("C(event_t, Treatment(reference=-1))[T.-1]:treated", design.columns)
        self.assertIn("C(grad_year)[T.2019]", design.columns)

    def test_prepare_did_regression_df_builds_interacted_fe_group_for_individual_spec(self) -> None:
        did_panel = pd.DataFrame(
            [
                {
                    "calendar_year": 2019,
                    "event_t": -1,
                    "treated": 1,
                    "unitid": 101,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "degree_type": "Master",
                    "opt_share": 0.0,
                    "total_grads": 1.0,
                },
                {
                    "calendar_year": 2020,
                    "event_t": 0,
                    "treated": 1,
                    "unitid": 101,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "degree_type": "Master",
                    "opt_share": 1.0,
                    "total_grads": 1.0,
                },
                {
                    "calendar_year": 2019,
                    "event_t": -1,
                    "treated": 0,
                    "unitid": 201,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "degree_type": "Master",
                    "opt_share": 0.0,
                    "total_grads": 1.0,
                },
                {
                    "calendar_year": 2020,
                    "event_t": 0,
                    "treated": 0,
                    "unitid": 201,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "degree_type": "Master",
                    "opt_share": 0.0,
                    "total_grads": 1.0,
                },
            ]
        )

        reg_df = generalized._prepare_did_regression_df(
            did_panel,
            yvar="opt_share",
            did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
        )

        self.assertIn("did_fe_group", reg_df.columns)
        self.assertEqual(
            set(reg_df["did_fe_group"].tolist()),
            {"101||econ_to_quant_econ||Master", "201||econ_to_quant_econ||Master"},
        )

    def test_generalized_did_formula_uses_interacted_fe_when_requested(self) -> None:
        reg_df = pd.DataFrame(
            [
                {"did_fe_group": "101||econ_to_quant_econ||Master", "unitid": 101, "grad_year": 2019},
                {"did_fe_group": "201||econ_to_quant_econ||Master", "unitid": 201, "grad_year": 2019},
            ]
        )

        formula = generalized._did_event_study_formula(
            "opt_share",
            reference_event_time=-1,
            reg_df=reg_df,
            did_spec=generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
        )

        self.assertIn("C(did_fe_group)", formula)
        self.assertNotIn("+ C(unitid) +", formula)

    def test_find_did_interaction_param_accepts_pyfixest_style_names(self) -> None:
        params = pd.Series(
            [0.1, 0.2],
            index=[
                "C(event_t,contr.treatment(base=-2))[T.-1]:treated",
                "C(event_t,contr.treatment(base=-2))[T.0]:treated",
            ],
        )

        self.assertEqual(
            generalized._find_did_interaction_param(
                params,
                event_t=0,
                reference_event_time=-2,
            ),
            "C(event_t,contr.treatment(base=-2))[T.0]:treated",
        )

    def test_write_master_broad_bin_did_appendix_creates_manifest_and_plots(self) -> None:
        did_panel = pd.DataFrame(
            [
                {"calendar_year": 2019, "relabel_year": 2020, "event_t": -1, "treated": 1, "unitid": 101, "relabel_type": "business_52_to_52", "total_grads": 10, "opt_share": 0.10, "opt_stem_share": 0.05, "avg_tuition": 10000.0},
                {"calendar_year": 2020, "relabel_year": 2020, "event_t": 0, "treated": 1, "unitid": 101, "relabel_type": "business_52_to_52", "total_grads": 10, "opt_share": 0.40, "opt_stem_share": 0.15, "avg_tuition": 12000.0},
                {"calendar_year": 2019, "relabel_year": 2020, "event_t": -1, "treated": 1, "unitid": 102, "relabel_type": "business_52_to_52", "total_grads": 12, "opt_share": 0.20, "opt_stem_share": 0.10, "avg_tuition": 11000.0},
                {"calendar_year": 2020, "relabel_year": 2020, "event_t": 0, "treated": 1, "unitid": 102, "relabel_type": "business_52_to_52", "total_grads": 12, "opt_share": 0.50, "opt_stem_share": 0.20, "avg_tuition": 13000.0},
                {"calendar_year": 2019, "relabel_year": 2020, "event_t": -1, "treated": 0, "unitid": 201, "relabel_type": "business_52_to_52", "total_grads": 11, "opt_share": 0.10, "opt_stem_share": 0.05, "avg_tuition": 9000.0},
                {"calendar_year": 2020, "relabel_year": 2020, "event_t": 0, "treated": 0, "unitid": 201, "relabel_type": "business_52_to_52", "total_grads": 11, "opt_share": 0.15, "opt_stem_share": 0.05, "avg_tuition": 9500.0},
                {"calendar_year": 2019, "relabel_year": 2020, "event_t": -1, "treated": 0, "unitid": 202, "relabel_type": "business_52_to_52", "total_grads": 9, "opt_share": 0.20, "opt_stem_share": 0.05, "avg_tuition": 9800.0},
                {"calendar_year": 2020, "relabel_year": 2020, "event_t": 0, "treated": 0, "unitid": 202, "relabel_type": "business_52_to_52", "total_grads": 9, "opt_share": 0.25, "opt_stem_share": 0.10, "avg_tuition": 10200.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            outputs = generalized.write_master_broad_bin_did_appendix(did_panel, out_dir=tmp)

            appendix_path = Path(tmp) / "master_broad_bin_did_appendix.md"
            self.assertIn(appendix_path, outputs)
            self.assertTrue(appendix_path.exists())
            appendix_text = appendix_path.read_text()
            self.assertIn("Master Broad-Bin DiD Appendix", appendix_text)
            self.assertIn("Share of F-1s using OPT", appendix_text)

            plot_path = Path(tmp) / "master_broad_bin_did_appendix" / "master_broad_bins_opt_share_did_event_time_never_treated.png"
            csv_path = plot_path.with_suffix(".csv")
            self.assertIn(plot_path, outputs)
            self.assertIn(csv_path, outputs)
            self.assertTrue(plot_path.exists())
            self.assertTrue(csv_path.exists())

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
            self.assertIsNone(args.candidate_path)
        finally:
            sys.argv = original_argv
            if original_ipykernel is None:
                sys.modules.pop("ipykernel", None)
            else:
                sys.modules["ipykernel"] = original_ipykernel

    def test_parse_args_accepts_did_spec_flag(self) -> None:
        original_argv = sys.argv[:]
        original_ipykernel = sys.modules.get("ipykernel")
        try:
            sys.modules.pop("ipykernel", None)
            sys.argv = [
                "relabel_events_generalized.py",
                "--did-spec",
                generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE,
            ]
            args = generalized._parse_args()
            self.assertEqual(args.did_spec, generalized.DID_SPEC_INDIVIDUAL_BIN_DEGREE_FE)
        finally:
            sys.argv = original_argv
            if original_ipykernel is not None:
                sys.modules["ipykernel"] = original_ipykernel

    def test_parse_args_accepts_include_degree_specific_plots_flag(self) -> None:
        original_argv = sys.argv[:]
        original_ipykernel = sys.modules.get("ipykernel")
        try:
            sys.modules.pop("ipykernel", None)
            sys.argv = [
                "relabel_events_generalized.py",
                "--include-degree-specific-plots",
            ]
            args = generalized._parse_args()
            self.assertTrue(args.include_degree_specific_plots)
        finally:
            sys.argv = original_argv
            if original_ipykernel is not None:
                sys.modules["ipykernel"] = original_ipykernel

    def test_write_grouped_did_appendices_skips_degree_level_plots_by_default(self) -> None:
        did_panel = pd.DataFrame(
            [
                {"broad_pair_bin": "econ_to_quant_econ", "relabel_type": "econ_to_quant_econ"},
            ]
        )
        degree_panels = {
            "Bachelor": pd.DataFrame([{"unitid": 1}]),
        }

        with tempfile.TemporaryDirectory() as tmp:
            broad_plot_path = Path(tmp) / "pooled_broad_bin_did_appendix" / "pooled_broad_bins_opt_share_did_event_time_never_treated.png"
            with mock.patch.object(
                generalized,
                "compute_did_event_study_generalized",
                return_value=pd.DataFrame({"event_t": [-2, 0], "coef": [0.0, 0.1], "se": [0.0, 0.05], "reference_event_t": [-2, -2]}),
            ), mock.patch.object(
                generalized,
                "build_did_summary_text",
                return_value="summary",
            ), mock.patch.object(
                generalized,
                "plot_broad_bin_did_event_study_generalized",
                return_value=broad_plot_path,
            ), mock.patch.object(
                generalized,
                "plot_degree_level_did_event_study_generalized",
            ) as degree_plot_mock, mock.patch.object(
                generalized,
                "compute_calendar_year_did_by_relabel_year",
                return_value=pd.DataFrame(),
            ):
                generalized.write_grouped_did_appendices(
                    did_panel,
                    degree_level_panels=degree_panels,
                    out_dir=tmp,
                    yvars=["opt_share"],
                    include_degree_level_plots=False,
                )

            degree_plot_mock.assert_not_called()

    def test_run_degree_plots_keeps_degree_level_appendix_when_degree_specific_plots_are_off(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "degree_type": "Master",
                    "event_origin_category": "ipeds_only",
                    "event_flag": 1,
                    "broad_bin_eligible": 1,
                }
            ]
        )
        pooled_did_panel = pd.DataFrame(
            [
                {
                    "calendar_year": 2018,
                    "event_t": -2,
                    "treated": 1,
                    "unitid": 101,
                    "relabel_year": 2020,
                    "broad_pair_bin": "econ_to_quant_econ",
                    "opt_share": 0.1,
                    "total_grads": 5.0,
                }
            ]
        )
        degree_did_panel = pooled_did_panel.assign(degree_type="Master")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                generalized,
                "plot_broad_bin_event_counts_by_year",
                return_value=None,
            ), mock.patch.object(
                generalized,
                "plot_relabel_broad_bin_degree_year_breakdown",
                return_value=None,
            ), mock.patch.object(
                generalized,
                "write_broad_bin_treated_control_school_samples",
                return_value=None,
            ), mock.patch.object(
                generalized,
                "compute_generalized_did_panel",
                side_effect=[degree_did_panel, pd.DataFrame(), pd.DataFrame(), pooled_did_panel],
            ), mock.patch.object(
                generalized,
                "write_grouped_did_appendices",
                return_value=[],
            ) as appendix_mock, mock.patch.object(
                generalized,
                "plot_event_time_with_control_generalized",
                return_value=(None, None),
            ), mock.patch.object(
                generalized,
                "compute_did_event_study_generalized",
                return_value=pd.DataFrame(),
            ), mock.patch.object(
                generalized,
                "build_did_summary_text",
                return_value=None,
            ):
                generalized.run_degree_plots(
                    duckdb.connect(),
                    pd.DataFrame(),
                    panel,
                    plots_dir=tmp,
                    yvars=["opt_share"],
                    include_degree_specific_plots=False,
                )

            self.assertTrue(appendix_mock.called)
            self.assertTrue(appendix_mock.call_args.kwargs["include_degree_level_plots"])
            degree_level_panels = appendix_mock.call_args.kwargs["degree_level_panels"]
            self.assertEqual(set(degree_level_panels.keys()), {"Bachelor"})

    def test_parse_args_accepts_employer_history_inputs(self) -> None:
        original_argv = sys.argv[:]
        original_ipykernel = sys.modules.get("ipykernel")
        try:
            sys.modules.pop("ipykernel", None)
            sys.argv = [
                "relabel_events_generalized.py",
                "--foia-person-panel-path",
                "/tmp/person_panel.parquet",
                "--employer-match-dir",
                "/tmp/employer_match",
            ]
            args = generalized._parse_args()
            self.assertEqual(args.foia_person_panel_path, "/tmp/person_panel.parquet")
            self.assertEqual(args.employer_match_dir, "/tmp/employer_match")
        finally:
            sys.argv = original_argv
            if original_ipykernel is not None:
                sys.modules["ipykernel"] = original_ipykernel

    def test_employer_history_yvars_have_labels_and_formatting(self) -> None:
        for yvar in generalized.EMPLOYER_HISTORY_YVARS:
            self.assertIn(yvar, generalized.DEFAULT_YVARS)
            self.assertNotEqual(generalized.base.yvar_label(yvar), yvar)

        self.assertEqual(generalized._format_outcome_value("unique_employers", 2.345), "2.35")
        self.assertEqual(generalized._format_outcome_value("auth_employment_tenure_years", 1.236), "1.24")
        self.assertEqual(generalized._format_outcome_value("employer_opt_intensity_pctile", 72.34), "72.3")
        self.assertEqual(generalized._format_outcome_value("internship_count", 1.236), "1.2")
        self.assertEqual(generalized._format_outcome_value("internship_opt_years", 0.456), "0.46")
        self.assertEqual(generalized._format_outcome_value("opt_share", 0.62), "62.0 pp")
        self.assertEqual(generalized._format_outcome_value("cnralt_share_of_ctotalt", 0.41), "41.0 pp")
        self.assertEqual(generalized._format_se_value("unique_opt_cities", 0.456), "0.46")
        self.assertEqual(generalized._format_se_value("employer_opt_intensity_pctile", 3.44), "3.4")
        self.assertEqual(generalized._format_se_value("internship_opt_years", 0.456), "0.46")
        self.assertEqual(generalized._format_se_value("cnralt_share_of_ctotalt", 0.012), "1.2 pp")
        self.assertEqual(
            generalized._outcome_ylabel("cnralt_share_of_ctotalt"),
            "Nonresident share of IPEDS completions (pp)",
        )
        self.assertEqual(generalized._format_percentage_point_tick(0.123, None), "12.3")

    def test_internship_event_time_outcomes_are_per_student_averages(self) -> None:
        opt_usage = pd.DataFrame(
            {
                "calendar_year": [2019, 2019],
                "relabel_year": [2020, 2020],
                "relabel_type": ["business_52_to_52", "business_52_to_52"],
                "degree_type": ["Master", "Master"],
                "total_grads": [10.0, 5.0],
                "opt_users": [2.0, 1.0],
                "opt_stem_users": [1.0, 0.0],
                "total_opt_years": [3.0, 1.0],
                "status_change_users": [1.0, 0.0],
                "unique_employers": [2.0, 1.0],
                "unique_opt_cities": [1.0, 1.0],
                "auth_employment_tenure_years": [0.5, 0.2],
                "employer_opt_intensity_pctile": [50.0, 25.0],
                "total_internships": [4.0, 2.0],
                "total_internship_opt_years": [1.5, 0.75],
                "tuition_total": [100000.0, 50000.0],
                "tuition_ipeds_total": [110000.0, 55000.0],
                "ctotalt": [20.0, 10.0],
                "cnralt": [12.0, 6.0],
            }
        )

        event_time = generalized.compute_opt_usage_event_time_generalized(opt_usage)

        self.assertEqual(len(event_time), 1)
        self.assertAlmostEqual(float(event_time.loc[0, "internship_count"]), 6.0 / 15.0)
        self.assertAlmostEqual(float(event_time.loc[0, "internship_opt_years"]), 2.25 / 15.0)

    def test_student_employer_outcomes_identify_pre_program_year_internships(self) -> None:
        con = duckdb.connect()
        try:
            con.execute(
                """
                CREATE OR REPLACE TEMP VIEW flagged AS
                SELECT 0 AS original_row_num, 's1' AS student_id, DATE '2020-05-15' AS program_end_date
                UNION ALL
                SELECT 1 AS original_row_num, 's2' AS student_id, DATE '2021-05-15' AS program_end_date
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP VIEW foia_person_employer_history AS
                SELECT
                    0 AS original_row_num,
                    'firm-a' AS foia_firm_uid,
                    'seattle' AS employer_city_clean,
                    'prior-spell' AS spell_key,
                    DATE '2019-06-01' AS spell_start_date,
                    DATE '2019-09-01' AS spell_end_date
                UNION ALL
                SELECT
                    0 AS original_row_num,
                    'firm-b' AS foia_firm_uid,
                    'seattle' AS employer_city_clean,
                    'same-year-spell' AS spell_key,
                    DATE '2020-01-15' AS spell_start_date,
                    DATE '2020-04-15' AS spell_end_date
                UNION ALL
                SELECT
                    1 AS original_row_num,
                    'firm-c' AS foia_firm_uid,
                    'austin' AS employer_city_clean,
                    'open-prior-spell' AS spell_key,
                    DATE '2020-08-01' AS spell_start_date,
                    NULL AS spell_end_date
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP VIEW foia_employer_intensity_pctiles AS
                SELECT 'firm-a' AS foia_firm_uid, 2 AS employer_opt_person_count, 50.0 AS employer_opt_intensity_pctile
                """
            )
            ctes = generalized._student_employer_outcome_ctes(
                source_table="flagged",
                group_cols=["student_id"],
            )
            rows = con.execute(
                f"""
                WITH student_level_base AS (
                    SELECT DISTINCT student_id FROM flagged
                ),
                {ctes}
                final AS (
                    SELECT student_id, internship_count, internship_opt_years
                    FROM student_level
                )
                SELECT student_id, internship_count, internship_opt_years
                FROM final
                ORDER BY student_id
                """
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(rows[0][0], "s1")
        self.assertEqual(rows[0][1], 1)
        self.assertAlmostEqual(rows[0][2], 92.0 / 365.25)
        self.assertEqual(rows[1], ("s2", 1, 0.0))

    def test_load_foia_base_stages_employer_history_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            foia_path = tmpdir / "foia.parquet"
            inst_cw_path = tmpdir / "inst_cw.parquet"
            person_panel_path = tmpdir / "foia_person_panel.parquet"
            employer_match_dir = tmpdir / "employer_match"
            employer_match_dir.mkdir()

            pd.DataFrame(
                [
                    {
                        "school_name": "Example University",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2020-05-15",
                        "student_key": "s1",
                        "tuition__fees": 10000.0,
                        "student_edu_level_desc": "MASTER'S",
                        "requested_status": "H1B",
                        "year": 2020,
                        "opt_authorization_end_date": "2021-05-14",
                    },
                    {
                        "school_name": "Example University",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2020-05-15",
                        "student_key": "s2",
                        "tuition__fees": 12000.0,
                        "student_edu_level_desc": "MASTER'S",
                        "requested_status": None,
                        "year": 2020,
                        "opt_authorization_end_date": "2021-06-01",
                    },
                    {
                        "school_name": "Example University",
                        "major_1_cip_code": "450601",
                        "program_end_date": "2020-05-15",
                        "student_key": "s3",
                        "tuition__fees": 9000.0,
                        "student_edu_level_desc": "MASTER'S",
                        "requested_status": None,
                        "year": 2020,
                        "opt_authorization_end_date": "2020-12-31",
                    },
                ]
            ).to_parquet(foia_path, index=False)
            pd.DataFrame([{"school_name": "Example University", "unitid": 1001}]).to_parquet(inst_cw_path, index=False)
            pd.DataFrame(
                [
                    {
                        "original_row_num": 0,
                        "person_id": "p1",
                        "spell_key": "spell-1",
                        "employer_name": "Acme Inc.",
                        "employer_city": "Seattle",
                        "employer_state": "WA",
                        "employer_zip_code": "98101",
                        "employment_opt_type": "STEM",
                        "authorization_start_date": "2020-06-01",
                        "authorization_end_date": "2021-05-31",
                        "opt_authorization_start_date": "2020-06-05",
                        "opt_authorization_end_date": "2021-05-31",
                        "opt_employer_start_date": "2020-06-10",
                        "opt_employer_end_date": "2021-05-30",
                    },
                    {
                        "original_row_num": 1,
                        "person_id": "p2",
                        "spell_key": "spell-2",
                        "employer_name": "Acme Inc.",
                        "employer_city": "Seattle",
                        "employer_state": "WA",
                        "employer_zip_code": "98101",
                        "employment_opt_type": "POST",
                        "authorization_start_date": "2020-07-01",
                        "authorization_end_date": "2021-06-30",
                        "opt_authorization_start_date": "2020-07-03",
                        "opt_authorization_end_date": "2021-06-30",
                        "opt_employer_start_date": None,
                        "opt_employer_end_date": None,
                    },
                    {
                        "original_row_num": 2,
                        "person_id": "p3",
                        "spell_key": "spell-3",
                        "employer_name": "Beta LLC",
                        "employer_city": "Austin",
                        "employer_state": "TX",
                        "employer_zip_code": "78701",
                        "employment_opt_type": "POST",
                        "authorization_start_date": "2020-08-01",
                        "authorization_end_date": "2020-12-31",
                        "opt_authorization_start_date": "2020-08-02",
                        "opt_authorization_end_date": "2020-12-31",
                        "opt_employer_start_date": None,
                        "opt_employer_end_date": None,
                    },
                ]
            ).to_parquet(person_panel_path, index=False)
            pd.DataFrame(
                [
                    {
                        "foia_row_uid": "row-1",
                        "raw_name_example": "Acme Inc.",
                        "row_name_clean": "acme",
                        "row_city_clean": "seattle",
                        "row_state_clean": "WA",
                        "row_zip_clean": "98101",
                    },
                    {
                        "foia_row_uid": "row-2",
                        "raw_name_example": "Beta LLC",
                        "row_name_clean": "beta",
                        "row_city_clean": "austin",
                        "row_state_clean": "TX",
                        "row_zip_clean": "78701",
                    },
                ]
            ).to_csv(employer_match_dir / "foia_row_entities.csv", index=False)
            pd.DataFrame(
                [
                    {"foia_row_uid": "row-1", "foia_firm_uid": "firm-a"},
                    {"foia_row_uid": "row-2", "foia_firm_uid": "firm-b"},
                ]
            ).to_csv(employer_match_dir / "foia_row_to_firm.csv", index=False)

            con = duckdb.connect()
            schema = generalized._load_foia_base(
                con,
                foia_path=foia_path,
                inst_cw_path=inst_cw_path,
                foia_person_panel_path=person_panel_path,
                employer_match_dir=employer_match_dir,
            )

            self.assertEqual(schema["foia_student_col"], "student_key")
            history = con.execute(
                """
                SELECT original_row_num, student_id, foia_firm_uid, CAST(spell_start_date AS VARCHAR) AS spell_start_date
                FROM foia_person_employer_history
                ORDER BY original_row_num
                """
            ).fetchall()
            self.assertEqual(
                history,
                [
                    (0, "s1", "firm-a", "2020-06-10"),
                    (1, "s2", "firm-a", "2020-07-03"),
                    (2, "s3", "firm-b", "2020-08-02"),
                ],
            )

            pctiles = con.execute(
                """
                SELECT foia_firm_uid, employer_opt_person_count
                FROM foia_employer_intensity_pctiles
                ORDER BY foia_firm_uid
                """
            ).fetchall()
            self.assertEqual(pctiles, [("firm-a", 2), ("firm-b", 1)])

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
                    {"unitid": 100, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270501, "cipcode_lab": "27.0501-Statistics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 100, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270501, "cipcode_lab": "27.0501-Statistics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 100, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270501, "cipcode_lab": "27.0501-Statistics, General", "ctotalt": 18, "cnralt": 11, "share_intl": 0.7},
                ],
                tmpdir,
            )
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertEqual(len(events), 1)
            self.assertEqual(int(events.loc[0, "unitid"]), 100)
            self.assertEqual(int(events.loc[0, "relabel_year"]), 2021)
            self.assertEqual(events.loc[0, "event_source_cip6"], "520801")
            self.assertEqual(events.loc[0, "target_cip6"], "270501")
            self.assertEqual(events.loc[0, "degree_type"], "Master")

    def test_detect_ipeds_relabels_uses_broad_source_aggregate_not_single_cip_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 101, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 20, "cnralt": 10, "share_intl": 0.6},
                    {"unitid": 101, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 20, "cnralt": 10, "share_intl": 0.6},
                    {"unitid": 101, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 20, "cnralt": 10, "share_intl": 0.6},
                    {"unitid": 101, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.6},
                    {"unitid": 101, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.6},
                    {"unitid": 101, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450601, "cipcode_lab": "45.0601-Economics, General", "ctotalt": 4, "cnralt": 2, "share_intl": 0.6},
                    {"unitid": 101, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450699, "cipcode_lab": "45.0699-Economics, Other", "ctotalt": 16, "cnralt": 8, "share_intl": 0.6},
                    {"unitid": 101, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450699, "cipcode_lab": "45.0699-Economics, Other", "ctotalt": 16, "cnralt": 8, "share_intl": 0.6},
                    {"unitid": 101, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450699, "cipcode_lab": "45.0699-Economics, Other", "ctotalt": 16, "cnralt": 8, "share_intl": 0.6},
                    {"unitid": 101, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450603, "cipcode_lab": "45.0603-Econometrics and Quantitative Economics", "ctotalt": 12, "cnralt": 6, "share_intl": 0.6},
                    {"unitid": 101, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450603, "cipcode_lab": "45.0603-Econometrics and Quantitative Economics", "ctotalt": 12, "cnralt": 6, "share_intl": 0.6},
                    {"unitid": 101, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 450603, "cipcode_lab": "45.0603-Econometrics and Quantitative Economics", "ctotalt": 12, "cnralt": 6, "share_intl": 0.6},
                ],
                tmpdir,
            )
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertTrue(events.empty)

    def test_detect_ipeds_relabels_drops_master_when_matching_phd_event_lags_2_to_4_years(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(self._econ_master_phd_guard_rows(), tmpdir)
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertEqual(len(events), 1)
            self.assertEqual(events.loc[0, "degree_type"], "Doctor")
            self.assertEqual(int(events.loc[0, "relabel_year"]), 2024)
            self.assertEqual(events.loc[0, "event_source_cip6"], "450601")
            self.assertEqual(events.loc[0, "target_cip6"], "450603")

    def test_detect_ipeds_relabels_keeps_master_when_matching_phd_event_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                self._econ_master_phd_guard_rows(
                    unitid=301,
                    doctor_source_pre=12,
                    doctor_source_curr=6,
                    doctor_target_curr=6,
                ),
                tmpdir,
            )
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertEqual(len(events), 2)
            self.assertEqual(set(events["degree_type"]), {"Master", "Doctor"})
            self.assertEqual(set(events["relabel_year"].astype(int)), {2021, 2024})

    def test_detect_ipeds_relabels_drops_master_when_later_doctorate_shift_is_same_broad_bin_but_different_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                self._econ_master_phd_guard_rows(
                    unitid=302,
                    master_source_cip=450699,
                    doctor_source_cip=450601,
                ),
                tmpdir,
            )
            con = duckdb.connect()

            events = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)

            self.assertEqual(len(events), 1)
            self.assertEqual(events.loc[0, "degree_type"], "Doctor")
            self.assertEqual(int(events.loc[0, "relabel_year"]), 2024)
            self.assertEqual(events.loc[0, "event_source_cip6"], "450601")
            self.assertEqual(events.loc[0, "target_cip6"], "450603")

    def test_broad_bin_control_matching_can_use_different_exact_source_cip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 9, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                ],
                tmpdir,
            )
            panel = pd.DataFrame(
                [
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2018,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 8.0,
                        "source_total_prev": 0.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2019,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 9.0,
                        "source_total_prev": 8.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2020,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 10.0,
                        "source_total_prev": 9.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2021,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 4.0,
                        "source_total_prev": 10.0,
                        "event_flag": 1,
                        "event_origin_category": "ipeds_only",
                    },
                ]
            )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(
                con,
                panel,
                ipeds_path=ipeds_path,
                control_guard_thresholds={
                    "min_source_pre_count": 5.0,
                    "min_source_drop_count": 3.0,
                    "min_source_drop_fraction": 0.35,
                    "min_target_growth_share_of_source_drop_threshold": 0.40,
                    "max_net_loss_share_of_source_drop": 0.75,
                    "min_persistent_source_drop_fraction": 0.15,
                    "min_persistent_target_gain_share_of_source_drop_threshold": 0.15,
                },
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches.loc[0, "broad_pair_bin"], "business_52_to_52")
            self.assertEqual(int(matches.loc[0, "control_unitid"]), 902)
            self.assertEqual(float(matches.loc[0, "treated_source_group_pre_size"]), 10.0)
            self.assertEqual(float(matches.loc[0, "control_source_group_pre_size"]), 11.0)
            self.assertAlmostEqual(float(matches.loc[0, "treated_pre_avg_level"]), 9.0)
            self.assertAlmostEqual(float(matches.loc[0, "control_pre_avg_level"]), 10.0)
            self.assertAlmostEqual(float(matches.loc[0, "treated_pre_avg_growth"]), 1.0)
            self.assertAlmostEqual(float(matches.loc[0, "control_pre_avg_growth"]), 1.0)

    def test_control_matching_excludes_treated_unitids_only_within_degree_broad_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 9, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                    {"unitid": 903, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 20, "cnralt": 10, "share_intl": 0.4},
                    {"unitid": 903, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 20, "cnralt": 10, "share_intl": 0.4},
                    {"unitid": 903, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 20, "cnralt": 10, "share_intl": 0.4},
                ],
                tmpdir,
            )
            rows: list[dict[str, object]] = []
            for year, source_total in [(2018, 8.0), (2019, 9.0), (2020, 10.0), (2021, 4.0)]:
                rows.append(
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": source_total,
                        "source_total_prev": 10.0 if year == 2021 else 0.0,
                        "event_flag": 1 if year == 2021 else 0,
                        "event_origin_category": "ipeds_only",
                    }
                )
            for year, source_total in [(2018, 5.0), (2019, 6.0), (2020, 7.0), (2021, 2.0)]:
                rows.append(
                    {
                        "unitid": 902,
                        "awlevel": 5,
                        "degree_type": "Bachelor",
                        "relabel_year": 2021,
                        "year": year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": source_total,
                        "source_total_prev": 7.0 if year == 2021 else 0.0,
                        "event_flag": 1 if year == 2021 else 0,
                        "event_origin_category": "ipeds_only",
                    }
                )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(con, pd.DataFrame(rows), ipeds_path=ipeds_path)
            master_match = matches.loc[matches["treated_unitid"].eq(901)].reset_index(drop=True)

            self.assertEqual(len(master_match), 1)
            self.assertEqual(int(master_match.loc[0, "control_unitid"]), 902)
            self.assertEqual(master_match.loc[0, "degree_type"], "Master")

    def test_control_matching_excludes_controls_with_same_year_source_drop_and_target_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 9, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                    {"unitid": 902, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 902, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                    {"unitid": 903, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 9, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 903, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                    {"unitid": 903, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                ],
                tmpdir,
            )
            panel = pd.DataFrame(
                [
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": source_total,
                        "source_total_prev": 10.0 if year == 2021 else 0.0,
                        "event_flag": 1 if year == 2021 else 0,
                        "event_origin_category": "ipeds_only",
                    }
                    for year, source_total in [(2018, 8.0), (2019, 9.0), (2020, 10.0), (2021, 4.0)]
                ]
            )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(
                con,
                panel,
                ipeds_path=ipeds_path,
                control_guard_thresholds={
                    "min_source_pre_count": 5.0,
                    "min_source_drop_count": 3.0,
                    "min_source_drop_fraction": 0.35,
                    "min_target_growth_share_of_source_drop_threshold": 0.40,
                    "max_net_loss_share_of_source_drop": 0.75,
                    "min_persistent_source_drop_fraction": 0.15,
                    "min_persistent_target_gain_share_of_source_drop_threshold": 0.15,
                },
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(int(matches.loc[0, "control_unitid"]), 903)

    def test_control_matching_keeps_controls_with_target_gain_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 9, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                    {"unitid": 902, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 7, "share_intl": 0.4},
                    {"unitid": 902, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 2, "cnralt": 1, "share_intl": 0.4},
                    {"unitid": 903, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 13, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 13, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 13, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 13, "cnralt": 6, "share_intl": 0.4},
                ],
                tmpdir,
            )
            panel = pd.DataFrame(
                [
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": source_total,
                        "source_total_prev": 10.0 if year == 2021 else 0.0,
                        "event_flag": 1 if year == 2021 else 0,
                        "event_origin_category": "ipeds_only",
                    }
                    for year, source_total in [(2018, 8.0), (2019, 9.0), (2020, 10.0), (2021, 4.0)]
                ]
            )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(
                con,
                panel,
                ipeds_path=ipeds_path,
                control_guard_thresholds={
                    "min_source_pre_count": 5.0,
                    "min_source_drop_count": 3.0,
                    "min_source_drop_fraction": 0.35,
                    "min_target_growth_share_of_source_drop_threshold": 0.40,
                    "max_net_loss_share_of_source_drop": 0.75,
                    "min_persistent_source_drop_fraction": 0.15,
                    "min_persistent_target_gain_share_of_source_drop_threshold": 0.15,
                },
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(int(matches.loc[0, "control_unitid"]), 902)

    def test_control_matching_excludes_controls_with_relabel_like_event_in_other_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2016, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2017, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 10, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 5, "cnralt": 3, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 4, "cnralt": 2, "share_intl": 0.4},
                    {"unitid": 903, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 12, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 12, "cnralt": 6, "share_intl": 0.4},
                    {"unitid": 903, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 12, "cnralt": 6, "share_intl": 0.4},
                ],
                tmpdir,
            )
            panel = pd.DataFrame(
                [
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": year,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": source_total,
                        "source_total_prev": 10.0 if year == 2021 else 0.0,
                        "event_flag": 1 if year == 2021 else 0,
                        "event_origin_category": "ipeds_only",
                    }
                    for year, source_total in [(2018, 8.0), (2019, 9.0), (2020, 10.0), (2021, 4.0)]
                ]
            )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(con, panel, ipeds_path=ipeds_path)

            self.assertEqual(len(matches), 1)
            self.assertEqual(int(matches.loc[0, "control_unitid"]), 903)

    def test_broad_bin_control_matching_uses_preperiod_level_and_growth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 902, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 902, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520301, "cipcode_lab": "52.0301-Accounting", "ctotalt": 11, "cnralt": 5, "share_intl": 0.4},
                    {"unitid": 903, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520401, "cipcode_lab": "52.0401-Administrative Assistant and Secretarial Science, General", "ctotalt": 6, "cnralt": 3, "share_intl": 0.4},
                    {"unitid": 903, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520401, "cipcode_lab": "52.0401-Administrative Assistant and Secretarial Science, General", "ctotalt": 8, "cnralt": 4, "share_intl": 0.4},
                    {"unitid": 903, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520401, "cipcode_lab": "52.0401-Administrative Assistant and Secretarial Science, General", "ctotalt": 10, "cnralt": 5, "share_intl": 0.4},
                ],
                tmpdir,
            )
            panel = pd.DataFrame(
                [
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2018,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 8.0,
                        "source_total_prev": 0.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2019,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 10.0,
                        "source_total_prev": 8.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2020,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 12.0,
                        "source_total_prev": 10.0,
                        "event_flag": 0,
                        "event_origin_category": "ipeds_only",
                    },
                    {
                        "unitid": 901,
                        "awlevel": 7,
                        "degree_type": "Master",
                        "relabel_year": 2021,
                        "year": 2021,
                        "relabel_type": "business_52_to_52",
                        "broad_pair_bin": "business_52_to_52",
                        "broad_bin_eligible": 1,
                        "event_source_cip6": "520201",
                        "target_cip6": "521301",
                        "source_total": 4.0,
                        "source_total_prev": 12.0,
                        "event_flag": 1,
                        "event_origin_category": "ipeds_only",
                    },
                ]
            )
            con = duckdb.connect()

            matches = generalized.match_treated_to_never_treated(con, panel, ipeds_path=ipeds_path)

            self.assertEqual(len(matches), 1)
            self.assertEqual(int(matches.loc[0, "control_unitid"]), 903)
            self.assertAlmostEqual(float(matches.loc[0, "treated_pre_avg_level"]), 10.0)
            self.assertAlmostEqual(float(matches.loc[0, "treated_pre_avg_growth"]), 2.0)
            self.assertAlmostEqual(float(matches.loc[0, "control_pre_avg_level"]), 8.0)
            self.assertAlmostEqual(float(matches.loc[0, "control_pre_avg_growth"]), 2.0)
            self.assertAlmostEqual(float(matches.loc[0, "abs_pre_growth_diff"]), 0.0)

    def test_strict_scan_prefers_best_allowed_target_with_candidate_family_restriction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(
                [
                    {"unitid": 110, "year": 2018, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 110, "year": 2019, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 110, "year": 2020, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 20, "cnralt": 12, "share_intl": 0.6},
                    {"unitid": 110, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 110, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 110, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 520201, "cipcode_lab": "52.0201-Business Administration and Management, General", "ctotalt": 4, "cnralt": 3, "share_intl": 0.6},
                    {"unitid": 110, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 10, "cnralt": 7, "share_intl": 0.7},
                    {"unitid": 110, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 10, "cnralt": 7, "share_intl": 0.7},
                    {"unitid": 110, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 521301, "cipcode_lab": "52.1301-Management Science", "ctotalt": 10, "cnralt": 7, "share_intl": 0.7},
                    {"unitid": 110, "year": 2021, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 16, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 110, "year": 2022, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 16, "cnralt": 11, "share_intl": 0.7},
                    {"unitid": 110, "year": 2023, "awlevel": 7, "awlevel_group": "Master", "cipcode": 270301, "cipcode_lab": "27.0301-Applied Mathematics, General", "ctotalt": 16, "cnralt": 11, "share_intl": 0.7},
                ],
                tmpdir,
            )
            allowed_configs = generalized.derive_allowable_pair_configs(
                pd.DataFrame(
                    [
                        {
                            "candidate_program_desc": "Full-Time MBA",
                            "candidate_source_cip6_hint": pd.NA,
                            "candidate_target_cip6_hint": pd.NA,
                        }
                    ]
                )
            )
            con = duckdb.connect()

            unrestricted = generalized.detect_ipeds_relabels(con, ipeds_path=ipeds_path)
            restricted = generalized.detect_ipeds_relabels(
                con,
                ipeds_path=ipeds_path,
                allowed_pair_configs=allowed_configs,
            )

            self.assertEqual(len(unrestricted), 1)
            self.assertEqual(unrestricted.loc[0, "target_cip6"], "521301")
            self.assertEqual(len(restricted), 1)
            self.assertEqual(restricted.loc[0, "event_source_cip6"], "520201")
            self.assertEqual(restricted.loc[0, "target_cip6"], "521301")

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

    def test_relaxed_candidate_verification_drops_master_when_matching_phd_event_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            ipeds_path = self._write_ipeds_fixture(self._econ_master_phd_guard_rows(), tmpdir)
            crosswalk_path = self._write_crosswalk_fixture(
                [{"UNITID": 300, "instname": "Example State University", "ALIAS": False}],
                tmpdir,
            )
            candidate_path = tmpdir / "candidates.csv"
            pd.DataFrame(
                [
                    {
                        "school_name": "Example State University",
                        "approx_year": 2021,
                        "description": "economics major",
                        "degree": "masters",
                        "initial_cip": "45.0601",
                        "new_cip": "45.0603",
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
            self.assertEqual(audit.loc[0, "verification_notes"], "no_relaxed_ipeds_match")

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

    def test_add_did_summary_text_places_annotation_inside_axes(self) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            generalized._add_did_summary_text(
                ax,
                "Baseline mean (t <= -2): 62.0 pp\nTreat x Post (t >= -1): 2.3 pp (NA)\nEffect size: 3.8%",
            )
            self.assertEqual(len(ax.texts), 1)
            annotation = ax.texts[0]
            self.assertEqual(annotation.get_position(), (0.02, 0.98))
            self.assertEqual(annotation.get_ha(), "left")
            self.assertEqual(annotation.get_va(), "top")
            self.assertIs(annotation.get_transform(), ax.transAxes)
            self.assertIsNotNone(annotation.get_bbox_patch())
        finally:
            plt.close(fig)

    def test_add_did_summary_text_skips_empty_summary(self) -> None:
        fig, ax = plt.subplots(figsize=(4, 3))
        try:
            generalized._add_did_summary_text(ax, None)
            self.assertEqual(len(ax.texts), 0)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
